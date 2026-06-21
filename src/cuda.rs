//! Optional native CUDA forest inference (`cudarc` + nvrtc).
//!
//! Feature-gated (`cuda`), native NVIDIA only — never compiled into the wasm or
//! default wheels. Sibling to the wgpu `gpu` backend: the recursive `TreeNode`
//! forest is flattened to struct-of-arrays, uploaded once, and traversed one
//! row per CUDA thread by a kernel compiled at runtime via nvrtc. f32 compute
//! (parity with the CPU `traverse` is exact). `CudaForest::new` returns `None`
//! when no CUDA device is available so callers fall back to CPU.
//!
//! Verified on an A100 (Colab): ~130-160x kernel-only / ~45-51x end-to-end over
//! a 12-core CPU on a 200-tree x depth-8 forest.

use crate::tree::TreeNode;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

#[derive(Default)]
struct Flat {
    feature: Vec<i32>, // -1 => leaf
    threshold: Vec<f32>,
    left: Vec<u32>,
    right: Vec<u32>,
    value: Vec<f32>,
    roots: Vec<u32>,
}

fn flatten_node(node: &TreeNode, f: &mut Flat) -> u32 {
    let id = f.feature.len() as u32;
    if node.left.is_some() && node.right.is_some() {
        f.feature.push(node.feature as i32);
        f.threshold.push(node.threshold as f32);
        f.left.push(0);
        f.right.push(0);
        f.value.push(0.0);
        let l = flatten_node(node.left.as_ref().unwrap(), f);
        let r = flatten_node(node.right.as_ref().unwrap(), f);
        f.left[id as usize] = l;
        f.right[id as usize] = r;
    } else {
        f.feature.push(-1);
        f.threshold.push(0.0);
        f.left.push(0);
        f.right.push(0);
        f.value.push(node.value as f32);
    }
    id
}

const KERNEL: &str = r#"
extern "C" __global__ void predict(
    const float* x, const int* feat, const float* thr,
    const unsigned* left, const unsigned* right, const float* val, const unsigned* roots,
    const int n_features, const int n_trees, const int n, float* out)
{
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if (s >= n) return;
    int base = s*n_features; float acc = 0.0f;
    for (int t=0; t<n_trees; ++t) {
        unsigned node = roots[t];
        for(;;){ int fe=feat[node]; if(fe<0){acc+=val[node];break;}
            float xv=x[base+fe]; node = (xv<=thr[node])? left[node]:right[node]; }
    }
    out[s] = acc/(float)n_trees;
}
"#;

/// A forest uploaded to a CUDA device, ready for repeated batched inference.
pub struct CudaForest {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    _ctx: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    feature: CudaSlice<i32>,
    threshold: CudaSlice<f32>,
    left: CudaSlice<u32>,
    right: CudaSlice<u32>,
    value: CudaSlice<f32>,
    roots: CudaSlice<u32>,
    n_features: i32,
    n_trees: i32,
}

impl CudaForest {
    /// Flatten `trees` and upload to CUDA device 0. Returns `None` if CUDA is
    /// unavailable (no driver/device, or kernel compile/upload failed).
    pub fn new(trees: &[TreeNode], n_features: usize) -> Option<CudaForest> {
        let mut flat = Flat::default();
        for t in trees {
            let r = flatten_node(t, &mut flat);
            flat.roots.push(r);
        }

        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(compile_ptx(KERNEL).ok()?).ok()?;
        let func = module.load_function("predict").ok()?;

        Some(CudaForest {
            feature: stream.memcpy_stod(&flat.feature).ok()?,
            threshold: stream.memcpy_stod(&flat.threshold).ok()?,
            left: stream.memcpy_stod(&flat.left).ok()?,
            right: stream.memcpy_stod(&flat.right).ok()?,
            value: stream.memcpy_stod(&flat.value).ok()?,
            roots: stream.memcpy_stod(&flat.roots).ok()?,
            n_features: n_features as i32,
            n_trees: trees.len() as i32,
            stream,
            func,
            _ctx: ctx,
            _module: module,
        })
    }

    /// Mean of `traverse` over all trees, one value per row. `x` is row-major
    /// f32 of shape `n_rows x n_features`.
    pub fn predict_avg(&self, x: &[f32], n_rows: usize) -> Vec<f32> {
        let x_d = self.stream.memcpy_stod(x).unwrap();
        let mut out = self.stream.alloc_zeros::<f32>(n_rows).unwrap();
        let ni = n_rows as i32;
        let cfg = LaunchConfig::for_num_elems(n_rows as u32);
        let mut b = self.stream.launch_builder(&self.func);
        b.arg(&x_d)
            .arg(&self.feature)
            .arg(&self.threshold)
            .arg(&self.left)
            .arg(&self.right)
            .arg(&self.value)
            .arg(&self.roots)
            .arg(&self.n_features)
            .arg(&self.n_trees)
            .arg(&ni)
            .arg(&mut out);
        unsafe {
            b.launch(cfg).unwrap();
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&out).unwrap()
    }
}
