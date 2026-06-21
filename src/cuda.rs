//! Optional native CUDA forest inference (`cudarc` + nvrtc).
//!
//! Feature-gated (`cuda`), native NVIDIA only — never in the wasm/default wheels.
//! Sibling to the wgpu `gpu` backend, same multi-output model: `out_dim = 1` for
//! the mean leaf value (regression / boosting round), `out_dim = n_classes` for
//! the mean leaf distribution (class probabilities). One nvrtc-compiled kernel,
//! one row per CUDA thread accumulating an `out_dim`-length vector. f32 (parity
//! with CPU exact to f32 rounding). `CudaForest::new` returns `None` when CUDA
//! is unavailable so callers fall back to CPU.

use crate::tree::TreeNode;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

/// Max supported output width (n_classes). Must match `acc[]` in the kernel.
pub const MAX_OUT: usize = 32;

#[derive(Default)]
struct Flat {
    feature: Vec<i32>, // -1 => leaf
    threshold: Vec<f32>,
    left: Vec<u32>,
    right: Vec<u32>,
    values: Vec<f32>, // n_nodes * out_dim; only leaf slots are meaningful
    roots: Vec<u32>,
}

fn flatten_node<F: Fn(&TreeNode, &mut [f32])>(
    node: &TreeNode,
    f: &mut Flat,
    out_dim: usize,
    leaf: &F,
) -> u32 {
    let id = f.feature.len() as u32;
    f.feature.push(0);
    f.threshold.push(0.0);
    f.left.push(0);
    f.right.push(0);
    let base = f.values.len();
    f.values.resize(base + out_dim, 0.0);
    if node.left.is_some() && node.right.is_some() {
        f.feature[id as usize] = node.feature as i32;
        f.threshold[id as usize] = node.threshold as f32;
        let l = flatten_node(node.left.as_ref().unwrap(), f, out_dim, leaf);
        let r = flatten_node(node.right.as_ref().unwrap(), f, out_dim, leaf);
        f.left[id as usize] = l;
        f.right[id as usize] = r;
    } else {
        f.feature[id as usize] = -1;
        leaf(node, &mut f.values[base..base + out_dim]);
    }
    id
}

const KERNEL: &str = r#"
#define MAX_OUT 32
extern "C" __global__ void predict(
    const float* x, const int* feat, const float* thr,
    const unsigned* left, const unsigned* right, const float* values, const unsigned* roots,
    const int n_features, const int n_trees, const int out_dim, const int n, float* out)
{
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if (s >= n) return;
    int base = s*n_features;
    float acc[MAX_OUT];
    for (int k=0;k<out_dim;++k) acc[k]=0.0f;
    for (int t=0; t<n_trees; ++t) {
        unsigned node = roots[t];
        for(;;){ int fe=feat[node]; if(fe<0) break;
            float xv=x[base+fe]; node = (xv<=thr[node])? left[node]:right[node]; }
        unsigned vb = node*out_dim;
        for (int k=0;k<out_dim;++k) acc[k]+=values[vb+k];
    }
    int ob = s*out_dim; float inv = 1.0f/(float)n_trees;
    for (int k=0;k<out_dim;++k) out[ob+k]=acc[k]*inv;
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
    values: CudaSlice<f32>,
    roots: CudaSlice<u32>,
    n_features: i32,
    n_trees: i32,
    out_dim: i32,
}

impl CudaForest {
    /// Flatten `trees` (each leaf producing an `out_dim`-length vector via
    /// `leaf`) and upload to CUDA device 0. `None` if `out_dim > MAX_OUT` or
    /// CUDA is unavailable.
    pub fn new<F: Fn(&TreeNode, &mut [f32])>(
        trees: &[TreeNode],
        n_features: usize,
        out_dim: usize,
        leaf: F,
    ) -> Option<CudaForest> {
        if out_dim == 0 || out_dim > MAX_OUT {
            return None;
        }
        let mut flat = Flat::default();
        for t in trees {
            let r = flatten_node(t, &mut flat, out_dim, &leaf);
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
            values: stream.memcpy_stod(&flat.values).ok()?,
            roots: stream.memcpy_stod(&flat.roots).ok()?,
            n_features: n_features as i32,
            n_trees: trees.len() as i32,
            out_dim: out_dim as i32,
            stream,
            func,
            _ctx: ctx,
            _module: module,
        })
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim as usize
    }

    /// Mean over trees of the leaf vector, per row. `x` is row-major f32 of shape
    /// `n_rows x n_features`; returns `n_rows * out_dim` f32 (row-major).
    pub fn predict(&self, x: &[f32], n_rows: usize) -> Vec<f32> {
        let x_d = self.stream.memcpy_stod(x).unwrap();
        let out_len = n_rows * self.out_dim as usize;
        let mut out = self.stream.alloc_zeros::<f32>(out_len).unwrap();
        let ni = n_rows as i32;
        let cfg = LaunchConfig::for_num_elems(n_rows as u32);
        let mut b = self.stream.launch_builder(&self.func);
        b.arg(&x_d)
            .arg(&self.feature)
            .arg(&self.threshold)
            .arg(&self.left)
            .arg(&self.right)
            .arg(&self.values)
            .arg(&self.roots)
            .arg(&self.n_features)
            .arg(&self.n_trees)
            .arg(&self.out_dim)
            .arg(&ni)
            .arg(&mut out);
        unsafe {
            b.launch(cfg).unwrap();
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&out).unwrap()
    }
}
