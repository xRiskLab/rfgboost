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

        Self::from_flat(flat, n_features, out_dim)
    }

    /// Boosting helper: out_dim=1, each tree's leaf values scaled by `weights[t]`,
    /// then the standard mean kernel — folds per-round learning-rate / tree-count
    /// factors into one forest (the caller adds the bias afterward).
    pub fn new_scaled(
        trees: &[&TreeNode],
        n_features: usize,
        weights: &[f32],
    ) -> Option<CudaForest> {
        let mut flat = Flat::default();
        for (t, &w) in trees.iter().zip(weights) {
            let leaf = move |node: &TreeNode, out: &mut [f32]| out[0] = (node.value as f32) * w;
            let r = flatten_node(t, &mut flat, 1, &leaf);
            flat.roots.push(r);
        }
        Self::from_flat(flat, n_features, 1)
    }

    fn from_flat(flat: Flat, n_features: usize, out_dim: usize) -> Option<CudaForest> {
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
            n_trees: flat.roots.len() as i32,
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

// ----------------------------------------------------------------- TreeSHAP
// Exact Shapley by 2^k coalition enumeration, one CUDA thread per sample;
// `evaluate_coalition`'s recursion is an explicit per-thread weight-stack. f32.
pub const SHAP_MAX_K: usize = 16;

const SHAP_KERNEL: &str = r#"
#define MAXD 64
__device__ float eval_coalition(const float* x, int base, unsigned mask, int c, int nc,
    const int* feat, const float* thr, const unsigned* nleft, const unsigned* nright,
    const float* pL, const float* pR, const int* uidx, const float* leafval)
{
    unsigned sn[MAXD]; float sw[MAXD]; int sp = 0;
    sn[0] = 0u; sw[0] = 1.0f; sp = 1;
    float acc = 0.0f;
    while (sp > 0) {
        sp--; unsigned node = sn[sp]; float w = sw[sp];
        int f = feat[node];
        if (f < 0) { acc += w * leafval[node*nc + c]; }
        else {
            unsigned revealed = (mask >> (unsigned)uidx[node]) & 1u;
            if (revealed) {
                bool goleft = x[base + f] <= thr[node];
                sn[sp] = goleft ? nleft[node] : nright[node]; sw[sp] = w; sp++;
            } else {
                sn[sp] = nleft[node];  sw[sp] = w * pL[node]; sp++;
                sn[sp] = nright[node]; sw[sp] = w * pR[node]; sp++;
            }
        }
    }
    return acc;
}

extern "C" __global__ void shap(
    const float* x, const int* feat, const float* thr, const unsigned* nleft, const unsigned* nright,
    const float* pL, const float* pR, const int* uidx, const float* leafval, const unsigned* ufeat,
    const float* fact, const int n_samples, const int n_features, const int n_classes, const int k, float* out)
{
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if (s >= n_samples) return;
    int base = s*n_features;
    unsigned ncoal = 1u << k;
    for (int c = 0; c < n_classes; c++) {
        for (int jj = 0; jj < k; jj++) {
            float contrib = 0.0f;
            for (unsigned mask = 0; mask < ncoal; mask++) {
                if ((mask >> jj) & 1u) continue;
                float fs0 = eval_coalition(x, base, mask, c, n_classes, feat, thr, nleft, nright, pL, pR, uidx, leafval);
                float fs1 = eval_coalition(x, base, mask | (1u << jj), c, n_classes, feat, thr, nleft, nright, pL, pR, uidx, leafval);
                int ssize = __popc(mask);
                float wgt = 1.0f;
                if (k > 1) wgt = fact[ssize]*fact[k-ssize-1]/fact[k];
                contrib += wgt*(fs1 - fs0);
            }
            out[(s*n_classes + c)*n_features + ufeat[jj]] = contrib;
        }
    }
}
"#;

/// One CUDA thread per sample; returns `n_samples * n_classes * n_features` f32
/// SHAP values (row-major). `None` if `k > SHAP_MAX_K` or CUDA unavailable.
#[allow(clippy::too_many_arguments)]
pub fn shap_explain(
    x: &[f32],
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    k: usize,
    feat: &[i32],
    thr: &[f32],
    left: &[u32],
    right: &[u32],
    pl: &[f32],
    pr: &[f32],
    uidx: &[i32],
    leafval: &[f32],
    ufeat: &[u32],
    fact: &[f32],
) -> Option<Vec<f32>> {
    if k > SHAP_MAX_K {
        return None;
    }
    let ctx = CudaContext::new(0).ok()?;
    let stream = ctx.default_stream();
    let func = ctx
        .load_module(compile_ptx(SHAP_KERNEL).ok()?)
        .ok()?
        .load_function("shap")
        .ok()?;

    let d_x = stream.memcpy_stod(x).ok()?;
    let d_feat = stream.memcpy_stod(feat).ok()?;
    let d_thr = stream.memcpy_stod(thr).ok()?;
    let d_left = stream.memcpy_stod(left).ok()?;
    let d_right = stream.memcpy_stod(right).ok()?;
    let d_pl = stream.memcpy_stod(pl).ok()?;
    let d_pr = stream.memcpy_stod(pr).ok()?;
    let d_uidx = stream.memcpy_stod(uidx).ok()?;
    let d_leaf = stream.memcpy_stod(leafval).ok()?;
    let d_ufeat = stream.memcpy_stod(ufeat).ok()?;
    let d_fact = stream.memcpy_stod(fact).ok()?;
    let out_len = n_samples * n_classes * n_features;
    let mut d_out = stream.alloc_zeros::<f32>(out_len).ok()?;
    let (ns, nf, nc, kk) = (
        n_samples as i32,
        n_features as i32,
        n_classes as i32,
        k as i32,
    );

    let cfg = LaunchConfig::for_num_elems(n_samples as u32);
    let mut b = stream.launch_builder(&func);
    b.arg(&d_x)
        .arg(&d_feat)
        .arg(&d_thr)
        .arg(&d_left)
        .arg(&d_right)
        .arg(&d_pl)
        .arg(&d_pr)
        .arg(&d_uidx)
        .arg(&d_leaf)
        .arg(&d_ufeat)
        .arg(&d_fact)
        .arg(&ns)
        .arg(&nf)
        .arg(&nc)
        .arg(&kk)
        .arg(&mut d_out);
    unsafe {
        b.launch(cfg).ok()?;
    }
    stream.synchronize().ok()?;
    stream.memcpy_dtov(&d_out).ok()
}
