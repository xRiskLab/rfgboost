//! Optional GPU forest inference (`wgpu` -> Metal / Vulkan / DX12).
//!
//! Feature-gated (`gpu`), native-only — never compiled into the wasm wheel.
//! Multi-output: `out_dim = 1` gives the mean leaf value (regression / boosting
//! round contribution); `out_dim = n_classes` gives the mean leaf distribution
//! (class probabilities). One kernel covers both — one row per GPU thread,
//! accumulating an `out_dim`-length vector. f32 compute; parity with the CPU
//! `traverse`/`traverse_proba` is exact to f32 rounding. `GpuForest::new`
//! returns `None` when no adapter is available so callers fall back to CPU.

use crate::tree::TreeNode;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Max supported output width (n_classes). Forests above this fall back to CPU.
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

// Flattens the recursive forest to SoA. `leaf` fills the out_dim-length slot for
// each leaf. Nodes are pushed in id order with out_dim values each, so node `id`
// owns values[id*out_dim .. id*out_dim+out_dim].
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n_samples: u32,
    n_features: u32,
    n_trees: u32,
    out_dim: u32,
}

const SHADER: &str = r#"
struct Params { n_samples: u32, n_features: u32, n_trees: u32, out_dim: u32 };
@group(0) @binding(0) var<storage, read> features: array<f32>;
@group(0) @binding(1) var<storage, read> n_feature: array<i32>;
@group(0) @binding(2) var<storage, read> n_threshold: array<f32>;
@group(0) @binding(3) var<storage, read> n_left: array<u32>;
@group(0) @binding(4) var<storage, read> n_right: array<u32>;
@group(0) @binding(5) var<storage, read> values: array<f32>;
@group(0) @binding(6) var<storage, read> roots: array<u32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.n_samples) { return; }
    let od = params.out_dim;
    let base = s * params.n_features;
    var acc: array<f32, 32>;
    for (var k = 0u; k < od; k = k + 1u) { acc[k] = 0.0; }
    for (var t = 0u; t < params.n_trees; t = t + 1u) {
        var node = roots[t];
        loop {
            let f = n_feature[node];
            if (f < 0) { break; }
            let xv = features[base + u32(f)];
            if (xv <= n_threshold[node]) { node = n_left[node]; }
            else { node = n_right[node]; }
        }
        let vb = node * od;
        for (var k = 0u; k < od; k = k + 1u) { acc[k] = acc[k] + values[vb + k]; }
    }
    let ob = s * od;
    let inv = 1.0 / f32(params.n_trees);
    for (var k = 0u; k < od; k = k + 1u) { out[ob + k] = acc[k] * inv; }
}
"#;

/// A forest uploaded to the GPU, ready for repeated batched inference.
pub struct GpuForest {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    b_feature: wgpu::Buffer,
    b_threshold: wgpu::Buffer,
    b_left: wgpu::Buffer,
    b_right: wgpu::Buffer,
    b_values: wgpu::Buffer,
    b_roots: wgpu::Buffer,
    n_features: u32,
    n_trees: u32,
    out_dim: u32,
}

impl GpuForest {
    /// Flatten `trees` (each leaf producing an `out_dim`-length vector via
    /// `leaf`) and upload to the GPU. `None` if `out_dim > MAX_OUT` or no
    /// adapter is available.
    pub fn new<F: Fn(&TreeNode, &mut [f32])>(
        trees: &[TreeNode],
        n_features: usize,
        out_dim: usize,
        leaf: F,
    ) -> Option<GpuForest> {
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
    pub fn new_scaled(trees: &[&TreeNode], n_features: usize, weights: &[f32]) -> Option<GpuForest> {
        let mut flat = Flat::default();
        for (t, &w) in trees.iter().zip(weights) {
            let leaf = move |node: &TreeNode, out: &mut [f32]| out[0] = (node.value as f32) * w;
            let r = flatten_node(t, &mut flat, 1, &leaf);
            flat.roots.push(r);
        }
        Self::from_flat(flat, n_features, 1)
    }

    fn from_flat(flat: Flat, n_features: usize, out_dim: usize) -> Option<GpuForest> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("rfgboost-gpu"),
                ..Default::default()
            }),
        )
        .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forest"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("forest"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let mk = |bytes: &[u8], label: &str| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };

        Some(GpuForest {
            b_feature: mk(bytemuck::cast_slice(&flat.feature), "feature"),
            b_threshold: mk(bytemuck::cast_slice(&flat.threshold), "threshold"),
            b_left: mk(bytemuck::cast_slice(&flat.left), "left"),
            b_right: mk(bytemuck::cast_slice(&flat.right), "right"),
            b_values: mk(bytemuck::cast_slice(&flat.values), "values"),
            b_roots: mk(bytemuck::cast_slice(&flat.roots), "roots"),
            n_features: n_features as u32,
            n_trees: flat.roots.len() as u32,
            out_dim: out_dim as u32,
            device,
            queue,
            pipeline,
        })
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim as usize
    }

    /// Mean over trees of the leaf vector, per row. `x` is row-major f32 of shape
    /// `n_rows x n_features`; returns `n_rows * out_dim` f32 (row-major).
    pub fn predict(&self, x: &[f32], n_rows: usize) -> Vec<f32> {
        let b_x = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let out_len = n_rows * self.out_dim as usize;
        let out_size = (out_len * 4) as u64;
        let b_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = Params {
            n_samples: n_rows as u32,
            n_features: self.n_features,
            n_trees: self.n_trees,
            out_dim: self.out_dim,
        };
        let b_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = self.pipeline.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.b_feature.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.b_threshold.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.b_left.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.b_right.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.b_values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.b_roots.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: b_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: b_params.as_entire_binding() },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(((n_rows + 63) / 64) as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&b_out, 0, &staging, 0, out_size);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }
}

// ----------------------------------------------------------------- TreeSHAP
// Exact Shapley by 2^k coalition enumeration, one GPU thread per sample. The
// recursive `evaluate_coalition` (hidden feature -> weight both children) is an
// explicit weight-stack. Caller flattens the SHAP tree to SoA. f32.
pub const SHAP_MAX_K: usize = 16;
const SHAP_MAX_DEPTH: u32 = 64;

const SHAP_SHADER: &str = r#"
struct P { n_samples:u32, n_features:u32, n_classes:u32, k:u32 };
@group(0) @binding(0) var<storage, read> features: array<f32>;
@group(0) @binding(1) var<storage, read> feat: array<i32>;
@group(0) @binding(2) var<storage, read> thr: array<f32>;
@group(0) @binding(3) var<storage, read> nleft: array<u32>;
@group(0) @binding(4) var<storage, read> nright: array<u32>;
@group(0) @binding(5) var<storage, read> pL: array<f32>;
@group(0) @binding(6) var<storage, read> pR: array<f32>;
@group(0) @binding(7) var<storage, read> uidx: array<i32>;
@group(0) @binding(8) var<storage, read> leafval: array<f32>;
@group(0) @binding(9) var<storage, read> ufeat: array<u32>;
@group(0) @binding(10) var<storage, read> fact: array<f32>;
@group(0) @binding(11) var<storage, read_write> out: array<f32>;
@group(0) @binding(12) var<uniform> params: P;

fn eval_coalition(s: u32, mask: u32, c: u32) -> f32 {
    var sn: array<u32, 64>;
    var sw: array<f32, 64>;
    sn[0] = 0u; sw[0] = 1.0;
    var sp = 1u;
    var acc = 0.0;
    let nc = params.n_classes;
    let base = s * params.n_features;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node = sn[sp];
        let w = sw[sp];
        let f = feat[node];
        if (f < 0) {
            acc = acc + w * leafval[node * nc + c];
        } else {
            let revealed = (mask >> u32(uidx[node])) & 1u;
            if (revealed == 1u) {
                let goleft = features[base + u32(f)] <= thr[node];
                sn[sp] = select(nright[node], nleft[node], goleft);
                sw[sp] = w;
                sp = sp + 1u;
            } else {
                sn[sp] = nleft[node]; sw[sp] = w * pL[node]; sp = sp + 1u;
                sn[sp] = nright[node]; sw[sp] = w * pR[node]; sp = sp + 1u;
            }
        }
    }
    return acc;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.n_samples) { return; }
    let k = params.k;
    let nc = params.n_classes;
    let nf = params.n_features;
    let ncoal = 1u << k;
    for (var c = 0u; c < nc; c = c + 1u) {
        for (var jj = 0u; jj < k; jj = jj + 1u) {
            var contrib = 0.0;
            for (var mask = 0u; mask < ncoal; mask = mask + 1u) {
                if (((mask >> jj) & 1u) == 1u) { continue; }
                let fs0 = eval_coalition(s, mask, c);
                let fs1 = eval_coalition(s, mask | (1u << jj), c);
                let ssize = countOneBits(mask);
                var wgt = 1.0;
                if (k > 1u) { wgt = fact[ssize] * fact[k - ssize - 1u] / fact[k]; }
                contrib = contrib + wgt * (fs1 - fs0);
            }
            out[(s * nc + c) * nf + ufeat[jj]] = contrib;
        }
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShapParams {
    n_samples: u32,
    n_features: u32,
    n_classes: u32,
    k: u32,
}

/// One GPU thread per sample; returns `n_samples * n_classes * n_features` f32
/// SHAP values (row-major). `None` if `k > SHAP_MAX_K` or no adapter.
#[allow(clippy::too_many_arguments)]
pub fn shap_explain(
    x: &[f32], n_samples: usize, n_features: usize, n_classes: usize, k: usize,
    feat: &[i32], thr: &[f32], left: &[u32], right: &[u32], pl: &[f32], pr: &[f32],
    uidx: &[i32], leafval: &[f32], ufeat: &[u32], fact: &[f32],
) -> Option<Vec<f32>> {
    if k > SHAP_MAX_K || (SHAP_MAX_DEPTH as usize) < 1 {
        return None;
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("rfgboost-shap"),
        required_limits: adapter.limits(), // SHAP needs >8 storage buffers
        ..Default::default()
    }))
    .ok()?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shap"),
        source: wgpu::ShaderSource::Wgsl(SHAP_SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("shap"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let mk = |bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    let out_len = n_samples * n_classes * n_features;
    let out_size = (out_len * 4) as u64;
    let b_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = ShapParams {
        n_samples: n_samples as u32,
        n_features: n_features as u32,
        n_classes: n_classes as u32,
        k: k as u32,
    };
    let b_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let b = [
        mk(bytemuck::cast_slice(x)),
        mk(bytemuck::cast_slice(feat)),
        mk(bytemuck::cast_slice(thr)),
        mk(bytemuck::cast_slice(left)),
        mk(bytemuck::cast_slice(right)),
        mk(bytemuck::cast_slice(pl)),
        mk(bytemuck::cast_slice(pr)),
        mk(bytemuck::cast_slice(uidx)),
        mk(bytemuck::cast_slice(leafval)),
        mk(bytemuck::cast_slice(ufeat)),
        mk(bytemuck::cast_slice(fact)),
    ];
    let bgl = pipeline.get_bind_group_layout(0);
    let mut entries: Vec<wgpu::BindGroupEntry> = b
        .iter()
        .enumerate()
        .map(|(i, buf)| wgpu::BindGroupEntry { binding: i as u32, resource: buf.as_entire_binding() })
        .collect();
    entries.push(wgpu::BindGroupEntry { binding: 11, resource: b_out.as_entire_binding() });
    entries.push(wgpu::BindGroupEntry { binding: 12, resource: b_params.as_entire_binding() });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &entries });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(((n_samples + 63) / 64) as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&b_out, 0, &staging, 0, out_size);
    queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::traverse;

    fn leaf(v: f64) -> TreeNode {
        TreeNode { feature: 0, threshold: 0.0, left: None, right: None, value: v, samples: 1, class_counts: None }
    }
    fn node(feature: usize, threshold: f64, l: TreeNode, r: TreeNode) -> TreeNode {
        TreeNode { feature, threshold, left: Some(Box::new(l)), right: Some(Box::new(r)), value: 0.0, samples: 1, class_counts: None }
    }

    // Needs a GPU adapter; run locally: `cargo test --features gpu -- --ignored`
    #[test]
    #[ignore]
    fn gpu_matches_cpu_traverse() {
        let trees = vec![
            node(0, 0.0, node(1, 0.5, leaf(1.0), leaf(2.0)), leaf(3.0)),
            node(1, -0.2, leaf(-1.0), node(0, 1.0, leaf(0.5), leaf(4.0))),
            node(0, 1.5, leaf(0.0), leaf(7.0)),
        ];
        let rows: Vec<[f64; 2]> = vec![[-1.0, 0.1], [0.3, 0.9], [2.0, -0.5], [0.0, 0.0], [1.7, 1.0]];
        let xf: Vec<f32> = rows.iter().flat_map(|r| r.iter().map(|&v| v as f32)).collect();
        let g = GpuForest::new(&trees, 2, 1, |n, out| out[0] = n.value as f32).expect("no GPU adapter");
        let gpu = g.predict(&xf, rows.len());
        for (i, r) in rows.iter().enumerate() {
            let cpu = trees.iter().map(|t| traverse(t, r)).sum::<f64>() / trees.len() as f64;
            assert!((cpu as f32 - gpu[i]).abs() < 1e-5, "row {i}: cpu={cpu} gpu={}", gpu[i]);
        }
    }
}
