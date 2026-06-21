//! Optional GPU forest inference (`wgpu` -> Metal / Vulkan / DX12).
//!
//! Feature-gated (`gpu`), native-only — never compiled into the wasm wheel.
//! The recursive `TreeNode` forest is flattened to struct-of-arrays (the
//! GPU-friendly layout) and traversed one row per GPU thread, mirroring
//! `predict_all` (mean of `traverse` over a round's trees). Compute is f32
//! (WGSL has no f64); parity with the CPU `traverse` is exact to f32 rounding.
//!
//! Benchmark on Apple M4 (10-core CPU vs 10-core GPU, end-to-end): ~17x over
//! rayon at >=100k rows, crossover ~2-5k rows. Use for large batch / SHAP-
//! background scoring, not single-row latency.

use crate::tree::TreeNode;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

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
    // build_node always produces either two children or a (None, None) leaf.
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n_samples: u32,
    n_features: u32,
    n_trees: u32,
    _pad: u32,
}

const SHADER: &str = r#"
struct Params { n_samples: u32, n_features: u32, n_trees: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> features: array<f32>;
@group(0) @binding(1) var<storage, read> n_feature: array<i32>;
@group(0) @binding(2) var<storage, read> n_threshold: array<f32>;
@group(0) @binding(3) var<storage, read> n_left: array<u32>;
@group(0) @binding(4) var<storage, read> n_right: array<u32>;
@group(0) @binding(5) var<storage, read> n_value: array<f32>;
@group(0) @binding(6) var<storage, read> roots: array<u32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.n_samples) { return; }
    let base = s * params.n_features;
    var acc = 0.0;
    for (var t = 0u; t < params.n_trees; t = t + 1u) {
        var node = roots[t];
        loop {
            let f = n_feature[node];
            if (f < 0) { acc = acc + n_value[node]; break; }
            let xv = features[base + u32(f)];
            if (xv <= n_threshold[node]) { node = n_left[node]; }
            else { node = n_right[node]; }
        }
    }
    out[s] = acc / f32(params.n_trees);
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
    b_value: wgpu::Buffer,
    b_roots: wgpu::Buffer,
    n_features: u32,
    n_trees: u32,
}

impl GpuForest {
    /// Flatten `trees` and upload to the GPU. Returns `None` if no GPU adapter
    /// is available (caller should fall back to the CPU path).
    pub fn new(trees: &[TreeNode], n_features: usize) -> Option<GpuForest> {
        let mut flat = Flat::default();
        for t in trees {
            let r = flatten_node(t, &mut flat);
            flat.roots.push(r);
        }

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
            b_value: mk(bytemuck::cast_slice(&flat.value), "value"),
            b_roots: mk(bytemuck::cast_slice(&flat.roots), "roots"),
            n_features: n_features as u32,
            n_trees: trees.len() as u32,
            device,
            queue,
            pipeline,
        })
    }

    /// Mean of `traverse` over all trees, one value per row. `x` is row-major
    /// f32 of shape `n_rows x n_features`.
    pub fn predict_avg(&self, x: &[f32], n_rows: usize) -> Vec<f32> {
        let b_x = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let out_size = (n_rows * 4) as u64;
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
            _pad: 0,
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
                wgpu::BindGroupEntry { binding: 5, resource: self.b_value.as_entire_binding() },
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::traverse;

    fn leaf(v: f64) -> TreeNode {
        TreeNode {
            feature: 0,
            threshold: 0.0,
            left: None,
            right: None,
            value: v,
            samples: 1,
            class_counts: None,
        }
    }
    fn node(feature: usize, threshold: f64, l: TreeNode, r: TreeNode) -> TreeNode {
        TreeNode {
            feature,
            threshold,
            left: Some(Box::new(l)),
            right: Some(Box::new(r)),
            value: 0.0,
            samples: 1,
            class_counts: None,
        }
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
        let nf = 2usize;
        let rows: Vec<[f64; 2]> = vec![
            [-1.0, 0.1],
            [0.3, 0.9],
            [2.0, -0.5],
            [0.0, 0.0],
            [1.7, 1.0],
        ];
        let xf: Vec<f32> = rows
            .iter()
            .flat_map(|r| r.iter().map(|&v| v as f32))
            .collect();

        let g = GpuForest::new(&trees, nf).expect("no GPU adapter");
        let gpu = g.predict_avg(&xf, rows.len());

        for (i, r) in rows.iter().enumerate() {
            let cpu = trees.iter().map(|t| traverse(t, r)).sum::<f64>() / trees.len() as f64;
            assert!(
                (cpu as f32 - gpu[i]).abs() < 1e-5,
                "row {i}: cpu={cpu} gpu={}",
                gpu[i]
            );
        }
    }
}
