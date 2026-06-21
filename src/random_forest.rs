use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
#[cfg(target_os = "emscripten")]
use rand_pcg::Pcg32 as Pcg64;
#[cfg(not(target_os = "emscripten"))]
use rand_pcg::Pcg64;
use crate::par::*;
use std::collections::HashMap;

use crate::histogram::HistogramData;
use crate::tree::{
    build_tree_on_bootstrap, resolve_max_features, traverse, traverse_proba, TreeConfig, TreeNode,
};

// With the `cuda` feature the struct holds a CUDA context (thread-affine), so
// the pyclass is `unsendable` there; default/wasm builds stay plain `#[pyclass]`.
#[cfg_attr(feature = "cuda", pyclass(unsendable))]
#[cfg_attr(not(feature = "cuda"), pyclass)]
pub struct RandomForestRegressor {
    n_estimators: usize,
    max_depth: Option<usize>,
    max_features: Option<String>,
    bootstrap: bool,
    random_state: Option<u64>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    trees: Vec<TreeNode>,
    is_fitted: bool,
    // Lazily built once, reused across predict_cuda calls; reset on fit.
    #[cfg(feature = "cuda")]
    cuda_cache: std::cell::RefCell<Option<crate::cuda::CudaForest>>,
    #[cfg(feature = "gpu")]
    gpu_cache: std::cell::RefCell<Option<crate::gpu::GpuForest>>,
}

#[pymethods]
impl RandomForestRegressor {
    #[new]
    #[pyo3(signature = (
        n_estimators=50, max_depth=None, max_features=None,
        bootstrap=true, random_state=None, min_samples_split=2, min_samples_leaf=1
    ))]
    fn new(
        n_estimators: usize, max_depth: Option<usize>, max_features: Option<String>,
        bootstrap: bool, random_state: Option<u64>, min_samples_split: usize, min_samples_leaf: usize,
    ) -> Self {
        Self {
            n_estimators, max_depth, max_features, bootstrap, random_state,
            min_samples_split, min_samples_leaf, trees: Vec::new(), is_fitted: false,
            #[cfg(feature = "cuda")]
            cuda_cache: std::cell::RefCell::new(None),
            #[cfg(feature = "gpu")]
            gpu_cache: std::cell::RefCell::new(None),
        }
    }

    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let x_arr = x.as_array();
        let y_vec: Vec<f64> = y.as_array().to_vec();
        crate::tree::validate_finite(&x_arr.view(), &y_vec)
            .map_err(PyValueError::new_err)?;
        let n_samples = x_arr.nrows();
        let n_features = x_arr.ncols();
        let weights: Vec<f64> = match sample_weight {
            Some(arr) => {
                let v: Vec<f64> = arr.as_array().to_vec();
                crate::tree::validate_weights(&v, n_samples).map_err(PyValueError::new_err)?;
                v
            }
            None => vec![1.0; n_samples],
        };
        let max_feat = resolve_max_features(&self.max_features, n_features);
        let mut rng = crate::tree::seed_rng(self.random_state);

        let config = TreeConfig {
            max_depth: self.max_depth, min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf, is_classification: false,
            max_features: if max_feat < n_features { Some(max_feat) } else { None },
        };

        let tree_params: Vec<(Vec<usize>, u64)> = (0..self.n_estimators)
            .map(|_| {
                let boot = if self.bootstrap { (0..n_samples).map(|_| rng.gen_range(0..n_samples)).collect() } else { (0..n_samples).collect() };
                (boot, rng.gen())
            })
            .collect();

        let x_owned = x_arr.to_owned();
        let global_hist = HistogramData::build(&x_owned.view(), n_samples, n_features);

        self.trees = tree_params
            .into_par_iter()
            .map(|(boot, seed)| {
                let mut tree_rng = Pcg64::seed_from_u64(seed);
                build_tree_on_bootstrap(&x_owned.view(), &y_vec, &weights, &boot, &config, &mut tree_rng, &global_hist)
            })
            .collect();

        #[cfg(feature = "cuda")]
        { *self.cuda_cache.borrow_mut() = None; }
        #[cfg(feature = "gpu")]
        { *self.gpu_cache.borrow_mut() = None; }

        self.is_fitted = true;
        Ok(())
    }

    /// Predict on a chosen device. `device`: `"cpu"` (default), `"cuda"`
    /// (NVIDIA, needs the `cuda` feature) or `"mps"`/`"metal"`/`"gpu"` (wgpu →
    /// Metal on Apple, needs the `gpu` feature). Results match across devices to
    /// f32 rounding. Requesting a device this wheel wasn't built for, or one
    /// with no available hardware, raises a clear error.
    #[pyo3(signature = (x, device="cpu"))]
    fn predict(&self, x: PyReadonlyArray2<f64>, device: &str) -> PyResult<Vec<f64>> {
        if !self.is_fitted { return Err(PyValueError::new_err("RandomForestRegressor has not been fitted")); }
        match device {
            "cpu" => {
                let x_arr = x.as_array();
                let n_trees = self.trees.len() as f64;
                let trees = &self.trees;
                Ok(x_arr.outer_iter().collect::<Vec<_>>().into_par_iter()
                    .map(|row| {
                        let sample = row.as_slice().unwrap();
                        trees.iter().map(|tree| traverse(tree, sample)).sum::<f64>() / n_trees
                    })
                    .collect())
            }
            #[cfg(feature = "cuda")]
            "cuda" => self.predict_cuda_impl(x),
            #[cfg(feature = "gpu")]
            "mps" | "metal" | "gpu" => self.predict_gpu_impl(x),
            other => Err(PyValueError::new_err(format!(
                "device '{}' is not available in this build. Available: {}.",
                other, Self::available_devices()
            ))),
        }
    }

    fn get_info(&self) -> PyResult<HashMap<String, usize>> {
        let mut info = HashMap::new();
        info.insert("n_estimators".to_string(), self.n_estimators);
        info.insert("is_fitted".to_string(), if self.is_fitted { 1 } else { 0 });
        Ok(info)
    }

    #[getter] fn n_estimators(&self) -> usize { self.n_estimators }
    #[getter] fn is_fitted(&self) -> bool { self.is_fitted }
}

// Device backends behind `predict(device=...)` — not exposed as Python methods.
impl RandomForestRegressor {
    fn available_devices() -> String {
        #[allow(unused_mut)]
        let mut d = vec!["cpu"];
        #[cfg(feature = "cuda")] d.push("cuda");
        #[cfg(feature = "gpu")] d.push("mps");
        d.join(", ")
    }

    #[cfg(feature = "cuda")]
    fn predict_cuda_impl(&self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
        let x_arr = x.as_array();
        let (n, nf) = (x_arr.nrows(), x_arr.ncols());
        let xf: Vec<f32> = match x_arr.as_slice() {
            Some(s) => s.par_iter().map(|&v| v as f32).collect(),
            None => x_arr.iter().map(|&v| v as f32).collect(),
        };
        let mut cache = self.cuda_cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(
                crate::cuda::CudaForest::new(&self.trees, nf, 1, |node, out| out[0] = node.value as f32)
                    .ok_or_else(|| PyValueError::new_err("CUDA device unavailable"))?,
            );
        }
        Ok(cache.as_ref().unwrap().predict(&xf, n).into_iter().map(|v| v as f64).collect())
    }

    #[cfg(feature = "gpu")]
    fn predict_gpu_impl(&self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
        let x_arr = x.as_array();
        let (n, nf) = (x_arr.nrows(), x_arr.ncols());
        let xf: Vec<f32> = match x_arr.as_slice() {
            Some(s) => s.par_iter().map(|&v| v as f32).collect(),
            None => x_arr.iter().map(|&v| v as f32).collect(),
        };
        let mut cache = self.gpu_cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(
                crate::gpu::GpuForest::new(&self.trees, nf, 1, |node, out| out[0] = node.value as f32)
                    .ok_or_else(|| PyValueError::new_err("GPU (wgpu) device unavailable"))?,
            );
        }
        Ok(cache.as_ref().unwrap().predict(&xf, n).into_iter().map(|v| v as f64).collect())
    }
}

#[cfg_attr(feature = "cuda", pyclass(unsendable))]
#[cfg_attr(not(feature = "cuda"), pyclass)]
pub struct RandomForestClassifier {
    n_estimators: usize,
    max_depth: Option<usize>,
    max_features: Option<String>,
    bootstrap: bool,
    random_state: Option<u64>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    trees: Vec<TreeNode>,
    n_classes: usize,
    classes_: Option<Vec<usize>>,
    is_fitted: bool,
    #[cfg(feature = "cuda")]
    cuda_cache: std::cell::RefCell<Option<crate::cuda::CudaForest>>,
    #[cfg(feature = "gpu")]
    gpu_cache: std::cell::RefCell<Option<crate::gpu::GpuForest>>,
}

#[pymethods]
impl RandomForestClassifier {
    #[new]
    #[pyo3(signature = (
        n_estimators=100, max_depth=None, max_features=None,
        bootstrap=true, random_state=None, min_samples_split=2, min_samples_leaf=1
    ))]
    fn new(
        n_estimators: usize, max_depth: Option<usize>, max_features: Option<String>,
        bootstrap: bool, random_state: Option<u64>, min_samples_split: usize, min_samples_leaf: usize,
    ) -> Self {
        Self {
            n_estimators, max_depth, max_features, bootstrap, random_state,
            min_samples_split, min_samples_leaf, trees: Vec::new(), n_classes: 0,
            classes_: None, is_fitted: false,
            #[cfg(feature = "cuda")]
            cuda_cache: std::cell::RefCell::new(None),
            #[cfg(feature = "gpu")]
            gpu_cache: std::cell::RefCell::new(None),
        }
    }

    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let x_arr = x.as_array();
        let y_vec: Vec<f64> = y.as_array().to_vec();
        crate::tree::validate_finite(&x_arr.view(), &y_vec)
            .map_err(PyValueError::new_err)?;
        let n_samples = x_arr.nrows();
        let n_features = x_arr.ncols();
        let weights: Vec<f64> = match sample_weight {
            Some(arr) => {
                let v: Vec<f64> = arr.as_array().to_vec();
                crate::tree::validate_weights(&v, n_samples).map_err(PyValueError::new_err)?;
                v
            }
            None => vec![1.0; n_samples],
        };

        let mut classes: Vec<usize> = y_vec.iter().map(|&v| v as usize).collect();
        classes.sort();
        classes.dedup();
        self.classes_ = Some(classes.clone());
        self.n_classes = classes.len();

        let max_feat = resolve_max_features(&self.max_features, n_features);
        let mut rng = crate::tree::seed_rng(self.random_state);

        let config = TreeConfig {
            max_depth: self.max_depth, min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf, is_classification: true,
            max_features: if max_feat < n_features { Some(max_feat) } else { None },
        };

        let tree_params: Vec<(Vec<usize>, u64)> = (0..self.n_estimators)
            .map(|_| {
                let boot = if self.bootstrap { (0..n_samples).map(|_| rng.gen_range(0..n_samples)).collect() } else { (0..n_samples).collect() };
                (boot, rng.gen())
            })
            .collect();

        let x_owned = x_arr.to_owned();
        let global_hist = HistogramData::build(&x_owned.view(), n_samples, n_features);

        self.trees = tree_params
            .into_par_iter()
            .map(|(boot, seed)| {
                let mut tree_rng = Pcg64::seed_from_u64(seed);
                build_tree_on_bootstrap(&x_owned.view(), &y_vec, &weights, &boot, &config, &mut tree_rng, &global_hist)
            })
            .collect();

        #[cfg(feature = "cuda")]
        { *self.cuda_cache.borrow_mut() = None; }
        #[cfg(feature = "gpu")]
        { *self.gpu_cache.borrow_mut() = None; }

        self.is_fitted = true;
        Ok(())
    }

    /// Predicted class index. `device`: "cpu" (default), "cuda" or "mps"/"metal"/"gpu".
    #[pyo3(signature = (x, device="cpu"))]
    fn predict(&self, x: PyReadonlyArray2<f64>, device: &str) -> PyResult<Vec<f64>> {
        let proba = self.predict_proba_impl(x, device)?;
        Ok(proba.iter().map(|p| {
            p.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map_or(0.0, |(i, _)| i as f64)
        }).collect())
    }

    /// Class probabilities (n_rows x n_classes). `device`: "cpu" (default),
    /// "cuda" or "mps"/"metal"/"gpu".
    #[pyo3(signature = (x, device="cpu"))]
    fn predict_proba(&self, x: PyReadonlyArray2<f64>, device: &str) -> PyResult<Vec<Vec<f64>>> {
        self.predict_proba_impl(x, device)
    }

    fn score(&self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let preds = self.predict(x, "cpu")?;
        let y_vec = y.as_array().to_vec();
        let correct = preds.iter().zip(y_vec.iter()).filter(|(&p, &t)| (p as usize) == (t as usize)).count();
        Ok(correct as f64 / y_vec.len() as f64)
    }

    fn get_info(&self) -> PyResult<HashMap<String, usize>> {
        let mut info = HashMap::new();
        info.insert("n_estimators".to_string(), self.n_estimators);
        info.insert("n_classes".to_string(), self.n_classes);
        info.insert("is_fitted".to_string(), if self.is_fitted { 1 } else { 0 });
        Ok(info)
    }

    #[getter] fn n_estimators(&self) -> usize { self.n_estimators }
    #[getter] fn n_classes(&self) -> usize { self.n_classes }
    #[getter] fn classes_(&self) -> Option<Vec<usize>> { self.classes_.clone() }
    #[getter] fn is_fitted(&self) -> bool { self.is_fitted }
}

// Leaf -> class-probability vector (matches traverse_proba): out[cls] = count/total.
#[cfg(any(feature = "cuda", feature = "gpu"))]
fn leaf_proba(node: &TreeNode, n_classes: usize, out: &mut [f32]) {
    if let Some(counts) = &node.class_counts {
        let total: f64 = counts.values().sum();
        if total > 0.0 {
            for (&cls, &cnt) in counts {
                if cls < n_classes { out[cls] = (cnt / total) as f32; }
            }
        }
    }
}

// Device backends behind predict/predict_proba(device=...) — not Python methods.
impl RandomForestClassifier {
    fn available_devices() -> String {
        #[allow(unused_mut)]
        let mut d = vec!["cpu"];
        #[cfg(feature = "cuda")] d.push("cuda");
        #[cfg(feature = "gpu")] d.push("mps");
        d.join(", ")
    }

    fn predict_proba_impl(&self, x: PyReadonlyArray2<f64>, device: &str) -> PyResult<Vec<Vec<f64>>> {
        if !self.is_fitted { return Err(PyValueError::new_err("RandomForestClassifier has not been fitted")); }
        match device {
            "cpu" => {
                let x_arr = x.as_array();
                let n_classes = self.n_classes;
                let n_trees = self.trees.len() as f64;
                let trees = &self.trees;
                Ok(x_arr.outer_iter().collect::<Vec<_>>().into_par_iter()
                    .map(|row| {
                        let sample = row.as_slice().unwrap();
                        let mut agg = vec![0.0; n_classes];
                        for tree in trees {
                            let probs = traverse_proba(tree, sample, n_classes);
                            for (i, p) in probs.iter().enumerate() { agg[i] += p; }
                        }
                        agg.iter().map(|&v| v / n_trees).collect()
                    })
                    .collect())
            }
            #[cfg(feature = "cuda")]
            "cuda" => self.proba_cuda_impl(x),
            #[cfg(feature = "gpu")]
            "mps" | "metal" | "gpu" => self.proba_gpu_impl(x),
            other => Err(PyValueError::new_err(format!(
                "device '{}' is not available in this build. Available: {}.",
                other, Self::available_devices()
            ))),
        }
    }

    #[cfg(feature = "cuda")]
    fn proba_cuda_impl(&self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<f64>>> {
        let nc = self.n_classes;
        let x_arr = x.as_array();
        let (n, nf) = (x_arr.nrows(), x_arr.ncols());
        let xf: Vec<f32> = match x_arr.as_slice() {
            Some(s) => s.par_iter().map(|&v| v as f32).collect(),
            None => x_arr.iter().map(|&v| v as f32).collect(),
        };
        let mut cache = self.cuda_cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(
                crate::cuda::CudaForest::new(&self.trees, nf, nc, move |node, out| leaf_proba(node, nc, out))
                    .ok_or_else(|| PyValueError::new_err(format!(
                        "CUDA unavailable or n_classes>{} unsupported", crate::cuda::MAX_OUT)))?,
            );
        }
        Ok(cache.as_ref().unwrap().predict(&xf, n).chunks(nc)
            .map(|c| c.iter().map(|&v| v as f64).collect()).collect())
    }

    #[cfg(feature = "gpu")]
    fn proba_gpu_impl(&self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<f64>>> {
        let nc = self.n_classes;
        let x_arr = x.as_array();
        let (n, nf) = (x_arr.nrows(), x_arr.ncols());
        let xf: Vec<f32> = match x_arr.as_slice() {
            Some(s) => s.par_iter().map(|&v| v as f32).collect(),
            None => x_arr.iter().map(|&v| v as f32).collect(),
        };
        let mut cache = self.gpu_cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(
                crate::gpu::GpuForest::new(&self.trees, nf, nc, move |node, out| leaf_proba(node, nc, out))
                    .ok_or_else(|| PyValueError::new_err(format!(
                        "GPU unavailable or n_classes>{} unsupported", crate::gpu::MAX_OUT)))?,
            );
        }
        Ok(cache.as_ref().unwrap().predict(&xf, n).chunks(nc)
            .map(|c| c.iter().map(|&v| v as f64).collect()).collect())
    }
}
