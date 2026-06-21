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

        self.is_fitted = true;
        Ok(())
    }

    fn predict(&self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
        if !self.is_fitted { return Err(PyValueError::new_err("RandomForestRegressor has not been fitted")); }
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

    /// GPU-accelerated `predict` via the native CUDA backend (`cuda` feature).
    /// Same result as `predict`; a throughput win for large batches.
    #[cfg(feature = "cuda")]
    fn predict_cuda(&self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
        if !self.is_fitted { return Err(PyValueError::new_err("RandomForestRegressor has not been fitted")); }
        let x_arr = x.as_array();
        let n = x_arr.nrows();
        let nf = x_arr.ncols();
        // row-major f32 copy (ndarray .iter() yields logical row-major order)
        // f64->f32 is the dominant host cost at scale; parallelize it (rayon).
        let xf: Vec<f32> = match x_arr.as_slice() {
            Some(s) => s.par_iter().map(|&v| v as f32).collect(),
            None => x_arr.iter().map(|&v| v as f32).collect(), // non-contiguous fallback
        };
        // Build the CUDA forest once (context + nvrtc kernel + upload) and reuse.
        let mut cache = self.cuda_cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(
                crate::cuda::CudaForest::new(&self.trees, nf)
                    .ok_or_else(|| PyValueError::new_err("CUDA device unavailable"))?,
            );
        }
        let out = cache.as_ref().unwrap().predict_avg(&xf, n);
        Ok(out.into_iter().map(|v| v as f64).collect())
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

#[pyclass]
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
        Self { n_estimators, max_depth, max_features, bootstrap, random_state, min_samples_split, min_samples_leaf, trees: Vec::new(), n_classes: 0, classes_: None, is_fitted: false }
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

        self.is_fitted = true;
        Ok(())
    }

    fn predict(&self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
        if !self.is_fitted { return Err(PyValueError::new_err("RandomForestClassifier has not been fitted")); }
        let proba = self.predict_proba(x)?;
        Ok(proba.iter().map(|p| {
            p.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map_or(0.0, |(i, _)| i as f64)
        }).collect())
    }

    fn predict_proba(&self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<f64>>> {
        if !self.is_fitted { return Err(PyValueError::new_err("RandomForestClassifier has not been fitted")); }
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

    fn score(&self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let preds = self.predict(x)?;
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
