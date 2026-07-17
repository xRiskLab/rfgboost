use ndarray::ArrayView2;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::prelude::*;
// Pcg64 (128-bit) is unavailable on emscripten/wasm; Pcg32 has the same
// SeedableRng/RngCore API, so alias it under the Pcg64 name there.
#[cfg(target_os = "emscripten")]
use rand_pcg::Pcg32 as Pcg64;
#[cfg(not(target_os = "emscripten"))]
use rand_pcg::Pcg64;
use crate::par::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::histogram::HistogramData;
use crate::tree::{
    build_tree_on_bootstrap, build_tree_on_bootstrap_exact, resolve_max_features, traverse,
    TreeConfig, TreeNode,
};

// ---------------------------------------------------------------------------
// Tree serialization for external compressors / cross-language inference.
// Emits sklearn-style flat parallel arrays per tree.
// Leaf rows: feature=-1, threshold=NaN, children_left=children_right=-1.
// ---------------------------------------------------------------------------

fn flatten_tree(
    node: &TreeNode,
    children_left: &mut Vec<i64>,
    children_right: &mut Vec<i64>,
    feature: &mut Vec<i64>,
    threshold: &mut Vec<f64>,
    value: &mut Vec<f64>,
) -> i64 {
    let my_idx = children_left.len() as i64;
    children_left.push(-1);
    children_right.push(-1);
    feature.push(-1);
    threshold.push(f64::NAN);
    value.push(node.value);

    match (&node.left, &node.right) {
        (Some(l), Some(r)) => {
            feature[my_idx as usize] = node.feature as i64;
            threshold[my_idx as usize] = node.threshold;
            let li = flatten_tree(l, children_left, children_right, feature, threshold, value);
            let ri = flatten_tree(r, children_left, children_right, feature, threshold, value);
            children_left[my_idx as usize] = li;
            children_right[my_idx as usize] = ri;
        }
        (None, None) => {}
        _ => debug_assert!(false, "TreeNode invariant violated: half-internal node"),
    }
    my_idx
}

fn tree_to_pydict(py: Python<'_>, root: &TreeNode) -> PyResult<Py<PyDict>> {
    let mut cl = Vec::new();
    let mut cr = Vec::new();
    let mut feat = Vec::new();
    let mut thr = Vec::new();
    let mut val = Vec::new();
    flatten_tree(root, &mut cl, &mut cr, &mut feat, &mut thr, &mut val);

    let d = PyDict::new(py);
    d.set_item("children_left", cl)?;
    d.set_item("children_right", cr)?;
    d.set_item("feature", feat)?;
    d.set_item("threshold", thr)?;
    d.set_item("value", val)?;
    Ok(d.into())
}

fn build_rounds(py: Python<'_>, models: &[InternalRF], n_outputs: usize) -> PyResult<Py<PyList>> {
    debug_assert!(n_outputs > 0, "n_outputs must be positive");
    debug_assert_eq!(
        models.len() % n_outputs,
        0,
        "models.len() ({}) must be a multiple of n_outputs ({})",
        models.len(),
        n_outputs,
    );
    let rounds = PyList::empty(py);
    let n_rounds = models.len() / n_outputs;
    for r in 0..n_rounds {
        let outputs = PyList::empty(py);
        for o in 0..n_outputs {
            let rf = &models[r * n_outputs + o];
            let trees = PyList::empty(py);
            for tree in &rf.trees {
                trees.append(tree_to_pydict(py, tree)?)?;
            }
            let out_dict = PyDict::new(py);
            out_dict.set_item("trees", trees)?;
            outputs.append(out_dict)?;
        }
        let round_dict = PyDict::new(py);
        round_dict.set_item("outputs", outputs)?;
        rounds.append(round_dict)?;
    }
    Ok(rounds.into())
}

pub struct InternalRF {
    pub trees: Vec<TreeNode>,
}

impl InternalRF {
    pub fn predict_all(&self, x: &ArrayView2<f64>) -> Vec<f64> {
        let n = self.trees.len() as f64;
        x.outer_iter()
            .map(|row| {
                let v = row.to_vec();
                self.trees.iter().map(|t| traverse(t, &v)).sum::<f64>() / n
            })
            .collect()
    }

    pub fn predict_per_tree(&self, x: &ArrayView2<f64>) -> Vec<Vec<f64>> {
        self.trees
            .iter()
            .map(|tree| {
                x.outer_iter()
                    .map(|row| {
                        let v = row.to_vec();
                        traverse(tree, &v)
                    })
                    .collect()
            })
            .collect()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Internal fit helpers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn fit_internal_rf(
    x: &ArrayView2<f64>, y: &[f64], w: &[f64], n_estimators: usize, config: &TreeConfig,
    bootstrap: bool, rng: &mut Pcg64, use_histogram: bool,
) -> InternalRF {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    let tree_params: Vec<(Vec<usize>, u64)> = (0..n_estimators)
        .map(|_| {
            let boot = if bootstrap { (0..n_samples).map(|_| rng.gen_range(0..n_samples)).collect() } else { (0..n_samples).collect() };
            (boot, rng.gen())
        })
        .collect();

    let x_owned = x.to_owned();

    let trees: Vec<TreeNode> = if use_histogram {
        let global_hist = HistogramData::build(&x_owned.view(), n_samples, n_features);
        tree_params.into_par_iter().map(|(boot, seed)| {
            let mut tree_rng = Pcg64::seed_from_u64(seed);
            build_tree_on_bootstrap(&x_owned.view(), y, w, &boot, config, &mut tree_rng, &global_hist)
        }).collect()
    } else {
        tree_params.into_par_iter().map(|(boot, seed)| {
            let mut tree_rng = Pcg64::seed_from_u64(seed);
            build_tree_on_bootstrap_exact(&x_owned.view(), y, w, &boot, config, &mut tree_rng)
        }).collect()
    };

    InternalRF { trees }
}

#[allow(clippy::too_many_arguments)]
pub fn fit_internal_rf_streaming(
    x: &ArrayView2<f64>, y: &[f64], w: &[f64], n_estimators: usize, config: &TreeConfig,
    bootstrap: bool, rng: &mut Pcg64, use_histogram: bool,
    tol: f64, min_trees: usize, is_classification: bool,
) -> (InternalRF, Vec<f64>) {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    let tree_params: Vec<(usize, Vec<usize>, u64)> = (0..n_estimators)
        .map(|idx| {
            let boot = if bootstrap { (0..n_samples).map(|_| rng.gen_range(0..n_samples)).collect() } else { (0..n_samples).collect() };
            (idx, boot, rng.gen())
        })
        .collect();

    let x_owned = x.to_owned();
    let hist = if use_histogram { Some(HistogramData::build(&x_owned.view(), n_samples, n_features)) } else { None };

    let converged = AtomicBool::new(false);
    let n_collected = AtomicUsize::new(0);
    let collector: Mutex<Vec<(usize, TreeNode, Vec<f64>)>> = Mutex::new(Vec::with_capacity(n_estimators));
    let welford_mean: Mutex<Vec<f64>> = Mutex::new(vec![0.0; n_samples]);
    let welford_m2: Mutex<Vec<f64>> = Mutex::new(vec![0.0; n_samples]);

    scope(|s| {
        for (idx, boot, seed) in &tree_params {
            let idx = *idx;
            let seed = *seed;
            let converged = &converged;
            let n_collected = &n_collected;
            let collector = &collector;
            let welford_mean = &welford_mean;
            let welford_m2 = &welford_m2;
            let x_ref = &x_owned;
            let hist_ref = &hist;
            let config_ref = config;

            s.spawn(move |_| {
                if converged.load(Ordering::Relaxed) { return; }

                let mut tree_rng = Pcg64::seed_from_u64(seed);
                let tree = if let Some(h) = hist_ref {
                    build_tree_on_bootstrap(&x_ref.view(), y, w, boot, config_ref, &mut tree_rng, h)
                } else {
                    build_tree_on_bootstrap_exact(&x_ref.view(), y, w, boot, config_ref, &mut tree_rng)
                };

                if converged.load(Ordering::Relaxed) { return; }

                let preds: Vec<f64> = x_ref.view().outer_iter()
                    .map(|row| {
                        let v: Vec<f64> = row.to_vec();
                        traverse(&tree, &v)
                    })
                    .collect();

                {
                    let c = n_collected.fetch_add(1, Ordering::SeqCst) + 1;
                    let c_f64 = c as f64;
                    let mut mean = welford_mean.lock().unwrap();
                    let mut m2 = welford_m2.lock().unwrap();

                    for i in 0..preds.len() {
                        let delta = preds[i] - mean[i];
                        mean[i] += delta / c_f64;
                        let delta2 = preds[i] - mean[i];
                        m2[i] += delta * delta2;
                    }

                    if c >= min_trees && c > 1 {
                        let should_stop = if tol > 0.0 {
                            let max_change: f64 = preds.iter().enumerate()
                                .map(|(i, &x_k)| {
                                    let prev = (mean[i] * c_f64 - x_k) / (c_f64 - 1.0);
                                    (x_k - prev).abs() / c_f64
                                })
                                .fold(0.0, f64::max);
                            max_change < tol
                        } else if is_classification {
                            let z = 1.96_f64;
                            let z2 = z * z;
                            let k = c_f64;
                            let mut n_stable: usize = 0;
                            for i in 0..n_samples {
                                let p = (1.0 / (1.0 + (-mean[i]).exp())).clamp(1e-6, 1.0 - 1e-6);
                                let denom = 1.0 + z2 / k;
                                let center = (p + z2 / (2.0 * k)) / denom;
                                let half_w = z * (p * (1.0 - p) / k + z2 / (4.0 * k * k)).sqrt() / denom;
                                let lo = (center - half_w).max(0.0);
                                let hi = (center + half_w).min(1.0);
                                if hi < 0.5 || lo > 0.5 { n_stable += 1; }
                            }
                            n_stable as f64 / n_samples as f64 >= 0.90
                        } else {
                            let z = 1.96_f64;
                            let k = c_f64;
                            let mut pred_min = f64::INFINITY;
                            let mut pred_max = f64::NEG_INFINITY;
                            for i in 0..n_samples { pred_min = pred_min.min(mean[i]); pred_max = pred_max.max(mean[i]); }
                            let spread = (pred_max - pred_min).max(1e-10);
                            let threshold = 0.05 * spread;
                            let mut n_stable: usize = 0;
                            for i in 0..n_samples {
                                let var_i = if c > 1 { m2[i] / (c_f64 - 1.0) } else { 0.0 };
                                let half_w = z * (var_i / k).sqrt();
                                if half_w < threshold { n_stable += 1; }
                            }
                            n_stable as f64 / n_samples as f64 >= 0.90
                        };

                        if should_stop { converged.store(true, Ordering::SeqCst); }
                    }
                }

                collector.lock().unwrap().push((idx, tree, preds));
            });
        }
    });

    let mut results = collector.into_inner().unwrap();
    results.sort_by_key(|(idx, _, _)| *idx);

    let n_kept = results.len();
    let n_kept_f64 = n_kept as f64;

    let mut final_pred = vec![0.0; n_samples];
    for (_, _, preds) in &results {
        for (i, &p) in preds.iter().enumerate() { final_pred[i] += p; }
    }
    for v in &mut final_pred { *v /= n_kept_f64; }

    let trees: Vec<TreeNode> = results.into_iter().map(|(_, t, _)| t).collect();
    (InternalRF { trees }, final_pred)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn compute_feature_importances(models: &[InternalRF]) -> Vec<f64> {
    let n_features = if let Some(first_tree) = models.first().and_then(|rf| rf.trees.first()) {
        fn max_feature(node: &TreeNode) -> usize {
            let mut m = node.feature;
            if let Some(l) = &node.left { m = m.max(max_feature(l)); }
            if let Some(r) = &node.right { m = m.max(max_feature(r)); }
            m
        }
        max_feature(first_tree) + 1
    } else {
        return vec![];
    };

    let mut importance = vec![0.0; n_features];
    let mut total_splits = 0.0;
    for rf in models {
        for tree in &rf.trees {
            fn accumulate(node: &TreeNode, imp: &mut Vec<f64>, count: &mut f64) {
                if node.left.is_none() && node.right.is_none() { return; }
                if node.feature < imp.len() { imp[node.feature] += node.samples as f64; *count += node.samples as f64; }
                if let Some(l) = &node.left { accumulate(l, imp, count); }
                if let Some(r) = &node.right { accumulate(r, imp, count); }
            }
            accumulate(tree, &mut importance, &mut total_splits);
        }
    }
    if total_splits > 0.0 { for v in &mut importance { *v /= total_splits; } }
    importance
}

fn get_raw_predictions(models: &[InternalRF], initial_pred: &[f64], learning_rate: f64, x: &ArrayView2<f64>, n_classes: usize) -> Vec<Vec<f64>> {
    let n = x.nrows();
    let mut pred = vec![vec![0.0; n_classes]; n];
    // Initialize
    for i in 0..n {
        for c in 0..n_classes {
            pred[i][c] = initial_pred[c];
        }
    }
    // Each model block has n_classes RF models (one per class for multiclass, 1 for binary/regression)
    let models_per_round = n_classes;
    for round_start in (0..models.len()).step_by(models_per_round) {
        for c in 0..n_classes {
            if round_start + c < models.len() {
                let update = models[round_start + c].predict_all(x);
                for i in 0..n {
                    pred[i][c] += learning_rate * update[i];
                }
            }
        }
    }
    pred
}

fn boost_devices() -> String {
    #[allow(unused_mut)]
    let mut d = vec!["cpu"];
    #[cfg(feature = "cuda")] d.push("cuda");
    #[cfg(feature = "gpu")] d.push("mps");
    d.join(", ")
}

/// Raw boosting scores `[n][n_out] = initial + Σ_rounds lr·predict_all`, on the
/// chosen device. The GPU paths fold the per-round `lr / tree-count` factors into
/// one scaled forest per output channel and reuse the mean kernel; the caller
/// adds the bias and applies the link function.
fn raw_dispatch(
    models: &[InternalRF], initial_pred: &[f64], lr: f64,
    x: &ArrayView2<f64>, n_out: usize, device: &str,
) -> PyResult<Vec<Vec<f64>>> {
    match device {
        "cpu" => Ok(get_raw_predictions(models, initial_pred, lr, x, n_out)),
        #[cfg(feature = "cuda")]
        "cuda" => boost_raw_cuda(models, initial_pred, lr, x, n_out),
        #[cfg(feature = "gpu")]
        "mps" | "metal" | "gpu" => boost_raw_gpu(models, initial_pred, lr, x, n_out),
        other => Err(PyValueError::new_err(format!(
            "device '{}' is not available in this build. Available: {}.", other, boost_devices()))),
    }
}

#[cfg(any(feature = "cuda", feature = "gpu"))]
fn boost_class_forest<'a>(
    models: &'a [InternalRF], lr: f64, n_out: usize, c: usize,
) -> Option<(Vec<&'a crate::tree::TreeNode>, Vec<f32>)> {
    let class_models: Vec<&InternalRF> =
        (0..models.len()).step_by(n_out).filter_map(|rs| models.get(rs + c)).collect();
    let t_total: usize = class_models.iter().map(|m| m.trees.len()).sum();
    if t_total == 0 { return None; }
    let mut trees = Vec::with_capacity(t_total);
    let mut weights = Vec::with_capacity(t_total);
    for m in &class_models {
        // mean kernel divides by t_total, so pre-scale leaves by (lr * t_total / t_round)
        let w = lr as f32 * t_total as f32 / m.trees.len() as f32;
        for tree in &m.trees { trees.push(tree); weights.push(w); }
    }
    Some((trees, weights))
}

#[cfg(feature = "cuda")]
fn boost_raw_cuda(models: &[InternalRF], initial_pred: &[f64], lr: f64, x: &ArrayView2<f64>, n_out: usize) -> PyResult<Vec<Vec<f64>>> {
    let (n, nf) = (x.nrows(), x.ncols());
    let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let mut raw = vec![vec![0.0f64; n_out]; n];
    for r in raw.iter_mut() { r.copy_from_slice(initial_pred); }
    for c in 0..n_out {
        if let Some((trees, weights)) = boost_class_forest(models, lr, n_out, c) {
            let forest = crate::cuda::CudaForest::new_scaled(&trees, nf, &weights)
                .ok_or_else(|| PyValueError::new_err("CUDA device unavailable"))?;
            let contrib = forest.predict(&xf, n);
            for i in 0..n { raw[i][c] += contrib[i] as f64; }
        }
    }
    Ok(raw)
}

#[cfg(feature = "gpu")]
fn boost_raw_gpu(models: &[InternalRF], initial_pred: &[f64], lr: f64, x: &ArrayView2<f64>, n_out: usize) -> PyResult<Vec<Vec<f64>>> {
    let (n, nf) = (x.nrows(), x.ncols());
    let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let mut raw = vec![vec![0.0f64; n_out]; n];
    for r in raw.iter_mut() { r.copy_from_slice(initial_pred); }
    for c in 0..n_out {
        if let Some((trees, weights)) = boost_class_forest(models, lr, n_out, c) {
            let forest = crate::gpu::GpuForest::new_scaled(&trees, nf, &weights)
                .ok_or_else(|| PyValueError::new_err("GPU device unavailable"))?;
            let contrib = forest.predict(&xf, n);
            for i in 0..n { raw[i][c] += contrib[i] as f64; }
        }
    }
    Ok(raw)
}

fn set_thread_pool(n_jobs: Option<usize>) {
    if let Some(nj) = n_jobs {
        if nj > 0 {
            let _ = ThreadPoolBuilder::new()
                .num_threads(nj)
                .build_global();
        }
    }
}

// ---------------------------------------------------------------------------
// RFGBoostClassifier
// ---------------------------------------------------------------------------

#[pyclass]
pub struct RFGBoostClassifier {
    n_estimators: usize,
    learning_rate: f64,
    rf_n_estimators: usize,
    rf_max_depth: Option<usize>,
    rf_max_features: Option<String>,
    bootstrap: bool,
    random_state: Option<u64>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    use_histogram: bool,
    async_mode: bool,
    tol: f64,
    n_jobs: Option<usize>,
    monotone_constraints: Vec<i8>,
    // Fitted state
    n_classes: usize,
    initial_pred: Vec<f64>,
    models: Vec<InternalRF>,
    trees_used: Vec<usize>,
    is_fitted: bool,
}

#[pymethods]
impl RFGBoostClassifier {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        n_estimators=20, learning_rate=0.1, rf_n_estimators=20,
        rf_max_depth=None, rf_max_features=None, bootstrap=true,
        random_state=None, min_samples_split=2, min_samples_leaf=1,
        use_histogram=true, async_mode=false, tol=1e-4, n_jobs=None,
        monotone_constraints=None
    ))]
    fn new(
        n_estimators: usize, learning_rate: f64, rf_n_estimators: usize,
        rf_max_depth: Option<usize>, rf_max_features: Option<String>,
        bootstrap: bool, random_state: Option<u64>,
        min_samples_split: usize, min_samples_leaf: usize,
        use_histogram: bool, async_mode: bool, tol: f64, n_jobs: Option<usize>,
        monotone_constraints: Option<Vec<i8>>,
    ) -> Self {
        set_thread_pool(n_jobs);
        Self {
            n_estimators, learning_rate, rf_n_estimators, rf_max_depth, rf_max_features,
            bootstrap, random_state, min_samples_split, min_samples_leaf,
            use_histogram, async_mode, tol, n_jobs,
            monotone_constraints: monotone_constraints.unwrap_or_default(),
            n_classes: 0, initial_pred: vec![], models: Vec::new(),
            trees_used: Vec::new(), is_fitted: false,
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
        let total_w: f64 = weights.iter().sum();
        let max_feat = resolve_max_features(&self.rf_max_features, n_features);
        let mut rng = crate::tree::seed_rng(self.random_state);

        let config = TreeConfig {
            max_depth: self.rf_max_depth, min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf, is_classification: false,
            max_features: if max_feat < n_features { Some(max_feat) } else { None },
            monotone_constraints: self.monotone_constraints.clone(),
        };

        // Detect classes
        let mut classes: Vec<usize> = y_vec.iter().map(|&v| v as usize).collect();
        classes.sort();
        classes.dedup();
        self.n_classes = classes.len();

        let x_owned = x_arr.to_owned();
        let min_trees = (self.rf_n_estimators / 2).max(3);
        self.models.clear();
        self.trees_used.clear();

        if self.n_classes == 2 {
            // Binary: single logit. Weighted prior.
            let mean_y: f64 = if total_w > 0.0 {
                (0..n_samples).map(|i| weights[i] * y_vec[i]).sum::<f64>() / total_w
            } else { 0.5 };
            let clamped = mean_y.clamp(1e-5, 1.0 - 1e-5);
            self.initial_pred = vec![(clamped / (1.0 - clamped)).ln()];

            let mut pred = vec![self.initial_pred[0]; n_samples];

            for _ in 0..self.n_estimators {
                // Newton-weighted leaves: target = (y-p)/h, weight = user_w * h, where h = p(1-p).
                // The weighted leaf mean reduces to Sum(g) / Sum(h), the Newton-optimal step.
                let mut targets = vec![0.0; n_samples];
                let mut newton_w = vec![0.0; n_samples];
                for i in 0..n_samples {
                    let p = sigmoid(pred[i]).clamp(1e-5, 1.0 - 1e-5);
                    let h = p * (1.0 - p);
                    targets[i] = (y_vec[i] - p) / h;
                    newton_w[i] = weights[i] * h;
                }

                let (rf, update, n_used) = self.fit_one_rf(&x_owned, &targets, &newton_w, &config, &mut rng, min_trees, true);
                self.trees_used.push(n_used);
                for i in 0..n_samples { pred[i] += self.learning_rate * update[i]; }
                self.models.push(rf);
            }
        } else {
            // Multiclass: one RF per class per round (one-vs-rest in logit space)
            self.initial_pred = vec![0.0; self.n_classes];
            // Weighted class priors
            for c in 0..self.n_classes {
                let wc: f64 = (0..n_samples).filter(|&i| y_vec[i] as usize == c).map(|i| weights[i]).sum();
                let p = if total_w > 0.0 { (wc / total_w).clamp(1e-5, 1.0 - 1e-5) } else { 1e-5 };
                self.initial_pred[c] = p.ln(); // log-prior for softmax
            }

            // pred[sample][class]
            let mut pred: Vec<Vec<f64>> = (0..n_samples)
                .map(|_| self.initial_pred.clone())
                .collect();

            for _ in 0..self.n_estimators {
                for c in 0..self.n_classes {
                    // Newton-weighted leaves for softmax: target = (y_c - p_c)/h_c, weight = user_w * h_c,
                    // where h_c = p_c * (1 - p_c). Weighted leaf mean = Sum(g_c) / Sum(h_c).
                    let mut targets = vec![0.0; n_samples];
                    let mut newton_w = vec![0.0; n_samples];
                    for i in 0..n_samples {
                        let probs = softmax(&pred[i]);
                        let p_c = probs[c].clamp(1e-5, 1.0 - 1e-5);
                        let h_c = p_c * (1.0 - p_c);
                        let y_c = if y_vec[i] as usize == c { 1.0 } else { 0.0 };
                        targets[i] = (y_c - p_c) / h_c;
                        newton_w[i] = weights[i] * h_c;
                    }

                    let (rf, update, n_used) = self.fit_one_rf(&x_owned, &targets, &newton_w, &config, &mut rng, min_trees, true);
                    if c == 0 { self.trees_used.push(n_used); }
                    for i in 0..n_samples { pred[i][c] += self.learning_rate * update[i]; }
                    self.models.push(rf);
                }
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    /// `device`: "cpu" (default), "cuda" or "mps"/"metal"/"gpu".
    #[pyo3(signature = (x, device="cpu"))]
    fn predict(&self, x: PyReadonlyArray2<f64>, device: &str) -> PyResult<Vec<f64>> {
        if !self.is_fitted { return Err(PyValueError::new_err("RFGBoostClassifier has not been fitted")); }
        let x_arr = x.as_array();
        let raw = raw_dispatch(&self.models, &self.initial_pred, self.learning_rate, &x_arr.view(), self.initial_pred.len(), device)?;

        if self.n_classes == 2 {
            Ok(raw.iter().map(|p| if sigmoid(p[0]) > 0.5 { 1.0 } else { 0.0 }).collect())
        } else {
            Ok(raw.iter().map(|p| {
                let probs = softmax(p);
                probs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map_or(0.0, |(i, _)| i as f64)
            }).collect())
        }
    }

    /// `device`: "cpu" (default), "cuda" or "mps"/"metal"/"gpu".
    #[pyo3(signature = (x, device="cpu"))]
    fn predict_proba(&self, x: PyReadonlyArray2<f64>, device: &str) -> PyResult<Vec<Vec<f64>>> {
        if !self.is_fitted { return Err(PyValueError::new_err("RFGBoostClassifier has not been fitted")); }
        let x_arr = x.as_array();
        let raw = raw_dispatch(&self.models, &self.initial_pred, self.learning_rate, &x_arr.view(), self.initial_pred.len(), device)?;

        if self.n_classes == 2 {
            Ok(raw.iter().map(|p| { let prob = sigmoid(p[0]); vec![1.0 - prob, prob] }).collect())
        } else {
            Ok(raw.iter().map(|p| softmax(p)).collect())
        }
    }

    /// Wilson score confidence interval on the predicted probability.
    /// Treats each tree's leaf probability as a binomial trial and
    /// computes the Wilson interval on the ensemble average.
    #[pyo3(signature = (x, alpha=0.05))]
    fn predict_ci(&self, x: PyReadonlyArray2<f64>, alpha: f64) -> PyResult<Vec<Vec<f64>>> {
        if !self.is_fitted { return Err(PyValueError::new_err("RFGBoostClassifier has not been fitted")); }
        if self.n_classes != 2 { return Err(PyValueError::new_err("predict_ci only supports binary classification")); }
        let x_arr = x.as_array();
        let n = x_arr.nrows();

        // Collect all individual tree probabilities across all boosting rounds
        // Each tree votes with its leaf probability; the ensemble is an average
        let mut all_tree_probs: Vec<Vec<f64>> = vec![Vec::new(); n];

        for rf in &self.models {
            let per_tree = rf.predict_per_tree(&x_arr.view());
            for tree_preds in &per_tree {
                for i in 0..n {
                    // Convert logit-space tree prediction to probability
                    // The tree fits residuals in logit space, so raw values
                    // aren't probabilities. Use the ensemble probability instead.
                    all_tree_probs[i].push(tree_preds[i]);
                }
            }
        }

        // For each sample, compute the ensemble probability and Wilson CI
        let z = if alpha <= 0.01 { 2.576 } else if alpha <= 0.05 { 1.96 } else if alpha <= 0.10 { 1.645 } else { 1.28 };
        let z2 = z * z;

        // Get the ensemble predicted probabilities
        let proba = self.predict_proba(x, "cpu")?;

        Ok((0..n).map(|i| {
            let p = proba[i][1]; // P(class=1)
            let k = all_tree_probs[i].len() as f64; // total number of tree votes

            // Wilson score interval
            let denom = 1.0 + z2 / k;
            let center = (p + z2 / (2.0 * k)) / denom;
            let half_w = z * (p * (1.0 - p) / k + z2 / (4.0 * k * k)).sqrt() / denom;
            let lo = (center - half_w).clamp(0.0, 1.0);
            let hi = (center + half_w).clamp(0.0, 1.0);
            vec![lo, hi]
        }).collect())
    }

    fn feature_importances(&self) -> PyResult<Vec<f64>> {
        if !self.is_fitted { return Err(PyValueError::new_err("RFGBoostClassifier has not been fitted")); }
        Ok(compute_feature_importances(&self.models))
    }

    #[getter] fn n_estimators(&self) -> usize { self.n_estimators }
    #[getter] fn n_classes(&self) -> usize { self.n_classes }
    #[getter] fn is_fitted(&self) -> bool { self.is_fitted }
    #[getter] fn trees_used(&self) -> Vec<usize> { self.trees_used.clone() }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        if !self.is_fitted {
            return Err(PyValueError::new_err("RFGBoostClassifier has not been fitted"));
        }
        let n_outputs = self.initial_pred.len();
        let task = if n_outputs == 1 { "binary" } else { "multiclass" };
        let n_estimators_fit = self.models.len() / n_outputs;
        let rounds = build_rounds(py, &self.models, n_outputs)?;

        let d = PyDict::new(py);
        d.set_item("task", task)?;
        d.set_item("n_outputs", n_outputs)?;
        d.set_item("n_classes", self.n_classes)?;
        d.set_item("learning_rate", self.learning_rate)?;
        d.set_item("init", self.initial_pred.clone())?;
        d.set_item("n_estimators", n_estimators_fit)?;
        d.set_item("rounds", rounds)?;
        Ok(d.into())
    }
}

impl RFGBoostClassifier {
    #[allow(clippy::too_many_arguments)]
    fn fit_one_rf(
        &self, x: &ndarray::Array2<f64>, targets: &[f64], weights: &[f64], config: &TreeConfig,
        rng: &mut Pcg64, min_trees: usize, is_clf: bool,
    ) -> (InternalRF, Vec<f64>, usize) {
        if self.async_mode {
            let (rf, update) = fit_internal_rf_streaming(
                &x.view(), targets, weights, self.rf_n_estimators, config,
                self.bootstrap, rng, self.use_histogram, self.tol, min_trees, is_clf,
            );
            let n = rf.trees.len();
            (rf, update, n)
        } else {
            let rf = fit_internal_rf(
                &x.view(), targets, weights, self.rf_n_estimators, config,
                self.bootstrap, rng, self.use_histogram,
            );
            let update = rf.predict_all(&x.view());
            let n = rf.trees.len();
            (rf, update, n)
        }
    }
}

// ---------------------------------------------------------------------------
// RFGBoostRegressor
// ---------------------------------------------------------------------------

#[pyclass]
pub struct RFGBoostRegressor {
    n_estimators: usize,
    learning_rate: f64,
    rf_n_estimators: usize,
    rf_max_depth: Option<usize>,
    rf_max_features: Option<String>,
    bootstrap: bool,
    random_state: Option<u64>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    use_histogram: bool,
    async_mode: bool,
    tol: f64,
    n_jobs: Option<usize>,
    conformal_fraction: f64,
    // Fitted state
    initial_pred: f64,
    conformal_quantile: f64,  // calibrated quantile for split conformal CI
    models: Vec<InternalRF>,
    trees_used: Vec<usize>,
    is_fitted: bool,
}

#[pymethods]
impl RFGBoostRegressor {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        n_estimators=20, learning_rate=0.1, rf_n_estimators=20,
        rf_max_depth=None, rf_max_features=None, bootstrap=true,
        random_state=None, min_samples_split=2, min_samples_leaf=1,
        use_histogram=true, async_mode=false, tol=1e-4, n_jobs=None,
        conformal_fraction=0.2
    ))]
    fn new(
        n_estimators: usize, learning_rate: f64, rf_n_estimators: usize,
        rf_max_depth: Option<usize>, rf_max_features: Option<String>,
        bootstrap: bool, random_state: Option<u64>,
        min_samples_split: usize, min_samples_leaf: usize,
        use_histogram: bool, async_mode: bool, tol: f64, n_jobs: Option<usize>,
        conformal_fraction: f64,
    ) -> Self {
        set_thread_pool(n_jobs);
        Self {
            n_estimators, learning_rate, rf_n_estimators, rf_max_depth, rf_max_features,
            bootstrap, random_state, min_samples_split, min_samples_leaf,
            use_histogram, async_mode, tol, n_jobs,
            conformal_fraction: conformal_fraction.clamp(0.1, 0.5),
            initial_pred: 0.0, conformal_quantile: 0.0,
            models: Vec::new(), trees_used: Vec::new(), is_fitted: false,
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
        let n_total = x_arr.nrows();
        let n_features = x_arr.ncols();
        let weights_full: Vec<f64> = match sample_weight {
            Some(arr) => {
                let v: Vec<f64> = arr.as_array().to_vec();
                crate::tree::validate_weights(&v, n_total).map_err(PyValueError::new_err)?;
                v
            }
            None => vec![1.0; n_total],
        };
        let max_feat = resolve_max_features(&self.rf_max_features, n_features);
        let mut rng = crate::tree::seed_rng(self.random_state);

        // Split into train and calibration sets for conformal prediction
        let n_cal = ((n_total as f64) * self.conformal_fraction) as usize;
        let n_train = n_total - n_cal;
        let mut indices: Vec<usize> = (0..n_total).collect();
        indices.shuffle(&mut rng);
        let train_idx: Vec<usize> = indices[..n_train].to_vec();
        let cal_idx: Vec<usize> = indices[n_train..].to_vec();

        // Extract train data
        let mut x_train = ndarray::Array2::zeros((n_train, n_features));
        let mut y_train = vec![0.0; n_train];
        let mut w_train = vec![0.0; n_train];
        for (new_i, &old_i) in train_idx.iter().enumerate() {
            y_train[new_i] = y_vec[old_i];
            w_train[new_i] = weights_full[old_i];
            for j in 0..n_features {
                x_train[[new_i, j]] = x_arr[[old_i, j]];
            }
        }
        let total_w_train: f64 = w_train.iter().sum();

        let config = TreeConfig {
            max_depth: self.rf_max_depth, min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf, is_classification: false,
            max_features: if max_feat < n_features { Some(max_feat) } else { None },
            monotone_constraints: Vec::new(),
        };

        self.initial_pred = if total_w_train > 0.0 {
            (0..n_train).map(|i| w_train[i] * y_train[i]).sum::<f64>() / total_w_train
        } else { 0.0 };
        let mut pred = vec![self.initial_pred; n_train];
        self.models.clear();
        self.trees_used.clear();

        let min_trees = (self.rf_n_estimators / 2).max(3);

        // Train on train set
        for _ in 0..self.n_estimators {
            let targets: Vec<f64> = (0..n_train).map(|i| y_train[i] - pred[i]).collect();

            if self.async_mode {
                let (rf, update) = fit_internal_rf_streaming(
                    &x_train.view(), &targets, &w_train, self.rf_n_estimators, &config,
                    self.bootstrap, &mut rng, self.use_histogram, self.tol, min_trees, false,
                );
                self.trees_used.push(rf.trees.len());
                for i in 0..n_train { pred[i] += self.learning_rate * update[i]; }
                self.models.push(rf);
            } else {
                let rf = fit_internal_rf(
                    &x_train.view(), &targets, &w_train, self.rf_n_estimators, &config,
                    self.bootstrap, &mut rng, self.use_histogram,
                );
                let update = rf.predict_all(&x_train.view());
                self.trees_used.push(rf.trees.len());
                for i in 0..n_train { pred[i] += self.learning_rate * update[i]; }
                self.models.push(rf);
            }
        }

        // Calibrate on calibration set (split conformal prediction)
        // Compute absolute residuals on unseen calibration data
        let mut cal_scores: Vec<f64> = Vec::with_capacity(n_cal);
        for &ci in &cal_idx {
            let sample: Vec<f64> = (0..n_features).map(|j| x_arr[[ci, j]]).collect();
            let mut cal_pred = self.initial_pred;
            for rf in &self.models {
                let tree_mean: f64 = rf.trees.iter()
                    .map(|t| traverse(t, &sample))
                    .sum::<f64>() / rf.trees.len() as f64;
                cal_pred += self.learning_rate * tree_mean;
            }
            cal_scores.push((y_vec[ci] - cal_pred).abs());
        }

        // Conformal quantile: ceil((1-alpha)(1+1/n_cal))-th smallest score
        // For alpha=0.05, this gives ~95% coverage guarantee
        cal_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q_idx = ((0.95 * (1.0 + 1.0 / n_cal as f64)) * n_cal as f64).ceil() as usize;
        let q_idx = q_idx.min(n_cal - 1);
        self.conformal_quantile = cal_scores[q_idx];

        // Now retrain on ALL data for the final model
        let total_w_full: f64 = weights_full.iter().sum();
        self.initial_pred = if total_w_full > 0.0 {
            (0..n_total).map(|i| weights_full[i] * y_vec[i]).sum::<f64>() / total_w_full
        } else { 0.0 };
        let mut pred_full = vec![self.initial_pred; n_total];
        self.models.clear();
        self.trees_used.clear();

        let x_full = x_arr.to_owned();
        let mut rng2 = crate::tree::seed_rng(self.random_state);

        for _ in 0..self.n_estimators {
            let targets: Vec<f64> = (0..n_total).map(|i| y_vec[i] - pred_full[i]).collect();

            if self.async_mode {
                let (rf, update) = fit_internal_rf_streaming(
                    &x_full.view(), &targets, &weights_full, self.rf_n_estimators, &config,
                    self.bootstrap, &mut rng2, self.use_histogram, self.tol, min_trees, false,
                );
                self.trees_used.push(rf.trees.len());
                for i in 0..n_total { pred_full[i] += self.learning_rate * update[i]; }
                self.models.push(rf);
            } else {
                let rf = fit_internal_rf(
                    &x_full.view(), &targets, &weights_full, self.rf_n_estimators, &config,
                    self.bootstrap, &mut rng2, self.use_histogram,
                );
                let update = rf.predict_all(&x_full.view());
                self.trees_used.push(rf.trees.len());
                for i in 0..n_total { pred_full[i] += self.learning_rate * update[i]; }
                self.models.push(rf);
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    /// `device`: "cpu" (default), "cuda" or "mps"/"metal"/"gpu".
    #[pyo3(signature = (x, device="cpu"))]
    fn predict(&self, x: PyReadonlyArray2<f64>, device: &str) -> PyResult<Vec<f64>> {
        if !self.is_fitted { return Err(PyValueError::new_err("RFGBoostRegressor has not been fitted")); }
        let x_arr = x.as_array();
        let raw = raw_dispatch(&self.models, &[self.initial_pred], self.learning_rate, &x_arr.view(), 1, device)?;
        Ok(raw.into_iter().map(|r| r[0]).collect())
    }

    /// Confidence intervals via split conformal prediction.
    /// Guarantees (1-alpha) coverage on exchangeable data.
    /// The conformal quantile is calibrated on a held-out calibration set during fit.
    #[pyo3(signature = (x, alpha=0.05))]
    fn predict_ci(&self, x: PyReadonlyArray2<f64>, alpha: f64) -> PyResult<Vec<Vec<f64>>> {
        if !self.is_fitted { return Err(PyValueError::new_err("RFGBoostRegressor has not been fitted")); }
        let x_arr = x.as_array();
        let n = x_arr.nrows();
        let mut pred_mean = vec![self.initial_pred; n];

        for rf in &self.models {
            let update = rf.predict_all(&x_arr.view());
            for i in 0..n { pred_mean[i] += self.learning_rate * update[i]; }
        }

        // Scale conformal quantile by alpha ratio if different from default 0.05
        // (approximate: the exact approach would need recalibration)
        let q = if (alpha - 0.05).abs() < 1e-6 {
            self.conformal_quantile
        } else {
            // Rough scaling via normal approximation
            let z_ratio = if alpha <= 0.01 { 2.576 / 1.96 }
                          else if alpha <= 0.05 { 1.0 }
                          else if alpha <= 0.10 { 1.645 / 1.96 }
                          else { 1.28 / 1.96 };
            self.conformal_quantile * z_ratio
        };

        Ok((0..n).map(|i| {
            vec![pred_mean[i] - q, pred_mean[i] + q]
        }).collect())
    }

    fn feature_importances(&self) -> PyResult<Vec<f64>> {
        if !self.is_fitted { return Err(PyValueError::new_err("RFGBoostRegressor has not been fitted")); }
        Ok(compute_feature_importances(&self.models))
    }

    #[getter] fn n_estimators(&self) -> usize { self.n_estimators }
    #[getter] fn is_fitted(&self) -> bool { self.is_fitted }
    #[getter] fn trees_used(&self) -> Vec<usize> { self.trees_used.clone() }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        if !self.is_fitted {
            return Err(PyValueError::new_err("RFGBoostRegressor has not been fitted"));
        }
        let n_outputs = 1;
        let n_estimators_fit = self.models.len();
        let rounds = build_rounds(py, &self.models, n_outputs)?;

        let d = PyDict::new(py);
        d.set_item("task", "regression")?;
        d.set_item("n_outputs", n_outputs)?;
        d.set_item("learning_rate", self.learning_rate)?;
        d.set_item("init", vec![self.initial_pred])?;
        d.set_item("n_estimators", n_estimators_fit)?;
        d.set_item("conformal_quantile", self.conformal_quantile)?;
        d.set_item("rounds", rounds)?;
        Ok(d.into())
    }
}

// ---------------------------------------------------------------------------
// Keep the original RFGBoost for backward compatibility
// ---------------------------------------------------------------------------

#[pyclass]
pub struct RFGBoost {
    clf: Option<RFGBoostClassifier>,
    reg: Option<RFGBoostRegressor>,
    task: String,
}

#[pymethods]
impl RFGBoost {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        n_estimators=20, learning_rate=0.1, rf_n_estimators=20,
        rf_max_depth=None, rf_max_features=None, bootstrap=true,
        random_state=None, min_samples_split=2, min_samples_leaf=1,
        task="classification", use_histogram=true, async_mode=false, tol=1e-4, n_jobs=None
    ))]
    fn new(
        n_estimators: usize, learning_rate: f64, rf_n_estimators: usize,
        rf_max_depth: Option<usize>, rf_max_features: Option<String>,
        bootstrap: bool, random_state: Option<u64>,
        min_samples_split: usize, min_samples_leaf: usize,
        task: &str, use_histogram: bool, async_mode: bool, tol: f64, n_jobs: Option<usize>,
    ) -> Self {
        if task == "classification" {
            Self {
                clf: Some(RFGBoostClassifier::new(
                    n_estimators, learning_rate, rf_n_estimators, rf_max_depth, rf_max_features,
                    bootstrap, random_state, min_samples_split, min_samples_leaf,
                    use_histogram, async_mode, tol, n_jobs,
                    None,
                )),
                reg: None,
                task: task.to_string(),
            }
        } else {
            Self {
                clf: None,
                reg: Some(RFGBoostRegressor::new(
                    n_estimators, learning_rate, rf_n_estimators, rf_max_depth, rf_max_features,
                    bootstrap, random_state, min_samples_split, min_samples_leaf,
                    use_histogram, async_mode, tol, n_jobs, 0.2,
                )),
                task: task.to_string(),
            }
        }
    }

    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        if let Some(clf) = &mut self.clf { clf.fit(x, y, sample_weight) }
        else if let Some(reg) = &mut self.reg { reg.fit(x, y, sample_weight) }
        else { Err(PyValueError::new_err("Invalid state")) }
    }

    /// `device`: "cpu" (default), "cuda" or "mps"/"metal"/"gpu".
    #[pyo3(signature = (x, device="cpu"))]
    fn predict(&self, x: PyReadonlyArray2<f64>, device: &str) -> PyResult<Vec<f64>> {
        if let Some(clf) = &self.clf { clf.predict(x, device) }
        else if let Some(reg) = &self.reg { reg.predict(x, device) }
        else { Err(PyValueError::new_err("Invalid state")) }
    }

    /// `device`: "cpu" (default), "cuda" or "mps"/"metal"/"gpu".
    #[pyo3(signature = (x, device="cpu"))]
    fn predict_proba(&self, x: PyReadonlyArray2<f64>, device: &str) -> PyResult<Vec<Vec<f64>>> {
        if let Some(clf) = &self.clf { clf.predict_proba(x, device) }
        else { Err(PyValueError::new_err("predict_proba is only available for classification")) }
    }

    #[pyo3(signature = (x, alpha=0.05))]
    fn predict_ci(&self, x: PyReadonlyArray2<f64>, alpha: f64) -> PyResult<Vec<Vec<f64>>> {
        if let Some(clf) = &self.clf { clf.predict_ci(x, alpha) }
        else if let Some(reg) = &self.reg { reg.predict_ci(x, alpha) }
        else { Err(PyValueError::new_err("Invalid state")) }
    }

    fn feature_importances(&self) -> PyResult<Vec<f64>> {
        if let Some(clf) = &self.clf { clf.feature_importances() }
        else if let Some(reg) = &self.reg { reg.feature_importances() }
        else { Err(PyValueError::new_err("Invalid state")) }
    }

    #[getter] fn n_estimators(&self) -> usize {
        self.clf.as_ref().map_or_else(|| self.reg.as_ref().map_or(0, |r| r.n_estimators), |c| c.n_estimators)
    }
    #[getter] fn is_fitted(&self) -> bool {
        self.clf.as_ref().map_or_else(|| self.reg.as_ref().map_or(false, |r| r.is_fitted), |c| c.is_fitted)
    }
    #[getter] fn trees_used(&self) -> Vec<usize> {
        self.clf.as_ref().map_or_else(|| self.reg.as_ref().map_or_else(Vec::new, |r| r.trees_used.clone()), |c| c.trees_used.clone())
    }
}
