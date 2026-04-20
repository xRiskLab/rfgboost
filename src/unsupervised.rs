use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_pcg::Pcg64;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::histogram::HistogramData;
use crate::tree::{
    build_tree_on_bootstrap, resolve_max_features, traverse, TreeConfig, TreeNode,
};

/// Find which terminal node a sample lands in.
fn terminal_node_id(node: &TreeNode, sample: &[f64], id: usize) -> usize {
    if node.left.is_none() && node.right.is_none() {
        return id;
    }
    if sample[node.feature] <= node.threshold {
        terminal_node_id(node.left.as_ref().unwrap(), sample, 2 * id + 1)
    } else {
        terminal_node_id(node.right.as_ref().unwrap(), sample, 2 * id + 2)
    }
}

/// Sparse proximity entry: stores k nearest neighbors per sample
#[derive(Clone)]
struct SparseProximityRow {
    neighbors: Vec<(usize, f64)>, // (index, proximity)
}

impl SparseProximityRow {
    fn new() -> Self {
        Self { neighbors: Vec::new() }
    }

    fn insert(&mut self, idx: usize, prox: f64, max_neighbors: usize) {
        // Find if neighbor already exists
        if let Some(pos) = self.neighbors.iter().position(|(i, _)| *i == idx) {
            self.neighbors[pos].1 += prox;
        } else {
            self.neighbors.push((idx, prox));
        }
        
        // Keep only top k neighbors if exceeded
        if self.neighbors.len() > max_neighbors * 2 {
            self.neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            self.neighbors.truncate(max_neighbors);
        }
    }

    fn finalize(&mut self, max_neighbors: usize, n_trees: f64) {
        // Normalize by number of trees and keep top k
        for (_, p) in &mut self.neighbors {
            *p /= n_trees;
        }
        self.neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        self.neighbors.truncate(max_neighbors);
    }

    #[allow(dead_code)]
    fn get(&self, idx: usize) -> f64 {
        self.neighbors.iter()
            .find(|(i, _)| *i == idx)
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    }

    fn sum_squared(&self) -> f64 {
        self.neighbors.iter().map(|(_, p)| p * p).sum()
    }
}

#[pyclass]
pub struct RandomForestUnsupervised {
    n_estimators: usize,
    max_depth: Option<usize>,
    max_features: Option<String>,
    random_state: Option<u64>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    n_jobs: Option<usize>,
    n_neighbors: usize, // For sparse proximity storage
    // Fitted state
    trees: Vec<TreeNode>,
    oob_indices: Vec<HashSet<usize>>, // OOB samples per tree (for original data only)
    x_train: Option<Array2<f64>>,
    n_original: usize, // Number of original samples (before synthetic)
    is_fitted: bool,
    // Cached sparse proximity
    sparse_prox: Option<Vec<SparseProximityRow>>,
}

#[pymethods]
impl RandomForestUnsupervised {
    #[new]
    #[pyo3(signature = (
        n_estimators=500, max_depth=None, max_features=None,
        random_state=None, min_samples_split=2, min_samples_leaf=1, 
        n_jobs=None, n_neighbors=None
    ))]
    fn new(
        n_estimators: usize, max_depth: Option<usize>, max_features: Option<String>,
        random_state: Option<u64>, min_samples_split: usize, min_samples_leaf: usize,
        n_jobs: Option<usize>, n_neighbors: Option<usize>,
    ) -> Self {
        if let Some(nj) = n_jobs {
            if nj > 0 {
                let _ = rayon::ThreadPoolBuilder::new().num_threads(nj).build_global();
            }
        }
        Self {
            n_estimators, max_depth, max_features, random_state,
            min_samples_split, min_samples_leaf, n_jobs,
            n_neighbors: n_neighbors.unwrap_or(0), // 0 means full matrix
            trees: Vec::new(), 
            oob_indices: Vec::new(),
            x_train: None, 
            n_original: 0,
            is_fitted: false,
            sparse_prox: None,
        }
    }

    /// Fit the unsupervised random forest using Breiman's synthetic data method.
    /// Creates a synthetic dataset by permuting each feature independently,
    /// labels original data as class 1 and synthetic as class 0,
    /// then trains a classification RF.
    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_arr = x.as_array();
        let n_samples = x_arr.nrows();
        let n_features = x_arr.ncols();
        let mut rng = Pcg64::seed_from_u64(self.random_state.unwrap_or(42));

        self.n_original = n_samples;

        // Generate synthetic data by permuting each column independently
        let mut x_synth = Array2::zeros((n_samples, n_features));
        for col in 0..n_features {
            let mut col_vals: Vec<f64> = (0..n_samples).map(|i| x_arr[[i, col]]).collect();
            col_vals.shuffle(&mut rng);
            for i in 0..n_samples {
                x_synth[[i, col]] = col_vals[i];
            }
        }

        // Combine: original (label=1) + synthetic (label=0)
        let n_total = 2 * n_samples;
        let mut x_combined = Array2::zeros((n_total, n_features));
        let mut y_combined = vec![0.0; n_total];

        for i in 0..n_samples {
            for j in 0..n_features {
                x_combined[[i, j]] = x_arr[[i, j]];
                x_combined[[n_samples + i, j]] = x_synth[[i, j]];
            }
            y_combined[i] = 1.0;
            y_combined[n_samples + i] = 0.0;
        }

        let max_feat = resolve_max_features(&self.max_features, n_features);
        let config = TreeConfig {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            is_classification: true,
            max_features: if max_feat < n_features { Some(max_feat) } else { None },
        };

        // Pre-compute bootstrap params and track OOB for original samples
        let mut tree_params: Vec<(Vec<usize>, u64, HashSet<usize>)> = Vec::with_capacity(self.n_estimators);
        for _ in 0..self.n_estimators {
            let boot: Vec<usize> = (0..n_total)
                .map(|_| rng.gen_range(0..n_total))
                .collect();
            
            // Track which original samples (0..n_samples) are OOB
            let in_bag: HashSet<usize> = boot.iter()
                .filter(|&&idx| idx < n_samples)
                .cloned()
                .collect();
            let oob: HashSet<usize> = (0..n_samples)
                .filter(|i| !in_bag.contains(i))
                .collect();
            
            tree_params.push((boot, rng.gen(), oob));
        }

        let global_hist = HistogramData::build(&x_combined.view(), n_total, n_features);

        // Build trees in parallel
        let results: Vec<(TreeNode, HashSet<usize>)> = tree_params
            .into_par_iter()
            .map(|(boot, seed, oob)| {
                let mut tree_rng = Pcg64::seed_from_u64(seed);
                let tree = build_tree_on_bootstrap(
                    &x_combined.view(), &y_combined, &boot, &config, &mut tree_rng, &global_hist,
                );
                (tree, oob)
            })
            .collect();

        self.trees = results.iter().map(|(t, _)| t.clone()).collect();
        self.oob_indices = results.into_iter().map(|(_, oob)| oob).collect();

        self.x_train = Some(x_arr.to_owned());
        self.is_fitted = true;
        self.sparse_prox = None; // Reset cached proximity
        Ok(())
    }

    /// Compute the proximity matrix (n x n) using OOB-based method.
    /// Following Breiman: proximity is computed only when at least one sample is OOB.
    /// prox(i, j) = (weighted count of co-occurrences) / (number of valid comparisons)
    #[pyo3(signature = (use_oob=true))]
    fn proximity_matrix(&self, use_oob: bool) -> PyResult<Vec<Vec<f64>>> {
        if !self.is_fitted {
            return Err(PyValueError::new_err("Model has not been fitted"));
        }
        let x = self.x_train.as_ref().unwrap();
        let n = x.nrows();

        // For each tree, find the terminal node for each original sample
        let leaf_assignments: Vec<Vec<usize>> = self.trees
            .par_iter()
            .map(|tree| {
                (0..n)
                    .map(|i| {
                        let row = x.row(i);
                        terminal_node_id(tree, row.as_slice().unwrap(), 0)
                    })
                    .collect()
            })
            .collect();

        // Build proximity matrix with OOB weighting
        let mut prox = vec![vec![0.0; n]; n];
        let mut counts = vec![vec![0usize; n]; n]; // Number of valid comparisons

        for (t, leaves) in leaf_assignments.iter().enumerate() {
            let oob = &self.oob_indices[t];
            
            for i in 0..n {
                for j in i..n {
                    // Breiman's method: count only when at least one is OOB
                    let valid = if use_oob {
                        oob.contains(&i) || oob.contains(&j)
                    } else {
                        true
                    };
                    
                    if valid && leaves[i] == leaves[j] {
                        prox[i][j] += 1.0;
                        if i != j {
                            prox[j][i] += 1.0;
                        }
                    }
                    if valid {
                        counts[i][j] += 1;
                        if i != j {
                            counts[j][i] += 1;
                        }
                    }
                }
            }
        }

        // Normalize by number of valid comparisons
        for i in 0..n {
            for j in 0..n {
                if counts[i][j] > 0 {
                    prox[i][j] /= counts[i][j] as f64;
                }
            }
        }

        Ok(prox)
    }

    /// Compute sparse proximity (k nearest neighbors per sample).
    /// More memory efficient for large datasets.
    /// Returns list of (neighbor_index, proximity) tuples per sample.
    #[pyo3(signature = (k=None))]
    fn proximity_sparse(&mut self, k: Option<usize>) -> PyResult<Vec<Vec<(usize, f64)>>> {
        if !self.is_fitted {
            return Err(PyValueError::new_err("Model has not been fitted"));
        }
        
        let x = self.x_train.as_ref().unwrap();
        let n = x.nrows();
        let max_neighbors = k.unwrap_or(self.n_neighbors).max(1).min(n - 1);
        let n_trees = self.trees.len() as f64;

        // Build sparse proximity incrementally
        let mut sparse: Vec<SparseProximityRow> = (0..n)
            .map(|_| SparseProximityRow::new())
            .collect();

        for (t, tree) in self.trees.iter().enumerate() {
            // Group samples by terminal node
            let mut node_to_samples: HashMap<usize, Vec<usize>> = HashMap::new();
            for i in 0..n {
                let row = x.row(i);
                let node_id = terminal_node_id(tree, row.as_slice().unwrap(), 0);
                node_to_samples.entry(node_id).or_default().push(i);
            }

            let oob = &self.oob_indices[t];

            // Add proximity for samples in same terminal node
            for samples in node_to_samples.values() {
                for &i in samples {
                    for &j in samples {
                        if i != j {
                            // OOB check: at least one should be OOB
                            if oob.contains(&i) || oob.contains(&j) {
                                sparse[i].insert(j, 1.0, max_neighbors);
                            }
                        }
                    }
                }
            }
        }

        // Finalize: normalize and keep top k
        for row in &mut sparse {
            row.finalize(max_neighbors, n_trees);
        }

        self.sparse_prox = Some(sparse.clone());

        Ok(sparse.into_iter()
            .map(|r| r.neighbors)
            .collect())
    }

    /// Compute outlier scores following Breiman's method.
    /// outlier(n) = n / sum(prox(n, k)^2) for all k, then standardized
    /// by (score - median) / MAD.
    /// Higher values = more outlier.
    #[pyo3(signature = (use_sparse=false))]
    fn outlier_scores(&self, use_sparse: bool) -> PyResult<Vec<f64>> {
        if !self.is_fitted {
            return Err(PyValueError::new_err("Model has not been fitted"));
        }
        
        let n = self.n_original;
        let mut raw = vec![0.0; n];

        if use_sparse && self.sparse_prox.is_some() {
            // Use cached sparse proximity
            let sparse = self.sparse_prox.as_ref().unwrap();
            for i in 0..n {
                let sum_sq = sparse[i].sum_squared();
                raw[i] = if sum_sq > 0.0 { n as f64 / sum_sq } else { 0.0 };
            }
        } else {
            // Compute full proximity
            let prox = self.proximity_matrix(true)?;
            for i in 0..n {
                let sum_sq: f64 = prox[i].iter().map(|&p| p * p).sum();
                raw[i] = if sum_sq > 0.0 { n as f64 / sum_sq } else { 0.0 };
            }
        }

        // Standardize: (raw - median) / MAD, capped at 20
        let mut sorted = raw.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[n / 2];
        let mut abs_devs: Vec<f64> = raw.iter()
            .map(|&r| (r - median).abs().min(5.0 * median))
            .collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = abs_devs.iter().sum::<f64>() / n as f64;
        let mad = if mad > 0.0 { mad } else { 1.0 };

        Ok(raw.iter().map(|&r| ((r - median) / mad).min(20.0)).collect())
    }

    /// Compute variable importance via permutation.
    /// For unsupervised RF, importance is measured by how much permuting
    /// a variable decreases the ability to distinguish real from synthetic data.
    fn feature_importances(&self) -> PyResult<Vec<f64>> {
        if !self.is_fitted {
            return Err(PyValueError::new_err("Model has not been fitted"));
        }
        
        let x = self.x_train.as_ref().unwrap();
        let n = x.nrows();
        let n_features = x.ncols();
        let n_trees = self.trees.len();
        
        // Base accuracy: fraction of original samples correctly classified as class 1
        let base_correct: f64 = self.trees.par_iter()
            .zip(self.oob_indices.par_iter())
            .map(|(tree, oob)| {
                let mut correct = 0.0;
                let mut count = 0.0;
                for &i in oob {
                    let row = x.row(i);
                    let pred = traverse(tree, row.as_slice().unwrap());
                    if pred > 0.5 { // Class 1 = original
                        correct += 1.0;
                    }
                    count += 1.0;
                }
                if count > 0.0 { correct / count } else { 0.0 }
            })
            .sum::<f64>() / n_trees as f64;

        // Permutation importance per feature
        let mut rng = Pcg64::seed_from_u64(self.random_state.unwrap_or(42));
        let mut importances = vec![0.0; n_features];

        for feat in 0..n_features {
            // Create permuted version of feature
            let mut perm_indices: Vec<usize> = (0..n).collect();
            perm_indices.shuffle(&mut rng);

            let perm_correct: f64 = self.trees.iter()
                .zip(self.oob_indices.iter())
                .map(|(tree, oob)| {
                    let mut correct = 0.0;
                    let mut count = 0.0;
                    for &i in oob {
                        // Create permuted sample
                        let mut sample: Vec<f64> = x.row(i).to_vec();
                        sample[feat] = x[[perm_indices[i], feat]];
                        
                        let pred = traverse(tree, &sample);
                        if pred > 0.5 {
                            correct += 1.0;
                        }
                        count += 1.0;
                    }
                    if count > 0.0 { correct / count } else { 0.0 }
                })
                .sum::<f64>() / n_trees as f64;

            // Importance = decrease in accuracy when feature is permuted
            importances[feat] = base_correct - perm_correct;
        }

        // Normalize to sum to 1 (only positive importances)
        let sum: f64 = importances.iter().map(|&v| v.max(0.0)).sum();
        if sum > 0.0 {
            for imp in &mut importances {
                *imp = (*imp).max(0.0) / sum;
            }
        }

        Ok(importances)
    }

    /// Compute low-dimensional embedding via classical MDS on the proximity matrix.
    /// Follows Breiman's myscale subroutine: power iteration on double-centered
    /// proximity matrix to find eigenvalues/eigenvectors.
    #[pyo3(signature = (n_components=2))]
    fn transform(&self, n_components: usize) -> PyResult<Vec<Vec<f64>>> {
        if !self.is_fitted {
            return Err(PyValueError::new_err("Model has not been fitted"));
        }
        let prox = self.proximity_matrix(true)?;
        let n = prox.len();

        // Row means and grand mean (for double centering)
        let mut row_mean = vec![0.0; n];
        for i in 0..n {
            row_mean[i] = prox[i].iter().sum::<f64>() / n as f64;
        }
        let grand_mean: f64 = row_mean.iter().sum::<f64>() / n as f64;

        // Power iteration to find top eigenvalues/eigenvectors
        // of the double-centered proximity matrix B where
        // B[i,j] = 0.5 * (prox[i,j] - row_mean[i] - row_mean[j] + grand_mean)
        let mut eigenvectors: Vec<Vec<f64>> = Vec::new();
        let mut eigenvalues: Vec<f64> = Vec::new();

        for comp in 0..n_components {
            // Initialize with alternating signs
            let mut y: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

            let mut eigenvalue: f64 = 0.0;

            for _iter in 0..1000 {
                // Normalize y
                let norm: f64 = y.iter().map(|&v| v * v).sum::<f64>().sqrt();
                if norm < 1e-10 {
                    break;
                }
                let u: Vec<f64> = y.iter().map(|&v| v / norm).collect();

                // Multiply by double-centered proximity: y = B * u
                y = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        y[i] += prox[i][j] * u[j];
                    }
                }

                // Double centering: subtract row/col means, add grand mean
                let eu: f64 = u.iter().sum::<f64>();
                let ru: f64 = row_mean.iter().zip(u.iter()).map(|(&r, &ui)| r * ui).sum();
                for i in 0..n {
                    y[i] = 0.5 * (y[i] - (row_mean[i] - grand_mean) * eu - ru);
                }

                // Deflate: remove components of previous eigenvectors
                for prev in 0..comp {
                    let dot: f64 = u.iter().zip(eigenvectors[prev].iter()).map(|(a, b)| a * b).sum();
                    for i in 0..n {
                        y[i] -= dot * eigenvalues[prev] * eigenvectors[prev][i];
                    }
                }

                // Rayleigh quotient
                eigenvalue = y.iter().zip(u.iter()).map(|(a, b)| a * b).sum();

                // Check convergence
                let residual: f64 = y.iter().zip(u.iter())
                    .map(|(&yi, &ui)| (yi - eigenvalue * ui).powi(2))
                    .sum();
                if residual < eigenvalue.abs() * 1e-7 {
                    let scale = eigenvalue.abs().sqrt();
                    eigenvectors.push(u.iter().map(|&v| v * scale).collect());
                    eigenvalues.push(eigenvalue);
                    break;
                }
            }

            if eigenvectors.len() <= comp {
                // Didn't converge, push zeros
                eigenvectors.push(vec![0.0; n]);
                eigenvalues.push(0.0);
            }
        }

        // Return as (n_samples, n_components)
        let mut result = vec![vec![0.0; n_components]; n];
        for i in 0..n {
            for c in 0..n_components {
                result[i][c] = eigenvectors[c][i];
            }
        }

        Ok(result)
    }

    /// Convenience method: fit the model and return the embedding.
    #[pyo3(signature = (x, n_components=2))]
    fn fit_transform(&mut self, x: PyReadonlyArray2<f64>, n_components: usize) -> PyResult<Vec<Vec<f64>>> {
        self.fit(x)?;
        self.transform(n_components)
    }

    /// Compute proximity of new samples to training samples.
    /// Returns (n_new, n_train) proximity matrix.
    fn predict_proximity(&self, x_new: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<f64>>> {
        if !self.is_fitted {
            return Err(PyValueError::new_err("Model has not been fitted"));
        }
        
        let x_train = self.x_train.as_ref().unwrap();
        let x_new_arr = x_new.as_array();
        let n_train = x_train.nrows();
        let n_new = x_new_arr.nrows();
        let n_trees = self.trees.len() as f64;

        // Get terminal nodes for training data
        let train_nodes: Vec<Vec<usize>> = self.trees.par_iter()
            .map(|tree| {
                (0..n_train)
                    .map(|i| {
                        let row = x_train.row(i);
                        terminal_node_id(tree, row.as_slice().unwrap(), 0)
                    })
                    .collect()
            })
            .collect();

        // Get terminal nodes for new data and compute proximity
        let mut prox = vec![vec![0.0; n_train]; n_new];
        
        for (t, tree) in self.trees.iter().enumerate() {
            for i in 0..n_new {
                let row = x_new_arr.row(i);
                let new_node = terminal_node_id(tree, row.as_slice().unwrap(), 0);
                
                for j in 0..n_train {
                    if train_nodes[t][j] == new_node {
                        prox[i][j] += 1.0;
                    }
                }
            }
        }

        // Normalize
        for i in 0..n_new {
            for j in 0..n_train {
                prox[i][j] /= n_trees;
            }
        }

        Ok(prox)
    }

    /// Get outlier scores for new samples based on proximity to training data.
    fn predict_outlier_scores(&self, x_new: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
        if !self.is_fitted {
            return Err(PyValueError::new_err("Model has not been fitted"));
        }
        
        let prox = self.predict_proximity(x_new)?;
        let n_train = self.n_original;
        
        // Raw outlier scores: n / sum(prox^2)
        let raw: Vec<f64> = prox.iter()
            .map(|row| {
                let sum_sq: f64 = row.iter().map(|&p| p * p).sum();
                if sum_sq > 0.0 { n_train as f64 / sum_sq } else { f64::MAX }
            })
            .collect();

        // Get training outlier statistics for normalization
        let train_outliers = self.outlier_scores(false)?;
        let mut sorted = train_outliers.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let mut abs_devs: Vec<f64> = train_outliers.iter()
            .map(|&r| (r - median).abs().min(5.0 * median))
            .collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = abs_devs.iter().sum::<f64>() / train_outliers.len() as f64;
        let mad = if mad > 0.0 { mad } else { 1.0 };

        Ok(raw.iter().map(|&r| ((r - median) / mad).min(20.0)).collect())
    }

    #[getter] fn n_estimators_(&self) -> usize { self.n_estimators }
    #[getter] fn is_fitted_(&self) -> bool { self.is_fitted }
    #[getter] fn n_samples_(&self) -> usize { self.n_original }
    #[getter] fn n_features_(&self) -> PyResult<usize> {
        if !self.is_fitted {
            return Err(PyValueError::new_err("Model has not been fitted"));
        }
        Ok(self.x_train.as_ref().unwrap().ncols())
    }
}
