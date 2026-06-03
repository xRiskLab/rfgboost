use ndarray::{Array2, ArrayView2};
use rand::prelude::*;
use rand_pcg::Pcg64;
use std::collections::HashMap;

use crate::histogram::HistogramData;

/// Seed an RNG. If random_state is None, use OS entropy for non-determinism.
/// If Some, use the value as a fixed seed for reproducibility.
pub fn seed_rng(random_state: Option<u64>) -> Pcg64 {
    match random_state {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    }
}

/// Check that X and y contain no NaN or infinite values.
/// Returns Err with a descriptive message if invalid values are found.
pub fn validate_finite(x: &ArrayView2<f64>, y: &[f64]) -> Result<(), String> {
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            let v = x[[i, j]];
            if !v.is_finite() {
                return Err(format!(
                    "X contains non-finite value at row {}, column {} ({}). \
                     Impute or drop missing values before fitting.",
                    i, j, v
                ));
            }
        }
    }
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!(
                "y contains non-finite value at index {} ({}).",
                i, v
            ));
        }
    }
    Ok(())
}

/// Validate sample_weight: must be finite and non-negative, length must match n.
pub fn validate_weights(w: &[f64], n: usize) -> Result<(), String> {
    if w.len() != n {
        return Err(format!(
            "sample_weight length ({}) does not match number of samples ({})",
            w.len(), n
        ));
    }
    for (i, &v) in w.iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            return Err(format!(
                "sample_weight[{}] = {} is not a non-negative finite number",
                i, v
            ));
        }
    }
    Ok(())
}

#[derive(Clone)]
pub struct TreeNode {
    pub feature: usize,
    pub threshold: f64,
    pub left: Option<Box<TreeNode>>,
    pub right: Option<Box<TreeNode>>,
    pub value: f64,
    pub samples: usize,
    /// Weighted class counts at this node (sum of sample_weight per class).
    pub class_counts: Option<HashMap<usize, f64>>,
}

#[derive(Clone)]
pub struct TreeConfig {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub is_classification: bool,
    pub max_features: Option<usize>,
    /// Per-feature monotonic constraint: +1 (prediction non-decreasing in the
    /// feature), -1 (non-increasing), 0 (unconstrained). Empty = all 0.
    pub monotone_constraints: Vec<i8>,
}

/// Monotone constraint for a feature (0 if unconstrained or out of range).
#[inline]
fn constraint_of(config: &TreeConfig, feat: usize) -> i8 {
    config.monotone_constraints.get(feat).copied().unwrap_or(0)
}

fn weighted_mean(y: &[f64], w: &[f64], indices: &[usize]) -> f64 {
    let mut sw = 0.0;
    let mut swy = 0.0;
    for &i in indices {
        sw += w[i];
        swy += w[i] * y[i];
    }
    if sw > 0.0 { swy / sw } else { 0.0 }
}

pub fn resolve_max_features(spec: &Option<String>, n_features: usize) -> usize {
    match spec {
        Some(s) if s == "sqrt" => ((n_features as f64).sqrt() as usize).max(1),
        Some(s) if s == "log2" => ((n_features as f64).log2() as usize).max(1),
        Some(s) => s
            .parse::<f64>()
            .map(|v| {
                if v <= 1.0 {
                    ((v * n_features as f64) as usize).max(1)
                } else {
                    v as usize
                }
            })
            .unwrap_or(n_features),
        None => n_features,
    }
}

#[allow(dead_code)]
pub fn sample_features(max_feat: usize, n_features: usize, rng: &mut Pcg64) -> Vec<usize> {
    let mut features: Vec<usize> = (0..n_features).collect();
    features.shuffle(rng);
    features.truncate(max_feat);
    features.sort();
    features
}

pub fn build_tree_on_bootstrap(
    x: &ArrayView2<f64>,
    y: &[f64],
    weights: &[f64],
    bootstrap_indices: &[usize],
    config: &TreeConfig,
    rng: &mut Pcg64,
    global_hist: &HistogramData,
) -> TreeNode {
    let n_boot = bootstrap_indices.len();
    let n_feat = x.ncols();
    let mut x_boot = Array2::zeros((n_boot, n_feat));
    let mut y_boot = vec![0.0; n_boot];
    let mut w_boot = vec![0.0; n_boot];

    let mut boot_hist_bins = Vec::with_capacity(n_feat);
    for feat in 0..n_feat {
        let feat_bins: Vec<u8> = bootstrap_indices
            .iter()
            .map(|&old_row| global_hist.bin_indices[feat][old_row])
            .collect();
        boot_hist_bins.push(feat_bins);
    }

    for (new_row, &old_row) in bootstrap_indices.iter().enumerate() {
        y_boot[new_row] = y[old_row];
        w_boot[new_row] = weights[old_row];
        for col in 0..n_feat {
            x_boot[[new_row, col]] = x[[old_row, col]];
        }
    }

    let boot_hist = HistogramData {
        bin_indices: boot_hist_bins,
        bin_edges: global_hist.bin_edges.clone(),
        n_bins: global_hist.n_bins.clone(),
    };

    let indices: Vec<usize> = (0..n_boot).collect();
    build_node(&x_boot.view(), &y_boot, &w_boot, &indices, 0, config, rng, &boot_hist,
               f64::NEG_INFINITY, f64::INFINITY)
}

pub fn build_tree_on_bootstrap_exact(
    x: &ArrayView2<f64>,
    y: &[f64],
    weights: &[f64],
    bootstrap_indices: &[usize],
    config: &TreeConfig,
    rng: &mut Pcg64,
) -> TreeNode {
    let n_boot = bootstrap_indices.len();
    let n_feat = x.ncols();
    let mut x_boot = Array2::zeros((n_boot, n_feat));
    let mut y_boot = vec![0.0; n_boot];
    let mut w_boot = vec![0.0; n_boot];

    for (new_row, &old_row) in bootstrap_indices.iter().enumerate() {
        y_boot[new_row] = y[old_row];
        w_boot[new_row] = weights[old_row];
        for col in 0..n_feat {
            x_boot[[new_row, col]] = x[[old_row, col]];
        }
    }

    let indices: Vec<usize> = (0..n_boot).collect();
    build_node_exact(&x_boot.view(), &y_boot, &w_boot, &indices, 0, config, rng,
                     f64::NEG_INFINITY, f64::INFINITY)
}

/// Given a split on `feat`, return (left_lower, left_upper, right_lower,
/// right_upper) enforcing the monotone constraint by value-bound propagation:
/// for an increasing feature every left-subtree leaf stays <= mid and every
/// right-subtree leaf stays >= mid, where mid is between the two child means.
fn child_bounds(
    config: &TreeConfig, feat: usize, y: &[f64], w: &[f64],
    left_idx: &[usize], right_idx: &[usize], lower: f64, upper: f64,
) -> (f64, f64, f64, f64) {
    let c = constraint_of(config, feat);
    if c == 0 {
        return (lower, upper, lower, upper);
    }
    let lv = weighted_mean(y, w, left_idx);
    let rv = weighted_mean(y, w, right_idx);
    let mut mid = 0.5 * (lv + rv);
    if mid < lower { mid = lower; }
    if mid > upper { mid = upper; }
    if c > 0 {
        (lower, mid, mid, upper) // increasing: left <= mid <= right
    } else {
        (mid, upper, lower, mid) // decreasing: left >= mid >= right
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_node(
    x: &ArrayView2<f64>,
    y: &[f64],
    w: &[f64],
    indices: &[usize],
    depth: usize,
    config: &TreeConfig,
    rng: &mut Pcg64,
    hist: &HistogramData,
    lower: f64,
    upper: f64,
) -> TreeNode {
    let n = indices.len();

    if n < config.min_samples_split || config.max_depth.is_some_and(|md| depth >= md) {
        return create_leaf(y, w, indices, config, lower, upper);
    }
    if config.is_classification && is_pure(y, indices) {
        return create_leaf(y, w, indices, config, lower, upper);
    }

    let (best_feat, best_thresh, best_gain) = find_best_split_hist(x, y, w, indices, config, rng, hist);
    if best_gain <= 0.0 {
        return create_leaf(y, w, indices, config, lower, upper);
    }

    let (mut left_idx, mut right_idx) = (Vec::new(), Vec::new());
    for &i in indices {
        if x[[i, best_feat]] <= best_thresh {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    }

    if left_idx.len() < config.min_samples_leaf || right_idx.len() < config.min_samples_leaf {
        return create_leaf(y, w, indices, config, lower, upper);
    }

    let (ll, lu, rl, ru) = child_bounds(config, best_feat, y, w, &left_idx, &right_idx, lower, upper);

    TreeNode {
        feature: best_feat,
        threshold: best_thresh,
        left: Some(Box::new(build_node(x, y, w, &left_idx, depth + 1, config, rng, hist, ll, lu))),
        right: Some(Box::new(build_node(x, y, w, &right_idx, depth + 1, config, rng, hist, rl, ru))),
        value: 0.0,
        samples: n,
        class_counts: None,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_node_exact(
    x: &ArrayView2<f64>,
    y: &[f64],
    w: &[f64],
    indices: &[usize],
    depth: usize,
    config: &TreeConfig,
    rng: &mut Pcg64,
    lower: f64,
    upper: f64,
) -> TreeNode {
    let n = indices.len();

    if n < config.min_samples_split || config.max_depth.is_some_and(|md| depth >= md) {
        return create_leaf(y, w, indices, config, lower, upper);
    }
    if config.is_classification && is_pure(y, indices) {
        return create_leaf(y, w, indices, config, lower, upper);
    }

    let (best_feat, best_thresh, best_gain) = find_best_split_exact(x, y, w, indices, config, rng);
    if best_gain <= 0.0 {
        return create_leaf(y, w, indices, config, lower, upper);
    }

    let (mut left_idx, mut right_idx) = (Vec::new(), Vec::new());
    for &i in indices {
        if x[[i, best_feat]] <= best_thresh {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    }

    if left_idx.len() < config.min_samples_leaf || right_idx.len() < config.min_samples_leaf {
        return create_leaf(y, w, indices, config, lower, upper);
    }

    let (ll, lu, rl, ru) = child_bounds(config, best_feat, y, w, &left_idx, &right_idx, lower, upper);

    TreeNode {
        feature: best_feat,
        threshold: best_thresh,
        left: Some(Box::new(build_node_exact(x, y, w, &left_idx, depth + 1, config, rng, ll, lu))),
        right: Some(Box::new(build_node_exact(x, y, w, &right_idx, depth + 1, config, rng, rl, ru))),
        value: 0.0,
        samples: n,
        class_counts: None,
    }
}

fn is_pure(y: &[f64], indices: &[usize]) -> bool {
    if indices.is_empty() {
        return true;
    }
    let first = y[indices[0]] as usize;
    indices.iter().all(|&i| (y[i] as usize) == first)
}

fn create_leaf(y: &[f64], w: &[f64], indices: &[usize], config: &TreeConfig, lower: f64, upper: f64) -> TreeNode {
    let n = indices.len();
    if config.is_classification {
        let mut counts: HashMap<usize, f64> = HashMap::new();
        for &i in indices {
            *counts.entry(y[i] as usize).or_insert(0.0) += w[i];
        }
        let majority = counts
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(cls, _)| *cls)
            .unwrap_or(0);
        TreeNode {
            feature: 0,
            threshold: 0.0,
            left: None,
            right: None,
            value: majority as f64,
            samples: n,
            class_counts: Some(counts),
        }
    } else {
        // Weighted mean of the leaf (shares the zero-weight fallback with
        // weighted_mean used during split/bound computation).
        let mut value = weighted_mean(y, w, indices);
        // Monotone value bounds: clamp into the interval propagated from
        // constrained ancestor splits (lower=-inf/upper=+inf when unconstrained).
        if value < lower { value = lower; }
        if value > upper { value = upper; }
        TreeNode {
            feature: 0,
            threshold: 0.0,
            left: None,
            right: None,
            value,
            samples: n,
            class_counts: None,
        }
    }
}

fn find_best_split_hist(
    x: &ArrayView2<f64>,
    y: &[f64],
    w: &[f64],
    indices: &[usize],
    config: &TreeConfig,
    rng: &mut Pcg64,
    hist: &HistogramData,
) -> (usize, f64, f64) {
    let n = indices.len();
    let total_w: f64 = indices.iter().map(|&i| w[i]).sum();
    if total_w <= 0.0 {
        return (0, 0.0, 0.0);
    }
    let (mut best_feat, mut best_thresh, mut best_gain) = (0, 0.0, 0.0);
    let n_features = x.ncols();

    let features_to_try: Vec<usize> = match config.max_features {
        Some(mf) if mf < n_features => {
            let mut feats: Vec<usize> = (0..n_features).collect();
            feats.shuffle(rng);
            feats.truncate(mf);
            feats
        }
        _ => (0..n_features).collect(),
    };

    if config.is_classification {
        let mut total_counts: HashMap<usize, f64> = HashMap::new();
        for &i in indices {
            *total_counts.entry(y[i] as usize).or_insert(0.0) += w[i];
        }
        let parent_impurity = 1.0
            - total_counts.values().map(|&c| (c / total_w).powi(2)).sum::<f64>();

        for &feat in &features_to_try {
            let nb = hist.n_bins[feat];
            if nb <= 1 { continue; }

            let mut bin_counts: Vec<HashMap<usize, f64>> = vec![HashMap::new(); nb];
            let mut bin_totals = vec![0.0_f64; nb];
            let mut bin_n = vec![0usize; nb];
            for &i in indices {
                let b = hist.bin_indices[feat][i] as usize;
                let cls = y[i] as usize;
                *bin_counts[b].entry(cls).or_insert(0.0) += w[i];
                bin_totals[b] += w[i];
                bin_n[b] += 1;
            }

            let mut left_counts: HashMap<usize, f64> = HashMap::new();
            let mut left_w = 0.0_f64;
            let mut left_n: usize = 0;

            for b in 0..nb - 1 {
                for (&cls, &cnt) in &bin_counts[b] {
                    *left_counts.entry(cls).or_insert(0.0) += cnt;
                }
                left_w += bin_totals[b];
                left_n += bin_n[b];
                let right_n = n - left_n;
                let right_w = total_w - left_w;

                if left_n < config.min_samples_leaf || right_n < config.min_samples_leaf {
                    continue;
                }
                if left_w <= 0.0 || right_w <= 0.0 {
                    continue;
                }

                let left_gini = 1.0
                    - left_counts.values().map(|&c| (c / left_w).powi(2)).sum::<f64>();
                let right_gini = {
                    let mut rg = 1.0;
                    for (&cls, &tc) in &total_counts {
                        let lc = *left_counts.get(&cls).unwrap_or(&0.0);
                        let rc = tc - lc;
                        rg -= (rc / right_w).powi(2);
                    }
                    rg
                };
                let weighted = (left_w * left_gini + right_w * right_gini) / total_w;
                let gain = parent_impurity - weighted;

                if gain > best_gain {
                    best_gain = gain;
                    best_feat = feat;
                    best_thresh = hist.bin_edges[feat][b];
                }
            }
        }
    } else {
        let total_wy: f64 = indices.iter().map(|&i| w[i] * y[i]).sum();
        let total_wyy: f64 = indices.iter().map(|&i| w[i] * y[i] * y[i]).sum();
        let parent_var = total_wyy / total_w - (total_wy / total_w).powi(2);

        for &feat in &features_to_try {
            let nb = hist.n_bins[feat];
            if nb <= 1 { continue; }

            let mut bin_wy = vec![0.0; nb];
            let mut bin_wyy = vec![0.0; nb];
            let mut bin_w = vec![0.0_f64; nb];
            let mut bin_n = vec![0usize; nb];
            for &i in indices {
                let b = hist.bin_indices[feat][i] as usize;
                bin_wy[b] += w[i] * y[i];
                bin_wyy[b] += w[i] * y[i] * y[i];
                bin_w[b] += w[i];
                bin_n[b] += 1;
            }

            let mut left_wy = 0.0;
            let mut left_wyy = 0.0;
            let mut left_w = 0.0_f64;
            let mut left_n: usize = 0;

            for b in 0..nb - 1 {
                left_wy += bin_wy[b];
                left_wyy += bin_wyy[b];
                left_w += bin_w[b];
                left_n += bin_n[b];
                let right_n = n - left_n;
                let right_w = total_w - left_w;

                if left_n < config.min_samples_leaf || right_n < config.min_samples_leaf {
                    continue;
                }
                if left_w <= 0.0 || right_w <= 0.0 {
                    continue;
                }

                let right_wy = total_wy - left_wy;
                let right_wyy = total_wyy - left_wyy;

                // Monotone constraint: reject candidate splits whose child means
                // violate the required direction, so the chosen structure is
                // consistent with the value-bound propagation in build_node.
                let c = constraint_of(config, feat);
                if c != 0 {
                    let lmean = left_wy / left_w;
                    let rmean = right_wy / right_w;
                    if (c > 0 && lmean > rmean) || (c < 0 && lmean < rmean) {
                        continue;
                    }
                }

                let left_var = left_wyy / left_w - (left_wy / left_w).powi(2);
                let right_var = right_wyy / right_w - (right_wy / right_w).powi(2);
                let weighted = (left_w * left_var + right_w * right_var) / total_w;
                let gain = parent_var - weighted;

                if gain > best_gain {
                    best_gain = gain;
                    best_feat = feat;
                    best_thresh = hist.bin_edges[feat][b];
                }
            }
        }
    }
    (best_feat, best_thresh, best_gain)
}

fn find_best_split_exact(
    x: &ArrayView2<f64>,
    y: &[f64],
    w: &[f64],
    indices: &[usize],
    config: &TreeConfig,
    rng: &mut Pcg64,
) -> (usize, f64, f64) {
    let n = indices.len();
    let total_w: f64 = indices.iter().map(|&i| w[i]).sum();
    if total_w <= 0.0 {
        return (0, 0.0, 0.0);
    }
    let (mut best_feat, mut best_thresh, mut best_gain) = (0, 0.0, 0.0);
    let n_features = x.ncols();

    // Always shuffle feature order (matches sklearn's random tie-breaking)
    let features_to_try: Vec<usize> = {
        let mut feats: Vec<usize> = (0..n_features).collect();
        feats.shuffle(rng);
        match config.max_features {
            Some(mf) if mf < n_features => { feats.truncate(mf); }
            _ => {}
        }
        feats
    };

    if config.is_classification {
        let mut total_counts: HashMap<usize, f64> = HashMap::new();
        for &i in indices {
            *total_counts.entry(y[i] as usize).or_insert(0.0) += w[i];
        }
        let parent_impurity = 1.0
            - total_counts.values().map(|&c| (c / total_w).powi(2)).sum::<f64>();

        for &feat in &features_to_try {
            let mut sorted: Vec<usize> = indices.to_vec();
            sorted.sort_by(|&a, &b| x[[a, feat]].partial_cmp(&x[[b, feat]]).unwrap());

            let mut left_counts: HashMap<usize, f64> = HashMap::new();
            let mut right_counts = total_counts.clone();
            let mut left_w = 0.0_f64;
            let mut left_n: usize = 0;

            for i in 0..sorted.len() - 1 {
                let idx = sorted[i];
                let cls = y[idx] as usize;
                let wi = w[idx];
                *left_counts.entry(cls).or_insert(0.0) += wi;
                *right_counts.get_mut(&cls).unwrap() -= wi;
                left_w += wi;
                left_n += 1;
                let right_n = n - left_n;
                let right_w = total_w - left_w;

                if x[[idx, feat]] == x[[sorted[i + 1], feat]] {
                    continue;
                }
                if left_n < config.min_samples_leaf || right_n < config.min_samples_leaf {
                    continue;
                }
                if left_w <= 0.0 || right_w <= 0.0 {
                    continue;
                }

                let left_gini = 1.0
                    - left_counts.values().map(|&c| (c / left_w).powi(2)).sum::<f64>();
                let right_gini = 1.0
                    - right_counts.values().map(|&c| (c / right_w).powi(2)).sum::<f64>();
                let weighted = (left_w * left_gini + right_w * right_gini) / total_w;
                let gain = parent_impurity - weighted;

                if gain > best_gain {
                    best_gain = gain;
                    best_feat = feat;
                    best_thresh = (x[[idx, feat]] + x[[sorted[i + 1], feat]]) / 2.0;
                }
            }
        }
    } else {
        let total_wy: f64 = indices.iter().map(|&i| w[i] * y[i]).sum();
        let total_wyy: f64 = indices.iter().map(|&i| w[i] * y[i] * y[i]).sum();
        let parent_var = total_wyy / total_w - (total_wy / total_w).powi(2);

        for &feat in &features_to_try {
            let mut sorted: Vec<usize> = indices.to_vec();
            sorted.sort_by(|&a, &b| x[[a, feat]].partial_cmp(&x[[b, feat]]).unwrap());

            let mut left_wy: f64 = 0.0;
            let mut left_wyy: f64 = 0.0;
            let mut left_w = 0.0_f64;
            let mut left_n: usize = 0;

            for i in 0..sorted.len() - 1 {
                let idx = sorted[i];
                let val = y[idx];
                let wi = w[idx];
                left_wy += wi * val;
                left_wyy += wi * val * val;
                left_w += wi;
                left_n += 1;
                let right_n = n - left_n;
                let right_w = total_w - left_w;

                if x[[idx, feat]] == x[[sorted[i + 1], feat]] {
                    continue;
                }
                if left_n < config.min_samples_leaf || right_n < config.min_samples_leaf {
                    continue;
                }
                if left_w <= 0.0 || right_w <= 0.0 {
                    continue;
                }

                let right_wy = total_wy - left_wy;
                let right_wyy = total_wyy - left_wyy;

                // Monotone constraint: reject candidate splits whose child means
                // violate the required direction, so the chosen structure is
                // consistent with the value-bound propagation in build_node.
                let c = constraint_of(config, feat);
                if c != 0 {
                    let lmean = left_wy / left_w;
                    let rmean = right_wy / right_w;
                    if (c > 0 && lmean > rmean) || (c < 0 && lmean < rmean) {
                        continue;
                    }
                }

                let left_var = left_wyy / left_w - (left_wy / left_w).powi(2);
                let right_var = right_wyy / right_w - (right_wy / right_w).powi(2);
                let weighted = (left_w * left_var + right_w * right_var) / total_w;
                let gain = parent_var - weighted;

                if gain > best_gain {
                    best_gain = gain;
                    best_feat = feat;
                    best_thresh = (x[[idx, feat]] + x[[sorted[i + 1], feat]]) / 2.0;
                }
            }
        }
    }
    (best_feat, best_thresh, best_gain)
}

pub fn traverse(node: &TreeNode, sample: &[f64]) -> f64 {
    let mut cur = node;
    loop {
        match (&cur.left, &cur.right) {
            (None, None) => return cur.value,
            (Some(left), _) if sample[cur.feature] <= cur.threshold => cur = left,
            (_, Some(right)) => cur = right,
            // Edge case: only one child exists
            (Some(left), None) => cur = left,
            (None, Some(right)) => cur = right,
        }
    }
}

pub fn traverse_proba(node: &TreeNode, sample: &[f64], n_classes: usize) -> Vec<f64> {
    let mut cur = node;
    loop {
        if cur.left.is_none() && cur.right.is_none() {
            break;
        }
        cur = if sample[cur.feature] <= cur.threshold {
            cur.left.as_ref().unwrap()
        } else {
            cur.right.as_ref().unwrap()
        };
    }
    let mut probs = vec![0.0; n_classes];
    if let Some(counts) = &cur.class_counts {
        let total: f64 = counts.values().sum();
        if total > 0.0 {
            for (&cls, &cnt) in counts {
                if cls < n_classes {
                    probs[cls] = cnt / total;
                }
            }
        }
    }
    probs
}
