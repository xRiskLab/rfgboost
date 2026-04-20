use ndarray::{Array2, ArrayView2};
use rand::prelude::*;
use rand_pcg::Pcg64;
use std::collections::HashMap;

use crate::histogram::HistogramData;

#[derive(Clone)]
pub struct TreeNode {
    pub feature: usize,
    pub threshold: f64,
    pub left: Option<Box<TreeNode>>,
    pub right: Option<Box<TreeNode>>,
    pub value: f64,
    pub samples: usize,
    pub class_counts: Option<HashMap<usize, usize>>,
}

#[derive(Clone)]
pub struct TreeConfig {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub is_classification: bool,
    pub max_features: Option<usize>,
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
    bootstrap_indices: &[usize],
    config: &TreeConfig,
    rng: &mut Pcg64,
    global_hist: &HistogramData,
) -> TreeNode {
    let n_boot = bootstrap_indices.len();
    let n_feat = x.ncols();
    let mut x_boot = Array2::zeros((n_boot, n_feat));
    let mut y_boot = vec![0.0; n_boot];

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
    build_node(&x_boot.view(), &y_boot, &indices, 0, config, rng, &boot_hist)
}

pub fn build_tree_on_bootstrap_exact(
    x: &ArrayView2<f64>,
    y: &[f64],
    bootstrap_indices: &[usize],
    config: &TreeConfig,
    rng: &mut Pcg64,
) -> TreeNode {
    let n_boot = bootstrap_indices.len();
    let n_feat = x.ncols();
    let mut x_boot = Array2::zeros((n_boot, n_feat));
    let mut y_boot = vec![0.0; n_boot];

    for (new_row, &old_row) in bootstrap_indices.iter().enumerate() {
        y_boot[new_row] = y[old_row];
        for col in 0..n_feat {
            x_boot[[new_row, col]] = x[[old_row, col]];
        }
    }

    let indices: Vec<usize> = (0..n_boot).collect();
    build_node_exact(&x_boot.view(), &y_boot, &indices, 0, config, rng)
}

pub fn build_node(
    x: &ArrayView2<f64>,
    y: &[f64],
    indices: &[usize],
    depth: usize,
    config: &TreeConfig,
    rng: &mut Pcg64,
    hist: &HistogramData,
) -> TreeNode {
    let n = indices.len();

    if n < config.min_samples_split || config.max_depth.is_some_and(|md| depth >= md) {
        return create_leaf(y, indices, config);
    }
    if config.is_classification && is_pure(y, indices) {
        return create_leaf(y, indices, config);
    }

    let (best_feat, best_thresh, best_gain) = find_best_split_hist(x, y, indices, config, rng, hist);
    if best_gain <= 0.0 {
        return create_leaf(y, indices, config);
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
        return create_leaf(y, indices, config);
    }

    TreeNode {
        feature: best_feat,
        threshold: best_thresh,
        left: Some(Box::new(build_node(x, y, &left_idx, depth + 1, config, rng, hist))),
        right: Some(Box::new(build_node(x, y, &right_idx, depth + 1, config, rng, hist))),
        value: 0.0,
        samples: n,
        class_counts: None,
    }
}

pub fn build_node_exact(
    x: &ArrayView2<f64>,
    y: &[f64],
    indices: &[usize],
    depth: usize,
    config: &TreeConfig,
    rng: &mut Pcg64,
) -> TreeNode {
    let n = indices.len();

    if n < config.min_samples_split || config.max_depth.is_some_and(|md| depth >= md) {
        return create_leaf(y, indices, config);
    }
    if config.is_classification && is_pure(y, indices) {
        return create_leaf(y, indices, config);
    }

    let (best_feat, best_thresh, best_gain) = find_best_split_exact(x, y, indices, config, rng);
    if best_gain <= 0.0 {
        return create_leaf(y, indices, config);
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
        return create_leaf(y, indices, config);
    }

    TreeNode {
        feature: best_feat,
        threshold: best_thresh,
        left: Some(Box::new(build_node_exact(x, y, &left_idx, depth + 1, config, rng))),
        right: Some(Box::new(build_node_exact(x, y, &right_idx, depth + 1, config, rng))),
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

fn create_leaf(y: &[f64], indices: &[usize], config: &TreeConfig) -> TreeNode {
    let n = indices.len();
    if config.is_classification {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &i in indices {
            *counts.entry(y[i] as usize).or_insert(0) += 1;
        }
        let majority = counts
            .iter()
            .max_by_key(|(_, c)| *c)
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
        let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64;
        TreeNode {
            feature: 0,
            threshold: 0.0,
            left: None,
            right: None,
            value: mean,
            samples: n,
            class_counts: None,
        }
    }
}

fn find_best_split_hist(
    x: &ArrayView2<f64>,
    y: &[f64],
    indices: &[usize],
    config: &TreeConfig,
    rng: &mut Pcg64,
    hist: &HistogramData,
) -> (usize, f64, f64) {
    let n = indices.len();
    let n_f64 = n as f64;
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
        let mut total_counts: HashMap<usize, usize> = HashMap::new();
        for &i in indices {
            *total_counts.entry(y[i] as usize).or_insert(0) += 1;
        }
        let parent_impurity = 1.0
            - total_counts.values().map(|&c| (c as f64 / n_f64).powi(2)).sum::<f64>();

        for &feat in &features_to_try {
            let nb = hist.n_bins[feat];
            if nb <= 1 { continue; }

            let mut bin_counts: Vec<HashMap<usize, usize>> = vec![HashMap::new(); nb];
            let mut bin_totals = vec![0usize; nb];
            for &i in indices {
                let b = hist.bin_indices[feat][i] as usize;
                let cls = y[i] as usize;
                *bin_counts[b].entry(cls).or_insert(0) += 1;
                bin_totals[b] += 1;
            }

            let mut left_counts: HashMap<usize, usize> = HashMap::new();
            let mut left_n: usize = 0;

            for b in 0..nb - 1 {
                for (&cls, &cnt) in &bin_counts[b] {
                    *left_counts.entry(cls).or_insert(0) += cnt;
                }
                left_n += bin_totals[b];
                let right_n = n - left_n;

                if left_n < config.min_samples_leaf || right_n < config.min_samples_leaf {
                    continue;
                }

                let left_gini = 1.0
                    - left_counts.values().map(|&c| (c as f64 / left_n as f64).powi(2)).sum::<f64>();
                let right_gini = {
                    let mut rg = 1.0;
                    for (&cls, &tc) in &total_counts {
                        let lc = *left_counts.get(&cls).unwrap_or(&0);
                        let rc = tc - lc;
                        rg -= (rc as f64 / right_n as f64).powi(2);
                    }
                    rg
                };
                let weighted = (left_n as f64 * left_gini + right_n as f64 * right_gini) / n_f64;
                let gain = parent_impurity - weighted;

                if gain > best_gain {
                    best_gain = gain;
                    best_feat = feat;
                    best_thresh = hist.bin_edges[feat][b];
                }
            }
        }
    } else {
        let total_sum: f64 = indices.iter().map(|&i| y[i]).sum();
        let total_sq_sum: f64 = indices.iter().map(|&i| y[i] * y[i]).sum();
        let parent_var = total_sq_sum / n_f64 - (total_sum / n_f64).powi(2);

        for &feat in &features_to_try {
            let nb = hist.n_bins[feat];
            if nb <= 1 { continue; }

            let mut bin_sum = vec![0.0; nb];
            let mut bin_sq_sum = vec![0.0; nb];
            let mut bin_count = vec![0usize; nb];
            for &i in indices {
                let b = hist.bin_indices[feat][i] as usize;
                bin_sum[b] += y[i];
                bin_sq_sum[b] += y[i] * y[i];
                bin_count[b] += 1;
            }

            let mut left_sum = 0.0;
            let mut left_sq_sum = 0.0;
            let mut left_n: usize = 0;

            for b in 0..nb - 1 {
                left_sum += bin_sum[b];
                left_sq_sum += bin_sq_sum[b];
                left_n += bin_count[b];
                let right_n = n - left_n;

                if left_n < config.min_samples_leaf || right_n < config.min_samples_leaf {
                    continue;
                }

                let right_sum = total_sum - left_sum;
                let right_sq_sum = total_sq_sum - left_sq_sum;
                let left_var = left_sq_sum / left_n as f64 - (left_sum / left_n as f64).powi(2);
                let right_var = right_sq_sum / right_n as f64 - (right_sum / right_n as f64).powi(2);
                let weighted = (left_n as f64 * left_var + right_n as f64 * right_var) / n_f64;
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
    indices: &[usize],
    config: &TreeConfig,
    rng: &mut Pcg64,
) -> (usize, f64, f64) {
    let n = indices.len();
    let n_f64 = n as f64;
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
        let mut total_counts: HashMap<usize, usize> = HashMap::new();
        for &i in indices {
            *total_counts.entry(y[i] as usize).or_insert(0) += 1;
        }
        let parent_impurity = 1.0
            - total_counts.values().map(|&c| (c as f64 / n_f64).powi(2)).sum::<f64>();

        for &feat in &features_to_try {
            let mut sorted: Vec<usize> = indices.to_vec();
            sorted.sort_by(|&a, &b| x[[a, feat]].partial_cmp(&x[[b, feat]]).unwrap());

            let mut left_counts: HashMap<usize, usize> = HashMap::new();
            let mut right_counts = total_counts.clone();
            let mut left_n: usize = 0;

            for i in 0..sorted.len() - 1 {
                let idx = sorted[i];
                let cls = y[idx] as usize;
                *left_counts.entry(cls).or_insert(0) += 1;
                *right_counts.get_mut(&cls).unwrap() -= 1;
                left_n += 1;
                let right_n = n - left_n;

                if x[[idx, feat]] == x[[sorted[i + 1], feat]] {
                    continue;
                }
                if left_n < config.min_samples_leaf || right_n < config.min_samples_leaf {
                    continue;
                }

                let left_gini = 1.0
                    - left_counts.values().map(|&c| (c as f64 / left_n as f64).powi(2)).sum::<f64>();
                let right_gini = 1.0
                    - right_counts.values().map(|&c| (c as f64 / right_n as f64).powi(2)).sum::<f64>();
                let weighted = (left_n as f64 * left_gini + right_n as f64 * right_gini) / n_f64;
                let gain = parent_impurity - weighted;

                if gain > best_gain {
                    best_gain = gain;
                    best_feat = feat;
                    best_thresh = (x[[idx, feat]] + x[[sorted[i + 1], feat]]) / 2.0;
                }
            }
        }
    } else {
        let total_sum: f64 = indices.iter().map(|&i| y[i]).sum();
        let total_sq_sum: f64 = indices.iter().map(|&i| y[i] * y[i]).sum();
        let parent_var = total_sq_sum / n_f64 - (total_sum / n_f64).powi(2);

        for &feat in &features_to_try {
            let mut sorted: Vec<usize> = indices.to_vec();
            sorted.sort_by(|&a, &b| x[[a, feat]].partial_cmp(&x[[b, feat]]).unwrap());

            let mut left_sum: f64 = 0.0;
            let mut left_sq_sum: f64 = 0.0;
            let mut left_n: usize = 0;

            for i in 0..sorted.len() - 1 {
                let idx = sorted[i];
                let val = y[idx];
                left_sum += val;
                left_sq_sum += val * val;
                left_n += 1;
                let right_n = n - left_n;

                if x[[idx, feat]] == x[[sorted[i + 1], feat]] {
                    continue;
                }
                if left_n < config.min_samples_leaf || right_n < config.min_samples_leaf {
                    continue;
                }

                let right_sum = total_sum - left_sum;
                let right_sq_sum = total_sq_sum - left_sq_sum;
                let left_var = left_sq_sum / left_n as f64 - (left_sum / left_n as f64).powi(2);
                let right_var = right_sq_sum / right_n as f64 - (right_sum / right_n as f64).powi(2);
                let weighted = (left_n as f64 * left_var + right_n as f64 * right_var) / n_f64;
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
        let total: usize = counts.values().sum();
        if total > 0 {
            for (&cls, &cnt) in counts {
                if cls < n_classes {
                    probs[cls] = cnt as f64 / total as f64;
                }
            }
        }
    }
    probs
}
