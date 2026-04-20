use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::decision_tree::DecisionTree;
use crate::tree::TreeNode;

#[derive(Clone)]
struct SHAPNode {
    is_leaf: bool,
    feature: usize,
    threshold: f64,
    value: Vec<f64>,
    samples: usize,
    left_samples: usize,
    right_samples: usize,
    left: Option<Box<SHAPNode>>,
    right: Option<Box<SHAPNode>>,
}

#[pyclass]
pub struct TreeSHAP {
    tree: DecisionTree,
    x_train: Array2<f64>,
    model_type: String,
    n_features: usize,
    n_classes: usize,
    root: Option<Box<SHAPNode>>,
}

impl TreeSHAP {
    fn build_shap_tree(&self, node: Option<&TreeNode>, indices: &[usize]) -> Option<Box<SHAPNode>> {
        node.map(|n| {
            Box::new(if n.left.is_none() && n.right.is_none() {
                let value = if self.model_type == "classification" {
                    let mut probs = vec![0.0; self.n_classes];
                    if let Some(counts) = &n.class_counts {
                        let total: usize = counts.values().sum();
                        if total > 0 {
                            for (&cls, &cnt) in counts {
                                if cls < self.n_classes { probs[cls] = cnt as f64 / total as f64; }
                            }
                        }
                    }
                    probs
                } else {
                    vec![n.value]
                };
                SHAPNode { is_leaf: true, feature: 0, threshold: 0.0, value, samples: n.samples, left_samples: 0, right_samples: 0, left: None, right: None }
            } else {
                let (mut li, mut ri) = (Vec::new(), Vec::new());
                for &i in indices {
                    if self.x_train[[i, n.feature]] <= n.threshold { li.push(i); } else { ri.push(i); }
                }
                SHAPNode {
                    is_leaf: false, feature: n.feature, threshold: n.threshold,
                    value: vec![0.0; self.n_classes.max(1)], samples: n.samples,
                    left_samples: li.len(), right_samples: ri.len(),
                    left: self.build_shap_tree(n.left.as_deref(), &li),
                    right: self.build_shap_tree(n.right.as_deref(), &ri),
                }
            })
        })
    }

    fn evaluate_coalition(&self, x: &[f64], revealed: u64, unique_feats: &[usize], node: &SHAPNode, class_index: usize) -> f64 {
        if node.is_leaf {
            return node.value[class_index.min(node.value.len() - 1)];
        }
        let feat_revealed = unique_feats.iter().enumerate().any(|(j, &f)| f == node.feature && (revealed >> j) & 1 == 1);
        if feat_revealed {
            let child = if x[node.feature] <= node.threshold { node.left.as_deref().unwrap() } else { node.right.as_deref().unwrap() };
            self.evaluate_coalition(x, revealed, unique_feats, child, class_index)
        } else {
            let n = node.samples as f64;
            let p_left = if n > 0.0 { node.left_samples as f64 / n } else { 0.5 };
            let p_right = if n > 0.0 { node.right_samples as f64 / n } else { 0.5 };
            p_left * self.evaluate_coalition(x, revealed, unique_feats, node.left.as_deref().unwrap(), class_index)
            + p_right * self.evaluate_coalition(x, revealed, unique_feats, node.right.as_deref().unwrap(), class_index)
        }
    }

    fn collect_tree_features(node: &SHAPNode, feats: &mut Vec<usize>) {
        if node.is_leaf { return; }
        if !feats.contains(&node.feature) { feats.push(node.feature); }
        if let Some(left) = node.left.as_deref() { Self::collect_tree_features(left, feats); }
        if let Some(right) = node.right.as_deref() { Self::collect_tree_features(right, feats); }
    }

    fn explain_single_class(&self, x: &[f64], class_index: usize) -> Vec<f64> {
        let mut phi = vec![0.0; self.n_features];
        let root = match self.root.as_deref() { Some(r) => r, None => return phi };

        let mut unique_feats: Vec<usize> = Vec::new();
        Self::collect_tree_features(root, &mut unique_feats);
        let k = unique_feats.len();
        if k == 0 { return phi; }

        let mut fact = vec![1.0_f64; k + 1];
        for i in 1..=k { fact[i] = fact[i - 1] * i as f64; }

        let n_coalitions = 1u64 << k;
        let mut f_s = vec![0.0; n_coalitions as usize];
        for mask in 0..n_coalitions {
            f_s[mask as usize] = self.evaluate_coalition(x, mask, &unique_feats, root, class_index);
        }

        for (j, &feat) in unique_feats.iter().enumerate() {
            let mut contrib = 0.0;
            for mask in 0..n_coalitions {
                if (mask >> j) & 1 == 1 { continue; }
                let s_size = (mask as u32).count_ones() as usize;
                let w = if k <= 1 { 1.0 } else { fact[s_size] * fact[k - s_size - 1] / fact[k] };
                contrib += w * (f_s[(mask | (1 << j)) as usize] - f_s[mask as usize]);
            }
            phi[feat] += contrib;
        }
        phi
    }
}

#[pymethods]
impl TreeSHAP {
    #[new]
    fn new(tree: DecisionTree, x_train: PyReadonlyArray2<f64>, model_type: &str) -> PyResult<Self> {
        if !tree.is_fitted {
            return Err(PyValueError::new_err("Tree must be fitted before creating TreeSHAP"));
        }
        let x_arr = x_train.as_array();
        let n_features = x_arr.ncols();
        let n_samples = x_arr.nrows();
        let n_classes = if model_type == "classification" { tree.classes_.as_ref().map_or(2, |c| c.len()) } else { 1 };
        let indices: Vec<usize> = (0..n_samples).collect();

        let mut shap = Self { tree, x_train: x_arr.to_owned(), model_type: model_type.to_string(), n_features, n_classes, root: None };
        shap.root = shap.build_shap_tree(shap.tree.root.as_ref(), &indices);
        Ok(shap)
    }

    fn explain(&self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let x_arr = x.as_array();
        let n_samples = x_arr.nrows();

        if self.model_type == "classification" {
            let mut result = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let row = x_arr.row(i);
                let sample_slice = row.as_slice().unwrap();
                let mut per_class = Vec::with_capacity(self.n_classes);
                for c in 0..self.n_classes { per_class.push(self.explain_single_class(sample_slice, c)); }
                result.push(per_class);
            }
            Ok(result)
        } else {
            let mut result = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let row = x_arr.row(i);
                let sample_slice = row.as_slice().unwrap();
                result.push(vec![self.explain_single_class(sample_slice, 0)]);
            }
            Ok(result)
        }
    }
}
