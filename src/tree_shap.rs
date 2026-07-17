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
                        let total: f64 = counts.values().sum();
                        if total > 0.0 {
                            for (&cls, &cnt) in counts {
                                if cls < self.n_classes {
                                    probs[cls] = cnt / total;
                                }
                            }
                        }
                    }
                    probs
                } else {
                    vec![n.value]
                };
                SHAPNode {
                    is_leaf: true,
                    feature: 0,
                    threshold: 0.0,
                    value,
                    samples: n.samples,
                    left_samples: 0,
                    right_samples: 0,
                    left: None,
                    right: None,
                }
            } else {
                let (mut li, mut ri) = (Vec::new(), Vec::new());
                for &i in indices {
                    if self.x_train[[i, n.feature]] <= n.threshold {
                        li.push(i);
                    } else {
                        ri.push(i);
                    }
                }
                SHAPNode {
                    is_leaf: false,
                    feature: n.feature,
                    threshold: n.threshold,
                    value: vec![0.0; self.n_classes.max(1)],
                    samples: n.samples,
                    left_samples: li.len(),
                    right_samples: ri.len(),
                    left: self.build_shap_tree(n.left.as_deref(), &li),
                    right: self.build_shap_tree(n.right.as_deref(), &ri),
                }
            })
        })
    }

    fn evaluate_coalition(
        &self,
        x: &[f64],
        revealed: u64,
        unique_feats: &[usize],
        node: &SHAPNode,
        class_index: usize,
    ) -> f64 {
        if node.is_leaf {
            return node.value[class_index.min(node.value.len() - 1)];
        }
        let feat_revealed = unique_feats
            .iter()
            .enumerate()
            .any(|(j, &f)| f == node.feature && (revealed >> j) & 1 == 1);
        if feat_revealed {
            let child = if x[node.feature] <= node.threshold {
                node.left.as_deref().unwrap()
            } else {
                node.right.as_deref().unwrap()
            };
            self.evaluate_coalition(x, revealed, unique_feats, child, class_index)
        } else {
            let n = node.samples as f64;
            let p_left = if n > 0.0 {
                node.left_samples as f64 / n
            } else {
                0.5
            };
            let p_right = if n > 0.0 {
                node.right_samples as f64 / n
            } else {
                0.5
            };
            p_left
                * self.evaluate_coalition(
                    x,
                    revealed,
                    unique_feats,
                    node.left.as_deref().unwrap(),
                    class_index,
                )
                + p_right
                    * self.evaluate_coalition(
                        x,
                        revealed,
                        unique_feats,
                        node.right.as_deref().unwrap(),
                        class_index,
                    )
        }
    }

    fn collect_tree_features(node: &SHAPNode, feats: &mut Vec<usize>) {
        if node.is_leaf {
            return;
        }
        if !feats.contains(&node.feature) {
            feats.push(node.feature);
        }
        if let Some(left) = node.left.as_deref() {
            Self::collect_tree_features(left, feats);
        }
        if let Some(right) = node.right.as_deref() {
            Self::collect_tree_features(right, feats);
        }
    }

    fn explain_single_class(&self, x: &[f64], class_index: usize) -> Vec<f64> {
        let mut phi = vec![0.0; self.n_features];
        let root = match self.root.as_deref() {
            Some(r) => r,
            None => return phi,
        };

        let mut unique_feats: Vec<usize> = Vec::new();
        Self::collect_tree_features(root, &mut unique_feats);
        let k = unique_feats.len();
        if k == 0 {
            return phi;
        }

        let mut fact = vec![1.0_f64; k + 1];
        for i in 1..=k {
            fact[i] = fact[i - 1] * i as f64;
        }

        let n_coalitions = 1u64 << k;
        let mut f_s = vec![0.0; n_coalitions as usize];
        for mask in 0..n_coalitions {
            f_s[mask as usize] = self.evaluate_coalition(x, mask, &unique_feats, root, class_index);
        }

        for (j, &feat) in unique_feats.iter().enumerate() {
            let mut contrib = 0.0;
            for mask in 0..n_coalitions {
                if (mask >> j) & 1 == 1 {
                    continue;
                }
                let s_size = (mask as u32).count_ones() as usize;
                let w = if k <= 1 {
                    1.0
                } else {
                    fact[s_size] * fact[k - s_size - 1] / fact[k]
                };
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
            return Err(PyValueError::new_err(
                "Tree must be fitted before creating TreeSHAP",
            ));
        }
        let x_arr = x_train.as_array();
        let n_features = x_arr.ncols();
        let n_samples = x_arr.nrows();
        let n_classes = if model_type == "classification" {
            tree.classes_.as_ref().map_or(2, |c| c.len())
        } else {
            1
        };
        let indices: Vec<usize> = (0..n_samples).collect();

        let mut shap = Self {
            tree,
            x_train: x_arr.to_owned(),
            model_type: model_type.to_string(),
            n_features,
            n_classes,
            root: None,
        };
        shap.root = shap.build_shap_tree(shap.tree.root.as_ref(), &indices);
        Ok(shap)
    }

    /// SHAP values, shape `[n_samples][n_classes][n_features]`. `device`: "cpu"
    /// (default), "cuda" or "mps"/"metal"/"gpu". GPU parallelizes over samples;
    /// trees with more than SHAP_MAX_K unique features fall back to CPU.
    #[pyo3(signature = (x, device="cpu"))]
    fn explain(&self, x: PyReadonlyArray2<f64>, device: &str) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let x_arr = x.as_array();
        match device {
            "cpu" => Ok(self.explain_cpu(&x_arr)),
            #[cfg(feature = "cuda")]
            "cuda" => self.explain_device(&x_arr, Backend::Cuda),
            #[cfg(feature = "gpu")]
            "mps" | "metal" | "gpu" => self.explain_device(&x_arr, Backend::Wgpu),
            other => Err(PyValueError::new_err(format!(
                "device '{}' is not available in this build. Available: {}.",
                other,
                shap_devices()
            ))),
        }
    }
}

fn shap_devices() -> String {
    #[allow(unused_mut)]
    let mut d = ["cpu"];
    #[cfg(feature = "cuda")]
    d.push("cuda");
    #[cfg(feature = "gpu")]
    d.push("mps");
    d.join(", ")
}

impl TreeSHAP {
    fn explain_cpu(&self, x_arr: &ndarray::ArrayView2<f64>) -> Vec<Vec<Vec<f64>>> {
        let n_samples = x_arr.nrows();
        let n_out = if self.model_type == "classification" {
            self.n_classes
        } else {
            1
        };
        (0..n_samples)
            .map(|i| {
                let row = x_arr.row(i);
                let s = row.as_slice().unwrap();
                (0..n_out)
                    .map(|c| self.explain_single_class(s, c))
                    .collect()
            })
            .collect()
    }
}

#[cfg(any(feature = "cuda", feature = "gpu"))]
enum Backend {
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "gpu")]
    Wgpu,
}

#[cfg(any(feature = "cuda", feature = "gpu"))]
struct ShapFlat {
    feat: Vec<i32>,
    thr: Vec<f32>,
    left: Vec<u32>,
    right: Vec<u32>,
    pl: Vec<f32>,
    pr: Vec<f32>,
    uidx: Vec<i32>,
    leafval: Vec<f32>,
    ufeat: Vec<u32>,
    fact: Vec<f32>,
    k: usize,
}

#[cfg(any(feature = "cuda", feature = "gpu"))]
fn flatten_shap(node: &SHAPNode, a: &mut ShapFlat, ufeat: &[usize], nc: usize) -> u32 {
    let id = a.feat.len() as u32;
    a.feat.push(0);
    a.thr.push(0.0);
    a.left.push(0);
    a.right.push(0);
    a.pl.push(0.0);
    a.pr.push(0.0);
    a.uidx.push(-1);
    let base = a.leafval.len();
    a.leafval.resize(base + nc, 0.0);
    if node.is_leaf {
        a.feat[id as usize] = -1;
        let vlen = node.value.len();
        for c in 0..nc {
            a.leafval[base + c] = node.value[c.min(vlen.saturating_sub(1))] as f32;
        }
    } else {
        a.feat[id as usize] = node.feature as i32;
        a.thr[id as usize] = node.threshold as f32;
        let n = node.samples as f32;
        a.pl[id as usize] = if n > 0.0 {
            node.left_samples as f32 / n
        } else {
            0.5
        };
        a.pr[id as usize] = if n > 0.0 {
            node.right_samples as f32 / n
        } else {
            0.5
        };
        a.uidx[id as usize] = ufeat.iter().position(|&f| f == node.feature).unwrap() as i32;
        let l = flatten_shap(node.left.as_deref().unwrap(), a, ufeat, nc);
        let r = flatten_shap(node.right.as_deref().unwrap(), a, ufeat, nc);
        a.left[id as usize] = l;
        a.right[id as usize] = r;
    }
    id
}

#[cfg(any(feature = "cuda", feature = "gpu"))]
impl TreeSHAP {
    fn flatten_for_gpu(&self) -> Option<ShapFlat> {
        let root = self.root.as_deref()?;
        let mut ufeat_us: Vec<usize> = Vec::new();
        Self::collect_tree_features(root, &mut ufeat_us);
        let k = ufeat_us.len();
        let mut a = ShapFlat {
            feat: vec![],
            thr: vec![],
            left: vec![],
            right: vec![],
            pl: vec![],
            pr: vec![],
            uidx: vec![],
            leafval: vec![],
            ufeat: ufeat_us.iter().map(|&f| f as u32).collect(),
            fact: vec![],
            k,
        };
        flatten_shap(root, &mut a, &ufeat_us, self.n_classes.max(1));
        let mut fact = vec![1.0f32; k + 1];
        for i in 1..=k {
            fact[i] = fact[i - 1] * i as f32;
        }
        a.fact = fact;
        Some(a)
    }

    fn explain_device(
        &self,
        x_arr: &ndarray::ArrayView2<f64>,
        backend: Backend,
    ) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let n = x_arr.nrows();
        let nf = self.n_features;
        let nc = if self.model_type == "classification" {
            self.n_classes
        } else {
            1
        };
        let flat = match self.flatten_for_gpu() {
            Some(f) => f,
            None => return Ok(self.explain_cpu(x_arr)),
        };
        let xf: Vec<f32> = x_arr.iter().map(|&v| v as f32).collect();
        let out = match backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda => crate::cuda::shap_explain(
                &xf,
                n,
                nf,
                nc,
                flat.k,
                &flat.feat,
                &flat.thr,
                &flat.left,
                &flat.right,
                &flat.pl,
                &flat.pr,
                &flat.uidx,
                &flat.leafval,
                &flat.ufeat,
                &flat.fact,
            ),
            #[cfg(feature = "gpu")]
            Backend::Wgpu => crate::gpu::shap_explain(
                &xf,
                n,
                nf,
                nc,
                flat.k,
                &flat.feat,
                &flat.thr,
                &flat.left,
                &flat.right,
                &flat.pl,
                &flat.pr,
                &flat.uidx,
                &flat.leafval,
                &flat.ufeat,
                &flat.fact,
            ),
        };
        // k too large for the GPU kernel, or no device -> CPU fallback.
        let out = match out {
            Some(o) => o,
            None => return Ok(self.explain_cpu(x_arr)),
        };
        Ok((0..n)
            .map(|s| {
                (0..nc)
                    .map(|c| {
                        let off = (s * nc + c) * nf;
                        out[off..off + nf].iter().map(|&v| v as f64).collect()
                    })
                    .collect()
            })
            .collect())
    }
}
