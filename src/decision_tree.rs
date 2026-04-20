use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_pcg::Pcg64;

use crate::tree::{build_node_exact, traverse, traverse_proba, TreeConfig, TreeNode};

#[pyclass]
#[derive(Clone)]
pub struct DecisionTree {
    pub root: Option<TreeNode>,
    pub max_depth: Option<usize>,
    pub criterion: String,
    pub task: String,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub random_state: Option<u64>,
    pub classes_: Option<Vec<usize>>,
    pub is_fitted: bool,
}

#[pymethods]
impl DecisionTree {
    #[new]
    #[pyo3(signature = (
        max_depth=None,
        criterion="entropy",
        task="classification",
        random_state=None,
        min_samples_split=2,
        min_samples_leaf=1
    ))]
    fn new(
        max_depth: Option<usize>,
        criterion: &str,
        task: &str,
        random_state: Option<u64>,
        min_samples_split: usize,
        min_samples_leaf: usize,
    ) -> Self {
        if task != "classification" && task != "regression" {
            panic!("task must be 'classification' or 'regression'");
        }
        Self {
            root: None,
            max_depth,
            criterion: criterion.to_string(),
            task: task.to_string(),
            min_samples_split,
            min_samples_leaf,
            random_state,
            classes_: None,
            is_fitted: false,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_arr = x.as_array();
        let y_arr = y.as_array();

        if self.task == "classification" {
            let mut classes: Vec<usize> = y_arr.iter().map(|&v| v as usize).collect();
            classes.sort();
            classes.dedup();
            self.classes_ = Some(classes);
        }

        let config = TreeConfig {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            is_classification: self.task == "classification",
            max_features: None,
        };
        let y_slice = y_arr.as_slice().unwrap();
        let n_samples = x_arr.nrows();
        let indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = Pcg64::seed_from_u64(self.random_state.unwrap_or(42));
        self.root = Some(build_node_exact(&x_arr.view(), y_slice, &indices, 0, &config, &mut rng));
        self.is_fitted = true;
        Ok(())
    }

    fn predict(&self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
        if !self.is_fitted {
            return Err(PyValueError::new_err("DecisionTree has not been fitted"));
        }
        let x_arr = x.as_array();
        let root = self.root.as_ref().unwrap();
        Ok(x_arr
            .outer_iter()
            .map(|row| traverse(root, row.as_slice().unwrap()))
            .collect())
    }

    fn predict_proba(&self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<f64>>> {
        if !self.is_fitted {
            return Err(PyValueError::new_err("DecisionTree has not been fitted"));
        }
        if self.task != "classification" {
            return Err(PyValueError::new_err(
                "predict_proba only available for classification",
            ));
        }
        let x_arr = x.as_array();
        let root = self.root.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();
        let n_classes = classes.len();

        Ok(x_arr
            .outer_iter()
            .map(|row| {
                let raw = traverse_proba(root, row.as_slice().unwrap(), n_classes);
                classes.iter().map(|&c| if c < raw.len() { raw[c] } else { 0.0 }).collect()
            })
            .collect())
    }

    fn get_depth(&self) -> usize {
        fn depth(node: &TreeNode) -> usize {
            if node.left.is_none() && node.right.is_none() {
                0
            } else {
                1 + std::cmp::max(
                    node.left.as_ref().map_or(0, |n| depth(n)),
                    node.right.as_ref().map_or(0, |n| depth(n)),
                )
            }
        }
        self.root.as_ref().map_or(0, depth)
    }

    fn get_n_leaves(&self) -> usize {
        fn leaves(node: &TreeNode) -> usize {
            if node.left.is_none() && node.right.is_none() {
                1
            } else {
                node.left.as_ref().map_or(0, |n| leaves(n))
                    + node.right.as_ref().map_or(0, |n| leaves(n))
            }
        }
        self.root.as_ref().map_or(0, leaves)
    }

    #[getter]
    fn classes_(&self) -> Option<Vec<usize>> {
        self.classes_.clone()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.is_fitted
    }
}
