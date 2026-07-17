mod boosting;
#[cfg(feature = "cuda")]
mod cuda;
mod decision_tree;
#[cfg(feature = "gpu")]
mod gpu;
mod histogram;
mod par;
mod random_forest;
mod tree;
mod tree_shap;
mod unsupervised;

use pyo3::prelude::*;

use boosting::{RFGBoost, RFGBoostClassifier, RFGBoostRegressor};
use decision_tree::DecisionTree;
use random_forest::{RandomForestClassifier, RandomForestRegressor};
use tree_shap::TreeSHAP;
use unsupervised::RandomForestUnsupervised;

#[pymodule]
fn _rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Module name must match pyproject.toml module-name last segment
    m.add_class::<DecisionTree>()?;
    m.add_class::<RandomForestRegressor>()?;
    m.add_class::<RandomForestClassifier>()?;
    m.add_class::<TreeSHAP>()?;
    m.add_class::<RFGBoost>()?;
    m.add_class::<RFGBoostClassifier>()?;
    m.add_class::<RFGBoostRegressor>()?;
    m.add_class::<RandomForestUnsupervised>()?;
    Ok(())
}
