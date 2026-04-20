mod histogram;
mod tree;
mod decision_tree;
mod random_forest;
mod tree_shap;
mod boosting;
mod unsupervised;

use pyo3::prelude::*;

use decision_tree::DecisionTree;
use random_forest::{RandomForest, RandomForestRegressor};
use tree_shap::TreeSHAP;
use boosting::{RFGBoost, RFGBoostClassifier, RFGBoostRegressor};
use unsupervised::RandomForestUnsupervised;

#[pymodule]
fn _rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Module name must match pyproject.toml module-name last segment
    m.add_class::<DecisionTree>()?;
    m.add_class::<RandomForestRegressor>()?;
    m.add_class::<RandomForest>()?;
    m.add_class::<TreeSHAP>()?;
    m.add_class::<RFGBoost>()?;
    m.add_class::<RFGBoostClassifier>()?;
    m.add_class::<RFGBoostRegressor>()?;
    m.add_class::<RandomForestUnsupervised>()?;
    Ok(())
}
