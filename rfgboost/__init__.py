"""RFGBoost: async Random Forest + gradient boosting engine in Rust."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rfgboost")
except PackageNotFoundError:  # not installed (e.g. running from source)
    __version__ = "0.0.0+unknown"

from rfgboost._rs import (
    DecisionTree,
    RandomForestClassifier,
    RandomForestRegressor,
    RandomForestUnsupervised,
    TreeSHAP,
)
from rfgboost._estimators import RFGBoostClassifier, RFGBoostRegressor
from rfgboost._woe import WoeEncoder

__all__ = [
    "__version__",
    "DecisionTree",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "RandomForestUnsupervised",  # TODO: Evaluate if needed
    "TreeSHAP",
    "RFGBoostClassifier",
    "RFGBoostRegressor",
    "WoeEncoder",
]
