"""Scikit-learn estimator wrappers over the Rust RFGBoost engines.

These are pure numeric — categorical / Weight-of-Evidence encoding lives in the
standalone `rfgboost.WoeEncoder` transformer. For categorical data, compose with
a Pipeline::

    from sklearn.pipeline import make_pipeline
    from rfgboost import WoeEncoder, RFGBoostClassifier

    model = make_pipeline(WoeEncoder(cat_features=[0, 3]), RFGBoostClassifier())
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from rfgboost._rs import (
    RFGBoostClassifier as _RustClassifier,
)
from rfgboost._rs import (
    RFGBoostRegressor as _RustRegressor,
)


def _f64(X: ArrayLike) -> NDArray[np.float64]:
    return np.ascontiguousarray(X, dtype=np.float64)


class RFGBoostClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """RFGBoost gradient-boosting classifier over numeric features.

    For categorical features, WOE-encode first with `rfgboost.WoeEncoder` in a
    Pipeline.
    """

    def __init__(
        self,
        n_estimators: int = 20,
        learning_rate: float = 0.1,
        rf_n_estimators: int = 20,
        rf_max_depth: Optional[int] = None,
        rf_max_features: Optional[str] = None,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        use_histogram: bool = True,
        async_mode: bool = False,
        tol: float = 1e-4,
        n_jobs: Optional[int] = None,
        monotone_constraints: Optional[dict] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_max_features = rf_max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.use_histogram = use_histogram
        self.async_mode = async_mode
        self.tol = tol
        self.n_jobs = n_jobs
        # {column_index: +1|-1|0} over the numeric feature matrix this estimator
        # receives. WOE-encode categoricals upstream with WoeEncoder; constraints
        # then key on the encoded column positions.
        self.monotone_constraints = monotone_constraints

    def _build_rust_params(self) -> dict[str, Any]:
        return dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            rf_n_estimators=self.rf_n_estimators,
            rf_max_depth=self.rf_max_depth,
            rf_max_features=self.rf_max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            use_histogram=self.use_histogram,
            async_mode=self.async_mode,
            tol=self.tol,
            n_jobs=self.n_jobs,
            monotone_constraints=getattr(self, "_monotone_encoded", None),
        )

    def _monotone_vector(self, n_features: int) -> Optional[list[int]]:
        """Build the per-column monotonicity vector the Rust core expects from
        ``monotone_constraints={column_index: +1|-1|0}``.

        Keys index the columns of the matrix passed to ``fit`` (already numeric).
        Only the sign of each direction matters. Validated here at fit time,
        where ``n_features`` is known, so ``__init__`` stays a verbatim param
        store per the sklearn convention.
        """
        if not self.monotone_constraints:
            return None
        directions = [0] * n_features
        for col, direction in self.monotone_constraints.items():
            if not isinstance(col, (int, np.integer)) or not (0 <= int(col) < n_features):
                raise ValueError(
                    f"monotone_constraints key {col!r} is not a valid column index "
                    f"in [0, {n_features})."
                )
            directions[int(col)] = 1 if direction > 0 else -1 if direction < 0 else 0
        return directions

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> RFGBoostClassifier:
        X_arr = _f64(X)
        y_arr = _f64(y)
        sw = _f64(sample_weight) if sample_weight is not None else None
        self.classes_ = np.unique(y_arr)
        self._monotone_encoded = self._monotone_vector(X_arr.shape[1])
        self._model = _RustClassifier(**self._build_rust_params())
        self._model.fit(X_arr, y_arr, sw)
        return self

    def predict(self, X: ArrayLike, device: str = "cpu") -> NDArray[np.float64]:
        return np.array(self._model.predict(_f64(X), device), dtype=np.float64)

    def predict_proba(self, X: ArrayLike, device: str = "cpu") -> NDArray[np.float64]:
        return np.array(self._model.predict_proba(_f64(X), device), dtype=np.float64)

    def predict_ci(self, X: ArrayLike, alpha: float = 0.05) -> NDArray[np.float64]:
        return np.array(self._model.predict_ci(_f64(X), alpha), dtype=np.float64)

    def feature_importances(self) -> list[float]:
        return list(self._model.feature_importances())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fitted model to a JSON-friendly dict."""
        if not getattr(self, "_model", None) or not self._model.is_fitted:
            raise ValueError("RFGBoostClassifier has not been fitted")
        return dict(self._model.to_dict())

    @property
    def n_classes_(self) -> Optional[int]:
        return self._model.n_classes if hasattr(self, "_model") else None

    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "_model") and self._model.is_fitted

    @property
    def trees_used(self) -> list[int]:
        return list(self._model.trees_used)


class RFGBoostRegressor(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """RFGBoost gradient-boosting regressor over numeric features.

    For categorical features, WOE-encode first with `rfgboost.WoeEncoder` in a
    Pipeline.
    """

    def __init__(
        self,
        n_estimators: int = 20,
        learning_rate: float = 0.1,
        rf_n_estimators: int = 20,
        rf_max_depth: Optional[int] = None,
        rf_max_features: Optional[str] = None,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        use_histogram: bool = True,
        async_mode: bool = False,
        tol: float = 1e-4,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_max_features = rf_max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.use_histogram = use_histogram
        self.async_mode = async_mode
        self.tol = tol
        self.n_jobs = n_jobs

    def _build_rust_params(self) -> dict[str, Any]:
        return dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            rf_n_estimators=self.rf_n_estimators,
            rf_max_depth=self.rf_max_depth,
            rf_max_features=self.rf_max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            use_histogram=self.use_histogram,
            async_mode=self.async_mode,
            tol=self.tol,
            n_jobs=self.n_jobs,
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> RFGBoostRegressor:
        y_arr = _f64(y)
        sw = _f64(sample_weight) if sample_weight is not None else None
        self._model = _RustRegressor(**self._build_rust_params())
        self._model.fit(_f64(X), y_arr, sw)
        return self

    def predict(self, X: ArrayLike, device: str = "cpu") -> NDArray[np.float64]:
        return np.array(self._model.predict(_f64(X), device), dtype=np.float64)

    def predict_ci(self, X: ArrayLike, alpha: float = 0.05) -> NDArray[np.float64]:
        return np.array(self._model.predict_ci(_f64(X), alpha), dtype=np.float64)

    def feature_importances(self) -> list[float]:
        return list(self._model.feature_importances())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fitted model to a JSON-friendly dict."""
        if not getattr(self, "_model", None) or not self._model.is_fitted:
            raise ValueError("RFGBoostRegressor has not been fitted")
        return dict(self._model.to_dict())

    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "_model") and self._model.is_fitted

    @property
    def trees_used(self) -> list[int]:
        return list(self._model.trees_used)
