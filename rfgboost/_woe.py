"""Python wrappers that add WOE categorical encoding to Rust classifiers."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rfgboost._rs import (
    RFGBoostClassifier as _RustClassifier,
)
from rfgboost._rs import (
    RFGBoostRegressor as _RustRegressor,
)


def _to_cat_rows(X: ArrayLike, cat_indices: Iterable[int]) -> list[list[str]]:
    """Extract categorical columns as list of rows (list of list of str)."""
    arr = np.asarray(X)
    n = arr.shape[0]
    indices = list(cat_indices)
    return [[str(arr[i, c]) for c in indices] for i in range(n)]


def _encode_woe(
    woe_encoder: Any,
    X: ArrayLike,
    cat_indices: Iterable[int],
) -> NDArray[np.float64]:
    """Apply fitted WOE encoder to categorical columns, return float64 array."""
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    indices = list(cat_indices)
    cat_rows = _to_cat_rows(arr, indices)
    woe_rows = woe_encoder.transform_matrix(cat_rows)
    woe_arr: NDArray[np.float64] = np.array(woe_rows, dtype=np.float64)

    if num_indices := [i for i in range(arr.shape[1]) if i not in indices]:
        num_arr = arr[:, num_indices].astype(np.float64)
        return np.hstack([woe_arr, num_arr])
    return woe_arr


class RFGBoostClassifier:
    """RFGBoostClassifier with optional WOE encoding for categorical features.

    Parameters
    ----------
    cat_features : list of int, optional
        Column indices of categorical features. When set, these columns are
        WOE-encoded during fit/predict using fastwoe-rs.
    All other parameters are passed to the Rust RFGBoostClassifier.
    """

    _estimator_type: str = "classifier"

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        rf_n_estimators: int = 50,
        rf_max_depth: Optional[int] = None,
        rf_max_features: Optional[int] = None,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        use_histogram: bool = True,
        async_mode: bool = False,
        tol: float = 1e-4,
        n_jobs: Optional[int] = None,
        cat_features: Optional[Iterable[int]] = None,
    ) -> None:
        self._params: dict[str, Any] = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            rf_max_features=rf_max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            use_histogram=use_histogram,
            async_mode=async_mode,
            tol=tol,
            n_jobs=n_jobs,
        )
        self.cat_features: Optional[list[int]] = (
            list(cat_features) if cat_features is not None else None
        )
        self._model: Any = _RustClassifier(**self._params)
        self._woe: Any = None
        self.classes_: NDArray[Any]

    def fit(self, X: ArrayLike, y: ArrayLike) -> RFGBoostClassifier:
        X_arr = np.asarray(X)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)

        self.classes_ = np.unique(y_arr)

        if self.cat_features:
            from fastwoe import FastWoe

            cat_rows = _to_cat_rows(X_arr, self.cat_features)
            self._woe = FastWoe()
            self._woe.fit_matrix(cat_rows, y_arr.astype(int))
            X_encoded = _encode_woe(self._woe, X_arr, self.cat_features)
        else:
            X_encoded = np.ascontiguousarray(X_arr, dtype=np.float64)

        self._model = _RustClassifier(**self._params)
        self._model.fit(
            np.ascontiguousarray(X_encoded, dtype=np.float64),
            y_arr,
        )
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict(X_encoded))

    def predict_proba(self, X: ArrayLike) -> NDArray[Any]:
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict_proba(X_encoded))

    def predict_ci(self, X: ArrayLike, alpha: float = 0.05) -> NDArray[Any]:
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict_ci(X_encoded, alpha))

    def feature_importances(self) -> Any:
        return self._model.feature_importances()

    def get_iv_analysis(self) -> Any:
        if self._woe is None:
            raise ValueError("No categorical features were encoded with WOE")
        return self._woe.get_iv_analysis()

    def _prepare_X(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.cat_features and self._woe is not None:
            return np.ascontiguousarray(
                _encode_woe(self._woe, X, self.cat_features), dtype=np.float64
            )
        return np.ascontiguousarray(X, dtype=np.float64)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator (sklearn-compatible)."""
        params = self._params.copy()
        params["cat_features"] = self.cat_features
        return params

    def set_params(self, **params: Any) -> RFGBoostClassifier:
        """Set parameters for this estimator (sklearn-compatible)."""
        cat_features = params.pop("cat_features", None)
        if cat_features is not None:
            self.cat_features = list(cat_features) if cat_features else None
        self._params.update(params)
        self._model = _RustClassifier(**self._params)
        return self

    @property
    def n_estimators(self) -> Any:
        return self._model.n_estimators

    @property
    def n_classes(self) -> Any:
        return self._model.n_classes

    @property
    def is_fitted(self) -> Any:
        return self._model.is_fitted

    @property
    def trees_used(self) -> Any:
        return self._model.trees_used


class RFGBoostRegressor:
    """RFGBoostRegressor with optional WOE encoding for categorical features.

    Parameters
    ----------
    cat_features : list of int, optional
        Column indices of categorical features. When set, these columns are
        WOE-encoded during fit/predict using fastwoe-rs. Note: WOE is computed
        by binarizing the regression target at its median.
    All other parameters are passed to the Rust RFGBoostRegressor.
    """

    _estimator_type: str = "regressor"

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        rf_n_estimators: int = 50,
        rf_max_depth: Optional[int] = None,
        rf_max_features: Optional[int] = None,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        use_histogram: bool = True,
        async_mode: bool = False,
        tol: float = 1e-4,
        n_jobs: Optional[int] = None,
        cat_features: Optional[Iterable[int]] = None,
    ) -> None:
        self._params: dict[str, Any] = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            rf_max_features=rf_max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            use_histogram=use_histogram,
            async_mode=async_mode,
            tol=tol,
            n_jobs=n_jobs,
        )
        self.cat_features: Optional[list[int]] = (
            list(cat_features) if cat_features is not None else None
        )
        self._model: Any = _RustRegressor(**self._params)
        self._woe: Any = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> RFGBoostRegressor:
        X_arr = np.asarray(X)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)

        if self.cat_features:
            from fastwoe import FastWoe

            y_binary = (y_arr > np.median(y_arr)).astype(int)
            cat_rows = _to_cat_rows(X_arr, self.cat_features)
            self._woe = FastWoe()
            self._woe.fit_matrix(cat_rows, y_binary)
            X_encoded = _encode_woe(self._woe, X_arr, self.cat_features)
        else:
            X_encoded = np.ascontiguousarray(X_arr, dtype=np.float64)

        self._model = _RustRegressor(**self._params)
        self._model.fit(
            np.ascontiguousarray(X_encoded, dtype=np.float64),
            y_arr,
        )
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict(X_encoded))

    def predict_ci(self, X: ArrayLike, alpha: float = 0.05) -> NDArray[Any]:
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict_ci(X_encoded, alpha))

    def feature_importances(self) -> Any:
        return self._model.feature_importances()

    def _prepare_X(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.cat_features and self._woe is not None:
            return np.ascontiguousarray(
                _encode_woe(self._woe, X, self.cat_features), dtype=np.float64
            )
        return np.ascontiguousarray(X, dtype=np.float64)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator (sklearn-compatible)."""
        params = self._params.copy()
        params["cat_features"] = self.cat_features
        return params

    def set_params(self, **params: Any) -> RFGBoostRegressor:
        """Set parameters for this estimator (sklearn-compatible)."""
        cat_features = params.pop("cat_features", None)
        if cat_features is not None:
            self.cat_features = list(cat_features) if cat_features else None
        self._params.update(params)
        self._model = _RustRegressor(**self._params)
        return self

    @property
    def n_estimators(self) -> Any:
        return self._model.n_estimators

    @property
    def is_fitted(self) -> Any:
        return self._model.is_fitted

    @property
    def trees_used(self) -> Any:
        return self._model.trees_used
