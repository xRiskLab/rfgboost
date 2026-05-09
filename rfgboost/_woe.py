"""Python wrappers that add WOE categorical encoding to Rust classifiers."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

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


def _woe_bundle(
    woe_encoder: Any,
    cat_features: list[int],
    n_total_features: int,
    multiclass: bool,
    n_classes: int,
) -> dict[str, Any]:
    """Serialize fitted FastWoe state into JSON-friendly lookup tables.

    Layout matches `_encode_woe`: WOE columns first (in cat_features order;
    one column per class for multiclass, one per feature for binary), then
    numeric columns in their original index order.
    """
    numeric_features = [i for i in range(n_total_features) if i not in cat_features]
    bundle: dict[str, Any] = {
        "cat_features": list(cat_features),
        "numeric_features": numeric_features,
        "woe_multiclass": bool(multiclass),
    }

    tables: list[Any] = []
    if multiclass:
        bundle["n_woe_classes"] = int(n_classes)
        for i, _ in enumerate(cat_features):
            per_class: list[dict[str, float]] = []
            for c in range(n_classes):
                rows = woe_encoder.get_feature_mapping_multiclass(c, f"feature_{i}")
                per_class.append({str(r.category): float(r.woe) for r in rows})
            tables.append(per_class)
    else:
        for i, _ in enumerate(cat_features):
            rows = woe_encoder.get_feature_mapping(f"feature_{i}")
            tables.append({str(r.category): float(r.woe) for r in rows})

    bundle["woe_tables"] = tables
    return bundle


def _encode_woe(
    woe_encoder: Any,
    X: ArrayLike,
    cat_indices: Iterable[int],
    multiclass: bool = False,
) -> NDArray[np.float64]:
    """Apply fitted WOE encoder to categorical columns, return float64 array.

    For binary, returns one WOE column per categorical feature.
    For multiclass, returns one WOE column per (feature, class) pair.
    """
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    indices = list(cat_indices)
    cat_rows = _to_cat_rows(arr, indices)
    if multiclass:
        woe_rows = woe_encoder.transform_matrix_multiclass(cat_rows)
    else:
        woe_rows = woe_encoder.transform_matrix(cat_rows)
    woe_arr: NDArray[np.float64] = np.array(woe_rows, dtype=np.float64)

    if num_indices := [i for i in range(arr.shape[1]) if i not in indices]:
        num_arr = arr[:, num_indices].astype(np.float64)
        return np.hstack([woe_arr, num_arr])
    return woe_arr


class RFGBoostClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """RFGBoostClassifier with optional WOE encoding for categorical features.

    Parameters
    ----------
    cat_features : list of int, optional
        Column indices of categorical features. When set, these columns are
        WOE-encoded during fit/predict using fastwoe-rs. Multiclass targets
        produce one WOE column per (feature, class) pair.
    All other parameters are passed to the Rust RFGBoostClassifier.
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
        cat_features: Optional[Iterable[int]] = None,
    ) -> None:
        # Store hyperparameters as attributes (sklearn BaseEstimator convention)
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
        self.cat_features = cat_features

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
    ) -> RFGBoostClassifier:
        X_arr = np.asarray(X)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        sw = (
            np.ascontiguousarray(sample_weight, dtype=np.float64)
            if sample_weight is not None
            else None
        )

        self.classes_ = np.unique(y_arr)
        is_multiclass = len(self.classes_) > 2
        cat_features = list(self.cat_features) if self.cat_features else None

        if cat_features:
            from fastwoe import FastWoe

            cat_rows = _to_cat_rows(X_arr, cat_features)
            self._woe = FastWoe()
            if is_multiclass:
                self._woe.fit_matrix_multiclass(cat_rows, y_arr.astype(int))
            else:
                self._woe.fit_matrix(cat_rows, y_arr.astype(int))
            self._woe_multiclass = is_multiclass
            X_encoded = _encode_woe(self._woe, X_arr, cat_features, multiclass=is_multiclass)
        else:
            self._woe = None
            self._woe_multiclass = False
            X_encoded = np.ascontiguousarray(X_arr, dtype=np.float64)

        self._model = _RustClassifier(**self._build_rust_params())
        self._model.fit(
            np.ascontiguousarray(X_encoded, dtype=np.float64),
            y_arr,
            sw,
        )
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict(X_encoded), dtype=np.float64)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict_proba(X_encoded), dtype=np.float64)

    def predict_ci(self, X: ArrayLike, alpha: float = 0.05) -> NDArray[np.float64]:
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict_ci(X_encoded, alpha), dtype=np.float64)

    def feature_importances(self) -> list[float]:
        return list(self._model.feature_importances())

    def get_iv_analysis(self) -> Any:
        if self._woe is None:
            raise ValueError("No categorical features were encoded with WOE")
        return self._woe.get_iv_analysis()

    def to_dict(self, n_features: Optional[int] = None) -> dict[str, Any]:
        """Serialize the fitted model to a JSON-friendly dict.

        Includes WOE lookup tables when `cat_features` was used. The returned
        structure can be dumped via `json.dump` after replacing NaN thresholds
        if the consumer rejects them. `n_features` is required when
        `cat_features` was used so the numeric-column ordering can be recovered.
        """
        if not getattr(self, "_model", None) or not self._model.is_fitted:
            raise ValueError("RFGBoostClassifier has not been fitted")
        d: dict[str, Any] = dict(self._model.to_dict())
        if self.cat_features and self._woe is not None:
            cat_features = list(self.cat_features)
            if n_features is None:
                raise ValueError(
                    "n_features (original column count) is required when cat_features was used"
                )
            d["woe"] = _woe_bundle(
                self._woe,
                cat_features,
                n_features,
                multiclass=getattr(self, "_woe_multiclass", False),
                n_classes=int(len(self.classes_)),
            )
        return d

    def _prepare_X(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.cat_features and self._woe is not None:
            return np.ascontiguousarray(
                _encode_woe(
                    self._woe,
                    X,
                    self.cat_features,
                    multiclass=getattr(self, "_woe_multiclass", False),
                ),
                dtype=np.float64,
            )
        return np.ascontiguousarray(X, dtype=np.float64)

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
    """RFGBoostRegressor with optional WOE encoding for categorical features.

    Parameters
    ----------
    cat_features : list of int, optional
        Column indices of categorical features. When set, these columns are
        WOE-encoded during fit/predict using fastwoe-rs. Note: WOE is computed
        by binarizing the regression target at its median.
    All other parameters are passed to the Rust RFGBoostRegressor.
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
        cat_features: Optional[Iterable[int]] = None,
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
        self.cat_features = cat_features

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
        X_arr = np.asarray(X)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        sw = (
            np.ascontiguousarray(sample_weight, dtype=np.float64)
            if sample_weight is not None
            else None
        )
        cat_features = list(self.cat_features) if self.cat_features else None

        if cat_features:
            from fastwoe import FastWoe

            y_binary = (y_arr > np.median(y_arr)).astype(int)
            cat_rows = _to_cat_rows(X_arr, cat_features)
            self._woe = FastWoe()
            self._woe.fit_matrix(cat_rows, y_binary)
            X_encoded = _encode_woe(self._woe, X_arr, cat_features)
        else:
            self._woe = None
            X_encoded = np.ascontiguousarray(X_arr, dtype=np.float64)

        self._model = _RustRegressor(**self._build_rust_params())
        self._model.fit(
            np.ascontiguousarray(X_encoded, dtype=np.float64),
            y_arr,
            sw,
        )
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict(X_encoded), dtype=np.float64)

    def predict_ci(self, X: ArrayLike, alpha: float = 0.05) -> NDArray[np.float64]:
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict_ci(X_encoded, alpha), dtype=np.float64)

    def feature_importances(self) -> list[float]:
        return list(self._model.feature_importances())

    def to_dict(self, n_features: Optional[int] = None) -> dict[str, Any]:
        """Serialize the fitted model to a JSON-friendly dict.

        WOE for regression uses median-binarized targets; if cat_features was
        used, n_features (original column count) is required to recover the
        numeric-column ordering.
        """
        if not getattr(self, "_model", None) or not self._model.is_fitted:
            raise ValueError("RFGBoostRegressor has not been fitted")
        d: dict[str, Any] = dict(self._model.to_dict())
        if self.cat_features and self._woe is not None:
            cat_features = list(self.cat_features)
            if n_features is None:
                raise ValueError(
                    "n_features (original column count) is required when cat_features was used"
                )
            d["woe"] = _woe_bundle(
                self._woe,
                cat_features,
                n_features,
                multiclass=False,
                n_classes=2,
            )
        return d

    def _prepare_X(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.cat_features and self._woe is not None:
            return np.ascontiguousarray(
                _encode_woe(self._woe, X, self.cat_features), dtype=np.float64
            )
        return np.ascontiguousarray(X, dtype=np.float64)

    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "_model") and self._model.is_fitted

    @property
    def trees_used(self) -> list[int]:
        return list(self._model.trees_used)
