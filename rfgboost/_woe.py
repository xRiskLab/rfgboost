"""Python wrappers that add WOE categorical encoding to Rust classifiers."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from rfgboost._rs import (
    RFGBoostClassifier as _RustClassifier,
)
from rfgboost._rs import (
    RFGBoostRegressor as _RustRegressor,
)


def _to_cat_rows(X, cat_indices):
    """Extract categorical columns as list of rows (list of list of str)."""
    X = np.asarray(X)
    n = X.shape[0]
    return [[str(X[i, c]) for c in cat_indices] for i in range(n)]


def _encode_woe(woe_encoder, X, cat_indices, multiclass=False):
    """Apply fitted WOE encoder to categorical columns, return float64 array.

    For binary, returns one WOE column per categorical feature.
    For multiclass, returns one WOE column per (feature, class) pair.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    cat_rows = _to_cat_rows(X, cat_indices)
    if multiclass:
        woe_rows = woe_encoder.transform_matrix_multiclass(cat_rows)
    else:
        woe_rows = woe_encoder.transform_matrix(cat_rows)
    woe_arr = np.array(woe_rows)

    num_indices = [i for i in range(X.shape[1]) if i not in cat_indices]
    if num_indices:
        num_arr = X[:, num_indices].astype(np.float64)
        return np.hstack([woe_arr, num_arr])
    return woe_arr


class RFGBoostClassifier(ClassifierMixin, BaseEstimator):
    """RFGBoostClassifier with optional WOE encoding for categorical features.

    Parameters
    ----------
    cat_features : list of int, optional
        Column indices of categorical features. When set, these columns are
        WOE-encoded during fit/predict using fastwoe-rs.
    All other parameters are passed to the Rust RFGBoostClassifier.
    """

    def __init__(
        self,
        n_estimators=20,
        learning_rate=0.1,
        rf_n_estimators=20,
        rf_max_depth=None,
        rf_max_features=None,
        bootstrap=True,
        random_state=None,
        min_samples_split=2,
        min_samples_leaf=1,
        use_histogram=True,
        async_mode=False,
        tol=1e-4,
        n_jobs=None,
        cat_features=None,
    ):
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

    def _build_rust_params(self):
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

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        y = np.ascontiguousarray(y, dtype=np.float64)
        sw = (
            np.ascontiguousarray(sample_weight, dtype=np.float64)
            if sample_weight is not None
            else None
        )

        self.classes_ = np.unique(y)
        is_multiclass = len(self.classes_) > 2
        cat_features = list(self.cat_features) if self.cat_features else None

        if cat_features:
            from fastwoe import FastWoe

            cat_rows = _to_cat_rows(X, cat_features)
            self._woe = FastWoe()
            if is_multiclass:
                self._woe.fit_matrix_multiclass(cat_rows, y.astype(int))
            else:
                self._woe.fit_matrix(cat_rows, y.astype(int))
            self._woe_multiclass = is_multiclass
            X_encoded = _encode_woe(self._woe, X, cat_features, multiclass=is_multiclass)
        else:
            self._woe = None
            self._woe_multiclass = False
            X_encoded = np.ascontiguousarray(X, dtype=np.float64)

        self._model = _RustClassifier(**self._build_rust_params())
        self._model.fit(
            np.ascontiguousarray(X_encoded, dtype=np.float64),
            y,
            sw,
        )
        return self

    def predict(self, X):
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict(X_encoded))

    def predict_proba(self, X):
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict_proba(X_encoded))

    def predict_ci(self, X, alpha=0.05):
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict_ci(X_encoded, alpha))

    def feature_importances(self):
        return self._model.feature_importances()

    def get_iv_analysis(self):
        if self._woe is None:
            raise ValueError("No categorical features were encoded with WOE")
        return self._woe.get_iv_analysis()

    def _prepare_X(self, X):
        if self.cat_features and self._woe is not None:
            return np.ascontiguousarray(
                _encode_woe(
                    self._woe, X, self.cat_features,
                    multiclass=getattr(self, "_woe_multiclass", False),
                ),
                dtype=np.float64,
            )
        return np.ascontiguousarray(X, dtype=np.float64)

    @property
    def n_classes_(self):
        return self._model.n_classes if hasattr(self, "_model") else None

    @property
    def is_fitted(self):
        return hasattr(self, "_model") and self._model.is_fitted

    @property
    def trees_used(self):
        return self._model.trees_used


class RFGBoostRegressor(RegressorMixin, BaseEstimator):
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
        n_estimators=20,
        learning_rate=0.1,
        rf_n_estimators=20,
        rf_max_depth=None,
        rf_max_features=None,
        bootstrap=True,
        random_state=None,
        min_samples_split=2,
        min_samples_leaf=1,
        use_histogram=True,
        async_mode=False,
        tol=1e-4,
        n_jobs=None,
        cat_features=None,
    ):
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

    def _build_rust_params(self):
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

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        y = np.ascontiguousarray(y, dtype=np.float64)
        sw = (
            np.ascontiguousarray(sample_weight, dtype=np.float64)
            if sample_weight is not None
            else None
        )
        cat_features = list(self.cat_features) if self.cat_features else None

        if cat_features:
            from fastwoe import FastWoe

            y_binary = (y > np.median(y)).astype(int)
            cat_rows = _to_cat_rows(X, cat_features)
            self._woe = FastWoe()
            self._woe.fit_matrix(cat_rows, y_binary)
            X_encoded = _encode_woe(self._woe, X, cat_features)
        else:
            self._woe = None
            X_encoded = np.ascontiguousarray(X, dtype=np.float64)

        self._model = _RustRegressor(**self._build_rust_params())
        self._model.fit(
            np.ascontiguousarray(X_encoded, dtype=np.float64),
            y,
            sw,
        )
        return self

    def predict(self, X):
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict(X_encoded))

    def predict_ci(self, X, alpha=0.05):
        X_encoded = self._prepare_X(X)
        return np.array(self._model.predict_ci(X_encoded, alpha))

    def feature_importances(self):
        return self._model.feature_importances()

    def _prepare_X(self, X):
        if self.cat_features and self._woe is not None:
            return np.ascontiguousarray(
                _encode_woe(self._woe, X, self.cat_features), dtype=np.float64
            )
        return np.ascontiguousarray(X, dtype=np.float64)

    @property
    def is_fitted(self):
        return hasattr(self, "_model") and self._model.is_fitted

    @property
    def trees_used(self):
        return self._model.trees_used
