"""Standalone Weight-of-Evidence encoder for categorical features.

Decoupled from the estimators so the numeric `RFGBoost*` models carry no
categorical / fastwoe-rs dependency (important for environments without a
fastwoe-rs wheel, e.g. Pyodide/WASM). Compose with a scikit-learn Pipeline::

    from sklearn.pipeline import make_pipeline
    from rfgboost import WoeEncoder, RFGBoostClassifier

    model = make_pipeline(WoeEncoder(cat_features=[0, 3]), RFGBoostClassifier())
    model.fit(X, y)
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target

_FASTWOE_MISSING = (
    "WoeEncoder requires fastwoe-rs, which has no Pyodide/WASM wheel and so is "
    "unavailable in this environment (e.g. JupyterLite). It ships automatically "
    "with a normal install; otherwise run `pip install fastwoe-rs`."
)


def _to_cat_rows(X: ArrayLike, cat_indices: Iterable[int]) -> list[list[str]]:
    """Extract categorical columns as a list of rows (list of list of str)."""
    arr = np.asarray(X)
    n = arr.shape[0]
    indices = list(cat_indices)
    return [[str(arr[i, c]) for c in indices] for i in range(n)]


def _encode_woe(
    woe_encoder: Any,
    X: ArrayLike,
    cat_indices: Iterable[int],
    multiclass: bool = False,
) -> NDArray[np.float64]:
    """Apply a fitted WOE encoder to the categorical columns; return float64.

    WOE columns first (binary: one per feature; multiclass: one per
    (feature, class)), then the remaining numeric columns in original order.
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


def _woe_bundle(
    woe_encoder: Any,
    cat_features: list[int],
    n_total_features: int,
    multiclass: bool,
    n_classes: int,
) -> dict[str, Any]:
    """Serialize fitted FastWoe state into JSON-friendly lookup tables.

    Layout matches `_encode_woe`: WOE columns first (in cat_features order; one
    per class for multiclass, one per feature for binary), then numeric columns
    in original index order.
    """
    cat_set = set(cat_features)
    numeric_features = [i for i in range(n_total_features) if i not in cat_set]
    bundle: dict[str, Any] = {
        "cat_features": list(cat_features),
        "numeric_features": numeric_features,
        "woe_multiclass": multiclass,
    }

    tables: list[Any] = []
    if multiclass:
        bundle["n_woe_classes"] = n_classes
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


class WoeEncoder(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Supervised WOE encoder (fastwoe-rs) for the columns in `cat_features`.

    `fit(X, y)` learns the encoding; the task is inferred from `y` via
    scikit-learn's `type_of_target` — binary / multiclass classification use the
    class labels, continuous targets are binarized at the median. `transform`
    replaces the categorical columns with their WOE columns (one per feature for
    binary, one per (feature, class) for multiclass) and keeps the numeric
    columns. With no `cat_features` it is a numeric passthrough.
    """

    def __init__(self, cat_features: Optional[Iterable[int]] = None) -> None:
        self.cat_features = cat_features

    def fit(self, X: ArrayLike, y: ArrayLike) -> "WoeEncoder":
        self.cat_features_ = list(self.cat_features) if self.cat_features else []
        if not self.cat_features_:
            self.woe_ = None
            self.multiclass_ = False
            self.n_classes_ = 0
            return self

        try:
            from fastwoe import FastWoe
        except ImportError as exc:
            raise ImportError(_FASTWOE_MISSING) from exc

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        cat_rows = _to_cat_rows(X_arr, self.cat_features_)
        self.woe_ = FastWoe()
        if type_of_target(y_arr) == "multiclass":
            self.multiclass_ = True
            yi = y_arr.astype(int)
            self.n_classes_ = int(np.unique(yi).size)
            self.woe_.fit_matrix_multiclass(cat_rows, yi)
        else:
            self.multiclass_ = False
            self.n_classes_ = 2
            if type_of_target(y_arr) == "continuous":
                yb = (y_arr.astype(float) > np.median(y_arr)).astype(int)
            else:
                yb = y_arr.astype(int)
            self.woe_.fit_matrix(cat_rows, yb)
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        if not getattr(self, "cat_features_", None) or self.woe_ is None:
            return np.ascontiguousarray(X, dtype=np.float64)
        return np.ascontiguousarray(
            _encode_woe(self.woe_, X, self.cat_features_, multiclass=self.multiclass_),
            dtype=np.float64,
        )

    def get_iv_analysis(self) -> Any:
        if getattr(self, "woe_", None) is None:
            raise ValueError("WoeEncoder has no fitted WOE (no cat_features)")
        return self.woe_.get_iv_analysis()

    def to_dict(self, n_features: int) -> dict[str, Any]:
        """JSON-friendly WOE lookup-table bundle. `n_features` is the original
        (pre-encoding) column count, needed to recover the numeric-column order."""
        if getattr(self, "woe_", None) is None:
            raise ValueError("WoeEncoder has no fitted WOE (no cat_features)")
        return _woe_bundle(
            self.woe_, self.cat_features_, n_features,
            multiclass=self.multiclass_, n_classes=self.n_classes_,
        )
