"""Multiclass and cat_features integration tests for RFGBoostClassifier."""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from rfgboost import RFGBoostClassifier


def test_multiclass_numeric_only():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y.astype(float), random_state=0, stratify=y)
    clf = RFGBoostClassifier(n_estimators=10, rf_n_estimators=20, rf_max_depth=4, random_state=42)
    clf.fit(Xtr, ytr)
    assert clf.n_classes_ == 3
    assert clf.predict_proba(Xte).shape == (len(yte), 3)
    np.testing.assert_allclose(clf.predict_proba(Xte).sum(axis=1), 1.0, atol=1e-6)
    assert accuracy_score(yte, clf.predict(Xte)) > 0.85


def _make_mixed_multiclass(n=600, n_classes=4, seed=0):
    rng = np.random.RandomState(seed)
    X_num = rng.randn(n, 4)
    X_cat1 = rng.randint(0, 5, size=n).astype(float)
    X_cat2 = rng.randint(0, 3, size=n).astype(float)
    X = np.column_stack([X_num, X_cat1, X_cat2])
    y = ((X_cat1.astype(int) % n_classes) + (X_num[:, 0] > 0).astype(int)) % n_classes
    return X, y.astype(float)


def test_multiclass_with_cat_features():
    X, y = _make_mixed_multiclass(n=600, n_classes=4)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0, stratify=y)
    clf = RFGBoostClassifier(
        n_estimators=10,
        rf_n_estimators=20,
        rf_max_depth=4,
        random_state=42,
        cat_features=[4, 5],
    )
    clf.fit(Xtr, ytr)
    assert clf.n_classes_ == 4
    # WOE expands each cat feature into n_classes columns: 4 numeric + 2 cat * 4 classes = 12
    assert clf._prepare_X(Xte[:1]).shape == (1, 12)
    proba = clf.predict_proba(Xte)
    assert proba.shape == (len(yte), 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert accuracy_score(yte, clf.predict(Xte)) > 0.7


def test_binary_with_cat_features_unchanged():
    """Binary path must keep using the binary WOE encoder (one column per cat feature)."""
    X, y_mc = _make_mixed_multiclass(n=400, n_classes=4)
    y = (y_mc < 2).astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0, stratify=y)
    clf = RFGBoostClassifier(
        n_estimators=8,
        rf_n_estimators=15,
        rf_max_depth=4,
        random_state=42,
        cat_features=[4, 5],
    )
    clf.fit(Xtr, ytr)
    assert clf.n_classes_ == 2
    # Binary WOE: 4 numeric + 2 cat * 1 = 6 columns
    assert clf._prepare_X(Xte[:1]).shape == (1, 6)
    proba = clf.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    assert accuracy_score(yte, clf.predict(Xte)) > 0.7


def test_multiclass_with_sample_weight():
    """Multiclass + cat_features + sample_weight all together."""
    X, y = _make_mixed_multiclass(n=400, n_classes=3)
    sw = np.ones(len(y))
    sw[y == 0] = 3.0
    clf = RFGBoostClassifier(
        n_estimators=5,
        rf_n_estimators=10,
        rf_max_depth=3,
        random_state=0,
        cat_features=[4, 5],
    )
    clf.fit(X, y, sample_weight=sw)
    assert clf.n_classes_ == 3
    assert clf.predict_proba(X).shape == (len(y), 3)
