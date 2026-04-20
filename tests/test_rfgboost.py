"""Tests for RFGBoost: classification, regression, async, CI, feature importance."""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import r2_score, roc_auc_score

from rfgboost import RFGBoostClassifier, RFGBoostRegressor


def _make_clf_data(n=2000):
    """Generate synthetic classification data."""
    X, y = make_classification(n_samples=n, n_features=10, random_state=42)
    return X, y.astype(float)


def _make_reg_data(n=2000):
    """Generate synthetic regression data."""
    return make_regression(n_samples=n, n_features=10, random_state=42)


# -------------------------------------------------------------------------------
# Classification tests
# -------------------------------------------------------------------------------
def test_classification_standard():
    """Test standard classification performance."""
    X, y = _make_clf_data()
    m = RFGBoostClassifier(
        n_estimators=5,
        rf_n_estimators=30,
        rf_max_depth=6,
        random_state=42,
    )
    m.fit(X, y)
    auc = roc_auc_score(y, np.array(m.predict_proba(X))[:, 1])
    assert auc > 0.95


def test_classification_async():
    """Test async classification performance and early stopping."""
    X, y = _make_clf_data()
    m = RFGBoostClassifier(
        n_estimators=5,
        rf_n_estimators=100,
        rf_max_depth=6,
        random_state=42,
        async_mode=True,
        tol=0.0,
    )
    m.fit(X, y)
    auc = roc_auc_score(y, np.array(m.predict_proba(X))[:, 1])
    assert auc > 0.95
    assert sum(m.trees_used) < 500


def test_classification_exact_split():
    """Test classification with exact splits (no histogram)."""
    X, y = _make_clf_data()
    m = RFGBoostClassifier(
        n_estimators=5,
        rf_n_estimators=30,
        rf_max_depth=6,
        random_state=42,
        use_histogram=False,
    )
    m.fit(X, y)
    auc = roc_auc_score(y, np.array(m.predict_proba(X))[:, 1])
    assert auc > 0.95


def test_predict_returns_labels():
    """Test that predict returns valid class labels."""
    X, y = _make_clf_data()
    m = RFGBoostClassifier(
        n_estimators=3,
        rf_n_estimators=10,
        rf_max_depth=4,
        random_state=42,
    )
    m.fit(X, y)
    preds = np.array(m.predict(X))
    assert set(preds).issubset({0.0, 1.0})


def test_predict_proba_shape():
    """Test that predict_proba returns correct shape and valid probabilities."""
    X, y = _make_clf_data(500)
    m = RFGBoostClassifier(
        n_estimators=3,
        rf_n_estimators=10,
        rf_max_depth=4,
        random_state=42,
    )
    m.fit(X, y)
    proba = np.array(m.predict_proba(X))
    assert proba.shape == (500, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# -------------------------------------------------------------------------------
# Regression tests
# -------------------------------------------------------------------------------
def test_regression_standard():
    """test standard regression performance."""
    X, y = _make_reg_data()
    m = RFGBoostRegressor(
        n_estimators=5,
        rf_n_estimators=30,
        rf_max_depth=6,
        random_state=42,
    )
    m.fit(X, y)
    r2 = r2_score(y, m.predict(X))
    assert r2 > 0.3


def test_regression_async():
    """Test async regression performance and early stopping."""
    X, y = _make_reg_data()
    m = RFGBoostRegressor(
        n_estimators=5,
        rf_n_estimators=100,
        rf_max_depth=6,
        random_state=42,
        async_mode=True,
        tol=0.0,
    )
    m.fit(X, y)
    r2 = r2_score(y, m.predict(X))
    assert r2 > 0.3
    assert sum(m.trees_used) < 500


# -------------------------------------------------------------------------------
# CI and feature importance tests
# -------------------------------------------------------------------------------
def test_predict_ci_shape():
    """Test that predict_ci returns correct shape and valid intervals."""
    X, y = _make_clf_data(200)
    m = RFGBoostClassifier(
        n_estimators=3,
        rf_n_estimators=10,
        rf_max_depth=4,
        random_state=42,
    )
    m.fit(X, y)
    ci = np.array(m.predict_ci(X))
    assert ci.shape == (200, 2)
    assert np.all(ci[:, 0] <= ci[:, 1])


def test_predict_ci_regression():
    """Test that predict_ci returns correct shape and valid intervals for regression."""
    X, y = _make_reg_data(200)
    m = RFGBoostRegressor(
        n_estimators=3,
        rf_n_estimators=10,
        rf_max_depth=4,
        random_state=42,
    )
    m.fit(X, y)
    ci = np.array(m.predict_ci(X))
    assert ci.shape == (200, 2)
    assert np.all(ci[:, 0] <= ci[:, 1])


def test_predict_ci_bounds_classification():
    """Test that classification CI intervals are within [0, 1]."""
    X, y = _make_clf_data(200)
    m = RFGBoostClassifier(
        n_estimators=5,
        rf_n_estimators=20,
        rf_max_depth=6,
        random_state=42,
    )
    m.fit(X, y)
    ci = np.array(m.predict_ci(X))
    assert np.all(ci[:, 0] >= 0.0)
    assert np.all(ci[:, 1] <= 1.0)


def test_feature_importances():
    """Test that feature_importances returns valid importance scores."""
    X, y = _make_clf_data()
    m = RFGBoostClassifier(
        n_estimators=5,
        rf_n_estimators=30,
        rf_max_depth=6,
        random_state=42,
    )
    m.fit(X, y)
    imp = m.feature_importances()
    assert len(imp) >= 5
    assert sum(imp) > 0


# -------------------------------------------------------------------------------
# Getter tests
# -------------------------------------------------------------------------------
def test_getters():
    """Test that getters return expected values."""
    m = RFGBoostClassifier(n_estimators=7, rf_n_estimators=20)
    assert not m.is_fitted
    assert m.n_estimators == 7
    X, y = _make_clf_data(200)
    m.fit(X, y)
    assert m.is_fitted
    assert len(m.trees_used) == 7


def test_trees_used_async():
    """Test that trees_used reflects actual trees used in async mode."""
    X, y = _make_clf_data()
    m = RFGBoostClassifier(
        n_estimators=5,
        rf_n_estimators=100,
        rf_max_depth=6,
        random_state=42,
        async_mode=True,
        tol=0.0,
    )
    m.fit(X, y)
    assert len(m.trees_used) == 5
    for t in m.trees_used:
        assert t <= 100
        assert t >= 3
