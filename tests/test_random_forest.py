"""Tests for RandomForest and RandomForestRegressor."""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score

from rfgboost import RandomForestClassifier, RandomForestRegressor


def test_classifier_accuracy():
    """Test that the classifier achieves reasonable accuracy on a synthetic dataset."""
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, max_depth=6, max_features="sqrt", random_state=42)
    rf.fit(X, y.astype(float))
    preds = np.array(rf.predict(X))
    assert accuracy_score(y, preds) > 0.9


def test_classifier_predict_proba_shape():
    """Test that predict_proba returns probabilities with the correct shape."""
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)
    rf = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42)
    rf.fit(X, y.astype(float))
    proba = np.array(rf.predict_proba(X))
    assert proba.shape == (500, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_classifier_getters():
    """Test the getters of the RandomForest classifier."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    assert not rf.is_fitted
    rf.fit(X, y.astype(float))
    assert rf.is_fitted
    assert rf.n_estimators == 10
    assert rf.n_classes == 2
    assert rf.classes_ == [0, 1]


def test_classifier_score():
    """Test that the score method returns a reasonable accuracy."""
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    rf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42)
    rf.fit(X, y.astype(float))
    s = rf.score(X, y.astype(float))
    assert s > 0.85


def test_regressor_r2():
    """Test that the regressor achieves a reasonable R^2 score on a synthetic dataset."""
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    rfr = RandomForestRegressor(n_estimators=50, max_depth=6, max_features="sqrt", random_state=42)
    rfr.fit(X, y)
    pred = np.array(rfr.predict(X))
    assert r2_score(y, pred) > 0.3


def test_regressor_getters():
    """Test the getters of the RandomForestRegressor."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    rfr = RandomForestRegressor(n_estimators=10, random_state=42)
    assert not rfr.is_fitted
    rfr.fit(X, y)
    assert rfr.is_fitted
    assert rfr.n_estimators == 10


def test_no_bootstrap():
    """Test that the classifier can be trained without bootstrapping."""
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, bootstrap=False, random_state=42)
    rf.fit(X, y.astype(float))
    assert rf.is_fitted
    acc = accuracy_score(y, rf.predict(X))
    assert acc > 0.8
