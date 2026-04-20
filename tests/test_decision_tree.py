"""Tests for DecisionTree: sklearn match, classification, regression."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from rfgboost import DecisionTree


def _make_data():
    """Generate synthetic classification data."""
    np.random.seed(42)
    X = np.random.rand(200, 3) * 4
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 1.5).astype(float)
    return X, y


def test_predictions_match_sklearn():
    """Test that predictions match sklearn's DecisionTreeClassifier."""
    X, y = _make_data()
    rust = DecisionTree(max_depth=3, criterion="gini", task="classification", random_state=42)
    rust.fit(X, y)
    sk = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
    sk.fit(X, y.astype(int))

    rust_pred = np.array(rust.predict(X))
    sk_pred = sk.predict(X).astype(float)
    assert np.mean(rust_pred == sk_pred) == 1.0


def test_probabilities_match_sklearn():
    """Test that predicted probabilities match sklearn's DecisionTreeClassifier."""
    X, y = _make_data()
    rust = DecisionTree(max_depth=3, criterion="gini", task="classification", random_state=42)
    rust.fit(X, y)
    sk = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
    sk.fit(X, y.astype(int))

    rust_proba = np.array(rust.predict_proba(X))
    sk_proba = sk.predict_proba(X)
    assert np.max(np.abs(rust_proba - sk_proba)) < 1e-10


def test_depth_and_leaves_match_sklearn():
    """Test that tree depth and number of leaves match sklearn's DecisionTreeClassifier."""
    X, y = _make_data()
    rust = DecisionTree(max_depth=3, criterion="gini", task="classification", random_state=42)
    rust.fit(X, y)
    sk = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
    sk.fit(X, y.astype(int))

    assert rust.get_depth() == sk.get_depth()
    assert rust.get_n_leaves() == sk.get_n_leaves()


def test_is_fitted_flag():
    """Test the is_fitted flag of the DecisionTree."""
    rust = DecisionTree(max_depth=3, task="classification")
    assert not rust.is_fitted
    X, y = _make_data()
    rust.fit(X, y)
    assert rust.is_fitted


def test_regression():
    """Test that the DecisionTree can fit and predict regression data."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    rust = DecisionTree(max_depth=5, criterion="variance", task="regression", random_state=42)
    rust.fit(X, y)
    pred = np.array(rust.predict(X))
    assert rust.is_fitted
    assert len(pred) == 200
    assert np.isfinite(pred).all()


def test_classes_getter():
    """Test the classes_ attribute of the DecisionTree."""
    X, y = _make_data()
    rust = DecisionTree(max_depth=3, task="classification", random_state=42)
    rust.fit(X, y)
    assert rust.classes_ == [0, 1]
