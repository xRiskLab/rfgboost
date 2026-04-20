"""Tests for TreeSHAP: exact match with official SHAP package."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from rfgboost import DecisionTree, TreeSHAP

shap = pytest.importorskip("shap")


def _make_data():
    """Creates a simple synthetic dataset for testing."""
    np.random.seed(42)
    X = np.random.rand(200, 3) * 4
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 1.5).astype(float)
    return X, y


def test_exact_match_with_shap_package():
    """Tests that TreeSHAP values match those from the official SHAP package."""
    X, y = _make_data()

    rust_tree = DecisionTree(max_depth=3, criterion="gini", task="classification", random_state=42)
    rust_tree.fit(X, y)
    sk_tree = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
    sk_tree.fit(X, y.astype(int))

    rust_shap = np.array(TreeSHAP(rust_tree, X, "classification").explain(X))[:, 1, :]
    sk_shap = shap.TreeExplainer(sk_tree, feature_perturbation="tree_path_dependent").shap_values(
        X
    )[:, :, 1]

    assert np.allclose(rust_shap, sk_shap), f"max diff: {np.max(np.abs(rust_shap - sk_shap))}"


def test_efficiency_property():
    """Tests the efficiency property of SHAP values."""
    X, y = _make_data()

    tree = DecisionTree(max_depth=4, task="classification", random_state=42)
    tree.fit(X, y)

    shap_vals = np.array(TreeSHAP(tree, X, "classification").explain(X))
    train_proba = np.array(tree.predict_proba(X))

    for c in range(2):
        base = np.mean([p[c] for p in train_proba])
        shap_sum = shap_vals[:, c, :].sum(axis=1)
        actual = np.array([p[c] for p in train_proba])
        assert np.max(np.abs(base + shap_sum - actual)) < 1e-10


def test_output_shape_classification():
    """Tests that the output shape of TreeSHAP is correct for classification."""
    X, y = _make_data()
    tree = DecisionTree(max_depth=3, task="classification", random_state=42)
    tree.fit(X, y)

    result = np.array(TreeSHAP(tree, X, "classification").explain(X[:10]))
    assert result.shape == (10, 2, 3)


def test_output_shape_regression():
    """Tests that the output shape of TreeSHAP is correct for regression."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    tree = DecisionTree(max_depth=4, criterion="variance", task="regression", random_state=42)
    tree.fit(X, y)

    result = np.array(TreeSHAP(tree, X, "regression").explain(X[:10]))
    assert result.shape == (10, 1, 5)


def test_deeper_tree_exact_match():
    """Tests that TreeSHAP values match SHAP package exactly at depth 4."""
    X, y = _make_data()
    tree = DecisionTree(max_depth=4, criterion="gini", task="classification", random_state=42)
    tree.fit(X, y)
    sk = DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=42)
    sk.fit(X, y.astype(int))

    rust_shap = np.array(TreeSHAP(tree, X, "classification").explain(X))[:, 1, :]
    sk_shap = shap.TreeExplainer(sk, feature_perturbation="tree_path_dependent").shap_values(X)[
        :, :, 1
    ]

    assert np.allclose(rust_shap, sk_shap, atol=1e-10)


def test_deeper_tree_efficiency():
    """Tests that SHAP efficiency holds exactly at any depth."""
    X, y = _make_data()
    for d in [3, 4, 5, 6]:
        tree = DecisionTree(max_depth=d, criterion="gini", task="classification", random_state=42)
        tree.fit(X, y)

        shap_vals = np.array(TreeSHAP(tree, X, "classification").explain(X))
        train_proba = np.array(tree.predict_proba(X))

        for c in range(2):
            base = np.mean([p[c] for p in train_proba])
            shap_sum = shap_vals[:, c, :].sum(axis=1)
            actual = np.array([p[c] for p in train_proba])
            assert np.max(np.abs(base + shap_sum - actual)) < 1e-10, (
                f"Efficiency violated at depth={d}, class={c}"
            )
