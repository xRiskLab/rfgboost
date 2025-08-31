"""Test tree extraction functionality with different base learners."""

import numpy as np
import pandas as pd
import pytest

from rfgboost import RFGBoost

# Check if XGBoost is available
try:
    import xgboost

    XGBOOST_AVAILABLE = bool(xgboost)
except ImportError:
    XGBOOST_AVAILABLE = False


@pytest.fixture
def tree_test_data():
    """Create simple test data for tree extraction."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "cat_a": ["A", "B", "A", "B"] * 50,
            "num_1": np.random.normal(0, 1, 200),
            "target": np.random.binomial(1, 0.4, 200),
        }
    )

    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y


def test_sklearn_tree_extraction(tree_test_data):
    """Test tree extraction with sklearn base learner."""
    X, y = tree_test_data

    model = RFGBoost(
        n_estimators=3,
        task="classification",
        cat_features=["cat_a"],
        base_learner="sklearn",
        rf_params={"n_estimators": 2, "max_depth": 3, "random_state": 42},
    )

    model.fit(X, y)

    # Test tree data extraction methods
    tree_data = model.extract_tree_data_with_conditions()
    assert isinstance(tree_data, pd.DataFrame)
    assert len(tree_data) > 0

    leaf_data = model.extract_leaf_nodes_with_conditions()
    assert isinstance(leaf_data, pd.DataFrame)
    assert len(leaf_data) > 0

    # Test trees to dataframe without data
    tree_df = model.trees_to_dataframe()
    assert isinstance(tree_df, pd.DataFrame)
    assert len(tree_df) > 0

    expected_cols = ["Round", "Tree", "NodeID", "PathCondition", "Samples", "Value"]
    for col in expected_cols:
        assert col in tree_df.columns

    # Test trees to dataframe with data
    tree_df_with_data = model.trees_to_dataframe(X, y)
    assert isinstance(tree_df_with_data, pd.DataFrame)
    assert len(tree_df_with_data) > 0

    # Should have additional columns when data is provided
    additional_cols = ["Events", "NonEvents", "EventRate"]
    for col in additional_cols:
        assert col in tree_df_with_data.columns


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
def test_xgboost_tree_extraction(tree_test_data):
    """Test tree extraction with XGBoost base learner."""
    X, y = tree_test_data

    model = RFGBoost(
        n_estimators=3,
        task="classification",
        cat_features=["cat_a"],
        base_learner="xgboost",
        rf_params={"n_estimators": 2, "max_depth": 3, "random_state": 42},
    )

    model.fit(X, y)

    # Test each tree extraction method
    tree_data = model.extract_tree_data_with_conditions()
    assert isinstance(tree_data, pd.DataFrame)
    assert len(tree_data) > 0

    leaf_data = model.extract_leaf_nodes_with_conditions()
    assert isinstance(leaf_data, pd.DataFrame)
    assert len(leaf_data) > 0

    tree_df = model.trees_to_dataframe()
    assert isinstance(tree_df, pd.DataFrame)
    assert len(tree_df) > 0

    tree_df_with_data = model.trees_to_dataframe(X, y)
    assert isinstance(tree_df_with_data, pd.DataFrame)
    assert len(tree_df_with_data) > 0


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
def test_sklearn_vs_xgboost_tree_structure(tree_test_data):
    """Compare tree structure between sklearn and XGBoost."""
    X, y = tree_test_data

    models = {}
    tree_results = {}

    for base_learner in ["sklearn", "xgboost"]:
        model = RFGBoost(
            n_estimators=2,
            task="classification",
            cat_features=["cat_a"],
            base_learner=base_learner,
            rf_params={"n_estimators": 2, "max_depth": 3, "random_state": 42},
        )

        model.fit(X, y)
        models[base_learner] = model
        tree_results[base_learner] = model.trees_to_dataframe()

    # Both should produce tree results
    sklearn_trees = tree_results["sklearn"]
    xgboost_trees = tree_results["xgboost"]

    assert len(sklearn_trees) > 0
    assert len(xgboost_trees) > 0

    # Should have same core columns
    common_cols = {"Value", "Samples", "PathCondition", "Round", "NodeID", "Tree"}

    sklearn_cols = set(sklearn_trees.columns)
    xgboost_cols = set(xgboost_trees.columns)

    assert common_cols.issubset(sklearn_cols)
    assert common_cols.issubset(xgboost_cols)


def test_tree_path_conditions(tree_test_data):
    """Test that path conditions are properly generated."""
    X, y = tree_test_data

    model = RFGBoost(
        n_estimators=2,
        task="classification",
        cat_features=["cat_a"],
        base_learner="sklearn",
        rf_params={"n_estimators": 2, "max_depth": 3, "random_state": 42},
    )

    model.fit(X, y)

    tree_df = model.trees_to_dataframe()

    # Path conditions should be strings or None
    for condition in tree_df["PathCondition"]:
        assert condition is None or isinstance(condition, str)

    # At least some path conditions should exist
    non_null_conditions = tree_df["PathCondition"].dropna()
    assert len(non_null_conditions) > 0


def test_tree_data_with_event_analysis(tree_test_data):
    """Test tree data includes proper event analysis when data is provided."""
    X, y = tree_test_data

    model = RFGBoost(
        n_estimators=2,
        task="classification",
        cat_features=["cat_a"],
        base_learner="sklearn",
        rf_params={"n_estimators": 2, "max_depth": 3, "random_state": 42},
    )

    model.fit(X, y)

    tree_df = model.trees_to_dataframe(X, y)

    # Check event analysis columns
    assert "Events" in tree_df.columns
    assert "NonEvents" in tree_df.columns
    assert "EventRate" in tree_df.columns

    # Events and NonEvents should be non-negative integers
    assert (tree_df["Events"] >= 0).all()
    assert (tree_df["NonEvents"] >= 0).all()

    # Event rates should be between 0 and 1 (excluding NaN)
    event_rates = tree_df["EventRate"].dropna()
    assert ((event_rates >= 0) & (event_rates <= 1)).all()

    # Total events should make sense
    total_events = tree_df["Events"].sum()
    total_non_events = tree_df["NonEvents"].sum()

    # Should have some events
    assert total_events > 0
    assert total_non_events > 0
