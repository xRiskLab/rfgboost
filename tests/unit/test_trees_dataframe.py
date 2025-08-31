"""Test for trees_to_dataframe method."""

import numpy as np
import pandas as pd
import pytest

from rfgboost import RFGBoost


# pylint: disable=redefined-outer-name
@pytest.fixture
def sample_data():
    """Create sample test data."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "cat_a": np.random.choice(["A", "B"], 100),
            "num_1": np.random.normal(0, 1, 100),
        }
    )
    y = np.random.binomial(1, 0.4, 100)
    return X, y


@pytest.fixture
def fitted_model(sample_data):
    """Create a fitted RFGBoost model."""
    X, y = sample_data
    model = RFGBoost(
        n_estimators=2, task="classification", cat_features=["cat_a"], learning_rate=0.1
    )
    model.fit(X, y)
    return model


def test_trees_to_dataframe_without_data(fitted_model):
    """Test trees_to_dataframe method without providing X and y."""
    leaf_stats = fitted_model.trees_to_dataframe()

    assert isinstance(leaf_stats, pd.DataFrame), "Should return a DataFrame"
    assert len(leaf_stats) > 0, "Should have some leaf statistics"

    # Check for expected columns
    expected_cols = ["Round", "Tree", "NodeID", "PathCondition", "Samples", "Value"]
    for col in expected_cols:
        assert col in leaf_stats.columns, f"Missing column: {col}"


def test_trees_to_dataframe_with_data(fitted_model, sample_data):
    """Test trees_to_dataframe method with X and y provided."""
    X, y = sample_data
    leaf_stats_with_data = fitted_model.trees_to_dataframe(X, y)

    assert isinstance(leaf_stats_with_data, pd.DataFrame), "Should return a DataFrame"
    assert len(leaf_stats_with_data) > 0, "Should have some leaf statistics"

    # When data is provided, should have additional columns like Events, EventRate
    additional_cols = ["Events", "NonEvents", "EventRate"]
    # sourcery skip: no-loop-in-tests
    for col in additional_cols:
        assert col in leaf_stats_with_data.columns, (
            f"Missing column when data provided: {col}"
        )


def test_trees_to_dataframe_data_integrity(fitted_model, sample_data):
    """Test that the data in trees_to_dataframe makes sense."""
    X, y = sample_data
    leaf_stats = fitted_model.trees_to_dataframe(X, y)

    # Event counts should be non-negative integers
    assert (leaf_stats["Events"] >= 0).all(), "Events should be non-negative"
    assert (leaf_stats["NonEvents"] >= 0).all(), "NonEvents should be non-negative"

    # Event rate should be between 0 and 1 (excluding NaN)
    event_rates = leaf_stats["EventRate"].dropna()
    assert (event_rates >= 0).all() and (event_rates <= 1).all(), (
        "Event rates should be between 0 and 1"
    )
