"""Unit test fixtures for RFGBoost - lightweight and fast."""

import warnings

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config):
    """Configure pytest settings for unit tests."""
    # Register custom pytest markers
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "xgboost: mark test as requiring XGBoost")

    # Filter out expected warnings from dependencies
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="fastwoe")
    warnings.filterwarnings("ignore", message=".*numpy array.*", category=UserWarning)


@pytest.fixture
def simple_tree_data():
    """Create minimal test data for tree extraction tests."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "cat_a": ["A", "B", "A", "B"] * 25,
            "num_1": np.random.normal(0, 1, 100),
            "target": np.random.binomial(1, 0.4, 100),
        }
    )

    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y


@pytest.fixture
def minimal_woe_data():
    """Create minimal categorical data for WOE unit tests."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "category": ["A", "B", "C"] * 20,
            "target": [0, 1, 0] * 20,
        }
    )
    return data


@pytest.fixture
def mock_model_params():
    """Standard model parameters for unit tests."""
    return {
        "n_estimators": 2,
        "task": "classification",
        "rf_params": {"n_estimators": 3, "max_depth": 2, "random_state": 42},
    }
