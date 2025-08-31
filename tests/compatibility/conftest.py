"""
Compatibility test configuration and fixtures.

This module provides test fixtures and configuration specifically for
testing Python version compatibility and dependency combinations.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture(scope="session")
def compatibility_test_data():
    """Create test data for compatibility testing."""
    np.random.seed(42)

    # Classification data
    X_clf, y_clf = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Regression data
    X_reg, y_reg = make_regression(
        n_samples=200, n_features=5, n_informative=3, noise=0.1, random_state=42
    )

    # Categorical data
    categories = ["A", "B", "C", "D", "E"]
    X_cat = np.random.choice(categories, size=200)
    X_num = np.random.randn(200, 3)

    # Create target with some relationship to categories
    y_cat = (X_cat == "A").astype(int) + np.random.normal(0, 0.1, 200)
    y_cat = (y_cat > 0.5).astype(int)

    return {
        "classification": {
            "X": pd.DataFrame(X_clf, columns=[f"feature_{i}" for i in range(5)]),
            "y": y_clf,
        },
        "regression": {
            "X": pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(5)]),
            "y": y_reg,
        },
        "categorical": {
            "X": pd.DataFrame(
                {
                    "category": X_cat,
                    "num1": X_num[:, 0],
                    "num2": X_num[:, 1],
                    "num3": X_num[:, 2],
                }
            ),
            "y": y_cat,
        },
    }


@pytest.fixture(scope="session")
def small_test_data():
    """Create small test data for edge case testing."""
    np.random.seed(42)

    # Very small dataset
    X_small, y_small = make_classification(n_samples=10, n_features=3, random_state=42)

    return {
        "X": pd.DataFrame(X_small, columns=[f"feature_{i}" for i in range(3)]),
        "y": y_small,
    }


@pytest.fixture(scope="session")
def model_configurations():
    """Provide different model configurations for testing."""
    return {
        "sklearn_classification": {
            "n_estimators": 3,
            "learning_rate": 0.1,
            "task": "classification",
            "base_learner": "sklearn",
            "rf_params": {"n_estimators": 2, "max_depth": 3, "random_state": 42},
        },
        "sklearn_regression": {
            "n_estimators": 3,
            "learning_rate": 0.1,
            "task": "regression",
            "base_learner": "sklearn",
            "rf_params": {"n_estimators": 2, "max_depth": 3, "random_state": 42},
        },
        "xgboost_classification": {
            "n_estimators": 3,
            "learning_rate": 0.1,
            "task": "classification",
            "base_learner": "xgboost",
            "rf_params": {"n_estimators": 2, "max_depth": 3, "random_state": 42},
        },
        "xgboost_regression": {
            "n_estimators": 3,
            "learning_rate": 0.1,
            "task": "regression",
            "base_learner": "xgboost",
            "rf_params": {"n_estimators": 2, "max_depth": 3, "random_state": 42},
        },
        "woe_classification": {
            "n_estimators": 3,
            "learning_rate": 0.1,
            "task": "classification",
            "cat_features": ["category"],
            "woe_kwargs": {"min_count": 10, "random_state": 42},
            "base_learner": "sklearn",
            "rf_params": {"n_estimators": 2, "max_depth": 3, "random_state": 42},
        },
    }


def pytest_configure(config):
    """Configure pytest for compatibility testing."""
    config.addinivalue_line("markers", "compatibility: mark test as compatibility test")
    config.addinivalue_line("markers", "python38: mark test as Python 3.8+ compatible")
    config.addinivalue_line("markers", "python39: mark test as Python 3.9+ compatible")
    config.addinivalue_line(
        "markers", "python310: mark test as Python 3.10+ compatible"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for compatibility testing."""
    for item in items:
        # Mark all tests in compatibility module as compatibility tests
        if "compatibility" in item.nodeid:
            item.add_marker(pytest.mark.compatibility)

        # Mark tests based on Python version requirements
        if "python38" in item.nodeid:
            item.add_marker(pytest.mark.python38)
        elif "python39" in item.nodeid:
            item.add_marker(pytest.mark.python39)
        elif "python310" in item.nodeid:
            item.add_marker(pytest.mark.python310)
