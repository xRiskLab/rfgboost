"""Integration test fixtures for RFGBoost - comprehensive end-to-end data."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


def pytest_configure(config):
    """Configure pytest settings for integration tests."""
    # Register custom pytest markers
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "xgboost: mark test as requiring XGBoost")
    config.addinivalue_line(
        "markers", "plotting: mark test as involving plotting functionality"
    )

    # Filter out expected warnings from dependencies
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="fastwoe")
    warnings.filterwarnings("ignore", message=".*numpy array.*", category=UserWarning)


@pytest.fixture(scope="session")
def sample_data():
    """Generate comprehensive sample data for integration testing."""
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame(
        {
            "cat_feature_a": np.random.choice(["cat1", "cat2", "cat3"], n_samples),
            "cat_feature_b": np.random.choice(["red", "green", "blue"], n_samples),
            "numeric_1": np.random.normal(0, 1, n_samples),
            "numeric_2": np.random.normal(0, 1, n_samples),
            "target": np.random.binomial(1, 0.4, n_samples),
        }
    )

    X = data.drop("target", axis=1)
    y = data["target"]

    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture(scope="session")
def inference_test_data():
    """Generate data for testing inference with unseen categories."""
    np.random.seed(42)

    # Training data with known categories
    categories = [f"cat_{i}" for i in range(10)]
    n_samples = 1000

    train_data = pd.DataFrame(
        {
            "category": np.random.choice(
                categories,
                n_samples,
                p=[0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05],
            ),
            "target": np.random.binomial(1, 0.3, n_samples),
        }
    )

    # Test data with some unseen categories
    test_categories = categories + ["cat_UNKNOWN", "cat_NEW_1", "cat_NEW_2", "cat_RARE"]
    test_data = pd.DataFrame(
        {
            "category": np.random.choice(test_categories, 200),
            "target": np.random.binomial(1, 0.3, 200),
        }
    )

    return train_data, test_data


@pytest.fixture(scope="session")
def benchmark_data():
    """Generate larger dataset for performance benchmarking."""
    np.random.seed(42)
    n_samples = 4000

    data = pd.DataFrame(
        {
            "cat_a": np.random.choice(["A", "B", "C"], n_samples),
            "cat_b": np.random.choice(["X", "Y", "Z"], n_samples),
            "num_1": np.random.normal(0, 1, n_samples),
            "num_2": np.random.normal(0, 1, n_samples),
            "num_3": np.random.normal(0, 1, n_samples),
            "target": np.random.binomial(1, 0.4, n_samples),
        }
    )

    X = data.drop("target", axis=1)
    y = data["target"]

    return train_test_split(X, y, test_size=0.375, random_state=42)


@pytest.fixture(scope="session")
def plot_test_data():
    """Generate data for plotting and visualization tests."""
    np.random.seed(42)
    n_samples = 200

    x = np.linspace(-3, 3, n_samples)
    categories = np.random.choice(["A", "B", "C"], n_samples)
    noise = np.random.normal(0, 0.1, n_samples)

    # Create non-linear relationship
    y_prob = 1 / (1 + np.exp(-(x + noise)))
    y = np.random.binomial(1, y_prob)

    X = pd.DataFrame(
        {
            "x": x,
            "category": categories,
            "noise": noise,
        }
    )

    return X, y


@pytest.fixture
def ci_test_data():
    """Create test data for confidence interval testing."""
    np.random.seed(42)
    n_samples = 500

    data = pd.DataFrame(
        {
            "category_feature": np.random.choice(["A", "B", "C"], n_samples),
            "numeric_feature": np.random.normal(0, 1, n_samples),
            "target": np.random.binomial(1, 0.4, n_samples),
        }
    )

    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y
