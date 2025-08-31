"""Test XGBoost integration with RFGBoost."""

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Check if XGBoost is available
try:
    import xgboost

    XGBOOST_AVAILABLE = bool(xgboost)
except ImportError:
    XGBOOST_AVAILABLE = False

from rfgboost import RFGBoost

pytestmark = pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")


@pytest.fixture
def benchmark_data():
    """Create larger dataset for performance comparison."""
    np.random.seed(42)
    n_samples = 5000

    data = pd.DataFrame(
        {
            "cat_a": np.random.choice(
                ["A", "B", "C", "D"], n_samples, p=[0.4, 0.3, 0.2, 0.1]
            ),
            "cat_b": np.random.choice(["X", "Y", "Z"], n_samples, p=[0.5, 0.3, 0.2]),
            "num_1": np.random.normal(0, 1, n_samples),
            "num_2": np.random.uniform(-2, 2, n_samples),
            "num_3": np.random.exponential(1, n_samples),
        }
    )

    # Create target with some signal
    target_prob = (
        0.3
        + 0.2 * (data["cat_a"] == "A")
        + 0.15 * (data["cat_b"] == "X")
        + 0.1 * (data["num_1"] > 0)
        + 0.05 * (data["num_2"] > 0)
    )
    data["target"] = np.random.binomial(1, target_prob)

    X = data.drop("target", axis=1)
    y = data["target"]
    return train_test_split(X, y, test_size=0.3, random_state=42)


def test_xgboost_vs_sklearn_performance(benchmark_data):
    """Test XGBoost vs sklearn performance comparison."""
    X_train, X_test, y_train, y_test = benchmark_data
    cat_features = ["cat_a", "cat_b"]

    models = {}
    results = {}

    for base_learner in ["sklearn", "xgboost"]:
        # Train model and measure time
        start_time = time.time()

        model = RFGBoost(
            n_estimators=8,
            task="classification",
            cat_features=cat_features,
            base_learner=base_learner,
            rf_params={"n_estimators": 10, "max_depth": 4, "random_state": 42},
        )

        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Get predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > 0.5).astype(
            int
        )  # Convert probabilities to binary predictions

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        models[base_learner] = model
        results[base_learner] = {
            "training_time": training_time,
            "accuracy": accuracy,
            "auc": auc,
        }

    # Test that both models work
    assert results["sklearn"]["accuracy"] > 0.5
    assert results["xgboost"]["accuracy"] > 0.5
    assert results["sklearn"]["auc"] > 0.5
    assert results["xgboost"]["auc"] > 0.5

    # XGBoost should typically be faster
    speed_ratio = (
        results["sklearn"]["training_time"] / results["xgboost"]["training_time"]
    )
    assert speed_ratio > 0.5  # XGBoost should be at least somewhat competitive


def test_feature_importance_comparison(benchmark_data):
    """Test feature importance between sklearn and XGBoost."""
    X_train, _, y_train, _ = benchmark_data
    cat_features = ["cat_a", "cat_b"]

    models = {}

    for base_learner in {"sklearn", "xgboost"}:
        model = RFGBoost(
            n_estimators=5,
            task="classification",
            cat_features=cat_features,
            base_learner=base_learner,
            rf_params={"n_estimators": 5, "max_depth": 3, "random_state": 42},
        )

        model.fit(X_train, y_train)
        models[base_learner] = model

    # Both models should have feature importance
    # sourcery skip: no-loop-in-tests
    for _base_learner, model in models.items():
        # Should have as many features as input
        assert len(model.feature_names_) == X_train.shape[1]

        # Should have trained some models
        assert len(model.models) > 0


def test_woe_consistency(benchmark_data):
    """Test that WOE encoding is consistent between backends."""
    X_train, _, y_train, _ = benchmark_data
    cat_features = ["cat_a", "cat_b"]

    models = {}
    woe_stats = {}

    # sourcery skip: no-loop-in-tests
    for base_learner in {"sklearn", "xgboost"}:
        model = RFGBoost(
            n_estimators=3,
            task="classification",
            cat_features=cat_features,
            base_learner=base_learner,
            rf_params={"random_state": 42},
        )

        model.fit(X_train, y_train)
        models[base_learner] = model
        woe_stats[base_learner] = model.woe_encoder.get_feature_stats()

    # WOE statistics should be identical (same encoding process)
    sklearn_stats = woe_stats["sklearn"]
    xgboost_stats = woe_stats["xgboost"]

    assert len(sklearn_stats) == len(xgboost_stats)

    # Gini and IV should be identical
    # sourcery skip: no-loop-in-tests
    for feature in ["cat_a", "cat_b"]:
        sklearn_row = sklearn_stats[sklearn_stats["feature"] == feature].iloc[0]
        xgboost_row = xgboost_stats[xgboost_stats["feature"] == feature].iloc[0]

        np.testing.assert_almost_equal(
            sklearn_row["gini"], xgboost_row["gini"], decimal=6
        )
        np.testing.assert_almost_equal(sklearn_row["iv"], xgboost_row["iv"], decimal=6)


def test_confidence_intervals_both_backends(benchmark_data):
    """Test confidence intervals work with both backends."""
    X_train, X_test, y_train, _ = benchmark_data
    cat_features = ["cat_a", "cat_b"]

    test_sample = X_test.head(50)

    # sourcery skip: no-loop-in-tests
    for base_learner in {"sklearn", "xgboost"}:
        model = RFGBoost(
            n_estimators=5,
            task="classification",
            cat_features=cat_features,
            base_learner=base_learner,
            rf_params={"n_estimators": 8, "random_state": 42},
        )

        model.fit(X_train, y_train)

        # Test confidence intervals
        ci_bounds = model.predict_ci(test_sample, alpha=0.05)
        # For classification, compare CI with probabilities, not discrete predictions
        probabilities = model.predict_proba(test_sample)[:, 1]

        assert ci_bounds.shape == (len(test_sample), 2)

        # Lower bounds should be <= probabilities <= upper bounds
        assert np.all(ci_bounds[:, 0] <= probabilities)
        assert np.all(probabilities <= ci_bounds[:, 1])

        # CI widths should be positive
        ci_widths = ci_bounds[:, 1] - ci_bounds[:, 0]
        assert np.all(ci_widths >= 0)


def test_prediction_consistency(benchmark_data):
    """Test that predictions are consistent across multiple calls."""
    X_train, X_test, y_train, _ = benchmark_data

    # sourcery skip: no-loop-in-tests
    for base_learner in {"sklearn", "xgboost"}:
        model = RFGBoost(
            n_estimators=3,
            task="classification",
            cat_features=["cat_a", "cat_b"],
            base_learner=base_learner,
            rf_params={"random_state": 42},
        )

        model.fit(X_train, y_train)

        test_sample = X_test.head(10)

        # Multiple predictions should be identical
        pred1 = model.predict(test_sample)
        pred2 = model.predict(test_sample)
        proba1 = model.predict_proba(test_sample)
        proba2 = model.predict_proba(test_sample)

        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_almost_equal(proba1, proba2)
