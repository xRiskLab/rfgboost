"""Test plotting functionality for RFGBoost."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rfgboost import RFGBoost


@pytest.fixture
def plot_test_data():
    """Create test data for plotting scenarios."""
    np.random.seed(42)
    n_samples = 200

    # Create features with clear patterns
    data = pd.DataFrame(
        {
            "x": np.linspace(-3, 3, n_samples),
            "category": np.random.choice(["A", "B"], n_samples),
            "noise": np.random.normal(0, 0.1, n_samples),
        }
    )

    # Create target with clear relationship
    data["target"] = ((data["x"] > 0) & (data["category"] == "A")).astype(int)

    # Add some noise
    noise_mask = np.random.random(n_samples) < 0.1
    data.loc[noise_mask, "target"] = 1 - data.loc[noise_mask, "target"]

    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y


def test_prediction_plotting_data_generation(plot_test_data):
    """Test that models can generate data suitable for plotting."""
    X, y = plot_test_data

    for base_learner in ["sklearn", "xgboost"]:
        model = RFGBoost(
            n_estimators=5,
            task="classification",
            cat_features=["category"],
            base_learner=base_learner,
            rf_params={"n_estimators": 10, "random_state": 42},
        )

        model.fit(X, y)

        # Test prediction generation for plotting
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)

        # Predictions should be probabilities for regression task, binary for classification
        if model.task == "classification":
            # predict() returns probabilities, not binary predictions
            assert np.all((predictions >= 0) & (predictions <= 1))
        else:
            assert set(predictions).issubset({0, 1})

        # Probabilities should be valid
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_confidence_interval_plotting_data(plot_test_data):
    """Test confidence interval data generation for plotting."""
    X, y = plot_test_data

    model = RFGBoost(
        n_estimators=5,
        task="classification",
        cat_features=["category"],
        base_learner="sklearn",
        rf_params={"n_estimators": 15, "random_state": 42},
    )

    model.fit(X, y)

    # Generate CI data for plotting
    ci_bounds = model.predict_ci(X, alpha=0.05)
    probabilities = model.predict_proba(X)[:, 1]  # Get probability of class 1

    assert ci_bounds.shape == (len(X), 2)

    # Lower bounds <= probabilities <= upper bounds
    assert np.all(ci_bounds[:, 0] <= probabilities)
    assert np.all(probabilities <= ci_bounds[:, 1])

    # CI bounds should be in valid probability range
    assert np.all((ci_bounds >= 0) & (ci_bounds <= 1))


def test_feature_effect_simulation(plot_test_data):
    """Test simulation of feature effects for plotting."""
    X, y = plot_test_data

    model = RFGBoost(
        n_estimators=3,
        task="classification",
        cat_features=["category"],
        base_learner="sklearn",
        rf_params={"n_estimators": 8, "random_state": 42},
    )

    model.fit(X, y)

    # Simulate feature effect by varying one feature
    x_range = np.linspace(X["x"].min(), X["x"].max(), 50)

    # Create simulation data
    sim_data = pd.DataFrame(
        {
            "x": x_range,
            "category": ["A"] * len(x_range),  # Fix category
            "noise": [0.0] * len(x_range),  # Fix noise
        }
    )

    # Get predictions
    sim_pred = model.predict_proba(sim_data)[:, 1]
    sim_ci = model.predict_ci(sim_data, alpha=0.1)

    assert len(sim_pred) == len(x_range)
    assert sim_ci.shape == (len(x_range), 2)

    # Predictions should vary smoothly (no dramatic jumps)
    pred_diff = np.abs(np.diff(sim_pred))
    assert np.max(pred_diff) < 0.5, "Predictions should vary smoothly"


def test_categorical_feature_plotting_data(plot_test_data):
    """Test plotting data generation with categorical features."""
    X, y = plot_test_data

    model = RFGBoost(
        n_estimators=3,
        task="classification",
        cat_features=["category"],
        base_learner="sklearn",
        rf_params={"n_estimators": 5, "random_state": 42},
    )

    model.fit(X, y)

    # Test predictions for each category
    categories = ["A", "B"]
    category_results = {}

    for cat in categories:
        cat_data = X.copy()
        cat_data["category"] = cat

        cat_pred = model.predict_proba(cat_data)[:, 1]
        category_results[cat] = cat_pred

    # Results should be different for different categories
    # (since category affects the target)
    diff = np.mean(np.abs(category_results["A"] - category_results["B"]))
    assert diff > 0.01, "Different categories should produce different predictions"


def test_plotting_data_consistency(plot_test_data):
    """Test that plotting data generation is consistent."""
    X, y = plot_test_data

    model = RFGBoost(
        n_estimators=3,
        task="classification",
        cat_features=["category"],
        base_learner="sklearn",
        rf_params={"random_state": 42},
    )

    model.fit(X, y)

    test_sample = X.head(10)

    # Multiple calls should be identical
    pred1 = model.predict_proba(test_sample)
    pred2 = model.predict_proba(test_sample)

    ci1 = model.predict_ci(test_sample, alpha=0.05)
    ci2 = model.predict_ci(test_sample, alpha=0.05)

    np.testing.assert_array_equal(pred1, pred2)
    np.testing.assert_array_equal(ci1, ci2)


def test_edge_case_plotting_data():
    """Test plotting data generation with edge cases."""
    # Create edge case data
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "x": [0.0] * 100,  # All same value
            "category": ["A"] * 100,  # All same category
        }
    )
    y = np.random.binomial(1, 0.5, 100)  # Random target

    model = RFGBoost(
        n_estimators=2,
        task="classification",
        cat_features=["category"],
        base_learner="sklearn",
        rf_params={"n_estimators": 3, "random_state": 42},
    )

    model.fit(X, y)

    # Should still be able to predict
    predictions = model.predict_proba(X)
    ci_bounds = model.predict_ci(X, alpha=0.1)

    assert predictions.shape == (len(X), 2)
    assert ci_bounds.shape == (len(X), 2)

    # All predictions should be the same (since all inputs are the same)
    assert np.allclose(predictions[0], predictions[1:], atol=1e-10)


@pytest.mark.skipif(True, reason="Skip matplotlib tests by default")
def test_actual_plot_generation(plot_test_data):
    """Test actual plot generation (optional, requires matplotlib)."""
    X, y = plot_test_data

    model = RFGBoost(
        n_estimators=3,
        task="classification",
        cat_features=["category"],
        base_learner="sklearn",
        rf_params={"n_estimators": 8, "random_state": 42},
    )

    model.fit(X, y)

    # Create a simple plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot data points
    for cat in ["A", "B"]:
        mask = X["category"] == cat
        ax.scatter(X.loc[mask, "x"], y[mask], alpha=0.6, label=f"Category {cat}")

    # Plot predictions
    x_range = np.linspace(X["x"].min(), X["x"].max(), 100)
    for cat in ["A", "B"]:
        sim_data = pd.DataFrame(
            {
                "x": x_range,
                "category": [cat] * len(x_range),
                "noise": [0.0] * len(x_range),
            }
        )

        pred = model.predict_proba(sim_data)[:, 1]
        ci = model.predict_ci(sim_data, alpha=0.1)

        ax.plot(x_range, pred, label=f"{cat} Prediction")
        ax.fill_between(x_range, ci[:, 0], ci[:, 1], alpha=0.3)

    ax.set_xlabel("X")
    ax.set_ylabel("Probability")
    ax.legend()
    plt.close(fig)  # Clean up
