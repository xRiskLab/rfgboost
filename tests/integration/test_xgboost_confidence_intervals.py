"""Test improved XGBoost confidence intervals using individual tree predictions."""

import warnings

import numpy as np
import pandas as pd
import pytest

from rfgboost import RFGBoost

warnings.filterwarnings("ignore")

# Check if XGBoost is available
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


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


def test_sklearn_confidence_intervals(ci_test_data):
    """Test sklearn confidence intervals functionality."""
    X, y = ci_test_data

    model = RFGBoost(
        n_estimators=8,
        task="classification",
        cat_features=["category_feature"],
        base_learner="sklearn",
        rf_params={"n_estimators": 10, "max_depth": 5, "random_state": 42},
    )

    model.fit(X, y)

    # Test confidence intervals
    test_sample = X.head(100)
    ci_pred = model.predict_ci(test_sample, alpha=0.05)  # 95% CI
    # For classification, compare CI with probabilities, not discrete predictions
    regular_pred = model.predict_proba(test_sample)[:, 1]  # Get probability of class 1

    # Validate CI properties
    lower_bounds = ci_pred[:, 0]
    upper_bounds = ci_pred[:, 1]

    # Check that CI bounds make sense
    bounds_valid = np.all(lower_bounds <= regular_pred) and np.all(
        regular_pred <= upper_bounds
    )
    width_positive = np.all(ci_pred[:, 1] > ci_pred[:, 0])

    assert bounds_valid, "CI bounds should contain probabilities"
    assert width_positive, "CI widths should be positive"

    # CI should have reasonable width
    ci_width = np.mean(ci_pred[:, 1] - ci_pred[:, 0])
    assert ci_width > 0.01, "CI width should be meaningful"
    assert ci_width < 1.0, "CI width should not span entire probability space"


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
def test_xgboost_confidence_intervals(ci_test_data):
    """Test XGBoost confidence intervals functionality."""
    X, y = ci_test_data

    model = RFGBoost(
        n_estimators=8,
        task="classification",
        cat_features=["category_feature"],
        base_learner="xgboost",
        rf_params={"n_estimators": 10, "max_depth": 5, "random_state": 42},
    )

    model.fit(X, y)

    # Test confidence intervals
    test_sample = X.head(100)
    ci_pred = model.predict_ci(test_sample, alpha=0.05)  # 95% CI
    # For classification, compare CI with probabilities, not discrete predictions
    regular_pred = model.predict_proba(test_sample)[:, 1]  # Get probability of class 1

    # Validate CI properties
    lower_bounds = ci_pred[:, 0]
    upper_bounds = ci_pred[:, 1]

    # Check that CI bounds make sense
    bounds_valid = np.all(lower_bounds <= regular_pred) and np.all(
        regular_pred <= upper_bounds
    )
    width_positive = np.all(ci_pred[:, 1] > ci_pred[:, 0])

    assert bounds_valid, "CI bounds should contain probabilities"
    assert width_positive, "CI widths should be positive"

    # XGBoost CI should be reasonably sized
    ci_width = np.mean(ci_pred[:, 1] - ci_pred[:, 0])
    assert ci_width > 0.001, "XGBoost CI width should be meaningful"


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
def test_sklearn_vs_xgboost_ci_comparison(ci_test_data):
    """Compare confidence intervals between sklearn and XGBoost."""
    X, y = ci_test_data

    models = {}
    ci_results = {}

    for base_learner in ["sklearn", "xgboost"]:
        model = RFGBoost(
            n_estimators=8,
            task="classification",
            cat_features=["category_feature"],
            base_learner=base_learner,
            rf_params={"n_estimators": 10, "max_depth": 5, "random_state": 42},
        )

        model.fit(X, y)
        models[base_learner] = model

        # Get CI for comparison
        test_sample = X.head(100)
        ci_pred = model.predict_ci(test_sample, alpha=0.05)
        # For classification, use probabilities, not discrete predictions
        regular_pred = model.predict_proba(test_sample)[
            :, 1
        ]  # Get probability of class 1

        ci_results[base_learner] = {
            "ci_pred": ci_pred,
            "regular_pred": regular_pred,
            "ci_width": ci_pred[:, 1] - ci_pred[:, 0],
        }

    # Compare results
    sklearn_ci = ci_results["sklearn"]
    xgboost_ci = ci_results["xgboost"]

    # Predictions should be reasonably correlated
    pred_correlation = np.corrcoef(
        sklearn_ci["regular_pred"], xgboost_ci["regular_pred"]
    )[0, 1]
    assert pred_correlation > 0.3, "Predictions should be somewhat correlated"

    # Both should have valid CI widths
    assert np.all(sklearn_ci["ci_width"] > 0)
    assert np.all(xgboost_ci["ci_width"] > 0)

    # CI widths should be finite
    assert np.all(np.isfinite(sklearn_ci["ci_width"]))
    assert np.all(np.isfinite(xgboost_ci["ci_width"]))


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
def test_xgboost_tree_variance_extraction():
    """Test XGBoost individual tree variance extraction."""
    # Create simple test data
    np.random.seed(42)
    X = pd.DataFrame({"category": ["A"] * 50, "numeric": np.random.normal(0, 1, 50)})
    y = np.random.binomial(1, 0.4, 50)

    model = RFGBoost(
        n_estimators=3,
        task="classification",
        cat_features=["category"],
        base_learner="xgboost",
        rf_params={"n_estimators": 5, "random_state": 42},
    )

    model.fit(X, y)

    # Test individual tree prediction extraction
    X_test_small = X.head(10)
    X_encoded = model._woe_encode(X_test_small, fit=False)
    rf = model.models[0]  # First boosting round

    # Use the pred_leaf method to get leaf indices
    booster = rf.get_booster()
    X_dmatrix = xgb.DMatrix(X_encoded)

    leaf_indices = booster.predict(X_dmatrix, pred_leaf=True)
    tree_df = booster.trees_to_dataframe()

    # Validate tree structure extraction
    assert isinstance(leaf_indices, np.ndarray)
    assert isinstance(tree_df, pd.DataFrame)
    assert len(tree_df) > 0

    # Should be able to extract individual tree predictions
    tree_preds = []
    for tree_id in range(min(rf.n_estimators, 3)):  # Test first few trees
        tree_subset = tree_df[tree_df["Tree"] == tree_id]
        if not tree_subset.empty:
            leaf_nodes = tree_subset[tree_subset["Feature"] == "Leaf"]
            if not leaf_nodes.empty:
                leaf_to_value = dict(zip(leaf_nodes["Node"], leaf_nodes["Gain"]))

                # Map leaf indices to values
                sample_leaf_indices = (
                    leaf_indices[:, tree_id] if leaf_indices.ndim > 1 else leaf_indices
                )
                tree_pred = np.array(
                    [leaf_to_value.get(int(idx), 0.0) for idx in sample_leaf_indices]
                )
                tree_preds.append(tree_pred)

    if tree_preds:
        tree_preds = np.array(tree_preds)
        # Should have extracted predictions for multiple trees
        assert tree_preds.shape[0] > 0
        assert tree_preds.shape[1] == len(X_test_small)

        # Tree predictions should be finite
        assert np.all(np.isfinite(tree_preds))


def test_ci_alpha_parameter_effects(ci_test_data):
    """Test that alpha parameter affects confidence interval width."""
    X, y = ci_test_data

    model = RFGBoost(
        n_estimators=5,
        task="classification",
        cat_features=["category_feature"],
        base_learner="sklearn",
        rf_params={"n_estimators": 8, "random_state": 42},
    )

    model.fit(X, y)

    test_sample = X.head(20)

    # Test different alpha values
    alphas = [0.01, 0.05, 0.1, 0.2]
    ci_widths = []

    for alpha in alphas:
        ci_pred = model.predict_ci(test_sample, alpha=alpha)
        ci_width = np.mean(ci_pred[:, 1] - ci_pred[:, 0])
        ci_widths.append(ci_width)

    # Higher confidence (lower alpha) should give wider intervals
    assert ci_widths[0] >= ci_widths[1], "99% CI should be wider than 95% CI"
    assert ci_widths[1] >= ci_widths[2], "95% CI should be wider than 90% CI"
    assert ci_widths[2] >= ci_widths[3], "90% CI should be wider than 80% CI"


def test_ci_consistency_across_calls(ci_test_data):
    """Test that confidence intervals are consistent across multiple calls."""
    X, y = ci_test_data

    for base_learner in ["sklearn"] + (["xgboost"] if XGBOOST_AVAILABLE else []):
        model = RFGBoost(
            n_estimators=3,
            task="classification",
            cat_features=["category_feature"],
            base_learner=base_learner,
            rf_params={"random_state": 42},
        )

        model.fit(X, y)

        test_sample = X.head(10)

        # Multiple CI calls should be identical
        ci1 = model.predict_ci(test_sample, alpha=0.05)
        ci2 = model.predict_ci(test_sample, alpha=0.05)

        np.testing.assert_array_almost_equal(ci1, ci2, decimal=10)
