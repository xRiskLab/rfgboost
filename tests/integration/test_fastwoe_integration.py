"""Test FastWoe integration with RFGBoost."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from rfgboost import RFGBoost


@pytest.fixture
def sample_data():
    """Create sample data with categorical and numeric features."""
    np.random.seed(42)
    n_samples = 1000

    # Create categorical features
    categories_a = np.random.choice(
        ["cat1", "cat2", "cat3", "cat4"], n_samples, p=[0.4, 0.3, 0.2, 0.1]
    )
    categories_b = np.random.choice(
        ["red", "blue", "green"], n_samples, p=[0.5, 0.3, 0.2]
    )

    # Create numeric features
    numeric_1 = np.random.normal(0, 1, n_samples)
    numeric_2 = np.random.uniform(-1, 1, n_samples)

    # Create target with some relationship to features
    target_prob = (
        0.1
        + 0.2 * (categories_a == "cat1")
        + 0.3 * (categories_b == "red")
        + 0.1 * (numeric_1 > 0)
        + 0.1 * (numeric_2 > 0)
    )
    target = np.random.binomial(1, target_prob)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "cat_feature_a": categories_a,
            "cat_feature_b": categories_b,
            "numeric_1": numeric_1,
            "numeric_2": numeric_2,
            "target": target,
        }
    )

    X = data.drop("target", axis=1)
    y = data["target"]
    return train_test_split(X, y, test_size=0.3, random_state=42)


def test_fastwoe_integration_basic(sample_data):
    """Test basic FastWoe integration functionality."""
    X_train, X_test, y_train, y_test = sample_data

    cat_features = ["cat_feature_a", "cat_feature_b"]
    woe_kwargs = {"random_state": 42}

    model = RFGBoost(
        n_estimators=5,
        task="classification",
        cat_features=cat_features,
        woe_kwargs=woe_kwargs,
        rf_params={"n_estimators": 5, "max_depth": 3, "random_state": 42},
    )

    # Test model fitting
    model.fit(X_train, y_train)
    assert model.woe_encoder is not None
    assert model.prior is not None

    # Test predictions
    probabilities = model.predict_proba(X_test)
    predictions = (probabilities[:, 1] > 0.5).astype(
        int
    )  # Convert probabilities to binary

    assert len(predictions) == len(X_test)
    assert probabilities.shape == (len(X_test), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))

    # Test performance is reasonable
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities[:, 1])

    assert accuracy > 0.5  # Better than random
    assert auc > 0.5  # Better than random


def test_woe_mappings_quality(sample_data):
    """Test that WOE mappings are generated correctly."""
    X_train, _, y_train, _ = sample_data

    model = RFGBoost(
        n_estimators=3,
        task="classification",
        cat_features=["cat_feature_a", "cat_feature_b"],
        rf_params={"n_estimators": 3, "random_state": 42},
    )

    model.fit(X_train, y_train)

    # Test WOE mappings exist and have expected structure
    woe_mappings = model.woe_encoder.get_all_mappings()

    assert "cat_feature_a" in woe_mappings
    assert "cat_feature_b" in woe_mappings

    # Check WOE mapping structure
    for _feature, mapping in woe_mappings.items():
        assert isinstance(mapping, pd.DataFrame)
        required_cols = ["category", "count", "good_count", "bad_count", "woe"]
        for col in required_cols:
            assert col in mapping.columns

        # WOE values should be finite
        assert np.all(np.isfinite(mapping["woe"]))


def test_feature_statistics(sample_data):
    """Test that feature statistics are computed correctly."""
    X_train, _, y_train, _ = sample_data

    model = RFGBoost(
        n_estimators=3,
        task="classification",
        cat_features=["cat_feature_a", "cat_feature_b"],
        rf_params={"n_estimators": 3, "random_state": 42},
    )

    model.fit(X_train, y_train)

    # Test feature statistics
    feature_stats = model.woe_encoder.get_feature_stats()

    assert isinstance(feature_stats, pd.DataFrame)
    assert len(feature_stats) == 2  # Two categorical features

    required_cols = ["feature", "n_categories", "total_observations", "gini", "iv"]
    for col in required_cols:
        assert col in feature_stats.columns

    # All values should be positive and finite
    assert np.all(feature_stats["n_categories"] > 0)
    assert np.all(feature_stats["total_observations"] > 0)
    assert np.all(np.isfinite(feature_stats["gini"]))
    assert np.all(np.isfinite(feature_stats["iv"]))


def test_inverse_transformation(sample_data):
    """Test WOE inverse transformation functionality."""
    X_train, X_test, y_train, _ = sample_data

    model = RFGBoost(
        n_estimators=3,
        task="classification",
        cat_features=["cat_feature_a", "cat_feature_b"],
        rf_params={"n_estimators": 3, "random_state": 42},
    )

    model.fit(X_train, y_train)

    # Get WOE encoded data
    X_woe = model._woe_encode(X_test.head(10), fit=False)

    # Test inverse transformation
    X_proba = model.inverse_woe_transform(X_woe)

    assert X_proba.shape == (10, X_test.shape[1])
    # Probabilities should be between 0 and 1
    cat_cols = ["cat_feature_a", "cat_feature_b"]
    for col in cat_cols:
        if col in X_proba.columns:
            col_values = X_proba[col].values
            assert np.all((col_values >= 0) & (col_values <= 1))


def test_model_persistence(sample_data):
    """Test that trained model maintains state correctly."""
    X_train, X_test, y_train, _ = sample_data

    model = RFGBoost(
        n_estimators=3,
        task="classification",
        cat_features=["cat_feature_a", "cat_feature_b"],
        rf_params={"n_estimators": 3, "random_state": 42},
    )

    model.fit(X_train, y_train)

    # Make predictions
    pred1 = model.predict(X_test.head(5))
    pred2 = model.predict(X_test.head(5))

    # Predictions should be consistent
    np.testing.assert_array_equal(pred1, pred2)

    # Feature names should be preserved
    assert model.feature_names_ is not None
    assert len(model.feature_names_) == X_train.shape[1]
