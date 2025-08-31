"""Test inference with new/unseen data using WoePreprocessor + FastWoe."""

import numpy as np
import pandas as pd
import pytest
from fastwoe import FastWoe, WoePreprocessor


@pytest.fixture
def inference_test_data():
    """Create training and test data with known and unknown categories."""
    np.random.seed(42)
    n_train = 1000

    # Training categories
    train_categories = [f"cat_{i}" for i in range(20)]
    train_probs = np.exp(-np.arange(20) * 0.3)
    train_probs = train_probs / train_probs.sum()

    train_cats = np.random.choice(train_categories, n_train, p=train_probs)
    train_target = np.random.binomial(1, 0.3 + 0.4 * (train_cats == "cat_0"), n_train)

    train_data = pd.DataFrame({"category": train_cats, "target": train_target})

    # Create test data with NEW categories + some original ones
    np.random.seed(123)  # Different seed for test data
    n_test = 200

    # Mix of seen and NEW categories
    test_categories = [
        # Some original categories
        "cat_0",
        "cat_1",
        "cat_5",
        "cat_10",
        # NEW categories not seen in training
        "cat_NEW_1",
        "cat_NEW_2",
        "cat_UNKNOWN",
        "cat_RARE",
    ]

    test_cats = np.random.choice(test_categories, n_test)
    test_target = np.random.binomial(1, 0.35, n_test)  # Different base rate

    test_data = pd.DataFrame({"category": test_cats, "target": test_target})

    return train_data, test_data


def test_basic_woe_preprocessing_inference(inference_test_data):
    """Test basic WOE preprocessing and inference with unseen categories."""
    train_data, test_data = inference_test_data

    # Fit preprocessor and WOE encoder
    preprocessor = WoePreprocessor(top_p=0.9, min_count=5)
    woe_encoder = FastWoe()

    X_train = train_data[["category"]]
    y_train = train_data["target"]

    # Preprocessing
    X_train_processed = preprocessor.fit_transform(X_train, cat_features=["category"])

    # Should have reduced the number of categories
    original_cats = len(X_train["category"].unique())
    processed_cats = len(X_train_processed["category"].unique())
    assert processed_cats <= original_cats

    # WOE encoding
    woe_encoder.fit(X_train_processed, y_train)

    # Test inference with new data
    X_test = test_data[["category"]]
    X_test_processed = preprocessor.transform(X_test)
    X_test_woe = woe_encoder.transform(X_test_processed)

    # Should successfully process test data
    assert len(X_test_woe) == len(X_test)
    assert isinstance(X_test_woe, pd.DataFrame)

    # WOE values should be finite
    assert np.all(np.isfinite(X_test_woe["category"]))


def test_unseen_category_handling(inference_test_data):
    """Test how unseen categories are handled during inference."""
    train_data, test_data = inference_test_data

    preprocessor = WoePreprocessor(top_p=0.8, min_count=10)

    X_train = train_data[["category"]]
    X_test = test_data[["category"]]

    # Fit on training data
    X_train_processed = preprocessor.fit_transform(X_train, cat_features=["category"])

    # Get the kept categories
    kept_categories = set(X_train_processed["category"].unique())

    # Transform test data
    X_test_processed = preprocessor.transform(X_test)

    # All test categories should be mapped to known categories
    test_categories_after = set(X_test_processed["category"].unique())
    assert test_categories_after.issubset(kept_categories)

    # Should have handled unknown categories gracefully
    assert len(X_test_processed) == len(X_test)


def test_woe_inference_consistency():
    """Test that WOE inference is consistent across multiple calls."""
    np.random.seed(42)

    # Simple training data
    train_data = pd.DataFrame(
        {"category": ["A", "B", "C"] * 100, "target": [0, 1, 0] * 100}
    )

    # Test data with mix of known and unknown
    test_data = pd.DataFrame({"category": ["A", "B", "UNKNOWN", "NEW_CAT"] * 10})

    preprocessor = WoePreprocessor(min_count=10)
    woe_encoder = FastWoe()

    # Fit
    X_train = train_data[["category"]]
    y_train = train_data["target"]

    X_train_processed = preprocessor.fit_transform(X_train, cat_features=["category"])
    woe_encoder.fit(X_train_processed, y_train)

    # Test consistency
    X_test = test_data[["category"]]

    result1 = preprocessor.transform(X_test)
    result2 = preprocessor.transform(X_test)

    # Results should be identical
    pd.testing.assert_frame_equal(result1, result2)

    # WOE transform should also be consistent
    woe1 = woe_encoder.transform(result1)
    woe2 = woe_encoder.transform(result2)

    pd.testing.assert_frame_equal(woe1, woe2)


def test_woe_value_ranges(inference_test_data):
    """Test that WOE values are in reasonable ranges."""
    train_data, test_data = inference_test_data

    preprocessor = WoePreprocessor(top_p=0.9, min_count=5)
    woe_encoder = FastWoe()

    X_train = train_data[["category"]]
    y_train = train_data["target"]

    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train, cat_features=["category"])
    woe_encoder.fit(X_train_processed, y_train)

    X_test = test_data[["category"]]
    X_test_processed = preprocessor.transform(X_test)
    X_test_woe = woe_encoder.transform(X_test_processed)

    # WOE values should be reasonable (typically between -5 and 5)
    woe_values = X_test_woe["category"]
    assert np.all(woe_values >= -10), "WOE values should not be extremely negative"
    assert np.all(woe_values <= 10), "WOE values should not be extremely positive"

    # Should have some variation in WOE values
    assert np.std(woe_values) > 0.01, "WOE values should have some variation"


def test_preprocessing_category_mapping(inference_test_data):
    """Test that category mapping works correctly during preprocessing."""
    train_data, test_data = inference_test_data

    preprocessor = WoePreprocessor(top_p=0.7, min_count=15)

    X_train = train_data[["category"]]
    X_test = test_data[["category"]]

    # Fit and get mapping
    X_train_processed = preprocessor.fit_transform(X_train, cat_features=["category"])
    category_mapping = preprocessor.get_category_mapping()["category"]

    # Mapping should contain only the kept categories
    assert isinstance(category_mapping, set)
    assert len(category_mapping) > 0

    # Check that categories were processed
    train_categories_after = set(X_train_processed["category"].unique())
    assert len(train_categories_after) > 0

    # Check that there's overlap between mapping and processed categories
    # (excluding '__other__' which is added during processing)
    non_other_processed = train_categories_after - {"__other__"}
    assert len(non_other_processed.intersection(category_mapping)) > 0

    # Transform test data and verify
    X_test_processed = preprocessor.transform(X_test)
    test_categories_after = set(X_test_processed["category"].unique())

    # Test data should be processed to known categories
    assert len(test_categories_after) > 0


def test_end_to_end_inference_pipeline(inference_test_data):
    """Test complete end-to-end inference pipeline."""
    train_data, test_data = inference_test_data

    # Complete pipeline
    preprocessor = WoePreprocessor(top_p=0.8, min_count=8)
    woe_encoder = FastWoe()

    # Training phase
    X_train = train_data[["category"]]
    y_train = train_data["target"]

    X_train_processed = preprocessor.fit_transform(X_train, cat_features=["category"])
    woe_encoder.fit(X_train_processed, y_train)
    X_train_woe = woe_encoder.transform(X_train_processed)

    # Inference phase
    X_test = test_data[["category"]]
    X_test_processed = preprocessor.transform(X_test)
    X_test_woe = woe_encoder.transform(X_test_processed)

    # Validate complete pipeline
    assert isinstance(X_train_woe, pd.DataFrame)
    assert isinstance(X_test_woe, pd.DataFrame)

    assert len(X_train_woe) == len(train_data)
    assert len(X_test_woe) == len(test_data)

    # Both should have the same column structure
    assert list(X_train_woe.columns) == list(X_test_woe.columns)

    # All values should be finite
    assert np.all(np.isfinite(X_train_woe["category"]))
    assert np.all(np.isfinite(X_test_woe["category"]))


def test_small_dataset_edge_case():
    """Test preprocessing and WOE with very small datasets."""
    # Very small training set
    train_data = pd.DataFrame(
        {"category": ["A", "B", "A", "B", "C"], "target": [1, 0, 1, 0, 1]}
    )

    # Test data with known and unknown categories
    test_data = pd.DataFrame({"category": ["A", "B", "UNKNOWN"]})

    preprocessor = WoePreprocessor(min_count=1, top_p=1.0)  # Keep all categories
    woe_encoder = FastWoe()

    X_train = train_data[["category"]]
    y_train = train_data["target"]

    # Should handle small dataset gracefully
    X_train_processed = preprocessor.fit_transform(X_train, cat_features=["category"])
    woe_encoder.fit(X_train_processed, y_train)

    X_test = test_data[["category"]]
    X_test_processed = preprocessor.transform(X_test)
    X_test_woe = woe_encoder.transform(X_test_processed)

    assert len(X_test_woe) == len(test_data)
    assert np.all(np.isfinite(X_test_woe["category"]))
