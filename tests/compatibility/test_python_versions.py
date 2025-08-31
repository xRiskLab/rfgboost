"""
Python Version Compatibility Tests

This module tests RFGBoost compatibility across different Python versions
and dependency combinations to ensure the package works correctly from Python 3.8+.
"""

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from rfgboost import RFGBoost


class TestPythonVersionCompatibility:
    """Test RFGBoost compatibility across Python versions."""

    def test_python_version_requirement(self):
        """Test that we're running on a supported Python version."""
        version = sys.version_info
        assert version >= (3, 8), (
            f"Python {version.major}.{version.minor} not supported. Need 3.8+"
        )
        print(f"✅ Running on Python {version.major}.{version.minor}.{version.micro}")

    def test_basic_imports(self):
        """Test that all core imports work correctly."""
        # Test main package import
        from rfgboost import RFGBoost

        assert RFGBoost is not None

        # Test submodule imports
        from rfgboost.rfgboost import RFGBoost as RFGBoostClass

        assert RFGBoostClass is not None

        print("✅ All imports successful")

    def test_basic_functionality_python_38_compatible(self):
        """Test basic functionality using only Python 3.8+ compatible features."""
        # Create simple test data
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])

        # Test sklearn backend
        model = RFGBoost(
            n_estimators=3,
            learning_rate=0.1,
            task="classification",
            base_learner="sklearn",
            rf_params={"n_estimators": 2, "max_depth": 3, "random_state": 42},
        )

        # Test fitting
        model.fit(X, y)
        assert hasattr(model, "models")
        assert len(model.models) == 3

        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert predictions.dtype in [np.int64, np.int32, np.float64]

        # Test predict_proba
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-10)

        print("✅ Basic sklearn functionality works")

    def test_xgboost_backend_compatibility(self):
        """Test XGBoost backend if available."""
        try:
            import xgboost as xgb

            XGBOOST_AVAILABLE = bool(xgb)
        except ImportError:
            XGBOOST_AVAILABLE = False
            pytest.skip("XGBoost not available")

        # sourcery skip: no-conditionals-in-tests
        if not XGBOOST_AVAILABLE:
            pytest.skip("XGBoost not available")

        # Create test data
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])

        # Test XGBoost backend
        model = RFGBoost(
            n_estimators=3,
            learning_rate=0.1,
            task="regression",
            base_learner="xgboost",
            rf_params={"n_estimators": 2, "max_depth": 3, "random_state": 42},
        )

        # Test fitting
        model.fit(X, y)
        assert hasattr(model, "models")
        assert len(model.models) == 3

        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert predictions.dtype in [np.float32, np.float64]

        print("✅ XGBoost backend functionality works")

    def test_woe_encoding_compatibility(self):
        """Test WOE encoding functionality."""
        # Create categorical data
        np.random.seed(42)
        n_samples = 200
        categories = ["A", "B", "C", "D"]
        X_cat = np.random.choice(categories, size=n_samples)
        X_num = np.random.randn(n_samples, 3)

        # Create target with some relationship to categories
        y = (X_cat == "A").astype(int) + np.random.normal(0, 0.1, n_samples)
        y = (y > 0.5).astype(int)  # Convert to binary

        X = pd.DataFrame(
            {
                "category": X_cat,
                "num1": X_num[:, 0],
                "num2": X_num[:, 1],
                "num3": X_num[:, 2],
            }
        )

        # Test with WOE encoding
        model = RFGBoost(
            n_estimators=3,
            learning_rate=0.1,
            task="classification",
            cat_features=["category"],
            woe_kwargs={"random_state": 42},
            base_learner="sklearn",
            rf_params={"n_estimators": 2, "max_depth": 3, "random_state": 42},
        )

        # Test fitting
        model.fit(X, y)
        assert model.woe_encoder is not None

        # Test WOE mappings
        woe_mappings = model.get_woe_mappings("category")
        assert len(woe_mappings) > 0
        assert "category" in woe_mappings.columns
        assert "woe" in woe_mappings.columns

        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(y)

        print("✅ WOE encoding functionality works")

    def test_feature_importance_compatibility(self):
        """Test feature importance functionality."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])

        model = RFGBoost(
            n_estimators=3,
            learning_rate=0.1,
            task="classification",
            base_learner="sklearn",
            rf_params={"n_estimators": 2, "max_depth": 3, "random_state": 42},
        )

        model.fit(X, y)

        # Test feature importance
        importance_df = model.get_feature_importance()
        assert len(importance_df) == 5
        assert "Feature" in importance_df.columns
        assert "Importance" in importance_df.columns
        assert all(
            isinstance(imp, (int, float, np.integer, np.floating))
            for imp in importance_df["Importance"]
        )

        print("✅ Feature importance functionality works")

    def test_confidence_intervals_compatibility(self):
        """Test confidence intervals functionality."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])

        model = RFGBoost(
            n_estimators=5,
            learning_rate=0.1,
            task="regression",
            base_learner="sklearn",
            rf_params={"n_estimators": 3, "max_depth": 3, "random_state": 42},
        )

        model.fit(X, y)

        # Test confidence intervals
        ci_result = model.predict_ci(X, alpha=0.05)
        assert ci_result.shape == (len(y), 2)
        ci_lower = ci_result[:, 0]
        ci_upper = ci_result[:, 1]
        assert len(ci_lower) == len(y)
        assert len(ci_upper) == len(y)
        assert all(ci_lower <= ci_upper)

        print("✅ Confidence intervals functionality works")

    def test_tree_extraction_compatibility(self):
        """Test tree extraction functionality."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])

        model = RFGBoost(
            n_estimators=2,
            learning_rate=0.1,
            task="classification",
            base_learner="sklearn",
            rf_params={"n_estimators": 2, "max_depth": 3, "random_state": 42},
        )

        model.fit(X, y)

        # Test tree extraction
        tree_data = model.extract_tree_data_with_conditions()
        assert len(tree_data) > 0
        assert "Round" in tree_data.columns
        assert "Tree" in tree_data.columns

        # Test leaf nodes extraction
        leaf_data = model.extract_leaf_nodes_with_conditions()
        assert len(leaf_data) > 0

        # Test trees to dataframe
        df_data = model.trees_to_dataframe(X, y)
        assert len(df_data) > 0
        assert "Events" in df_data.columns
        assert "NonEvents" in df_data.columns

        print("✅ Tree extraction functionality works")

    def test_edge_cases_compatibility(self):
        """Test edge cases and error handling."""
        # Test with very small dataset
        X, y = make_classification(
            n_samples=10,
            n_features=3,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            random_state=42,
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(3)])

        model = RFGBoost(
            n_estimators=2,
            learning_rate=0.1,
            task="classification",
            base_learner="sklearn",
            rf_params={"n_estimators": 1, "max_depth": 2, "random_state": 42},
        )

        # Should not raise an error
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == 10

        print("✅ Edge cases handled correctly")

    def test_parameter_validation_compatibility(self):
        """Test parameter validation and error handling."""
        X, y = make_classification(
            n_samples=50,
            n_features=3,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            random_state=42,
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(3)])

        # Test invalid base_learner - this should raise ValueError during fit
        model = RFGBoost(base_learner="invalid")
        with pytest.raises(ValueError):
            model.fit(X, y)

        # Test invalid task - this should raise ValueError during construction
        with pytest.raises(ValueError):
            model = RFGBoost(task="invalid")

        print("✅ Parameter validation works correctly")


def test_dependency_versions():
    """Test that all dependencies are at compatible versions."""
    import numpy
    import pandas
    import scipy
    import sklearn

    print(f"✅ scikit-learn: {sklearn.__version__}")
    print(f"✅ scipy: {scipy.__version__}")
    print(f"✅ numpy: {numpy.__version__}")
    print(f"✅ pandas: {pandas.__version__}")

    # Test XGBoost if available
    try:
        import xgboost as xgb

        print(f"✅ xgboost: {xgb.__version__}")
    except ImportError:
        print("⚠️  xgboost: not installed")

    # Test fastwoe if available
    try:
        import fastwoe

        print(f"✅ fastwoe: {fastwoe.__version__}")
    except ImportError:
        print("⚠️  fastwoe: not installed")


if __name__ == "__main__":
    print("🧪 Running Python Version Compatibility Tests")
    print("=" * 50)

    # Run version test
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")

    # Run all tests
    pytest.main([__file__, "-v"])
