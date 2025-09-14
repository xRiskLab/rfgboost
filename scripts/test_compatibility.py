#!/usr/bin/env python3
"""
Compatibility testing script for RFGBoost across different Python versions.
"""

import sys


def test_imports():  # sourcery skip: extract-duplicate-method, extract-method
    """Test that all main imports work correctly."""
    print("Testing imports...")

    try:
        # Test core package import
        import rfgboost

        print(f"fgboost v{rfgboost.__version__} imported successfully")

        # Test main class import
        from rfgboost import RFGBoost  # noqa: F401

        print("RFGBoost class imported successfully")

        # Test key dependencies
        import pandas as pd

        print(f"pandas v{pd.__version__} available")

        import numpy as np

        print(f"numpy v{np.__version__} available")

        import sklearn

        print(f"scikit-learn v{sklearn.__version__} available")

        import xgboost as xgb

        print(f"xgboost v{xgb.__version__} available")

        import fastwoe

        print(f"fastwoe v{fastwoe.__version__} available")

        return True

    except ImportError as e:
        print(f"Import failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during imports: {e}")
        return False


def test_basic_functionality():  # sourcery skip: extract-method
    """Test basic RFGBoost functionality."""
    print("\nTesting basic functionality...")

    try:
        import numpy as np
        import pandas as pd

        from rfgboost import RFGBoost

        # Create simple test data
        data = pd.DataFrame(
            {
                "cat_feature": ["A", "B", "C"] * 10,
                "num_feature": np.random.normal(0, 1, 30),
                "target": np.random.binomial(1, 0.5, 30),
            }
        )

        X = data.drop("target", axis=1)
        y = data["target"]

        # Test sklearn base learner
        model_sklearn = RFGBoost(
            n_estimators=2,
            task="classification",
            cat_features=["cat_feature"],
            base_learner="sklearn",
        )
        model_sklearn.fit(X, y)
        predictions_sklearn = model_sklearn.predict_proba(X)
        print("sklearn base learner working")

        # Test xgboost base learner
        model_xgb = RFGBoost(
            n_estimators=2,
            task="classification",
            cat_features=["cat_feature"],
            base_learner="xgboost",
        )
        model_xgb.fit(X, y)
        predictions_xgb = model_xgb.predict_proba(X)
        print("xgboost base learner working")

        # Test feature importance
        importance = model_sklearn.get_feature_importance()
        print("feature importance extraction working")

        # Test confidence intervals
        ci = model_sklearn.predict_ci(X)
        print("confidence intervals working")

        return True

    except Exception as e:
        print(f"Basic functionality test failed: {e}")
        return False


def test_python_version_compatibility():
    """Test Python version compatibility."""
    print(f"\nTesting Python {sys.version}")

    major, minor = sys.version_info[:2]

    # sourcery skip: no-conditionals-in-tests
    if major != 3:
        print(f"Python {major}.{minor} not supported (requires Python 3.9+)")
        return False

    if minor < 9:
        print(f"Python 3.{minor} not supported (requires Python 3.9+)")
        return False

    print(f"Python 3.{minor} is supported")
    return True


def main():
    """Main compatibility test runner."""
    print("=" * 60)
    print("RFGBoost Compatibility Test")
    print("=" * 60)

    if len(sys.argv) > 1 and sys.argv[1] == "imports":
        # Just test imports
        success = test_imports()
        sys.exit(0 if success else 1)

    # Run all tests
    tests = [test_python_version_compatibility, test_imports, test_basic_functionality]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("COMPATIBILITY TEST SUMMARY")
    print("=" * 60)

    all_passed = all(results)

    if all_passed:
        print("All compatibility tests PASSED")
        print(f"RFGBoost is compatible with Python {sys.version}")
    else:
        print("Some compatibility tests FAILED")
        print("RFGBoost may not work properly with this configuration")

    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
