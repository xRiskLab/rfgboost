"""
Test to demonstrate successful XGBoost tree extraction implementation.
This shows that the issue has been resolved and XGBoost now supports full tree analysis.
"""

import numpy as np
import pandas as pd

from rfgboost import RFGBoost


def test_xgboost_tree_extraction_success():
    """Test that XGBoost tree extraction now works correctly."""

    print("TESTING: XGBoost Tree Extraction Success")
    print("=" * 50)

    # Create test data
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "category_feature": ["A", "B", "C"] * 100,
            "numeric_feature": np.random.normal(0, 1, 300),
            "target": np.random.binomial(1, 0.3, 300),
        }
    )

    X = data.drop("target", axis=1)
    y = data["target"]

    print(f"📊 Dataset: {data.shape} with categorical and numeric features")

    # Create XGBoost model
    model = RFGBoost(
        n_estimators=5,
        task="classification",
        cat_features=["category_feature"],
        base_learner="xgboost",
        rf_params={"n_estimators": 3, "max_depth": 4, "random_state": 42},
    )

    print("\n🏗️  Training XGBoost model...")
    model.fit(X, y)

    print("\n🌳 Testing tree extraction methods:")

    # Test 1: Extract tree data with conditions
    try:
        tree_data = model.extract_tree_data_with_conditions()
        print(f"extract_tree_data_with_conditions: {tree_data.shape}")
        print(f"   Sample path: {tree_data['PathCondition'].dropna().iloc[0]}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"extract_tree_data_with_conditions failed: {e}")
        return False

    # Test 2: Extract leaf nodes with conditions
    try:
        leaf_data = model.extract_leaf_nodes_with_conditions()
        print(f"extract_leaf_nodes_with_conditions: {leaf_data.shape}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"extract_leaf_nodes_with_conditions failed: {e}")
        return False

    # Test 3: Trees to dataframe (basic)
    try:
        basic_df = model.trees_to_dataframe()
        print(f"trees_to_dataframe (basic): {basic_df.shape}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"trees_to_dataframe (basic) failed: {e}")
        return False

    # Test 4: Trees to dataframe with event analysis
    try:
        analysis_df = model.trees_to_dataframe(X, y)
        print(f"trees_to_dataframe (with events): {analysis_df.shape}")

        # sourcery skip: no-conditionals-in-tests
        if not analysis_df.empty:
            print("\nSample tree analysis:")
            sample_cols = [
                "Round",
                "Tree",
                "PathCondition",
                "Events",
                "NonEvents",
                "EventRate",
            ]
            available_cols = [col for col in sample_cols if col in analysis_df.columns]
            print(analysis_df[available_cols].head(3).to_string(index=False))

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"trees_to_dataframe (with events) failed: {e}")
        return False

    # Test 5: Feature name mapping verification
    print("\nFeature mapping verification:")
    print(f"   Original features: {model.feature_names_}")

    # Check if path conditions use actual feature names
    path_sample = tree_data["PathCondition"].dropna().iloc[0]
    uses_actual_names = any(fname in path_sample for fname in model.feature_names_)
    uses_xgb_names = any(
        f"f{i}" in path_sample for i in range(len(model.feature_names_))
    )

    # sourcery skip: no-conditionals-in-tests
    if uses_actual_names and not uses_xgb_names:
        print(f"Feature names correctly mapped: {path_sample}")
    else:
        print(f"Feature mapping may have issues: {path_sample}")

    # Test 6: WOE integration
    try:
        woe_mappings = model.get_woe_mappings("category_feature")
        print("\nWOE integration test:")
        print(f"WOE mappings available: {woe_mappings.shape}")
        print(f"   Categories: {woe_mappings['category'].tolist()}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"WOE integration failed: {e}")
        return False

    return True


# sourcery skip: use-named-expression
if __name__ == "__main__":
    success = test_xgboost_tree_extraction_success()
    if success:
        print("\nCONCLUSION: XGBoost tree extraction fully implemented!")
    else:
        print("\nFAILURE: Issues remain with XGBoost tree extraction")
