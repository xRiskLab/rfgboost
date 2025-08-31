"""
Test WoePreprocessor logic for binary categories and min_count/top_p interaction.
"""

import pandas as pd
from fastwoe import WoePreprocessor


def test_binary_category_preservation():
    """Test that binary features (≤2 categories) are always preserved regardless of min_count."""

    # Create test data with binary feature
    data = pd.DataFrame(
        {
            "binary_feature": ["A"] * 50
            + ["B"] * 5,  # A=50, B=5 (B below min_count=10)
            "multi_feature": ["X"] * 30 + ["Y"] * 20 + ["Z"] * 5,  # X=30, Y=20, Z=5
        }
    )

    # Test with min_count=10 (should filter out B from binary if logic was wrong)
    preprocessor = WoePreprocessor(min_count=10, top_p=None, max_categories=10)
    preprocessor.fit(data)

    # Binary feature should keep BOTH categories despite B having only 5 counts < min_count=10
    binary_mapping = preprocessor.get_category_mapping()["binary_feature"]
    assert binary_mapping == {"A", "B"}, (
        f"Binary feature should keep both categories, got: {binary_mapping}"
    )

    # Multi feature should filter out Z (5 < 10)
    multi_mapping = preprocessor.get_category_mapping()["multi_feature"]
    assert multi_mapping == {"X", "Y"}, (
        f"Multi feature should filter out Z, got: {multi_mapping}"
    )

    print("✅ Binary category preservation test passed")


def test_min_count_and_top_p_interaction():
    """Test how min_count and top_p interact - top_p should be based on original total."""

    # Create test data: A=1000, B=500, C=8, D=5, E=2 (total=1515)
    data = pd.DataFrame(
        {"feature": ["A"] * 1000 + ["B"] * 500 + ["C"] * 8 + ["D"] * 5 + ["E"] * 2}
    )

    # Test 1: min_count=10, top_p=0.8 (want 80% of original = 1212)
    preprocessor1 = WoePreprocessor(min_count=10, top_p=0.8)
    preprocessor1.fit(data)

    mapping1 = preprocessor1.get_category_mapping()["feature"]

    # After min_count filter: A=1000, B=500 (C,D,E filtered out)
    # Cumulative based on original total (1515):
    # A: 1000/1515 = 0.66 ≤ 0.8 ✓
    # A+B: 1500/1515 = 0.99 > 0.8 ✗
    # So should keep only A
    expected1 = {"A"}
    assert mapping1 == expected1, f"Expected {expected1}, got {mapping1}"

    # Test 2: min_count=10, top_p=0.95 (want 95% of original = 1439)
    preprocessor2 = WoePreprocessor(min_count=10, top_p=0.95)
    preprocessor2.fit(data)

    mapping2 = preprocessor2.get_category_mapping()["feature"]

    # After min_count filter: A=1000, B=500
    # Cumulative based on original total (1515):
    # A: 1000/1515 = 0.66 ≤ 0.95 ✓
    # A+B: 1500/1515 = 0.99 > 0.95 ✗
    # So should keep only A (same as above because A+B exceeds 95%)
    expected2 = {"A"}
    assert mapping2 == expected2, f"Expected {expected2}, got {mapping2}"

    # Test 3: min_count=10, top_p=0.995 (want 99.5% of original, A+B=99.01% should fit)
    preprocessor3 = WoePreprocessor(min_count=10, top_p=0.995)
    preprocessor3.fit(data)

    mapping3 = preprocessor3.get_category_mapping()["feature"]

    # After min_count filter: A=1000, B=500
    # Cumulative based on original total (1515):
    # A: 1000/1515 = 0.66 ≤ 0.995 ✓
    # A+B: 1500/1515 = 0.9901 ≤ 0.995 ✓
    # So should keep A and B
    expected3 = {"A", "B"}
    assert mapping3 == expected3, f"Expected {expected3}, got {mapping3}"

    print("✅ min_count and top_p interaction test passed")


def test_fallback_logic():
    """Test fallback when no categories meet top_p threshold after min_count filtering."""

    # Create data where min_count filters everything except one category
    data = pd.DataFrame(
        {
            "feature": ["A"] * 100
            + ["B"] * 5
            + ["C"] * 3
            + ["D"] * 2  # Only A survives min_count=10
        }
    )

    # Test with very low top_p that even A doesn't meet
    preprocessor = WoePreprocessor(
        min_count=10, top_p=0.5
    )  # Want 50% of 110 = 55, but A=100 > 55
    preprocessor.fit(data)

    mapping = preprocessor.get_category_mapping()["feature"]

    # Should fallback to most frequent category (A) even though it exceeds top_p
    expected = {"A"}
    assert mapping == expected, (
        f"Fallback should keep most frequent category, got {mapping}"
    )

    print("✅ Fallback logic test passed")


def test_edge_cases():
    """Test various edge cases."""

    # Test 1: Single category
    data1 = pd.DataFrame({"feature": ["A"] * 100})
    preprocessor1 = WoePreprocessor(min_count=10, top_p=0.8)
    preprocessor1.fit(data1)
    mapping1 = preprocessor1.get_category_mapping()["feature"]
    assert mapping1 == {"A"}, "Single category should be preserved"

    # Test 2: All categories below min_count
    data2 = pd.DataFrame({"feature": ["A"] * 5 + ["B"] * 3 + ["C"] * 2})
    preprocessor2 = WoePreprocessor(min_count=10, top_p=0.8)
    preprocessor2.fit(data2)
    mapping2 = preprocessor2.get_category_mapping()["feature"]
    assert mapping2 == {"A"}, "Should keep most frequent when all below min_count"

    # Test 3: Exactly 2 categories (binary) with one below min_count
    data3 = pd.DataFrame({"feature": ["A"] * 100 + ["B"] * 5})
    preprocessor3 = WoePreprocessor(min_count=10, top_p=0.8)
    preprocessor3.fit(data3)
    mapping3 = preprocessor3.get_category_mapping()["feature"]
    assert mapping3 == {"A", "B"}, "Binary feature should preserve both categories"

    print("✅ Edge cases test passed")


def test_demonstrate_old_vs_new_behavior():
    """Demonstrate the difference between old (wrong) and new (correct) cumulative calculation."""

    data = pd.DataFrame(
        {"feature": ["A"] * 1000 + ["B"] * 500 + ["C"] * 8 + ["D"] * 5 + ["E"] * 2}
    )

    # Simulate old behavior (cumulative based on filtered total)
    vc = data["feature"].value_counts(dropna=False).sort_values(ascending=False)
    vc_filtered = vc[vc >= 10]  # A=1000, B=500

    old_cumulative = (
        vc_filtered.cumsum() / vc_filtered.sum()
    )  # Wrong: based on filtered total
    new_cumulative = vc_filtered.cumsum() / vc.sum()  # Correct: based on original total

    print("\nDemonstrating old vs new behavior:")
    print(f"Original data: {dict(vc)}")
    print(f"After min_count=10 filter: {dict(vc_filtered)}")
    print(f"Original total: {vc.sum()}")
    print(f"Filtered total: {vc_filtered.sum()}")
    print()
    print("Old behavior (wrong):")
    for cat, old_cum in old_cumulative.items():
        print(f"  {cat}: {old_cum:.3f} ({old_cum * 100:.1f}% of filtered data)")
    print()
    print("New behavior (correct):")
    for cat, new_cum in new_cumulative.items():
        print(f"  {cat}: {new_cum:.3f} ({new_cum * 100:.1f}% of original data)")

    # With top_p=0.8:
    # Old: A=0.667, A+B=1.0 → would keep both (99% of original!)
    # New: A=0.660, A+B=0.990 → keeps only A (66% of original, closer to 80%)

    old_top_cats_80 = old_cumulative[old_cumulative <= 0.8].index.tolist()
    new_top_cats_80 = new_cumulative[new_cumulative <= 0.8].index.tolist()

    print("\nWith top_p=0.8:")
    print(
        f"Old behavior would keep: {old_top_cats_80} ({vc[old_top_cats_80].sum()}/{vc.sum()} = {vc[old_top_cats_80].sum() / vc.sum():.1%})"
    )
    print(
        f"New behavior keeps: {new_top_cats_80} ({vc[new_top_cats_80].sum()}/{vc.sum()} = {vc[new_top_cats_80].sum() / vc.sum():.1%})"
    )


if __name__ == "__main__":
    print("Testing WoePreprocessor logic...\n")

    test_binary_category_preservation()
    test_min_count_and_top_p_interaction()
    test_fallback_logic()
    test_edge_cases()
    test_demonstrate_old_vs_new_behavior()

    print("\n🎉 All tests passed!")
