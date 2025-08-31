"""Test WoePreprocessor for high-cardinality categorical features."""

import numpy as np
import pandas as pd
from fastwoe import FastWoe, WoePreprocessor

# Create high-cardinality test data
np.random.seed(42)
n_samples = 1000

# Create high-cardinality categorical feature
# Some categories are frequent, others are rare
categories = [f"cat_{i}" for i in range(50)]  # 50 categories
probs = np.exp(-np.arange(50) * 0.2)  # Exponential decay for realistic distribution
probs = probs / probs.sum()

high_card_feature = np.random.choice(categories, n_samples, p=probs)
target = np.random.binomial(1, 0.3 + 0.4 * (high_card_feature == "cat_0"), n_samples)

data = pd.DataFrame({"high_card_cat": high_card_feature, "target": target})

print("Original data info:")
print(f"Total categories: {data['high_card_cat'].nunique()}")
print(f"Category distribution:\n{data['high_card_cat'].value_counts().head(10)}")

# Initialize preprocessor
preprocessor = WoePreprocessor(top_p=0.9, min_count=5)

# Fit and transform
X = data[["high_card_cat"]]
y = data["target"]

X_processed = preprocessor.fit_transform(X, cat_features=["high_card_cat"])

print("\nAfter preprocessing:")
print(f"Processed categories: {X_processed['high_card_cat'].nunique()}")
print(f"Category distribution:\n{X_processed['high_card_cat'].value_counts()}")

# Show reduction summary
reduction_summary = preprocessor.get_reduction_summary(X)
print("\nReduction summary:")
print(reduction_summary)

print("\n" + "=" * 50)
print("TESTING WITH FASTWOE")
print("=" * 50)

# Test with FastWoe - original vs preprocessed
print("\n1. FastWoe on original high-cardinality data:")
woe_original = FastWoe()
try:
    woe_original.fit(X, y)
    stats_original = woe_original.get_feature_stats()
    print(
        f"Original - Categories: {stats_original['n_categories'].iloc[0]}, Gini: {stats_original['gini'].iloc[0]:.4f}"
    )
except Exception as e:
    print(f"Error with original data: {e}")

print("\n2. FastWoe on preprocessed data:")
woe_processed = FastWoe()
woe_processed.fit(X_processed, y)
stats_processed = woe_processed.get_feature_stats()
print(
    f"Processed - Categories: {stats_processed['n_categories'].iloc[0]}, Gini: {stats_processed['gini'].iloc[0]:.4f}"
)

# Show WOE mappings
print("\n3. WOE mappings for processed data:")
mappings = woe_processed.get_mapping("high_card_cat")
print(mappings.sort_values("woe", ascending=False))

print("\nPreprocessor successfully reduced cardinality and improved stability!")
