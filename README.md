# RFGBoost

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful Python package combining **Random Forest Gradient Boosting (RFGBoost)** and **Fast Weight of Evidence (FastWoe)** encoding for interpretable machine learning with categorical features.

## �  What's in This Repository

**RFGBoost** is a gradient boosting implementation that uses Random Forest as base learners instead of traditional decision trees, providing better interpretability while maintaining strong predictive performance. The library integrates seamlessly with **FastWoe** encoding for categorical features, making it particularly valuable for financial and risk modeling applications.

### Repository Structure
- `rfgboost/` - Core RFGBoost implementation with dual base learner support
- `notebooks/` - Jupyter notebooks demonstrating categorical features, tuning, and uncertainty estimation
- `tests/` - Comprehensive test suite with unit and integration tests
- `scripts/` - Benchmarking and analysis utilities
- `data/` - Sample datasets for experimentation
- `marimo/` - Interactive notebook examples

### Ideal Use Cases
- **Financial Risk Modeling**: Credit scoring and risk assessment with interpretable results
- **High-Interpretability ML**: Applications requiring clear decision paths and feature explanations
- **Categorical-Heavy Datasets**: Automatic handling of high-cardinality categorical features
- **Uncertainty-Critical Applications**: Built-in confidence intervals for risk-aware predictions

## 🚀 Key Features

### RFGBoost
- **Interpretable Gradient Boosting**: Uses Random Forest as base learners instead of decision trees
- **Multiple Base Learners**: Support for both sklearn RandomForest and XGBoost RandomForest
- **Built-in WOE Encoding**: Seamless integration with categorical features
- **Confidence Intervals**: Uncertainty quantification for predictions
- **Tree Analysis**: Extract detailed tree structures and decision paths (sklearn only)
- **Flexible Tasks**: Supports both regression and classification

## 📦 Installation

```bash
pip install rfgboost
```

Or install from source:
```bash
git clone https://github.com/xRiskLab/rfgboost.git
cd rfgboost
pip install -e .
```

## 🏃 Quick Start

### Basic Classification Example

```python
import pandas as pd
import numpy as np
from rfgboost import RFGBoost
from fastwoe import FastWoe

# Sample data
data = pd.DataFrame({
    'category_a': ['A', 'B', 'C', 'A', 'B'] * 200,
    'category_b': ['X', 'Y', 'X', 'Y', 'X'] * 200,
    'numeric_1': np.random.normal(0, 1, 1000),
    'target': np.random.binomial(1, 0.4, 1000)
})

X = data.drop('target', axis=1)
y = data['target']

# Train RFGBoost with automatic WOE encoding
model = RFGBoost(
    n_estimators=10,
    task='classification',
    cat_features=['category_a', 'category_b'],
    learning_rate=0.1
)

model.fit(X, y)

# Make predictions
predictions = model.predict_proba(X)[:, 1]

# Get feature importance
importance = model.get_feature_importance()
print(importance)
```

## 📊 Advanced Features

### 1. Tree Analysis and Interpretability

```python
# Extract tree structures with decision paths (works with both base learners)
tree_data = model.extract_tree_data_with_conditions()

# Get leaf nodes with path conditions
leaf_data = model.extract_leaf_nodes_with_conditions()

# Convert to interpretable DataFrame with event rates
analysis = model.trees_to_dataframe(X, y)
print(analysis.head())

# Example output for both sklearn and XGBoost:
# Round  Tree                    PathCondition  Events  NonEvents  EventRate
#     0     0  feature_1 <= 0.5 and feature_2 > 1.2      23         45      0.338
#     0     1  feature_1 > 0.5 and feature_3 <= 0.8      18         32      0.360
```

### 2. Confidence Intervals

```python
# Get prediction confidence intervals
ci_predictions = model.predict_ci(X, alpha=0.05)  # 95% CI
print(f"Lower bounds: {ci_predictions[:, 0]}")
print(f"Upper bounds: {ci_predictions[:, 1]}")
```

### 3. High-Cardinality Feature Handling

```python
from fastwoe import WoePreprocessor

# Handle features with many categories
preprocessor = WoePreprocessor(
    top_p=0.9,           # Keep categories covering 90% of data
    min_count=5,         # Minimum observations per category
    other_token="__rare__"  # Token for rare categories
)

# See cardinality reduction
X_processed = preprocessor.fit_transform(X, cat_features=['high_card_feature'])
reduction_summary = preprocessor.get_reduction_summary(X)
print(reduction_summary)
```

### 4. WOE Feature Analysis

```python
# Comprehensive feature analysis
woe_stats = model.get_woe_feature_stats()
print("\nFeature Statistics:")
print(woe_stats[['feature', 'gini', 'information_value', 'n_categories']])

# Feature ranking by predictive power
feature_ranking = model.get_woe_feature_summary()
print("\nTop Features by Gini:")
print(feature_ranking.head())

# Detailed WOE mappings
for feature in ['category_a', 'category_b']:
    mapping = model.get_woe_mappings(feature)
    print(f"\n{feature} WOE Mapping:")
    print(mapping.sort_values('woe', ascending=False))
```

### 5. Base Learner Selection

```python
# sklearn RandomForest (default) - best for detailed interpretability
model_sklearn = RFGBoost(
    n_estimators=10,
    task='classification',
    cat_features=['category_a'],
    base_learner="sklearn",  # Default
    rf_params={'n_estimators': 100, 'max_depth': 5}
)

# XGBoost RandomForest - best for performance + interpretability
model_xgboost = RFGBoost(
    n_estimators=10,
    task='classification',
    cat_features=['category_a'],
    base_learner="xgboost",  # Faster training
    rf_params={'n_estimators': 100, 'max_depth': 6}
)

# Tree extraction now works with both base learners!
sklearn_trees = model_sklearn.trees_to_dataframe(X, y)  # ✅ Works - more detailed
xgboost_trees = model_xgboost.trees_to_dataframe(X, y)  # ✅ Works - faster training
```

## 🎯 Production Usage

### Inference Pipeline

```python
# Training phase
model = RFGBoost(
    n_estimators=20,
    task='classification',
    cat_features=['cat_col1', 'cat_col2'],
    woe_kwargs={'random_state': 42}
)
model.fit(X_train, y_train)

# Save model (using joblib or pickle)
import joblib
joblib.dump(model, 'rfgboost_model.pkl')

# Inference phase
model = joblib.load('rfgboost_model.pkl')
new_predictions = model.predict_proba(X_new)[:, 1]

# Handle unseen categories automatically
# New categories get mapped to __other__ and receive appropriate WOE values
```

### Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Create preprocessing + modeling pipeline
def create_pipeline():
    return Pipeline(
        [
            ("woe_preprocess", WoePreprocessor(top_p=0.95, min_count=10)),
            ("woe_encode", FastWoe()),
            # Add other preprocessing steps as needed
        ]
    )

# Use in cross-validation or grid search
from sklearn.model_selection import cross_val_score

pipeline = create_pipeline()
scores = cross_val_score(pipeline, X, y, cv=5)
```

## 📈 Performance Benefits

### Interpretability
- **Clear Decision Paths**: Extract exact conditions leading to predictions
- **WOE Mappings**: Understand how each category contributes to predictions
- **Feature Importance**: Rank features by predictive power (Gini, IV)
- **Statistical Insights**: Comprehensive feature statistics and diagnostics

### Robustness
- **Handles Missing Data**: Built-in missing value handling in WOE encoding
- **Unseen Categories**: Graceful handling of new categories in production
- **High Cardinality**: Automatic cardinality reduction for stable encoding
- **Confidence Intervals**: Uncertainty quantification for risk assessment

### Performance
- **Fast Training**: Efficient Random Forest base learners
- **Scalable**: Handles large datasets with many categorical features
- **Memory Efficient**: Optimized WOE computation and storage
- **Production Ready**: Consistent inference pipeline

## 🛠️ API Reference

### RFGBoost

```python
RFGBoost(
    n_estimators=10,           # Number of boosting rounds
    rf_params=None,            # RandomForest parameters
    learning_rate=0.1,         # Boosting learning rate
    task="regression",         # "regression" or "classification"
    cat_features=None,         # List of categorical feature names
    woe_kwargs=None,          # Parameters for FastWoe encoder
    base_learner="sklearn"     # "sklearn" or "xgboost"
)
```

**Key Methods:**
- `fit(X, y)`: Train the model
- `predict(X)`: Make predictions
- `predict_proba(X)`: Get prediction probabilities (classification)
- `predict_ci(X, alpha=0.05)`: Get confidence intervals
- `get_feature_importance()`: Feature importance scores
- `get_woe_mappings(feature=None)`: WOE mappings
- `get_woe_feature_stats()`: WOE feature statistics
- `trees_to_dataframe(X, y)`: Detailed tree analysis (sklearn only)

**Base Learner Comparison:**

| Feature | sklearn | xgboost |
|---------|---------|---------|
| Training Speed | Slower | **4-5x Faster** |
| Tree Extraction | ✅ Full Support | ✅ **Full Support** |
| Confidence Intervals | ✅ Accurate | ✅ **Individual Tree Variance** |
| Dependencies | Built-in | Requires XGBoost |
| Memory Usage | Higher | Lower |
| GPU Support | ❌ | ✅ Available |