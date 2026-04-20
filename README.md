# rfgboost

Gradient boosting with asynchronous random forests, implemented in Rust with Python bindings.

## Installation

```bash
pip install rfgboost
```

From source (requires Rust toolchain):

```bash
pip install maturin
maturin develop --release
```

## Quick Start

```python
from rfgboost import RFGBoostClassifier, RFGBoostRegressor

# Classification
clf = RFGBoostClassifier(n_estimators=20, rf_n_estimators=50, rf_max_depth=6)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
ci = clf.predict_ci(X_test)  # Wilson score intervals

# Regression
reg = RFGBoostRegressor(n_estimators=20, rf_n_estimators=50, rf_max_depth=6)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
ci = reg.predict_ci(X_test)  # Split conformal prediction intervals

# Async mode (adaptive early stopping via CI convergence)
clf = RFGBoostClassifier(async_mode=True, tol=0.0)

# Categorical features (WOE encoding via fastwoe-rs)
clf = RFGBoostClassifier(cat_features=[0, 1, 2])
```

## Components

| Class | Description |
|-------|-------------|
| `RFGBoostClassifier` | Gradient boosting with RF base learners (binary + multiclass) |
| `RFGBoostRegressor` | Gradient boosting with RF base learners (regression) |
| `RandomForestClassifier` | Standalone random forest classifier |
| `RandomForestRegressor` | Standalone random forest regressor |
| `RandomForestUnsupervised` | Breiman's unsupervised RF (proximity, outliers, MDS) |
| `DecisionTree` | Single decision tree (exact sklearn match) |
| `TreeSHAP` | Exact tree-path-dependent SHAP values |

## Key Features

- **Async tree building**: Rayon work-stealing with AtomicBool convergence flag. Unstarted trees skip once the ensemble converges.
- **CI-based stopping**: Wilson intervals (classification) and normal CI (regression) determine convergence automatically with `tol=0`.
- **Histogram splitting**: 256-bin quantile histograms for O(n + bins) split search.
- **Conformal prediction**: Split conformal CIs for regression with coverage guarantees.
- **Unsupervised RF**: Proximity matrix, outlier detection, MDS embedding, feature importance from Breiman's original method.
- **Exact TreeSHAP**: Matches the official SHAP package to machine precision.

## License

MIT
