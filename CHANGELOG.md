# Changelog

All notable changes to rfgboost will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-20

### Added
- Initial public release.
- `RFGBoostClassifier` and `RFGBoostRegressor`: gradient boosting with random forest base learners (binary, multiclass, regression).
- `RandomForestClassifier`, `RandomForestRegressor`, `RandomForestUnsupervised`: standalone Rust random forests.
- `DecisionTree`: single-tree implementation matching sklearn splits.
- `TreeSHAP`: exact tree-path-dependent SHAP values matching the official `shap` package.
- Async tree building with Rayon work-stealing and CI-based convergence (Wilson/normal intervals, `tol=0` auto-stop).
- 256-bin histogram splitting for O(n + bins) split search.
- Split conformal prediction intervals for regression.
- Categorical feature support via WOE encoding (`fastwoe-rs`).

[Unreleased]: https://github.com/xRiskLab/rfgboost/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/xRiskLab/rfgboost/releases/tag/v0.1.0
