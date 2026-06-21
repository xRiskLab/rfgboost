# Changelog

All notable changes to rfgboost will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-06-16

### Changed
- `fastwoe-rs` (which powers the categorical `cat_features` WOE path) now carries an environment marker: it still installs automatically with a normal `pip install rfgboost`, but is excluded on Pyodide/WASM (`sys_platform == 'emscripten'`), where it has no wheel. This lets `micropip.install("rfgboost")` resolve by name in JupyterLite; using `cat_features` there raises a clear error.
- The Pyodide WASM wheel is now built and published by the main release workflow on tag (emscripten 4.0.9 / build-std), replacing the standalone `pyodide-wheel.yml`.

## [0.1.1] - 2026-06-16

### Added
- WebAssembly / Pyodide support: rfgboost builds and runs in the browser (JupyterLite, PyScript). Includes an in-browser demo (`examples/web/`) and a `pyodide-wheel.yml` workflow that builds the `pyemscripten` wheel.

### Changed
- Bumped `pyo3`/`numpy` from 0.19 to 0.22.
- `rayon` is now a native-only dependency; on emscripten/wasm a sequential fallback is used (native parallelism unchanged, verified ~3.7x on 10 cores).

### Fixed
- Added `scikit-learn` to core dependencies (it is imported at runtime but was previously only declared in the `test` extra).

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
