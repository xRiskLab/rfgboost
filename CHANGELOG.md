# Changelog

All notable changes to rfgboost will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-15

### Added
- Optional, off-by-default **GPU acceleration** across the predict/explain API via a `device=` argument:
  - `device="cuda"` — native CUDA (cudarc + nvrtc, `cuda` Cargo feature, NVIDIA).
  - `device="mps"` / `"metal"` / `"gpu"` — wgpu → Metal/Vulkan/DX12 (`gpu` Cargo feature).
  - `device="cpu"` (default) unchanged.
  - Covered: `RandomForestRegressor.predict`, `RandomForestClassifier.predict`/`predict_proba`, `RFGBoost`/`RFGBoostRegressor`/`RFGBoostClassifier` `predict`/`predict_proba`, and `TreeSHAP.explain`.
- Native-only and excluded from the default and wasm wheels. A wheel carries whichever backend it was built with; requesting an unavailable device raises a clear error.
- Measured speedups over the CPU path: predict up to ~33× (A100 `cuda`) / ~12× (M4 `mps`); `TreeSHAP.explain` ~3× (exact 2^k Shapley — a correct GPU reference, not a substitute for the polynomial TreeSHAP algorithm).
- `rfgboost.__version__` (read from package metadata).
- `rfgboost.WoeEncoder` — standalone scikit-learn transformer for supervised Weight-of-Evidence encoding of categorical features (fastwoe-rs).
- `async_mode` (+ `tol`) on `RandomForestClassifier`/`RandomForestRegressor` — confidence-interval early stopping: build only as many trees as the predictions need (classification: top-2 class-vote Wilson CIs separate for ≥90% of samples; regression: per-sample prediction-CI half-width, or a `tol` mean-change threshold). The `trees_used` attribute reports how many were built. Matched a 300-tree forest's accuracy with ~54 trees (classification) / ~30 trees (regression). This surfaces, as a first-class RF feature, the same async-convergence rule rfgboost already uses inside its boosting rounds.

### Changed
- **Breaking:** WOE/categorical encoding is decoupled from the boosting estimators. `RFGBoostClassifier`/`RFGBoostRegressor` no longer accept `cat_features` and are pure-numeric — for categorical data, compose with a Pipeline: `make_pipeline(WoeEncoder(cat_features=...), RFGBoostClassifier())`. This keeps the estimators free of the `fastwoe-rs` dependency (e.g. in Pyodide/WASM, where it has no wheel). The serializable WOE bundle (`to_dict`) and `get_iv_analysis` now live on `WoeEncoder`; `to_dict` on the estimators no longer takes `n_features`.

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

[Unreleased]: https://github.com/xRiskLab/rfgboost/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/xRiskLab/rfgboost/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/xRiskLab/rfgboost/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/xRiskLab/rfgboost/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/xRiskLab/rfgboost/releases/tag/v0.1.0
