# RFGBoost Test Suite

This directory contains comprehensive tests for the RFGBoost package, organized into unit and integration tests with proper fixture separation.

## Test Organization

### Unit Tests (`tests/unit/`)
Tests for individual components in isolation:

- **`test_woe_preprocessor.py`** - Tests WOE preprocessor component functionality
- **`test_woe_preprocessor_logic.py`** - Tests WOE preprocessing logic and transformations  
- **`test_trees_dataframe.py`** - Tests tree-to-dataframe conversion functionality
- **`test_tree_extraction.py`** - Tests tree extraction methods for both sklearn and XGBoost

### Integration Tests (`tests/integration/`)
Tests for component interactions and end-to-end functionality:

- **`test_fastwoe_integration.py`** - Tests integration with FastWoe library
- **`test_xgboost_integration.py`** - Tests XGBoost backend integration and performance
- **`test_xgboost_confidence_intervals.py`** - Tests confidence interval functionality
- **`test_xgboost_tree_success.py`** - Tests XGBoost tree extraction and analysis
- **`test_inference.py`** - Tests complete inference pipeline with WOE encoding
- **`test_plot_scenario.py`** - Tests plotting and visualization scenarios

## Test Configuration Structure

### Complete Isolation - No Shared Configuration
Each test type is **completely self-contained** with its own configuration:

### Unit Test Configuration (`tests/unit/conftest.py`)
**Fast, isolated testing environment:**
- **Pytest configuration** - Unit-specific markers and warning filters
- **Lightweight fixtures:**
  - `simple_tree_data` - Minimal data for tree extraction tests  
  - `minimal_woe_data` - Small categorical dataset for WOE unit tests
  - `mock_model_params` - Standard model parameters for fast testing

### Integration Test Configuration (`tests/integration/conftest.py`)  
**Comprehensive, end-to-end testing environment:**
- **Pytest configuration** - Integration-specific markers and warning filters
- **Session-scoped fixtures:**
  - `sample_data` - Large classification dataset with train/test split
  - `inference_test_data` - Data with unseen categories for inference testing
  - `benchmark_data` - Performance testing dataset (4000 samples)
  - `plot_test_data` - Non-linear data for visualization testing
  - `ci_test_data` - Confidence interval testing data

## Running Tests

### All Tests
```bash
uv run python -m pytest tests/
```

### Unit Tests Only (Fast)
```bash
uv run python -m pytest tests/unit/
```

### Integration Tests Only (Comprehensive)
```bash
uv run python -m pytest tests/integration/
```

### Specific Test Categories
```bash
# WOE-related tests
uv run python -m pytest tests/unit/test_woe* tests/integration/test_fastwoe*

# XGBoost-related tests  
uv run python -m pytest tests/integration/test_xgboost*

# Tree analysis tests
uv run python -m pytest tests/unit/test_tree* tests/unit/test_trees*

# Fast tests only (unit tests)
uv run python -m pytest -m "not slow" tests/unit/

# XGBoost tests only (if available)
uv run python -m pytest -m xgboost tests/
```

## Test Coverage

- **44 total tests** across all components
- **13 unit tests**: Fast, isolated component testing
- **31 integration tests**: End-to-end workflow validation
- **Backend coverage**: Both sklearn and XGBoost base learners
- **Edge cases**: Unseen categories, small datasets, confidence intervals

## Benefits of This Structure

### True Separation of Concerns
- **Unit environment**: Completely isolated with lightweight fixtures and minimal configuration
- **Integration environment**: Self-contained with comprehensive fixtures and full configuration  
- **Zero coupling**: No shared dependencies between test types

### Performance Optimization  
- **Fast unit tests**: Run in ~13 seconds for quick feedback
- **Session-scoped integration fixtures**: Expensive data generated once
- **Selective test execution**: Run only what you need

### Maintainability
- **Clear fixture scope**: Easy to understand what data each test type uses
- **Isolated changes**: Modify unit fixtures without affecting integration tests
- **Consistent patterns**: Predictable structure across test types

## Dependencies

Tests require:
- `pytest` - Test framework
- `numpy`, `pandas` - Data handling
- `scikit-learn` - Machine learning components
- `xgboost` - XGBoost backend (optional, tests skip if not available)
- `fastwoe` - Weight of Evidence encoding

All test dependencies are managed through the project's `pyproject.toml`. 