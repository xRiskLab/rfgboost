# Python Version Compatibility Tests

This directory contains comprehensive tests to verify that RFGBoost works correctly across different Python versions and dependency combinations.

## Overview

The compatibility tests ensure that RFGBoost functions properly from Python 3.8+ and works with all required dependencies. These tests are designed to catch any version-specific issues early in the development process.

## Test Structure

### `test_python_versions.py`
Main compatibility test file that includes:

- **Python Version Requirements**: Verifies Python 3.8+ compatibility
- **Import Compatibility**: Tests all package imports work correctly
- **Basic Functionality**: Tests core RFGBoost functionality
- **XGBoost Backend**: Tests XGBoost integration (if available)
- **WOE Encoding**: Tests Weight of Evidence encoding functionality
- **Feature Importance**: Tests feature importance calculation
- **Confidence Intervals**: Tests uncertainty quantification
- **Tree Extraction**: Tests tree analysis functionality
- **Edge Cases**: Tests handling of small datasets and edge cases
- **Parameter Validation**: Tests error handling for invalid parameters

### `conftest.py`
Pytest configuration and fixtures for compatibility testing:

- **Test Data Fixtures**: Provides standardized test datasets
- **Model Configurations**: Predefined model configurations for testing
- **Pytest Markers**: Custom markers for different Python version requirements

## Running Compatibility Tests

### Quick Test (Current Python Version)
```bash
# From project root
./run_compatibility_tests.sh
```

### Using the Compatibility Script
```bash
# Run all compatibility tests
python scripts/test_compatibility.py

# Test only imports
python scripts/test_compatibility.py imports

# Run specific test
python scripts/test_compatibility.py test python_versions
```

### Using Pytest Directly
```bash
# Run all compatibility tests
uv run python -m pytest tests/compatibility/ -v

# Run specific test file
uv run python -m pytest tests/compatibility/test_python_versions.py -v

# Run with specific markers
uv run python -m pytest tests/compatibility/ -m python38 -v
```

### Using Tox (Multiple Python Versions)
```bash
# Install tox
pip install tox

# Run tests for specific Python version
tox -e py38
tox -e py39
tox -e py310
tox -e py311
tox -e py312
tox -e py313

# Run all Python versions
tox

# Run only compatibility tests
tox -e compatibility
```

## GitHub Actions

The repository includes a GitHub Actions workflow (`.github/workflows/compatibility.yml`) that automatically runs compatibility tests across all supported Python versions on:

- Push to main/develop branches
- Pull requests
- Weekly scheduled runs

## Supported Python Versions

| Version | Status | Notes |
|---------|--------|-------|
| Python 3.8 | ✅ Supported | Minimum supported version |
| Python 3.9 | ✅ Supported | Full compatibility |
| Python 3.10 | ✅ Supported | Full compatibility |
| Python 3.11 | ✅ Supported | Full compatibility |
| Python 3.12 | ✅ Supported | Full compatibility |
| Python 3.13 | ✅ Supported | Full compatibility |

## Dependency Compatibility

### Required Dependencies
- **scikit-learn>=1.7.1**: Supports Python 3.8+
- **scipy>=1.16.0**: Supports Python 3.8+
- **xgboost>=3.0.2**: Supports Python 3.8+
- **fastwoe>=0.1.1.post4**: Supports Python 3.8+
- **marimo>=0.14.12**: Supports Python 3.8+

### Optional Dependencies
- **pytest>=7.0**: For running tests
- **numpy**: For numerical operations
- **pandas**: For data manipulation

## Test Coverage

The compatibility tests cover:

1. **Core Functionality**
   - Model initialization and fitting
   - Prediction (classification and regression)
   - Probability prediction (classification)
   - Feature importance calculation

2. **Advanced Features**
   - WOE encoding with categorical features
   - Confidence interval estimation
   - Tree extraction and analysis
   - XGBoost backend integration

3. **Error Handling**
   - Invalid parameter validation
   - Missing dependency handling
   - Edge case handling

4. **Import Compatibility**
   - Package import verification
   - Submodule import verification
   - Instance creation verification

## Adding New Compatibility Tests

When adding new features to RFGBoost, consider adding compatibility tests:

1. **Add to existing test class** in `test_python_versions.py`
2. **Use Python 3.8+ compatible syntax** only
3. **Test both sklearn and XGBoost backends** if applicable
4. **Include edge cases** and error conditions
5. **Update this README** with new test descriptions

### Example Test Structure
```python
def test_new_feature_compatibility(self):
    """Test new feature functionality."""
    # Setup
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    
    # Test
    model = RFGBoost(
        n_estimators=3,
        learning_rate=0.1,
        task="classification",
        base_learner="sklearn",
        rf_params={"n_estimators": 2, "max_depth": 3, "random_state": 42}
    )
    
    model.fit(X, y)
    
    # Assert
    result = model.new_feature(X)
    assert result is not None
    assert len(result) == len(y)
    
    print("✅ New feature functionality works")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Version Conflicts**: Use `uv sync --dev` to install correct versions
3. **Python Version**: Verify you're using Python 3.8+
4. **XGBoost Issues**: XGBoost tests will be skipped if not installed

### Debug Mode
```bash
# Run with verbose output
uv run python -m pytest tests/compatibility/ -v -s

# Run specific test with debug output
uv run python -m pytest tests/compatibility/test_python_versions.py::TestPythonVersionCompatibility::test_basic_functionality_python_38_compatible -v -s
```

## Continuous Integration

The compatibility tests are integrated into the CI/CD pipeline to ensure:

- All new code works across supported Python versions
- Dependency updates don't break compatibility
- Releases are tested before deployment

For more information, see the main project README and the GitHub Actions workflow configuration. 