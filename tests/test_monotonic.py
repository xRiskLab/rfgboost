"""Tests for monotonic constraints in RFGBoostClassifier.

Constraints are enforced via value-bound propagation during tree growth, so
monotonicity is exact: sweeping a constrained feature with all others held
fixed never moves the prediction the wrong way.
"""

import numpy as np

from rfgboost import RFGBoostClassifier

PARAMS = dict(
    n_estimators=25,
    learning_rate=0.2,
    rf_n_estimators=30,
    rf_max_depth=5,
    random_state=42,
)


def _ushaped_data(n=2000, seed=0):
    """y depends on feature 0 in a U shape, so an UNCONSTRAINED model is
    non-monotone in feature 0 — a +1/-1 constraint must actively reshape it."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 3))
    z = 1.6 * (X[:, 0] ** 2) - 1.0 * X[:, 1]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(z - z.mean())))).astype(float)
    return X, y


def _sweep(model, base_rows, feat, grid):
    """Predicted P(y=1) as `feat` is swept over `grid`, all else fixed."""
    paths = []
    for base in base_rows:
        rows = np.tile(base, (len(grid), 1))
        rows[:, feat] = grid
        paths.append(np.array(model.predict_proba(rows))[:, 1])
    return paths


def test_no_constraints_is_backward_compatible():
    """Passing no constraints reproduces the default model exactly."""
    X, y = _ushaped_data()
    p_default = np.array(RFGBoostClassifier(**PARAMS).fit(X, y).predict_proba(X))[:, 1]
    p_none = np.array(
        RFGBoostClassifier(**PARAMS, monotone_constraints=None).fit(X, y).predict_proba(X)
    )[:, 1]
    assert np.allclose(p_default, p_none)


def test_increasing_constraint_holds():
    """With {0: +1}, prediction is non-decreasing in feature 0, all else fixed."""
    X, y = _ushaped_data()
    model = RFGBoostClassifier(**PARAMS, monotone_constraints={0: 1}).fit(X, y)
    grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    rng = np.random.default_rng(1)
    base_rows = [X[rng.integers(len(X))].copy() for _ in range(20)]
    for path in _sweep(model, base_rows, 0, grid):
        assert np.all(np.diff(path) >= -1e-9)


def test_decreasing_constraint_holds():
    """With {0: -1}, prediction is non-increasing in feature 0, all else fixed."""
    X, y = _ushaped_data()
    model = RFGBoostClassifier(**PARAMS, monotone_constraints={0: -1}).fit(X, y)
    grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    rng = np.random.default_rng(2)
    base_rows = [X[rng.integers(len(X))].copy() for _ in range(20)]
    for path in _sweep(model, base_rows, 0, grid):
        assert np.all(np.diff(path) <= 1e-9)


def test_constraint_value_normalized_by_sign():
    """Only the sign of a direction matters: {0: 5} behaves like {0: 1}."""
    X, y = _ushaped_data()
    p1 = np.array(
        RFGBoostClassifier(**PARAMS, monotone_constraints={0: 1}).fit(X, y).predict_proba(X)
    )[:, 1]
    p5 = np.array(
        RFGBoostClassifier(**PARAMS, monotone_constraints={0: 5}).fit(X, y).predict_proba(X)
    )[:, 1]
    assert np.allclose(p1, p5)


def test_invalid_constraint_key_raises():
    """A key that is not a valid column index fails fast at fit time."""
    X, y = _ushaped_data()  # 3 features, so column 99 is out of range
    raised = False
    try:
        RFGBoostClassifier(**PARAMS, monotone_constraints={99: 1}).fit(X, y)
    except ValueError:
        raised = True
    assert raised


def test_unconstrained_model_is_non_monotone():
    """Sanity: without the constraint the U-shaped data is non-monotone in
    feature 0, so the constraint tests above are not vacuous."""
    X, y = _ushaped_data()
    model = RFGBoostClassifier(**PARAMS).fit(X, y)
    grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    path = _sweep(model, [np.zeros(3)], 0, grid)[0]
    assert not np.all(np.diff(path) >= -1e-9)
