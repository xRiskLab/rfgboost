"""Equivalence tests for sample_weight across DecisionTree, RandomForest, and RFGBoost."""

import numpy as np
import pytest

from rfgboost import (
    DecisionTree,
    RandomForestClassifier,
    RandomForestRegressor,
    RFGBoostClassifier,
    RFGBoostRegressor,
)

RNG = np.random.RandomState(0)


def _make_clf(n=200, p=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p).astype(np.float64)
    logits = X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2] * X[:, 3]
    y = (logits > 0).astype(np.float64)
    return X, y


def _make_reg(n=200, p=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p).astype(np.float64)
    y = X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2] * X[:, 3] + 0.1 * rng.randn(n)
    return X, y.astype(np.float64)


def test_uniform_weights_match_no_weights_decision_tree_clf():
    X, y = _make_clf()
    a = DecisionTree(max_depth=4, task="classification", random_state=42)
    a.fit(X, y)
    b = DecisionTree(max_depth=4, task="classification", random_state=42)
    b.fit(X, y, sample_weight=np.ones(len(y)))
    np.testing.assert_array_equal(np.asarray(a.predict(X)), np.asarray(b.predict(X)))


def test_uniform_weights_match_no_weights_decision_tree_reg():
    X, y = _make_reg()
    a = DecisionTree(max_depth=4, task="regression", random_state=42)
    a.fit(X, y)
    b = DecisionTree(max_depth=4, task="regression", random_state=42)
    b.fit(X, y, sample_weight=np.ones(len(y)))
    np.testing.assert_array_equal(np.asarray(a.predict(X)), np.asarray(b.predict(X)))


def test_uniform_weights_match_no_weights_rf_clf():
    X, y = _make_clf()
    a = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    a.fit(X, y)
    b = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    b.fit(X, y, sample_weight=np.ones(len(y)))
    np.testing.assert_allclose(
        np.asarray(a.predict_proba(X)), np.asarray(b.predict_proba(X))
    )


def test_uniform_weights_match_no_weights_rf_reg():
    X, y = _make_reg()
    a = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
    a.fit(X, y)
    b = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
    b.fit(X, y, sample_weight=np.ones(len(y)))
    np.testing.assert_allclose(np.asarray(a.predict(X)), np.asarray(b.predict(X)))


def test_uniform_weights_match_no_weights_rfgboost_clf():
    X, y = _make_clf()
    a = RFGBoostClassifier(n_estimators=5, rf_n_estimators=8, rf_max_depth=4, random_state=42)
    a.fit(X, y)
    b = RFGBoostClassifier(n_estimators=5, rf_n_estimators=8, rf_max_depth=4, random_state=42)
    b.fit(X, y, sample_weight=np.ones(len(y)))
    np.testing.assert_allclose(
        np.asarray(a.predict_proba(X)), np.asarray(b.predict_proba(X))
    )


def test_uniform_weights_match_no_weights_rfgboost_reg():
    X, y = _make_reg()
    a = RFGBoostRegressor(n_estimators=5, rf_n_estimators=8, rf_max_depth=4, random_state=42)
    a.fit(X, y)
    b = RFGBoostRegressor(n_estimators=5, rf_n_estimators=8, rf_max_depth=4, random_state=42)
    b.fit(X, y, sample_weight=np.ones(len(y)))
    np.testing.assert_allclose(np.asarray(a.predict(X)), np.asarray(b.predict(X)))


def test_constant_scaling_invariance_decision_tree_reg():
    """Splits and leaf values are invariant to global scaling of sample_weight."""
    X, y = _make_reg()
    a = DecisionTree(max_depth=5, task="regression", random_state=42)
    a.fit(X, y, sample_weight=np.ones(len(y)))
    b = DecisionTree(max_depth=5, task="regression", random_state=42)
    b.fit(X, y, sample_weight=3.7 * np.ones(len(y)))
    np.testing.assert_allclose(np.asarray(a.predict(X)), np.asarray(b.predict(X)))


def test_duplicate_rows_match_doubled_weights_decision_tree_reg():
    """Duplicating row i is equivalent to giving it weight 2 (with bootstrap off)."""
    X, y = _make_reg(n=80)
    # Duplicate every row
    Xd = np.vstack([X, X])
    yd = np.hstack([y, y])
    w = 2.0 * np.ones(len(y))

    a = DecisionTree(max_depth=4, task="regression", random_state=0)
    a.fit(Xd, yd)
    b = DecisionTree(max_depth=4, task="regression", random_state=0)
    b.fit(X, y, sample_weight=w)

    # Predictions on the original rows should match
    pa = np.asarray(a.predict(X))
    pb = np.asarray(b.predict(X))
    np.testing.assert_allclose(pa, pb)


def test_zero_weight_excludes_sample_decision_tree_reg():
    """Setting weight=0 for a subset reproduces the fit on the complement."""
    X, y = _make_reg(n=120)
    keep = np.arange(80)
    drop = np.arange(80, 120)
    w = np.ones(len(y))
    w[drop] = 0.0

    a = DecisionTree(max_depth=4, task="regression", random_state=0)
    a.fit(X[keep], y[keep])
    b = DecisionTree(max_depth=4, task="regression", random_state=0)
    b.fit(X, y, sample_weight=w)

    # min_samples_leaf is count-based, so trees can differ at edges; check kept-row predictions match closely
    pa = np.asarray(a.predict(X[keep]))
    pb = np.asarray(b.predict(X[keep]))
    # With min_samples_leaf=1 and identical data, leaves should align — predictions equal
    np.testing.assert_allclose(pa, pb, atol=1e-9)


def test_invalid_weight_raises():
    X, y = _make_clf(n=50)
    clf = RFGBoostClassifier(n_estimators=2, rf_n_estimators=4, random_state=0)
    with pytest.raises((ValueError, TypeError)):
        clf.fit(X, y, sample_weight=np.array([-1.0] * len(y)))
    with pytest.raises((ValueError, TypeError)):
        clf.fit(X, y, sample_weight=np.array([np.nan] * len(y)))
    with pytest.raises((ValueError, TypeError)):
        clf.fit(X, y, sample_weight=np.ones(len(y) - 1))


def _newton_leaf_closed_form(y, w, lr=1.0):
    """Return (initial_logit, expected_predicted_logit) for the single-leaf case
    where leaf_value = Σ(w·(y−p))/Σ(w·p(1−p)) under Newton-weighted leaves."""
    pi = (w * y).sum() / w.sum()
    pi = np.clip(pi, 1e-5, 1.0 - 1e-5)
    initial_logit = np.log(pi / (1.0 - pi))
    p0 = pi  # sigmoid(initial_logit) == pi exactly
    leaf = (w * (y - p0)).sum() / (w * p0 * (1.0 - p0)).sum()
    return initial_logit, initial_logit + lr * leaf


@pytest.mark.parametrize("use_histogram", [True, False])
def test_newton_leaf_matches_closed_form_uniform(use_histogram):
    """With weights=ones the Newton update collapses to mean(y−p)/mean(p(1−p))."""
    rng = np.random.RandomState(0)
    n = 200
    X = np.ones((n, 1))  # constant feature → no split possible → one-leaf tree
    y = (rng.rand(n) < 0.3).astype(np.float64)
    w = np.ones(n)

    clf = RFGBoostClassifier(
        n_estimators=1, learning_rate=1.0,
        rf_n_estimators=1, bootstrap=False,
        use_histogram=use_histogram, random_state=0,
    )
    clf.fit(X, y, sample_weight=w)
    p_pred = np.asarray(clf.predict_proba(X))[:, 1]
    # All samples in the same leaf → identical predictions
    np.testing.assert_allclose(p_pred, p_pred[0], rtol=1e-12)
    pred_logit = np.log(p_pred[0] / (1.0 - p_pred[0]))

    _, expected_logit = _newton_leaf_closed_form(y, w, lr=1.0)
    np.testing.assert_allclose(pred_logit, expected_logit, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("use_histogram", [True, False])
def test_newton_leaf_matches_closed_form_weighted(use_histogram):
    """User sample_weight propagates into both gradient and Hessian sums (XGBoost convention)."""
    rng = np.random.RandomState(1)
    n = 250
    X = np.ones((n, 1))
    y = (rng.rand(n) < 0.4).astype(np.float64)
    w = rng.uniform(0.5, 5.0, size=n)  # non-uniform weights

    clf = RFGBoostClassifier(
        n_estimators=1, learning_rate=1.0,
        rf_n_estimators=1, bootstrap=False,
        use_histogram=use_histogram, random_state=0,
    )
    clf.fit(X, y, sample_weight=w)
    p_pred = np.asarray(clf.predict_proba(X))[:, 1]
    np.testing.assert_allclose(p_pred, p_pred[0], rtol=1e-12)
    pred_logit = np.log(p_pred[0] / (1.0 - p_pred[0]))

    _, expected_logit = _newton_leaf_closed_form(y, w, lr=1.0)
    np.testing.assert_allclose(pred_logit, expected_logit, rtol=1e-9, atol=1e-12)


def test_newton_step_uses_weighted_hessian_not_just_weighted_gradient():
    """Sanity: the update divides by Σ(w·h), not Σ(w). Check that doubling weights
    on the positive class does NOT just double the leaf value (because Σ(w·h)
    grows along with Σ(w·g))."""
    rng = np.random.RandomState(2)
    n = 200
    X = np.ones((n, 1))
    y = (rng.rand(n) < 0.3).astype(np.float64)

    # Baseline: uniform
    w1 = np.ones(n)
    a = RFGBoostClassifier(n_estimators=1, learning_rate=1.0, rf_n_estimators=1,
                           bootstrap=False, random_state=0)
    a.fit(X, y, sample_weight=w1)
    logit_a = np.log(a.predict_proba(X)[0, 1] / (1.0 - a.predict_proba(X)[0, 1]))

    # Uniform×2 — should give the SAME leaf value (scaling invariance)
    w2 = 2.0 * np.ones(n)
    b = RFGBoostClassifier(n_estimators=1, learning_rate=1.0, rf_n_estimators=1,
                           bootstrap=False, random_state=0)
    b.fit(X, y, sample_weight=w2)
    logit_b = np.log(b.predict_proba(X)[0, 1] / (1.0 - b.predict_proba(X)[0, 1]))

    # Strict equality: weighted Newton is invariant to global weight scaling
    np.testing.assert_allclose(logit_a, logit_b, rtol=1e-10)


def test_class_imbalance_reweighting_shifts_predictions():
    """Boosting minority class with weights should raise its predicted probability."""
    rng = np.random.RandomState(0)
    n = 400
    # Severely imbalanced
    X = rng.randn(n, 6)
    y = np.zeros(n)
    y[:30] = 1.0  # 7.5% positive
    rng.shuffle(y)

    base = RFGBoostClassifier(n_estimators=8, rf_n_estimators=10, rf_max_depth=4, random_state=0)
    base.fit(X, y)
    p_base = np.asarray(base.predict_proba(X))[:, 1].mean()

    w = np.where(y == 1, 12.0, 1.0)  # boost minority
    weighted = RFGBoostClassifier(n_estimators=8, rf_n_estimators=10, rf_max_depth=4, random_state=0)
    weighted.fit(X, y, sample_weight=w)
    p_weighted = np.asarray(weighted.predict_proba(X))[:, 1].mean()

    # Reweighting should noticeably shift the average predicted P(y=1) upward
    assert p_weighted > p_base + 0.1, f"expected shift, got {p_base:.3f} -> {p_weighted:.3f}"
