"""Round-trip: serialize fitted model via to_dict, predict in pure Python, match Rust."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rfgboost import RFGBoostClassifier, RFGBoostRegressor, WoeEncoder


def _traverse(tree: dict, x: np.ndarray) -> float:
    cl = tree["children_left"]
    cr = tree["children_right"]
    feat = tree["feature"]
    thr = tree["threshold"]
    val = tree["value"]
    i = 0
    while cl[i] != -1:
        i = cl[i] if x[feat[i]] <= thr[i] else cr[i]
    return float(val[i])


def _predict_raw(d: dict, X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    raw = np.tile(np.asarray(d["init"], dtype=np.float64), (n, 1))
    lr = d["learning_rate"]
    for rnd in d["rounds"]:
        for o, out in enumerate(rnd["outputs"]):
            ts = out["trees"]
            avgs = np.array([[_traverse(t, X[i]) for t in ts] for i in range(n)]).mean(axis=1)
            raw[:, o] += lr * avgs
    return raw


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _encode_with_woe(d: dict, X_raw: np.ndarray) -> np.ndarray:
    """Reproduce _encode_woe purely from the dict bundle."""
    woe = d["woe"]
    cat = woe["cat_features"]
    num = woe["numeric_features"]
    tables = woe["woe_tables"]
    n = X_raw.shape[0]

    rows = []
    for i in range(n):
        encoded: list[float] = []
        for ci, col in enumerate(cat):
            v = str(X_raw[i, col])
            if woe["woe_multiclass"]:
                for c in range(woe["n_woe_classes"]):
                    encoded.append(tables[ci][c].get(v, 0.0))
            else:
                encoded.append(tables[ci].get(v, 0.0))
        for j in num:
            encoded.append(float(X_raw[i, j]))
        rows.append(encoded)
    return np.asarray(rows, dtype=np.float64)


def test_to_dict_binary_round_trip() -> None:
    rng = np.random.default_rng(0)
    X = rng.random((200, 4))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(np.float64)
    Xt = rng.random((40, 4))

    m = RFGBoostClassifier(
        n_estimators=5,
        learning_rate=0.1,
        rf_n_estimators=6,
        rf_max_depth=4,
        random_state=42,
    ).fit(X, y)

    d = m.to_dict()
    assert d["task"] == "binary"
    assert d["n_outputs"] == 1
    json.dumps(d, default=lambda v: None)  # JSON serializable (NaN handled by default=)

    raw = _predict_raw(d, Xt)
    p1 = _sigmoid(raw[:, 0])
    proba_rust = m.predict_proba(Xt)[:, 1]
    np.testing.assert_allclose(p1, proba_rust, atol=1e-12)


def test_to_dict_multiclass_round_trip() -> None:
    rng = np.random.default_rng(1)
    X = rng.random((200, 4))
    y = (X[:, 0] * 3).astype(int).astype(np.float64)
    Xt = rng.random((40, 4))

    m = RFGBoostClassifier(
        n_estimators=4,
        learning_rate=0.1,
        rf_n_estimators=5,
        rf_max_depth=4,
        random_state=42,
    ).fit(X, y)

    d = m.to_dict()
    assert d["task"] == "multiclass"
    assert d["n_outputs"] == 3

    raw = _predict_raw(d, Xt)
    proba_py = _softmax(raw)
    proba_rust = m.predict_proba(Xt)
    np.testing.assert_allclose(proba_py, proba_rust, atol=1e-12)


def test_to_dict_regression_round_trip() -> None:
    rng = np.random.default_rng(2)
    X = rng.random((200, 4))
    y = X[:, 0] * 2.0 + X[:, 1] - 0.5
    Xt = rng.random((40, 4))

    m = RFGBoostRegressor(
        n_estimators=5,
        learning_rate=0.1,
        rf_n_estimators=6,
        rf_max_depth=4,
        random_state=42,
    ).fit(X, y)

    d = m.to_dict()
    assert d["task"] == "regression"
    assert d["n_outputs"] == 1

    raw = _predict_raw(d, Xt)
    pred_rust = m.predict(Xt)
    np.testing.assert_allclose(raw[:, 0], pred_rust, atol=1e-12)


def test_to_dict_with_cat_features_binary() -> None:
    rng = np.random.default_rng(3)
    n = 300
    cat0 = rng.choice(["a", "b", "c"], size=n)
    cat1 = rng.choice(["x", "y"], size=n)
    num0 = rng.random(n)
    num1 = rng.random(n)
    X = np.column_stack([cat0, num0, cat1, num1])
    y = ((cat0 == "a").astype(float) + num0 > 0.8).astype(np.float64)

    enc = WoeEncoder(cat_features=[0, 2]).fit(X, y)
    m = RFGBoostClassifier(
        n_estimators=4,
        learning_rate=0.1,
        rf_n_estimators=5,
        rf_max_depth=4,
        random_state=42,
    ).fit(enc.transform(X), y)

    woe = enc.to_dict(n_features=4)
    assert woe["cat_features"] == [0, 2]
    assert woe["numeric_features"] == [1, 3]
    assert woe["woe_multiclass"] is False
    assert len(woe["woe_tables"]) == 2

    Xt = X[:40]
    encoded = _encode_with_woe({"woe": woe}, Xt)
    raw = _predict_raw(m.to_dict(), encoded)
    proba_py = _sigmoid(raw[:, 0])
    proba_rust = m.predict_proba(enc.transform(Xt))[:, 1]
    np.testing.assert_allclose(proba_py, proba_rust, atol=1e-12)


def test_to_dict_with_cat_features_multiclass() -> None:
    rng = np.random.default_rng(4)
    n = 300
    cat0 = rng.choice(["a", "b", "c", "d"], size=n)
    num0 = rng.random(n)
    X = np.column_stack([cat0, num0])
    y = (rng.choice([0, 1, 2], size=n)).astype(np.float64)

    enc = WoeEncoder(cat_features=[0]).fit(X, y)
    m = RFGBoostClassifier(
        n_estimators=3,
        learning_rate=0.1,
        rf_n_estimators=4,
        rf_max_depth=3,
        random_state=42,
    ).fit(enc.transform(X), y)

    woe = enc.to_dict(n_features=2)
    assert woe["woe_multiclass"] is True
    assert woe["n_woe_classes"] == 3
    assert len(woe["woe_tables"]) == 1
    assert len(woe["woe_tables"][0]) == 3

    Xt = X[:40]
    encoded = _encode_with_woe({"woe": woe}, Xt)
    raw = _predict_raw(m.to_dict(), encoded)
    proba_py = _softmax(raw)
    proba_rust = m.predict_proba(enc.transform(Xt))
    np.testing.assert_allclose(proba_py, proba_rust, atol=1e-12)


def test_to_dict_unfitted_raises() -> None:
    m = RFGBoostClassifier(n_estimators=2, random_state=0)
    with pytest.raises(ValueError, match="not been fitted"):
        m.to_dict()
