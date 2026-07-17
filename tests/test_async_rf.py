"""async_mode: confidence-based early stopping for the standalone forests."""

from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from rfgboost import RandomForestClassifier, RandomForestRegressor


def test_default_builds_all_trees():
    X, y = make_classification(n_samples=400, n_features=8, random_state=0)
    clf = RandomForestClassifier(n_estimators=50, random_state=0)
    clf.fit(X, y.astype(float))
    assert clf.trees_used == 50
    reg = RandomForestRegressor(n_estimators=40, random_state=0)
    reg.fit(X, X[:, 0])
    assert reg.trees_used == 40


def test_async_classifier_stops_early_same_accuracy():
    X, y = make_classification(
        n_samples=1500, n_features=20, n_informative=8, n_classes=3, random_state=0
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y.astype(float), random_state=0)
    full = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0)
    full.fit(Xtr, ytr)
    asy = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0, async_mode=True)
    asy.fit(Xtr, ytr)

    assert full.trees_used == 300
    assert 3 <= asy.trees_used < 300  # stopped early, but built at least min_trees
    # accuracy stays close to the full forest
    assert accuracy_score(yte, asy.predict(Xte)) >= accuracy_score(yte, full.predict(Xte)) - 0.03


def test_async_regressor_stops_early_same_r2():
    X, y = make_regression(
        n_samples=1500, n_features=20, n_informative=10, noise=10.0, random_state=0
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
    full = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=0)
    full.fit(Xtr, ytr)
    asy = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=0, async_mode=True)
    asy.fit(Xtr, ytr)

    assert full.trees_used == 300
    assert 3 <= asy.trees_used < 300
    assert r2_score(yte, asy.predict(Xte)) >= r2_score(yte, full.predict(Xte)) - 0.05
