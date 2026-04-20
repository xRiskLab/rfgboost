#!/usr/bin/env python3
"""Generate boosting iteration animations: Rust RFGBoost vs XGBoost vs LightGBM vs CatBoost."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import catboost as cb
import imageio.v3 as iio
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split

from rfgboost import RFGBoost

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
N_ITERATIONS = 30
FPS = 3

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({"font.size": 13})

# Colormaps
cm_reg = plt.cm.RdYlBu_r
cm_cls = plt.cm.RdYlBu_r


def generate_classification_animation():
    print("Generating classification animation...")
    np.random.seed(42)
    X = np.random.rand(500, 2) * 6 - 3
    y = (X[:, 1] > np.sin(X[:, 0])).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    cb_model = None
    frames = []

    for iteration in range(1, N_ITERATIONS + 1):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
        fig.subplots_adjust(top=0.85, bottom=0.08, left=0.08, right=0.92, wspace=0.3, hspace=0.3)
        axs = axs.ravel()

        results = []

        # Rust RFGBoost
        m = RFGBoost(
            n_estimators=iteration,
            learning_rate=0.1,
            rf_n_estimators=20,
            rf_max_depth=5,
            rf_max_features="sqrt",
            random_state=42,
            task="classification",
        )
        m.fit(X_train, y_train.astype(float))
        preds = np.array(m.predict_proba(grid))[:, 1].reshape(xx.shape)
        test_proba = np.array(m.predict_proba(X_test))[:, 1]
        ll = log_loss(y_test, test_proba)
        results.append(("Rust RFGBoost", preds, ll))

        # XGBoost
        m = xgb.XGBClassifier(
            n_estimators=iteration,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric="logloss",
        )
        m.fit(X_train, y_train)
        preds = m.predict_proba(grid)[:, 1].reshape(xx.shape)
        ll = log_loss(y_test, m.predict_proba(X_test)[:, 1])
        results.append(("XGBoost", preds, ll))

        # LightGBM
        m = lgb.LGBMClassifier(
            n_estimators=iteration,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=-1,
        )
        m.fit(X_train, y_train)
        preds = m.predict_proba(grid)[:, 1].reshape(xx.shape)
        ll = log_loss(y_test, m.predict_proba(X_test)[:, 1])
        results.append(("LightGBM", preds, ll))

        # CatBoost
        if iteration == 1:
            cb_model = cb.CatBoostClassifier(
                iterations=1,
                learning_rate=0.1,
                depth=5,
                verbose=0,
                random_seed=42,
                allow_writing_files=False,
            )
            cb_model.fit(X_train, y_train)
        else:
            cb_model = cb.CatBoostClassifier(
                iterations=iteration,
                learning_rate=0.1,
                depth=5,
                verbose=0,
                random_seed=42,
                allow_writing_files=False,
            )
            cb_model.fit(X_train, y_train)
        preds = cb_model.predict_proba(grid)[:, 1].reshape(xx.shape)
        ll = log_loss(y_test, cb_model.predict_proba(X_test)[:, 1])
        results.append(("CatBoost", preds, ll))

        for idx, (name, preds, ll) in enumerate(results):
            axs[idx].contourf(xx, yy, preds, alpha=0.5, levels=10, cmap=cm_cls)
            axs[idx].scatter(
                X_train[:, 0],
                X_train[:, 1],
                c=y_train,
                edgecolor="k",
                cmap=cm_cls,
                alpha=0.6,
                s=30,
            )
            axs[idx].set_title(name)
            axs[idx].text(
                0.02,
                0.03,
                f"Log Loss: {ll:.4f}",
                transform=axs[idx].transAxes,
                fontsize=11,
            )
            axs[idx].set_xlim(x_min, x_max)
            axs[idx].set_ylim(y_min, y_max)

        fig.suptitle(f"Classification (Boosting Iteration {iteration})", fontsize=22, y=0.96)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        frames.append(image)
        plt.close(fig)
        print(f"  frame {iteration}/{N_ITERATIONS}")

    out_path = OUTPUT_DIR / "classification_animation_grid.gif"
    iio.imwrite(out_path, frames, fps=FPS)
    print(f"Saved {out_path}")


def generate_regression_animation():
    print("Generating regression animation...")
    X, y = make_regression(n_samples=500, n_features=2, noise=0.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    cb_model = None
    frames = []

    for iteration in range(1, N_ITERATIONS + 1):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
        fig.subplots_adjust(top=0.85, bottom=0.08, left=0.08, right=0.92, wspace=0.3, hspace=0.3)
        axs = axs.ravel()

        results = []

        # Rust RFGBoost
        m = RFGBoost(
            n_estimators=iteration,
            learning_rate=0.1,
            rf_n_estimators=20,
            rf_max_depth=5,
            rf_max_features="sqrt",
            random_state=42,
            task="regression",
        )
        m.fit(X_train, y_train)
        preds = np.array(m.predict(grid)).reshape(xx.shape)
        mse = mean_squared_error(y_test, m.predict(X_test))
        results.append(("Rust RFGBoost", preds, mse))

        # XGBoost
        m = xgb.XGBRegressor(
            n_estimators=iteration,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
        m.fit(X_train, y_train)
        preds = m.predict(grid).reshape(xx.shape)
        mse = mean_squared_error(y_test, m.predict(X_test))
        results.append(("XGBoost", preds, mse))

        # LightGBM
        m = lgb.LGBMRegressor(
            n_estimators=iteration,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=-1,
        )
        m.fit(X_train, y_train)
        preds = m.predict(grid).reshape(xx.shape)
        mse = mean_squared_error(y_test, m.predict(X_test))
        results.append(("LightGBM", preds, mse))

        # CatBoost
        cb_model = cb.CatBoostRegressor(
            iterations=iteration,
            learning_rate=0.1,
            depth=5,
            verbose=0,
            random_seed=42,
            allow_writing_files=False,
        )
        cb_model.fit(X_train, y_train)
        preds = cb_model.predict(grid).reshape(xx.shape)
        mse = mean_squared_error(y_test, cb_model.predict(X_test))
        results.append(("CatBoost", preds, mse))

        for idx, (name, preds, mse) in enumerate(results):
            axs[idx].contourf(xx, yy, preds, alpha=0.5, levels=10, cmap=cm_reg)
            axs[idx].scatter(
                X_train[:, 0],
                X_train[:, 1],
                c=y_train,
                edgecolor="k",
                cmap=cm_reg,
                alpha=0.8,
                s=30,
            )
            axs[idx].set_title(name)
            axs[idx].text(
                0.02,
                0.03,
                f"MSE: {mse:.2f}",
                transform=axs[idx].transAxes,
                fontsize=11,
            )
            axs[idx].set_xlim(x_min, x_max)
            axs[idx].set_ylim(y_min, y_max)

        fig.suptitle(f"Regression (Boosting Iteration {iteration})", fontsize=22, y=0.96)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        frames.append(image)
        plt.close(fig)
        print(f"  frame {iteration}/{N_ITERATIONS}")

    out_path = OUTPUT_DIR / "regression_animation_grid.gif"
    iio.imwrite(out_path, frames, fps=FPS)
    print(f"Saved {out_path}")


def generate_rf_comparison_animation():
    """Rust RFGBoost (histogram) vs Rust RFGBoost (exact) vs XGBoost vs CatBoost."""

    print("Generating RF comparison animation...")
    np.random.seed(42)
    X = np.random.rand(500, 2) * 6 - 3
    y = (X[:, 1] > np.sin(X[:, 0])).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    n_steps = 30
    frames = []

    for step in range(1, n_steps + 1):
        n_rounds = step

        fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
        fig.subplots_adjust(top=0.85, bottom=0.08, left=0.08, right=0.92, wspace=0.3, hspace=0.3)
        axs = axs.ravel()

        results = []

        # Rust RFGBoost (histogram)
        m = RFGBoost(
            n_estimators=n_rounds,
            learning_rate=0.1,
            rf_n_estimators=20,
            rf_max_depth=5,
            rf_max_features="sqrt",
            random_state=42,
            task="classification",
            use_histogram=True,
        )
        m.fit(X_train, y_train.astype(float))
        preds = np.array(m.predict_proba(grid))[:, 1].reshape(xx.shape)
        test_proba = np.array(m.predict_proba(X_test))[:, 1]
        ll = log_loss(y_test, test_proba)
        results.append(("Rust RFGBoost (hist)", preds, ll))

        # Rust RFGBoost (exact)
        m = RFGBoost(
            n_estimators=n_rounds,
            learning_rate=0.1,
            rf_n_estimators=20,
            rf_max_depth=5,
            rf_max_features="sqrt",
            random_state=42,
            task="classification",
            use_histogram=False,
        )
        m.fit(X_train, y_train.astype(float))
        preds = np.array(m.predict_proba(grid))[:, 1].reshape(xx.shape)
        test_proba = np.array(m.predict_proba(X_test))[:, 1]
        ll = log_loss(y_test, test_proba)
        results.append(("Rust RFGBoost (exact)", preds, ll))

        # XGBoost
        m = xgb.XGBClassifier(
            n_estimators=n_rounds * 20,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric="logloss",
        )
        m.fit(X_train, y_train)
        preds = m.predict_proba(grid)[:, 1].reshape(xx.shape)
        ll = log_loss(y_test, m.predict_proba(X_test)[:, 1])
        results.append(("XGBoost", preds, ll))

        # CatBoost
        cb_model = cb.CatBoostClassifier(
            iterations=n_rounds * 20,
            learning_rate=0.1,
            depth=5,
            verbose=0,
            random_seed=42,
            allow_writing_files=False,
        )
        cb_model.fit(X_train, y_train)
        preds = cb_model.predict_proba(grid)[:, 1].reshape(xx.shape)
        ll = log_loss(y_test, cb_model.predict_proba(X_test)[:, 1])
        results.append(("CatBoost", preds, ll))

        for idx, (name, preds, ll) in enumerate(results):
            axs[idx].contourf(xx, yy, preds, alpha=0.5, levels=10, cmap=cm_cls)
            axs[idx].scatter(
                X_train[:, 0],
                X_train[:, 1],
                c=y_train,
                edgecolor="k",
                cmap=cm_cls,
                alpha=0.6,
                s=30,
            )
            axs[idx].set_title(name, fontsize=12)
            axs[idx].text(
                0.02,
                0.03,
                f"Log Loss: {ll:.4f}",
                transform=axs[idx].transAxes,
                fontsize=11,
            )
            axs[idx].set_xlim(x_min, x_max)
            axs[idx].set_ylim(y_min, y_max)

        fig.suptitle(f"Boosting Iteration {n_rounds}", fontsize=22, y=0.96)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        frames.append(image)
        plt.close(fig)
        print(f"  frame {step}/{n_steps}")

    out_path = OUTPUT_DIR / "rf_comparison_animation.gif"
    iio.imwrite(out_path, frames, fps=FPS)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    generate_classification_animation()
    generate_regression_animation()
    generate_rf_comparison_animation()
    print("Done.")
