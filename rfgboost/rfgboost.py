"""rfgboost.py."""

import contextlib
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastwoe import FastWoe
from scipy.special import expit as sigmoid
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor

# Optional XGBoost import
try:
    import xgboost as xgb
    from xgboost import XGBRFRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class RFGBoost(BaseEstimator):
    """Random Forest Gradient Boosting (RFGBoost) model."""

    def __init__(
        self,
        n_estimators: int = 10,
        rf_params: Optional[dict[str, Any]] = None,
        learning_rate: float = 0.1,
        task: str = "regression",
        cat_features: Optional[list[str]] = None,
        woe_kwargs: Optional[dict[str, Any]] = None,
        base_learner: str = "sklearn",
    ):
        """RFGBoost constructor."""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        # Store original parameter value for sklearn compatibility
        self.rf_params: Optional[dict[str, Any]] = rf_params
        self.task = task
        self.models: list[Any] = []
        self.initial_pred: Optional[float] = None
        self.cat_features: Optional[list[str]] = cat_features
        self.woe_encoder: Optional[FastWoe] = None
        # Store original parameter value for sklearn compatibility
        self.woe_kwargs: Optional[dict[str, Any]] = woe_kwargs
        self.prior: Optional[float] = None
        self.label_mapping: Optional[dict[Any, int]] = None
        self.feature_names_: Optional[list[str]] = None

        # Base learner configuration
        self.base_learner = base_learner
        if base_learner == "xgboost" and not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )

        self._validate_base_learner_params()

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns:
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "rf_params": self.rf_params,
            "task": self.task,
            "cat_features": self.cat_features,
            "woe_kwargs": self.woe_kwargs,
            "base_learner": self.base_learner,
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Supports nested parameters using double underscore notation:
        - rf_params__max_depth sets max_depth inside rf_params dict
        - woe_kwargs__random_state sets random_state inside woe_kwargs dict

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns:
        -------
        self : estimator instance
            Estimator instance.
        """
        valid_params = self.get_params(deep=False)

        for key, value in params.items():
            # Handle nested parameters (e.g., rf_params__max_depth)
            if "__" in key:
                main_param, sub_param = key.split("__", 1)

                if main_param not in valid_params:
                    raise ValueError(
                        f"Invalid parameter {main_param} for estimator {type(self).__name__}. "
                        f"Valid parameters are: {sorted(valid_params.keys())}"
                    )

                # Get current value of main parameter (should be dict or None)
                current_value = getattr(self, main_param, None)
                if current_value is None or not isinstance(current_value, dict):
                    current_value = {}
                # Update the sub-parameter within the dict
                current_value[sub_param] = value
                setattr(self, main_param, current_value)

            elif key in valid_params:
                setattr(self, key, value)

            else:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {type(self).__name__}. "
                    f"Valid parameters are: {sorted(valid_params.keys())}"
                )
        # Re-validate parameters after setting
        self._validate_base_learner_params()

        return self

    def _get_rf_params(self):
        """Get a copy of rf_params, handling None case."""
        return (self.rf_params if self.rf_params is not None else {}).copy()

    def _get_woe_kwargs(self):
        """Get a copy of woe_kwargs, handling None case."""
        return (self.woe_kwargs if self.woe_kwargs is not None else {}).copy()

    def _validate_base_learner_params(self):
        """Validate and adjust parameters for the chosen base learner."""
        # Validate task
        if self.task not in ["regression", "classification"]:
            raise ValueError(
                f"Unknown task type: {self.task}. Use 'regression' or 'classification'"
            )

        if self.base_learner == "xgboost":
            # Get a working copy of rf_params
            if self.rf_params is None:
                self.rf_params = {}

            # Set XGBoost-specific defaults if not provided
            xgb_defaults = {
                "n_estimators": 100,  # XGBoost default
                "max_depth": 6,  # XGBoost default
                "learning_rate": 0.3,  # XGBoost default (different from our boosting LR)
                "subsample": 0.8,  # For randomness
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
            }
            # Only set defaults if not explicitly provided
            for key, value in xgb_defaults.items():
                if self.rf_params is not None and key not in self.rf_params:
                    self.rf_params[key] = value

    def _create_base_learner(self):
        """Create the appropriate base learner based on configuration."""
        rf_params = self._get_rf_params()

        # Filter parameters based on base learner to avoid invalid parameter errors
        if self.base_learner == "sklearn":
            # RandomForestRegressor doesn't accept XGBoost-specific parameters
            xgb_only_params = [
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "colsample_bylevel",
                "colsample_bynode",
                "reg_alpha",
                "reg_lambda",
                "scale_pos_weight",
            ]
            sklearn_params = {
                k: v for k, v in rf_params.items() if k not in xgb_only_params
            }
            return RandomForestRegressor(**sklearn_params)
        elif self.base_learner == "xgboost":
            # Use XGBRFRegressor for both regression and classification gradient fitting
            return XGBRFRegressor(**rf_params)
        else:
            raise ValueError(
                f"Unknown base_learner: {self.base_learner}. Use 'sklearn' or 'xgboost'"
            )

    def _ensure_numeric(self, y):
        """
        Ensure target values are numeric.

        Parameters:
        -----------
        y : array-like
            The target values.

        Returns:
        --------
        y_numeric : array-like
            Numeric representation of target values.
        """
        y = np.asarray(y)
        if y.dtype.kind in "OSU":  # Object, String, or Unicode
            unique_vals = np.unique(y)
            if len(unique_vals) == 2:  # Binary classification
                label_mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                self.label_mapping = label_mapping
                return np.array([label_mapping[val] for val in y])
        return y

    def _woe_encode(self, X, y=None, fit=False):  # pylint: disable=invalid-name
        """
        Weight of Evidence encoding for categorical features using FastWoe.

        Parameters:
        -----------
        X : DataFrame or ndarray
            The input samples.
        y : array-like, optional
            The target values. Only needed when fit=True.
        fit : bool, default=False
            Whether to fit the encoders or just transform.

        Returns:
        --------
        X_encoded : ndarray
            The transformed data.
        """
        if self.cat_features is None or len(self.cat_features) == 0:
            return np.array(X)

        # Ensure X is a DataFrame for easier handling of categorical features
        # pylint: disable=invalid-name
        if isinstance(X, pd.DataFrame):
            X_encoded = X.copy()
        elif hasattr(X, "columns") and hasattr(X, "iloc"):  # DataFrame-like object
            X_encoded = pd.DataFrame(X, columns=X.columns)
        else:
            # Numpy array or array-like without column names
            X_array = np.asarray(X)
            all_features = [str(i) for i in range(X_array.shape[1])]
            X_encoded = pd.DataFrame(X_array, columns=all_features)  # type: ignore

        if fit:
            self._fit_woe(y, X_encoded)
        elif self.woe_encoder is not None:
            # Extract categorical features only
            X_cat = X_encoded[self.cat_features]

            # Transform using fitted FastWoe encoder
            X_cat_woe = self.woe_encoder.transform(X_cat)  # type: ignore

            # Replace categorical columns with WOE values
            X_encoded[self.cat_features] = X_cat_woe

        # Convert to numpy array at the end
        return X_encoded.values

    def _fit_woe(self, y, X_encoded):  # pylint: disable=invalid-name
        """Fit the FastWoe encoder to categorical features."""
        # Ensure y is numeric before fitting
        y_numeric = self._ensure_numeric(y)
        self.prior = float(np.mean(y_numeric))  # type: ignore

        # Extract categorical features only
        X_cat = X_encoded[self.cat_features]  # pylint: disable=invalid-name

        # Initialize and fit FastWoe encoder
        woe_kwargs = self._get_woe_kwargs()
        woe_encoder = FastWoe(**woe_kwargs)
        woe_encoder.fit(X_cat, y_numeric)
        self.woe_encoder = woe_encoder

        # Transform categorical features to WOE
        X_cat_woe = self.woe_encoder.transform(X_cat)  # pylint: disable=invalid-name

        # Replace categorical columns with WOE values
        X_encoded[self.cat_features] = X_cat_woe

    # pylint: disable=invalid-name
    def inverse_woe_transform(self, X_woe, cat_columns=None):
        """
        Transform WOE values back to probabilities using FastWoe.

        Parameters:
        -----------
        X_woe : DataFrame or ndarray
            The WOE-encoded data.
        cat_columns : list, optional
            List of categorical columns to transform. Defaults to self.cat_features.

        Returns:
        --------
        X_proba : array-like
            The probability-encoded data.
        """
        if cat_columns is None:
            cat_columns = self.cat_features

        if cat_columns is None or len(cat_columns) == 0 or self.woe_encoder is None:
            return X_woe.copy()

        # Ensure X_woe is a DataFrame for easier handling
        if not isinstance(X_woe, pd.DataFrame):
            if hasattr(X_woe, "columns"):  # If it has column names
                X_woe = pd.DataFrame(X_woe, columns=X_woe.columns)
            else:
                # If numpy array without column names
                X_woe = pd.DataFrame(
                    X_woe,
                    columns=[f"feature_{i}" for i in range(X_woe.shape[1])],  # type: ignore
                )

        X_proba = X_woe.copy()

        # FastWoe directly provides methods to work with WOE scores
        if self.prior is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        odds_prior = self.prior / (1 - self.prior)

        for col in cat_columns:
            if col not in X_proba.columns:
                continue

            # Extract WOE values
            woe_values = X_proba[col].values
            exp_woe = np.exp(woe_values)
            odds_cat = odds_prior * exp_woe
            p = odds_cat / (1 + odds_cat)

            X_proba[col] = p

        return X_proba

    def get_woe_mappings(self, feature=None):
        """
        Get WOE mappings from the FastWoe encoder.

        Parameters:
        -----------
        feature : str, optional
            Specific feature to get mapping for. If None, returns all mappings.

        Returns:
        --------
        dict or DataFrame
            WOE mappings for specified feature or all features.
        """
        if self.woe_encoder is None:
            raise ValueError("Model has not been fitted with categorical features yet.")

        if feature is not None:
            return self.woe_encoder.get_mapping(feature)
        else:
            return self.woe_encoder.get_all_mappings()

    def get_woe_feature_stats(self, feature=None):
        """
        Get feature statistics from the FastWoe encoder.

        Parameters:
        -----------
        feature : str, optional
            Specific feature to get stats for. If None, returns stats for all features.

        Returns:
        --------
        DataFrame
            Feature statistics including Gini, IV, etc.
        """
        if self.woe_encoder is None:
            raise ValueError("Model has not been fitted with categorical features yet.")

        return self.woe_encoder.get_feature_stats(feature)

    def get_woe_feature_summary(self):
        """
        Get a summary of all categorical features ranked by predictive power.

        Returns:
        --------
        DataFrame
            Feature summary with Gini, IV, and number of categories.
        """
        if self.woe_encoder is None:
            raise ValueError("Model has not been fitted with categorical features yet.")

        return self.woe_encoder.get_feature_summary()

    def fit(self, X, y):
        """
        Fit the RFGBoost model.
        Parameters:
        -----------
        X : DataFrame or ndarray
            The input samples.
        y : array-like
            The target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        if self.learning_rate > 0.5:
            warnings.warn(
                f"Learning rate {self.learning_rate} is high and can cause instability. "
                "Typical values are 0.01 to 0.2.",
                stacklevel=2,
            )
        # Ensure y is numeric
        y_numeric = self._ensure_numeric(y)

        # Set feature names first, regardless of categorical features
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        elif hasattr(X, "columns") and hasattr(X, "iloc"):
            self.feature_names_ = list(X.columns)
        else:
            X_array = np.asarray(X)
            self.feature_names_ = [f"feature_{i}" for i in range(X_array.shape[1])]

        # Apply WOE encoding if categorical features are specified
        if self.cat_features is not None and len(self.cat_features) > 0:
            X_encoded = self._woe_encode(X, y_numeric, fit=True)
        else:
            X_encoded = np.array(X)

        n_samples = X_encoded.shape[0]

        # Initialize predictions
        if self.task == "regression":
            self.initial_pred = float(np.mean(y_numeric))  # type: ignore
            pred = np.full(n_samples, self.initial_pred)
        elif self.task == "classification":
            self.initial_pred = np.log(
                float(np.mean(y_numeric)) / (1 - float(np.mean(y_numeric)))  # type: ignore
            )  # Logit for binary classification
            pred = np.full(n_samples, self.initial_pred)

        self.models = []
        update = np.zeros(n_samples)

        for _ in range(self.n_estimators):
            # Compute residuals
            if self.task == "regression":
                residuals = y_numeric - pred
                rf = self._create_base_learner()
                rf.fit(X_encoded, residuals)
                update = rf.predict(X_encoded)
            elif self.task == "classification":
                p = sigmoid(pred)
                eps = 1e-5
                p = np.clip(p, eps, 1 - eps)
                variance = p * (1 - p)
                residuals = (y_numeric - p) / (
                    p * (1 - p)
                )  # Working response (FHT2000)
                rf = self._create_base_learner()
                rf.fit(X_encoded, residuals, sample_weight=variance)
                update = rf.predict(X_encoded)

            self.models.append(rf)
            pred += self.learning_rate * update  # type: ignore

        return self

    def predict(self, X):
        """
        Predict using the fitted model.

        For classification: returns class labels (0 or 1)
        For regression: returns continuous values

        Parameters:
        -----------
        X : DataFrame or ndarray
            The input samples.

        Returns:
        --------
        pred : array-like
            The predicted class labels (classification) or values (regression).
        """
        # Apply WOE encoding for prediction
        if self.cat_features is not None and len(self.cat_features) > 0:
            X_encoded = self._woe_encode(X, fit=False)
        else:
            X_encoded = np.array(X)

        pred = np.full(X_encoded.shape[0], self.initial_pred)

        for rf in self.models:
            pred += self.learning_rate * rf.predict(X_encoded)

        if self.task != "classification":
            # For regression, return continuous values
            return pred
        # For classification, return class labels (0 or 1)
        probabilities = sigmoid(pred)
        return (probabilities > 0.5).astype(int)

    def predict_proba(self, X):
        """
        Predict class probabilities using the fitted model.

        Parameters:
        -----------
        X : DataFrame or ndarray
            The input samples.

        Returns:
        --------
        proba : array-like
            The predicted class probabilities. Shape (n_samples, 2) for classification.
        """
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification tasks")

        # Apply WOE encoding for prediction
        if self.cat_features is not None and len(self.cat_features) > 0:
            X_encoded = self._woe_encode(X, fit=False)
        else:
            X_encoded = np.array(X)

        pred = np.full(X_encoded.shape[0], self.initial_pred)

        for rf in self.models:
            pred += self.learning_rate * rf.predict(X_encoded)

        # Convert logits to probabilities
        probabilities = sigmoid(pred)
        return np.column_stack((1 - probabilities, probabilities))

    def get_feature_importance(self):
        """
        Extract feature importance from the fitted RFGBoost model.

        Returns:
        --------
        feature_importance_df : DataFrame
            Feature names and their average importance across all internal RF models.
        """
        if not self.models:
            raise ValueError("The model has not been fitted yet.")

        if self.feature_names_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        feature_importance = np.zeros(len(self.feature_names_))

        for rf_model in self.models:
            if self.base_learner in {"sklearn", "xgboost"}:
                importance = rf_model.feature_importances_
            else:
                raise ValueError(f"Unknown base_learner: {self.base_learner}")

            feature_importance += importance

        feature_importance /= len(self.models)

        feature_importance_df = pd.DataFrame(
            {"Feature": self.feature_names_, "Importance": feature_importance}
        )

        return feature_importance_df.sort_values(
            "Importance", ascending=False
        ).reset_index(drop=True)

    def predict_ci(self, X, alpha=0.05):
        """
        Compute confidence intervals for predicted logits or probabilities.

        Parameters:
            X (DataFrame or array): Input features.
            alpha (float): Significance level (default 0.05 for 95% CI).

        Returns:
            np.ndarray: [lower_bound, upper_bound] for each sample.
        """
        if self.cat_features:
            X_encoded = self._woe_encode(X, fit=False)
        else:
            X_encoded = np.array(X)

        pred_mean = np.full(X_encoded.shape[0], self.initial_pred)
        pred_var = np.zeros(X_encoded.shape[0])

        for rf in self.models:
            if self.base_learner == "sklearn":
                # Get predictions from each tree in RF
                tree_preds = np.array(
                    [tree.predict(X_encoded) for tree in rf.estimators_]
                )
                tree_mean = tree_preds.mean(axis=0)
                tree_var = tree_preds.var(axis=0)

            elif self.base_learner == "xgboost":
                rf_pred = rf.predict(X_encoded)

                booster = rf.get_booster()
                X_dmatrix = xgb.DMatrix(X_encoded)

                # Get leaf indices for each sample and tree
                leaf_indices = booster.predict(X_dmatrix, pred_leaf=True)
                tree_df = booster.trees_to_dataframe()

                tree_preds = []
                for tree_id in range(rf.n_estimators):
                    tree_subset = tree_df[tree_df["Tree"] == tree_id]
                    if tree_subset.empty:
                        continue

                    leaf_nodes = tree_subset[tree_subset["Feature"] == "Leaf"]
                    leaf_to_value = dict(
                        zip(leaf_nodes["Node"], leaf_nodes["Gain"], strict=False)
                    )

                    sample_leaf_indices = (
                        leaf_indices[:, tree_id]
                        if leaf_indices.ndim > 1
                        else leaf_indices
                    )
                    tree_pred = np.array(
                        [
                            leaf_to_value.get(int(idx), 0.0)  # type: ignore
                            for idx in sample_leaf_indices
                        ]
                    )
                    tree_preds.append(tree_pred)  # type: ignore

                tree_preds = np.array(tree_preds)
                tree_mean = rf_pred
                tree_var = tree_preds.var(axis=0)

            else:
                raise ValueError(f"Unknown base_learner: {self.base_learner}")

            pred_mean += self.learning_rate * tree_mean
            pred_var += self.learning_rate * tree_var

        z_crit = norm.ppf(1 - alpha / 2)
        std = np.sqrt(pred_var)

        if self.task == "classification":
            # Delta method
            proba = sigmoid(pred_mean)
            grad = proba * (1 - proba)
            proba_std = grad * std
            lower = np.clip(proba - z_crit * proba_std, 0, 1)
            upper = np.clip(proba + z_crit * proba_std, 0, 1)
        elif self.task == "regression":
            lower = pred_mean - z_crit * std
            upper = pred_mean + z_crit * std
        else:
            raise ValueError(f"Unknown task type: {self.task}")

        return np.vstack([lower, upper]).T

    def _extract_xgboost_tree_data(self):
        """
        Extract tree data specifically for XGBoost models.
        Returns structured information about trees, nodes, and decision paths.
        """
        if self.base_learner != "xgboost":
            raise ValueError("This method only works with XGBoost base learner")

        tree_data = []

        for round_idx, rf_model in enumerate(self.models):
            # Get the booster from XGBRFRegressor
            booster = rf_model.get_booster()

            # Use XGBoost's built-in trees_to_dataframe method
            tree_df = booster.trees_to_dataframe()

            # Process each tree in this round
            for tree_id in tree_df["Tree"].unique():
                tree_subset = tree_df[tree_df["Tree"] == tree_id]

                # Extract leaf nodes with their information
                leaf_nodes = tree_subset[tree_subset["Feature"] == "Leaf"]

                for _, leaf_row in leaf_nodes.iterrows():
                    # Get path to this leaf by traversing up the tree
                    path_conditions = self._get_xgboost_path_to_leaf(
                        tree_subset, leaf_row["ID"]
                    )

                    tree_data.append(
                        {
                            "Round": round_idx,
                            "Tree": tree_id,
                            "NodeID": leaf_row["ID"],
                            "PathCondition": path_conditions,
                            "Samples": leaf_row.get(
                                "Count", 0
                            ),  # XGBoost may not have sample counts
                            "Value": leaf_row["Gain"],  # Leaf value/prediction
                            "Depth": leaf_row.get("Depth", 0),
                        }
                    )

        return pd.DataFrame(tree_data)

    def _get_xgboost_path_to_leaf(self, tree_df, leaf_id):
        """
        Reconstruct the path conditions to reach a specific leaf in XGBoost tree.
        """
        conditions = []
        current_id = leaf_id

        # Build a parent mapping
        parent_map = {}
        for _, row in tree_df.iterrows():
            if row["Feature"] != "Leaf":  # Decision nodes
                parent_map[row["Yes"]] = (row["ID"], "<=", row["Feature"], row["Split"])
                parent_map[row["No"]] = (row["ID"], ">", row["Feature"], row["Split"])

        # Traverse up to root
        while current_id in parent_map:
            parent_id, operator, feature, threshold = parent_map[current_id]
            if pd.notna(feature) and pd.notna(threshold):
                # Map XGBoost feature names (f0, f1, f2...) to actual feature names
                actual_feature_name = self._map_xgboost_feature_name(feature)
                condition = f"{actual_feature_name} {operator} {threshold:.4f}"
                conditions.insert(0, condition)
            current_id = parent_id

        return " and ".join(conditions) if conditions else None

    def _map_xgboost_feature_name(self, xgb_feature_name):
        """
        Map XGBoost feature names (f0, f1, f2...) to actual feature names.
        """
        if isinstance(xgb_feature_name, str) and xgb_feature_name.startswith("f"):
            with contextlib.suppress(ValueError, IndexError):
                # Extract the index from 'f0', 'f1', etc.
                feature_idx = int(xgb_feature_name[1:])
                if self.feature_names_ is not None and 0 <= feature_idx < len(
                    self.feature_names_
                ):
                    return self.feature_names_[feature_idx]
        # Return original if mapping fails
        return xgb_feature_name

    def extract_tree_data_with_conditions(self, tree_list=None, feature_names=None):
        """
        Extracts detailed information about all trees in the RFGBoost ensemble,
        including split conditions and child relationships.
        Returns a DataFrame with all nodes (splits and leaves).
        Columns: Round, Tree, NodeID, ...
        """
        if self.base_learner == "xgboost":
            # Use XGBoost-specific extraction
            return self._extract_xgboost_tree_data()
        elif self.base_learner == "sklearn":
            # Use original sklearn logic
            if tree_list is None:
                tree_list = self.models
            if feature_names is None:
                feature_names = self.feature_names_
            tree_data = []
            for round_idx, rf in enumerate(tree_list):
                for tree_idx, tree in enumerate(rf.estimators_):
                    tree_ = tree.tree_
                    for node_id in range(tree_.node_count):
                        value = tree_.value[node_id][0, 0]
                        impurity = tree_.impurity[node_id]
                        samples = tree_.n_node_samples[node_id]
                        if (
                            tree_.children_left[node_id]
                            != tree_.children_right[node_id]
                        ):
                            left_condition = "<="
                            right_condition = ">"
                            feature = (
                                feature_names[tree_.feature[node_id]]
                                if feature_names is not None
                                else f"feature_{tree_.feature[node_id]}"
                            )
                            threshold = tree_.threshold[node_id]
                            tree_data.extend(
                                [
                                    {
                                        "Round": round_idx,
                                        "Tree": tree_idx,
                                        "NodeID": node_id,
                                        "Feature": feature,
                                        "Condition": left_condition,
                                        "Threshold": threshold,
                                        "Impurity": impurity,
                                        "Samples": samples,
                                        "Value": value,
                                        "ChildType": "Left",
                                        "ChildNodeID": tree_.children_left[node_id],
                                    },
                                    {
                                        "Round": round_idx,
                                        "Tree": tree_idx,
                                        "NodeID": node_id,
                                        "Feature": feature,
                                        "Condition": right_condition,
                                        "Threshold": threshold,
                                        "Impurity": impurity,
                                        "Samples": samples,
                                        "Value": value,
                                        "ChildType": "Right",
                                        "ChildNodeID": tree_.children_right[node_id],
                                    },
                                ]
                            )
                        else:
                            tree_data.append(
                                {
                                    "Round": round_idx,
                                    "Tree": tree_idx,
                                    "NodeID": node_id,
                                    "Feature": "Leaf",
                                    "Condition": None,
                                    "Threshold": None,
                                    "Impurity": impurity,
                                    "Samples": samples,
                                    "Value": value,
                                    "ChildType": None,
                                    "ChildNodeID": None,
                                }
                            )
            return pd.DataFrame(tree_data)
        else:
            raise ValueError(f"Unknown base_learner: {self.base_learner}")

    # pylint: disable=cell-var-from-loop
    def extract_leaf_nodes_with_conditions(self, tree_list=None, feature_names=None):
        """
        Extracts detailed information about leaf nodes and their corresponding path conditions for all trees in the RFGBoost ensemble.
        Returns a DataFrame with leaves only and their path conditions.
        """
        if self.base_learner == "xgboost":
            # For XGBoost, extract_tree_data_with_conditions already returns leaf-only data
            return self._extract_xgboost_tree_data()
        elif self.base_learner == "sklearn":
            # Use original sklearn logic
            if tree_list is None:
                tree_list = self.models
            if feature_names is None:
                feature_names = self.feature_names_
            leaf_data = []
            for round_idx, rf in enumerate(tree_list):
                for tree_idx, tree in enumerate(rf.estimators_):
                    tree_ = tree.tree_

                    def trace_conditions(node_id, path_conditions):
                        value = tree_.value[node_id][0, 0]
                        impurity = tree_.impurity[node_id]
                        samples = tree_.n_node_samples[node_id]
                        if (
                            tree_.children_left[node_id]
                            == tree_.children_right[node_id]
                        ):
                            leaf_data.append(
                                {
                                    "Round": round_idx,
                                    "Tree": tree_idx,
                                    "NodeID": node_id,
                                    "PathCondition": " and ".join(path_conditions)
                                    if path_conditions
                                    else None,
                                    "Impurity": impurity,
                                    "Samples": samples,
                                    "Value": value,
                                }
                            )
                        else:
                            feature = (
                                feature_names[tree_.feature[node_id]]
                                if feature_names is not None
                                else f"feature_{tree_.feature[node_id]}"
                            )
                            threshold = tree_.threshold[node_id]
                            trace_conditions(
                                tree_.children_left[node_id],
                                path_conditions + [f"{feature} <= {threshold:.4f}"],
                            )
                            trace_conditions(
                                tree_.children_right[node_id],
                                path_conditions + [f"{feature} > {threshold:.4f}"],
                            )

                    trace_conditions(0, [])
            return pd.DataFrame(leaf_data)
        else:
            raise ValueError(f"Unknown base_learner: {self.base_learner}")

    def trees_to_dataframe(
        self, X: pd.DataFrame | None = None, y: pd.Series | None = None
    ):
        """
        Returns a DataFrame for each leaf node with path conditions.
        If X and y are provided, also includes event/non-event counts.
        Columns always: Round, Tree, NodeID, PathCondition, Samples, Value
        If X and y are provided: adds Events, NonEvents, EventRate
        """
        if self.base_learner == "xgboost":
            return self._extract_from_xgboost_trees_to_dataframe(X, y)
        elif self.base_learner == "sklearn":
            # Original sklearn implementation
            leaf_data = self.extract_leaf_nodes_with_conditions()
            results = []
            # If X and y are provided, preprocess and calculate event stats
            if X is not None and y is not None:
                X_encoded = self._woe_encode(X, fit=False)
                if self.feature_names_ is not None:
                    X_encoded = pd.DataFrame(X_encoded, columns=self.feature_names_)  # type: ignore
                else:
                    X_encoded = pd.DataFrame(
                        X_encoded,
                        columns=[f"feature_{i}" for i in range(X_encoded.shape[1])],  # type: ignore
                    )
                for _, row in leaf_data.iterrows():
                    path_cond = row["PathCondition"]
                    if path_cond is None:
                        mask = pd.Series([True] * len(X_encoded))
                    else:
                        try:
                            mask = X_encoded.query(path_cond).index  # type: ignore
                        except (ValueError, KeyError) as e:
                            print(
                                f"Skipping leaf {row['NodeID']} due to query error: {e}"
                            )
                            continue
                    y_leaf = y.loc[mask] if isinstance(y, pd.Series) else y[mask]
                    n_class_1 = (y_leaf == 1).sum()
                    n_class_0 = (y_leaf == 0).sum()
                    total = n_class_0 + n_class_1
                    event_rate = n_class_1 / total if total > 0 else float("nan")
                    results.append(
                        {
                            "Round": row["Round"],
                            "Tree": row["Tree"],
                            "NodeID": row["NodeID"],
                            "PathCondition": path_cond,
                            "Samples": row["Samples"],
                            "Value": row["Value"],
                            "Events": n_class_1,
                            "NonEvents": n_class_0,
                            "EventRate": event_rate,
                        }
                    )
            else:
                # Only report what we have from the tree structure
                results.extend(
                    {
                        "Round": row["Round"],
                        "Tree": row["Tree"],
                        "NodeID": row["NodeID"],
                        "PathCondition": row["PathCondition"],
                        "Samples": row["Samples"],
                        "Value": row["Value"],
                    }
                    for _, row in leaf_data.iterrows()
                )
            return pd.DataFrame(results)
        else:
            raise ValueError(f"Unknown base_learner: {self.base_learner}")

    def _extract_from_xgboost_trees_to_dataframe(self, X, y):
        # XGBoost implementation
        leaf_data = self._extract_xgboost_tree_data()

        if X is None or y is None:
            # Return basic structure without event analysis
            return leaf_data[
                ["Round", "Tree", "NodeID", "PathCondition", "Samples", "Value"]
            ]

        # Apply WOE encoding for data analysis
        X_encoded = self._woe_encode(X, fit=False)
        if self.feature_names_ is not None:
            X_encoded = pd.DataFrame(X_encoded, columns=self.feature_names_)  # type: ignore
        else:
            X_encoded = pd.DataFrame(
                X_encoded,
                columns=[f"feature_{i}" for i in range(X_encoded.shape[1])],  # type: ignore
            )
        results = []
        for _, row in leaf_data.iterrows():
            path_cond = row["PathCondition"]
            if path_cond is None:
                mask = pd.Series([True] * len(X_encoded))
            else:
                try:
                    # For XGBoost, we need to be careful with feature names
                    # and ensure the query works with the encoded data
                    mask = X_encoded.query(path_cond).index  # type: ignore
                except (
                    ValueError,
                    KeyError,
                    pd.errors.UndefinedVariableError,
                ) as e:
                    # XGBoost path conditions might reference features differently
                    # Skip problematic conditions with a warning
                    warnings.warn(
                        f"Skipping leaf {row['NodeID']} due to query error: {e}",
                        stacklevel=2,
                    )
                    continue

            y_leaf = y.loc[mask] if isinstance(y, pd.Series) else y[mask]
            n_class_1 = (y_leaf == 1).sum()
            n_class_0 = (y_leaf == 0).sum()
            total = n_class_0 + n_class_1
            event_rate = n_class_1 / total if total > 0 else float("nan")

            results.append(
                {
                    "Round": row["Round"],
                    "Tree": row["Tree"],
                    "NodeID": row["NodeID"],
                    "PathCondition": path_cond,
                    "Samples": row["Samples"],
                    "Value": row["Value"],
                    "Events": n_class_1,
                    "NonEvents": n_class_0,
                    "EventRate": event_rate,
                }
            )
        return pd.DataFrame(results)
