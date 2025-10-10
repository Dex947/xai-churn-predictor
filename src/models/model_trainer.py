"""
Model training module for the Churn Prediction System.

This module handles training various machine learning models for churn prediction.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier


class ModelTrainer:
    """Train and manage machine learning models for churn prediction."""

    def __init__(self, config: Dict[str, Any] = None, random_state: int = 42):
        """
        Initialize the ModelTrainer.

        Args:
            config: Configuration dictionary containing model parameters.
            random_state: Random seed for reproducibility.
        """
        self.config = config or {}
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}

    def get_model(self, model_name: str, params: Dict[str, Any] = None) -> Any:
        """
        Get a machine learning model instance.

        Args:
            model_name: Name of the model ('logistic_regression', 'random_forest', 'xgboost').
            params: Model parameters. If None, uses config or defaults.

        Returns:
            Initialized model instance.
        """
        if params is None:
            model_config = self.config.get("models", {}).get(model_name, {})
            params = model_config.get("params", {})

        # Ensure random_state is set
        if "random_state" not in params:
            params["random_state"] = self.random_state

        logger.info(f"Initializing {model_name} with params: {params}")

        if model_name == "logistic_regression":
            model = LogisticRegression(**params)

        elif model_name == "random_forest":
            model = RandomForestClassifier(**params)

        elif model_name == "xgboost":
            # XGBoost uses different parameter names
            xgb_params = params.copy()
            # Map common parameter names
            if "n_jobs" in xgb_params:
                xgb_params["n_jobs"] = xgb_params.pop("n_jobs")
            model = XGBClassifier(**xgb_params)

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return model

    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        params: Dict[str, Any] = None,
    ) -> Any:
        """
        Train a single model.

        Args:
            model_name: Name of the model.
            X_train: Training features.
            y_train: Training labels.
            params: Model parameters.

        Returns:
            Trained model instance.
        """
        logger.info(f"Training {model_name}")

        model = self.get_model(model_name, params)

        # Train the model
        model.fit(X_train, y_train)

        # Store the trained model
        self.models[model_name] = model

        logger.info(f"{model_name} training completed")

        return model

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Train all enabled models from configuration.

        Args:
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Dictionary of trained models.
        """
        logger.info("Training all enabled models")

        models_config = self.config.get("models", {})
        trained_models = {}

        for model_name, model_config in models_config.items():
            if model_config.get("enabled", False):
                try:
                    model = self.train_model(
                        model_name,
                        X_train,
                        y_train,
                        params=model_config.get("params"),
                    )
                    trained_models[model_name] = model
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")

        logger.info(f"Trained {len(trained_models)} models successfully")

        return trained_models

    def cross_validate(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: np.ndarray,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
    ) -> Dict[str, float]:
        """
        Perform cross-validation on a model.

        Args:
            model_name: Name of the model.
            X: Features.
            y: Labels.
            cv_folds: Number of cross-validation folds.
            scoring: Scoring metric.

        Returns:
            Dictionary with cross-validation scores.
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation on {model_name}")

        if model_name not in self.models:
            model = self.get_model(model_name)
        else:
            model = self.models[model_name]

        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)

        cv_results = {
            "scores": scores,
            "mean": scores.mean(),
            "std": scores.std(),
            "min": scores.min(),
            "max": scores.max(),
        }

        self.cv_scores[model_name] = cv_results

        logger.info(
            f"{model_name} CV {scoring}: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})"
        )

        return cv_results

    def tune_hyperparameters(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        param_grid: Dict[str, List[Any]] = None,
        method: str = "grid_search",
        cv_folds: int = 5,
        scoring: str = "roc_auc",
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform hyperparameter tuning.

        Args:
            model_name: Name of the model.
            X_train: Training features.
            y_train: Training labels.
            param_grid: Parameter grid for tuning.
            method: Tuning method ('grid_search' or 'random_search').
            cv_folds: Number of cross-validation folds.
            scoring: Scoring metric.

        Returns:
            Tuple of (best_model, best_params).
        """
        logger.info(f"Tuning hyperparameters for {model_name} using {method}")

        # Get base model
        model = self.get_model(model_name)

        # Get parameter grid from config if not provided
        if param_grid is None:
            tuning_config = self.config.get("hyperparameter_tuning", {})
            param_grid = tuning_config.get(f"{model_name}_grid", {})

        if not param_grid:
            logger.warning(f"No parameter grid found for {model_name}")
            return model, {}

        # Perform search
        if method == "grid_search":
            search = GridSearchCV(
                model,
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
            )
        elif method == "random_search":
            search = RandomizedSearchCV(
                model,
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                n_iter=20,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown search method: {method}")

        # Fit the search
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_params = search.best_params_

        # Store results
        self.models[model_name] = best_model
        self.best_params[model_name] = best_params

        logger.info(f"Best {scoring}: {search.best_score_:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return best_model, best_params

    def get_feature_importance(
        self,
        model_name: str,
        feature_names: List[str] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Get feature importance from a trained model.

        Args:
            model_name: Name of the model.
            feature_names: List of feature names.
            top_n: Number of top features to return.

        Returns:
            DataFrame with feature importance scores.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")

        model = self.models[model_name]

        # Get feature importance based on model type
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            # For linear models, use absolute coefficients
            importance = np.abs(model.coef_[0])
        else:
            logger.warning(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()

        # Create DataFrame
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
            }
        )

        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False)

        # Get top N
        if top_n:
            importance_df = importance_df.head(top_n)

        logger.info(f"Feature importance calculated for {model_name}")

        return importance_df

    def predict(
        self,
        model_name: str,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Make predictions using a trained model.

        Args:
            model_name: Name of the model.
            X: Features to predict on.

        Returns:
            Array of predictions.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")

        model = self.models[model_name]
        predictions = model.predict(X)

        return predictions

    def predict_proba(
        self,
        model_name: str,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Get prediction probabilities using a trained model.

        Args:
            model_name: Name of the model.
            X: Features to predict on.

        Returns:
            Array of prediction probabilities.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")

        model = self.models[model_name]

        if not hasattr(model, "predict_proba"):
            raise ValueError(f"Model {model_name} does not support probability predictions")

        probabilities = model.predict_proba(X)

        return probabilities

    def save_model(self, model_name: str, file_path: str) -> None:
        """
        Save a trained model to disk.

        Args:
            model_name: Name of the model.
            file_path: Path to save the model.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        model = self.models[model_name]
        joblib.dump(model, file_path)

        logger.info(f"Model {model_name} saved to {file_path}")

    def load_model(self, model_name: str, file_path: str) -> Any:
        """
        Load a trained model from disk.

        Args:
            model_name: Name of the model.
            file_path: Path to load the model from.

        Returns:
            Loaded model instance.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        model = joblib.load(file_path)
        self.models[model_name] = model

        logger.info(f"Model {model_name} loaded from {file_path}")

        return model

    def save_all_models(self, output_dir: str) -> None:
        """
        Save all trained models to a directory.

        Args:
            output_dir: Directory to save models to.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for model_name in self.models:
            file_path = output_dir / f"{model_name}.joblib"
            self.save_model(model_name, str(file_path))

        logger.info(f"All models saved to {output_dir}")

    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models.

        Returns:
            DataFrame with model summary information.
        """
        summary_data = []

        for model_name, model in self.models.items():
            model_info = {
                "model_name": model_name,
                "model_type": type(model).__name__,
                "n_parameters": len(str(model.get_params())),
            }

            # Add CV scores if available
            if model_name in self.cv_scores:
                model_info["cv_mean"] = self.cv_scores[model_name]["mean"]
                model_info["cv_std"] = self.cv_scores[model_name]["std"]

            # Add best params if available
            if model_name in self.best_params:
                model_info["tuned"] = True
            else:
                model_info["tuned"] = False

            summary_data.append(model_info)

        summary_df = pd.DataFrame(summary_data)

        return summary_df
