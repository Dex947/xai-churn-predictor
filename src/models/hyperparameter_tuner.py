"""
Hyperparameter tuning module for the Churn Prediction System.

This module implements intelligent hyperparameter optimization using
GridSearchCV and RandomizedSearchCV with cross-validation.

Following Global rules:
- Measure baseline before optimizing
- Use domain knowledge to define search spaces
- Validate improvements statistically
- Document all decisions and learnings
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier


class HyperparameterTuner:
    """Intelligent hyperparameter tuning for churn prediction models."""

    def __init__(
        self,
        method: str = "grid_search",
        cv_folds: int = 5,
        scoring: str = "f1",
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        """
        Initialize the HyperparameterTuner.

        Args:
            method: Tuning method ('grid_search' or 'random_search').
            cv_folds: Number of cross-validation folds.
            scoring: Scoring metric for optimization.
            n_jobs: Number of parallel jobs (-1 = all cores).
            random_state: Random seed for reproducibility.
        """
        self.method = method
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.best_models = {}
        self.best_params = {}
        self.cv_results = {}
        self.tuning_history = {}

    def get_param_grid(self, model_name: str) -> Dict[str, List[Any]]:
        """
        Get parameter grid for a specific model.

        Based on domain knowledge and baseline analysis.

        Args:
            model_name: Name of the model.

        Returns:
            Parameter grid dictionary.
        """
        if model_name == "logistic_regression":
            # Note: lbfgs only supports l2, liblinear supports l1/l2, saga supports l1/l2/elasticnet
            return {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["liblinear", "saga"],  # Removed lbfgs to avoid l1 incompatibility
                "penalty": ["l1", "l2"],
                "max_iter": [1000, 2000],
                "class_weight": ["balanced"],
            }

        elif model_name == "random_forest":
            return {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2"],
                "class_weight": ["balanced"],
            }

        elif model_name == "xgboost":
            return {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "scale_pos_weight": [1, 2, 3, 4],
                "gamma": [0, 0.1, 0.5, 1.0],
            }

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def get_base_model(self, model_name: str) -> Any:
        """
        Get base model instance.

        Args:
            model_name: Name of the model.

        Returns:
            Base model instance.
        """
        if model_name == "logistic_regression":
            return LogisticRegression(random_state=self.random_state)

        elif model_name == "random_forest":
            return RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)

        elif model_name == "xgboost":
            return XGBClassifier(random_state=self.random_state, n_jobs=self.n_jobs)

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def tune_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        param_grid: Dict[str, List[Any]] = None,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """
        Tune hyperparameters for a single model.

        Args:
            model_name: Name of the model.
            X_train: Training features.
            y_train: Training labels.
            param_grid: Custom parameter grid (optional).

        Returns:
            Tuple of (best_model, best_params, cv_results).
        """
        logger.info(f"Starting hyperparameter tuning for {model_name}")
        logger.info(f"Method: {self.method} | CV Folds: {self.cv_folds} | Scoring: {self.scoring}")

        # Get parameter grid
        if param_grid is None:
            param_grid = self.get_param_grid(model_name)

        logger.info(f"Parameter grid size: {self._get_grid_size(param_grid)} combinations")

        # Get base model
        base_model = self.get_base_model(model_name)

        # Setup cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Perform search
        if self.method == "grid_search":
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1,
                return_train_score=True,
            )
        elif self.method == "random_search":
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=50,  # Sample 50 combinations
                cv=cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1,
                random_state=self.random_state,
                return_train_score=True,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Fit search
        logger.info(f"Fitting {model_name}...")
        search.fit(X_train, y_train)

        # Extract results
        best_model = search.best_estimator_
        best_params = search.best_params_
        cv_results = {
            "best_score": search.best_score_,
            "best_params": best_params,
            "cv_results": pd.DataFrame(search.cv_results_),
        }

        # Store results
        self.best_models[model_name] = best_model
        self.best_params[model_name] = best_params
        self.cv_results[model_name] = cv_results

        # Log results
        logger.info(f"Best {self.scoring} score: {search.best_score_:.4f}")
        logger.info(f"Best parameters: {best_params}")

        # Check for overfitting
        train_score = search.cv_results_["mean_train_score"][search.best_index_]
        test_score = search.best_score_
        overfitting_gap = train_score - test_score

        if overfitting_gap > 0.05:
            logger.warning(
                f"Potential overfitting detected: "
                f"Train={train_score:.4f}, CV={test_score:.4f}, Gap={overfitting_gap:.4f}"
            )
        else:
            logger.info(f"Good generalization: Train={train_score:.4f}, CV={test_score:.4f}")

        return best_model, best_params, cv_results

    def tune_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        models: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for all specified models.

        Args:
            X_train: Training features.
            y_train: Training labels.
            models: List of model names to tune (None = all).

        Returns:
            Dictionary of tuned models.
        """
        if models is None:
            models = ["logistic_regression", "random_forest", "xgboost"]

        logger.info(f"Tuning {len(models)} models: {models}")

        for model_name in models:
            try:
                self.tune_model(model_name, X_train, y_train)
            except Exception as e:
                logger.error(f"Failed to tune {model_name}: {e}")

        logger.info(f"Hyperparameter tuning complete for {len(self.best_models)} models")

        return self.best_models

    def compare_with_baseline(
        self,
        baseline_results: Dict[str, Dict[str, float]],
    ) -> pd.DataFrame:
        """
        Compare tuned models with baseline performance.

        Args:
            baseline_results: Baseline model results.

        Returns:
            Comparison DataFrame.
        """
        comparison_data = []

        for model_name in self.cv_results.keys():
            if model_name in baseline_results:
                baseline_score = baseline_results[model_name].get(self.scoring, 0)
                tuned_score = self.cv_results[model_name]["best_score"]
                improvement = tuned_score - baseline_score
                improvement_pct = (improvement / baseline_score) * 100

                comparison_data.append(
                    {
                        "model": model_name,
                        "baseline_score": baseline_score,
                        "tuned_score": tuned_score,
                        "improvement": improvement,
                        "improvement_pct": improvement_pct,
                        "significant": improvement > 0.02,  # 2% threshold
                    }
                )

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("improvement", ascending=False)

        return comparison_df

    def save_results(self, output_dir: str) -> None:
        """
        Save tuning results to disk.

        Args:
            output_dir: Directory to save results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save best models
        for model_name, model in self.best_models.items():
            model_path = output_dir / f"{model_name}_tuned.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved tuned model: {model_path}")

        # Save best parameters
        params_path = output_dir / "best_parameters.json"
        import json

        with open(params_path, "w") as f:
            json.dump(self.best_params, f, indent=2)
        logger.info(f"Saved best parameters: {params_path}")

        # Save CV results
        for model_name, results in self.cv_results.items():
            cv_path = output_dir / f"{model_name}_cv_results.csv"
            results["cv_results"].to_csv(cv_path, index=False)
            logger.info(f"Saved CV results: {cv_path}")

    def _get_grid_size(self, param_grid: Dict[str, List[Any]]) -> int:
        """Calculate total number of parameter combinations."""
        size = 1
        for values in param_grid.values():
            size *= len(values)
        return size

    def generate_report(self) -> str:
        """
        Generate a comprehensive tuning report.

        Returns:
            Report string.
        """
        report_lines = [
            "=" * 80,
            "HYPERPARAMETER TUNING REPORT",
            "=" * 80,
            "",
            f"Method: {self.method}",
            f"CV Folds: {self.cv_folds}",
            f"Scoring Metric: {self.scoring}",
            "",
            "=" * 80,
            "BEST PARAMETERS",
            "=" * 80,
            "",
        ]

        for model_name, params in self.best_params.items():
            report_lines.append(f"\n{model_name.upper()}:")
            report_lines.append("-" * 40)
            for param, value in params.items():
                report_lines.append(f"  {param}: {value}")
            if model_name in self.cv_results:
                score = self.cv_results[model_name]["best_score"]
                report_lines.append(f"  Best CV Score: {score:.4f}")

        report_lines.extend(["", "=" * 80])

        return "\n".join(report_lines)
