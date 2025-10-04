"""
Explainability module for the Churn Prediction System.

This module provides SHAP and LIME explanations for model predictions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from loguru import logger


class ModelExplainer:
    """Generate explanations for churn prediction models."""

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        feature_names: List[str] = None,
        class_names: List[str] = None,
    ):
        """
        Initialize the ModelExplainer.

        Args:
            model: Trained model to explain.
            X_train: Training data (used as background for SHAP).
            feature_names: List of feature names.
            class_names: List of class names (e.g., ['No Churn', 'Churn']).
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or X_train.columns.tolist()
        self.class_names = class_names if class_names is not None else ["No Churn", "Churn"]

        # Convert numpy array to list if needed
        if hasattr(self.class_names, 'tolist'):
            self.class_names = self.class_names.tolist()

        # Initialize explainers (lazy loading)
        self._shap_explainer = None
        self._lime_explainer = None
        self._shap_values = None

    def _get_shap_explainer(self, n_samples: int = 100) -> shap.Explainer:
        """
        Get or create SHAP explainer.

        Args:
            n_samples: Number of background samples for SHAP.

        Returns:
            SHAP explainer instance.
        """
        if self._shap_explainer is None:
            logger.info(f"Initializing SHAP explainer with {n_samples} background samples")

            # Sample background data for efficiency
            if len(self.X_train) > n_samples:
                background_data = shap.sample(self.X_train, n_samples, random_state=42)
            else:
                background_data = self.X_train

            # Use TreeExplainer for tree-based models, otherwise use KernelExplainer
            try:
                # Try TreeExplainer first (faster for tree-based models)
                self._shap_explainer = shap.TreeExplainer(self.model)
                logger.info("Using SHAP TreeExplainer")
            except Exception:
                try:
                    # Fall back to Explainer (auto-detects best explainer)
                    self._shap_explainer = shap.Explainer(self.model, background_data)
                    logger.info("Using SHAP Explainer (auto-detected)")
                except Exception as e:
                    logger.warning(f"SHAP auto-detection failed: {e}. Using KernelExplainer")
                    # Fall back to KernelExplainer (slower but works for any model)
                    self._shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba, background_data
                    )
                    logger.info("Using SHAP KernelExplainer")

        return self._shap_explainer

    def _get_lime_explainer(self) -> LimeTabularExplainer:
        """
        Get or create LIME explainer.

        Returns:
            LIME explainer instance.
        """
        if self._lime_explainer is None:
            logger.info("Initializing LIME explainer")

            self._lime_explainer = LimeTabularExplainer(
                self.X_train.values,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode="classification",
                random_state=42,
            )

            logger.info("LIME explainer initialized")

        return self._lime_explainer

    def compute_shap_values(
        self,
        X: pd.DataFrame,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """
        Compute SHAP values for the given data.

        Args:
            X: Data to explain.
            n_samples: Number of background samples.

        Returns:
            SHAP values array.
        """
        logger.info(f"Computing SHAP values for {len(X)} samples")

        explainer = self._get_shap_explainer(n_samples)

        try:
            shap_values = explainer.shap_values(X)

            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For binary classification, use values for positive class
                shap_values = shap_values[1]

            self._shap_values = shap_values
            logger.info("SHAP values computed successfully")

            return shap_values

        except Exception as e:
            logger.error(f"Failed to compute SHAP values: {e}")
            raise

    def plot_shap_summary(
        self,
        X: pd.DataFrame,
        shap_values: np.ndarray = None,
        max_display: int = 20,
        plot_type: str = "dot",
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot SHAP summary plot.

        Args:
            X: Data that was explained.
            shap_values: SHAP values. If None, uses cached values.
            max_display: Maximum number of features to display.
            plot_type: Type of plot ('dot', 'bar', 'violin').
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        if shap_values is None:
            shap_values = self._shap_values

        if shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")

        logger.info("Generating SHAP summary plot")

        plt.figure(figsize=(12, 8))

        # Handle 3D SHAP values (for multi-class/multi-output)
        plot_shap_values = shap_values
        if len(shap_values.shape) == 3:
            # For binary classification, use positive class
            plot_shap_values = shap_values[:, :, 1]

        shap.summary_plot(
            plot_shap_values,
            X,
            max_display=max_display,
            plot_type=plot_type,
            show=False,
        )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"SHAP summary plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_shap_waterfall(
        self,
        instance_idx: int,
        X: pd.DataFrame,
        shap_values: np.ndarray = None,
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot SHAP waterfall plot for a single instance.

        Args:
            instance_idx: Index of the instance to explain.
            X: Data that was explained.
            shap_values: SHAP values. If None, uses cached values.
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        if shap_values is None:
            shap_values = self._shap_values

        if shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")

        logger.info(f"Generating SHAP waterfall plot for instance {instance_idx}")

        # Create explanation object
        explainer = self._get_shap_explainer()

        if hasattr(explainer, 'expected_value'):
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1]  # Positive class
        else:
            base_value = 0

        explanation = shap.Explanation(
            values=shap_values[instance_idx],
            base_values=base_value,
            data=X.iloc[instance_idx].values,
            feature_names=self.feature_names,
        )

        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, show=False)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"SHAP waterfall plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_shap_force(
        self,
        instance_idx: int,
        X: pd.DataFrame,
        shap_values: np.ndarray = None,
        matplotlib: bool = True,
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot SHAP force plot for a single instance.

        Args:
            instance_idx: Index of the instance to explain.
            X: Data that was explained.
            shap_values: SHAP values. If None, uses cached values.
            matplotlib: Whether to use matplotlib (True) or HTML (False).
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        if shap_values is None:
            shap_values = self._shap_values

        if shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")

        logger.info(f"Generating SHAP force plot for instance {instance_idx}")

        explainer = self._get_shap_explainer()

        if hasattr(explainer, 'expected_value'):
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1]  # Positive class
        else:
            base_value = 0

        if matplotlib:
            plt.figure(figsize=(20, 3))
            shap.plots.force(
                base_value,
                shap_values[instance_idx],
                X.iloc[instance_idx],
                matplotlib=True,
                show=False,
            )

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                logger.info(f"SHAP force plot saved to {save_path}")

            if show:
                plt.show()
            else:
                plt.close()
        else:
            # HTML version
            force_plot = shap.force_plot(
                base_value,
                shap_values[instance_idx],
                X.iloc[instance_idx],
            )

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                shap.save_html(str(save_path), force_plot)
                logger.info(f"SHAP force plot saved to {save_path}")

    def plot_shap_dependence(
        self,
        feature: str,
        X: pd.DataFrame,
        shap_values: np.ndarray = None,
        interaction_feature: str = "auto",
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot SHAP dependence plot for a feature.

        Args:
            feature: Feature name to plot.
            X: Data that was explained.
            shap_values: SHAP values. If None, uses cached values.
            interaction_feature: Feature to color by. 'auto' for automatic selection.
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        if shap_values is None:
            shap_values = self._shap_values

        if shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")

        logger.info(f"Generating SHAP dependence plot for {feature}")

        plt.figure(figsize=(12, 8))

        shap.dependence_plot(
            feature,
            shap_values,
            X,
            interaction_index=interaction_feature,
            show=False,
        )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"SHAP dependence plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def explain_instance_lime(
        self,
        instance: Union[np.ndarray, pd.Series],
        n_features: int = 10,
        n_samples: int = 5000,
    ) -> Any:
        """
        Explain a single instance using LIME.

        Args:
            instance: Instance to explain.
            n_features: Number of features to include in explanation.
            n_samples: Number of samples for LIME.

        Returns:
            LIME explanation object.
        """
        logger.info("Generating LIME explanation")

        explainer = self._get_lime_explainer()

        if isinstance(instance, pd.Series):
            instance = instance.values

        explanation = explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=n_features,
            num_samples=n_samples,
        )

        logger.info("LIME explanation generated")

        return explanation

    def plot_lime_explanation(
        self,
        explanation: Any,
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot LIME explanation.

        Args:
            explanation: LIME explanation object.
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        logger.info("Plotting LIME explanation")

        fig = explanation.as_pyplot_figure()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"LIME explanation saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def get_lime_explanation_df(self, explanation: Any) -> pd.DataFrame:
        """
        Convert LIME explanation to DataFrame.

        Args:
            explanation: LIME explanation object.

        Returns:
            DataFrame with feature contributions.
        """
        exp_list = explanation.as_list()

        df = pd.DataFrame(exp_list, columns=["feature", "contribution"])
        df = df.sort_values("contribution", key=abs, ascending=False)

        return df

    def get_feature_importance_shap(
        self,
        shap_values: np.ndarray = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Get feature importance from SHAP values.

        Args:
            shap_values: SHAP values. If None, uses cached values.
            top_n: Number of top features to return.

        Returns:
            DataFrame with feature importance.
        """
        if shap_values is None:
            shap_values = self._shap_values

        if shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")

        # Calculate mean absolute SHAP values
        # Handle different SHAP value shapes
        if len(shap_values.shape) == 3:
            # Multi-dimensional output (e.g., [samples, features, classes])
            # Use positive class (index 1) for binary classification
            mean_abs_shap = np.abs(shap_values[:, :, 1]).mean(axis=0)
        elif len(shap_values.shape) == 2:
            # Standard 2D shape [samples, features]
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

        # Flatten if necessary
        if len(mean_abs_shap.shape) > 1:
            mean_abs_shap = mean_abs_shap.flatten()

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": mean_abs_shap,
        })

        importance_df = importance_df.sort_values("importance", ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        return importance_df

    def explain_predictions(
        self,
        X: pd.DataFrame,
        instance_indices: List[int] = None,
        method: str = "shap",
        save_dir: str = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanations for predictions.

        Args:
            X: Data to explain.
            instance_indices: Specific instances to explain. If None, explains all.
            method: Explanation method ('shap', 'lime', or 'both').
            save_dir: Directory to save plots.

        Returns:
            Dictionary containing explanations.
        """
        logger.info(f"Generating explanations using {method}")

        explanations = {}

        if method in ["shap", "both"]:
            # Compute SHAP values
            shap_values = self.compute_shap_values(X)
            explanations["shap_values"] = shap_values

            # Feature importance
            explanations["feature_importance"] = self.get_feature_importance_shap(shap_values)

            # Summary plot
            if save_dir:
                save_path = Path(save_dir) / "shap_summary.png"
                self.plot_shap_summary(X, shap_values, show=False, save_path=save_path)

        if method in ["lime", "both"] and instance_indices:
            lime_explanations = []

            for idx in instance_indices[:5]:  # Limit to 5 instances for LIME
                explanation = self.explain_instance_lime(X.iloc[idx])
                lime_explanations.append(explanation)

                if save_dir:
                    save_path = Path(save_dir) / f"lime_instance_{idx}.png"
                    self.plot_lime_explanation(explanation, show=False, save_path=save_path)

            explanations["lime_explanations"] = lime_explanations

        logger.info("Explanations generated successfully")

        return explanations
