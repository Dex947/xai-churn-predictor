"""
Visualization module for the Churn Prediction System.

This module provides various plotting utilities for EDA and model evaluation.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from ..utils import constants


class ChurnVisualizer:
    """Create visualizations for churn prediction analysis."""

    def __init__(self, style: str = "seaborn-v0_8-darkgrid", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the ChurnVisualizer.

        Args:
            style: Matplotlib style to use.
            figsize: Default figure size.
        """
        try:
            plt.style.use(style)
        except OSError:
            logger.warning(f"Style '{style}' not found, using default")

        self.figsize = figsize
        self.color_palette = sns.color_palette("Set2")

        # Set seaborn defaults
        sns.set_context("notebook")
        sns.set_palette(self.color_palette)

    def plot_churn_distribution(
        self,
        df: pd.DataFrame,
        target_column: str = "Churn",
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot the distribution of churn vs non-churn.

        Args:
            df: DataFrame containing the target column.
            target_column: Name of the target column.
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        logger.info("Plotting churn distribution")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Count plot
        churn_counts = df[target_column].value_counts()
        ax1.bar(churn_counts.index, churn_counts.values, color=["#2ecc71", "#e74c3c"])
        ax1.set_xlabel(target_column)
        ax1.set_ylabel("Count")
        ax1.set_title("Churn Distribution (Counts)")
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(churn_counts.values):
            ax1.text(
                i,
                v + constants.PLOT_TEXT_OFFSET,
                str(v),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Pie chart
        churn_pct = df[target_column].value_counts(normalize=True) * 100
        colors = ["#2ecc71", "#e74c3c"]
        ax2.pie(
            churn_pct.values,
            labels=churn_pct.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax2.set_title("Churn Distribution (Percentage)")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches=constants.PLOT_BBOX_INCHES, dpi=constants.PLOT_DPI)
            logger.info(f"Churn distribution plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_numeric_distributions(
        self,
        df: pd.DataFrame,
        numeric_columns: List[str],
        target_column: str = "Churn",
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot distributions of numeric features by churn status.

        Args:
            df: DataFrame containing the data.
            numeric_columns: List of numeric column names.
            target_column: Name of the target column.
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        logger.info(f"Plotting distributions for {len(numeric_columns)} numeric features")

        n_cols = 3
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for idx, col in enumerate(numeric_columns):
            if idx < len(axes):
                for churn_value in df[target_column].unique():
                    data = df[df[target_column] == churn_value][col]
                    axes[idx].hist(
                        data,
                        alpha=0.6,
                        label=f"{target_column}={churn_value}",
                        bins=constants.PLOT_HISTOGRAM_BINS,
                    )

                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel("Frequency")
                axes[idx].set_title(f"Distribution of {col}")
                axes[idx].legend()
                axes[idx].grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(len(numeric_columns), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches=constants.PLOT_BBOX_INCHES, dpi=constants.PLOT_DPI)
            logger.info(f"Numeric distributions plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_categorical_distributions(
        self,
        df: pd.DataFrame,
        categorical_columns: List[str],
        target_column: str = "Churn",
        max_categories: int = constants.PLOT_MAX_CATEGORIES,
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot distributions of categorical features by churn status.

        Args:
            df: DataFrame containing the data.
            categorical_columns: List of categorical column names.
            target_column: Name of the target column.
            max_categories: Maximum number of categories to show per feature.
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        logger.info(f"Plotting distributions for {len(categorical_columns)} categorical features")

        n_cols = 2
        n_rows = (len(categorical_columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for idx, col in enumerate(categorical_columns):
            if idx < len(axes):
                # Get top categories
                top_categories = df[col].value_counts().head(max_categories).index
                df_filtered = df[df[col].isin(top_categories)]

                # Create crosstab
                ct = (
                    pd.crosstab(df_filtered[col], df_filtered[target_column], normalize="index")
                    * 100
                )

                ct.plot(kind="bar", ax=axes[idx], color=["#2ecc71", "#e74c3c"])
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel("Percentage (%)")
                axes[idx].set_title(f"Churn Rate by {col}")
                axes[idx].legend(title=target_column)
                axes[idx].grid(axis="y", alpha=0.3)
                axes[idx].tick_params(axis="x", rotation=45)

        # Hide unused subplots
        for idx in range(len(categorical_columns), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches=constants.PLOT_BBOX_INCHES, dpi=constants.PLOT_DPI)
            logger.info(f"Categorical distributions plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        numeric_columns: List[str] = None,
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot correlation heatmap for numeric features.

        Args:
            df: DataFrame containing the data.
            numeric_columns: List of numeric columns. If None, auto-detect.
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        logger.info("Plotting correlation heatmap")

        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        corr_matrix = df[numeric_columns].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches=constants.PLOT_BBOX_INCHES, dpi=constants.PLOT_DPI)
            logger.info(f"Correlation heatmap saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str] = None,
        title: str = "Confusion Matrix",
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix array.
            class_names: List of class names.
            title: Plot title.
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        logger.info("Plotting confusion matrix")

        if class_names is None:
            class_names = ["No Churn", "Churn"]

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Count"},
        )

        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches=constants.PLOT_BBOX_INCHES, dpi=constants.PLOT_DPI)
            logger.info(f"Confusion matrix saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_roc_curves(
        self,
        roc_data: Dict[str, Dict[str, np.ndarray]],
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot ROC curves for multiple models.

        Args:
            roc_data: Dictionary with model names as keys and ROC data as values.
                     Each value should contain 'fpr', 'tpr', and 'auc'.
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        logger.info(f"Plotting ROC curves for {len(roc_data)} models")

        plt.figure(figsize=self.figsize)

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=2)

        # Plot ROC curve for each model
        for model_name, data in roc_data.items():
            plt.plot(
                data["fpr"],
                data["tpr"],
                label=f"{model_name} (AUC = {data['auc']:.3f})",
                linewidth=2,
            )

        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves Comparison", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches=constants.PLOT_BBOX_INCHES, dpi=constants.PLOT_DPI)
            logger.info(f"ROC curves saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        title: str = "Feature Importance",
        top_n: int = 20,
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot feature importance.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns.
            title: Plot title.
            top_n: Number of top features to show.
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        logger.info("Plotting feature importance")

        # Get top N features
        plot_df = importance_df.head(top_n).copy()

        plt.figure(figsize=(10, max(8, top_n * 0.4)))

        # Create horizontal bar plot
        plt.barh(range(len(plot_df)), plot_df["importance"].values, color=self.color_palette[0])
        plt.yticks(range(len(plot_df)), plot_df["feature"].values)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches=constants.PLOT_BBOX_INCHES, dpi=constants.PLOT_DPI)
            logger.info(f"Feature importance plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_precision_recall_curve(
        self,
        pr_data: Dict[str, Dict[str, np.ndarray]],
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot precision-recall curves for multiple models.

        Args:
            pr_data: Dictionary with model names as keys and PR data as values.
                    Each value should contain 'precision' and 'recall'.
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        logger.info(f"Plotting precision-recall curves for {len(pr_data)} models")

        plt.figure(figsize=self.figsize)

        # Plot PR curve for each model
        for model_name, data in pr_data.items():
            plt.plot(
                data["recall"],
                data["precision"],
                label=model_name,
                linewidth=2,
            )

        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curves Comparison", fontsize=14, fontweight="bold")
        plt.legend(loc="best", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches=constants.PLOT_BBOX_INCHES, dpi=constants.PLOT_DPI)
            logger.info(f"Precision-recall curves saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_calibration_curve(
        self,
        calib_data: Dict[str, Dict[str, np.ndarray]],
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot calibration curves for multiple models.

        Args:
            calib_data: Dictionary with model names as keys and calibration data as values.
                       Each value should contain 'prob_true' and 'prob_pred'.
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        logger.info(f"Plotting calibration curves for {len(calib_data)} models")

        plt.figure(figsize=self.figsize)

        # Plot diagonal (perfectly calibrated)
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", linewidth=2)

        # Plot calibration curve for each model
        for model_name, data in calib_data.items():
            plt.plot(
                data["prob_pred"],
                data["prob_true"],
                marker="o",
                label=model_name,
                linewidth=2,
            )

        plt.xlabel("Mean Predicted Probability", fontsize=12)
        plt.ylabel("Fraction of Positives", fontsize=12)
        plt.title("Calibration Curves Comparison", fontsize=14, fontweight="bold")
        plt.legend(loc="best", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches=constants.PLOT_BBOX_INCHES, dpi=constants.PLOT_DPI)
            logger.info(f"Calibration curves saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = None,
        show: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot model comparison across multiple metrics.

        Args:
            comparison_df: DataFrame with model comparison data.
            metrics: List of metrics to plot. If None, plots all numeric columns.
            show: Whether to display the plot.
            save_path: Path to save the plot.
        """
        logger.info("Plotting model comparison")

        if metrics is None:
            metrics = comparison_df.select_dtypes(include=[np.number]).columns.tolist()

        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            comparison_df.plot(
                x="model",
                y=metric,
                kind="bar",
                ax=axes[idx],
                color=self.color_palette[idx % len(self.color_palette)],
                legend=False,
            )
            axes[idx].set_title(f"{metric.upper()}", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel("Model", fontsize=10)
            axes[idx].set_ylabel(metric.capitalize(), fontsize=10)
            axes[idx].grid(axis="y", alpha=0.3)
            axes[idx].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches=constants.PLOT_BBOX_INCHES, dpi=constants.PLOT_DPI)
            logger.info(f"Model comparison plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
