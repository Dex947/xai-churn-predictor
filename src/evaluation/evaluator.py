"""
Model evaluation module for the Churn Prediction System.

This module handles evaluation metrics, confusion matrices, and performance analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ModelEvaluator:
    """Evaluate machine learning models for churn prediction."""

    def __init__(self, threshold: float = 0.5):
        """
        Initialize the ModelEvaluator.

        Args:
            threshold: Classification threshold for probability predictions.
        """
        self.threshold = threshold
        self.results = {}

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a model's performance.

        Args:
            model_name: Name of the model being evaluated.
            y_true: True labels.
            y_pred: Predicted labels.
            y_pred_proba: Predicted probabilities (optional).

        Returns:
            Dictionary containing evaluation metrics.
        """
        logger.info(f"Evaluating {model_name}")

        metrics = {}

        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average="binary", zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average="binary", zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, average="binary", zero_division=0)

        # ROC-AUC (requires probabilities)
        if y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1:
                # Multi-class probabilities, use positive class
                y_pred_proba_pos = y_pred_proba[:, 1]
            else:
                y_pred_proba_pos = y_pred_proba

            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba_pos)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm
        metrics["tn"] = int(cm[0, 0])
        metrics["fp"] = int(cm[0, 1])
        metrics["fn"] = int(cm[1, 0])
        metrics["tp"] = int(cm[1, 1])

        # Derived metrics
        total = metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
        metrics["specificity"] = metrics["tn"] / (metrics["tn"] + metrics["fp"]) if (metrics["tn"] + metrics["fp"]) > 0 else 0
        metrics["false_positive_rate"] = metrics["fp"] / (metrics["fp"] + metrics["tn"]) if (metrics["fp"] + metrics["tn"]) > 0 else 0
        metrics["false_negative_rate"] = metrics["fn"] / (metrics["fn"] + metrics["tp"]) if (metrics["fn"] + metrics["tp"]) > 0 else 0

        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics["classification_report"] = report

        # Store results
        self.results[model_name] = metrics

        logger.info(
            f"{model_name} | Accuracy: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | ROC-AUC: {metrics.get('roc_auc', 'N/A')}"
        )

        return metrics

    def get_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate ROC curve data.

        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities.

        Returns:
            Dictionary with fpr, tpr, and thresholds.
        """
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": auc,
        }

    def get_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate precision-recall curve data.

        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities.

        Returns:
            Dictionary with precision, recall, and thresholds.
        """
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

        return {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
        }

    def get_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate calibration curve data.

        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities.
            n_bins: Number of bins for calibration.

        Returns:
            Dictionary with prob_true, prob_pred.
        """
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        prob_true, prob_pred = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy="uniform"
        )

        return {
            "prob_true": prob_true,
            "prob_pred": prob_pred,
        }

    def compare_models(
        self,
        metrics: List[str] = None,
    ) -> pd.DataFrame:
        """
        Compare metrics across all evaluated models.

        Args:
            metrics: List of metrics to compare. If None, uses all available metrics.

        Returns:
            DataFrame with model comparison.
        """
        if not self.results:
            logger.warning("No models have been evaluated yet")
            return pd.DataFrame()

        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        comparison_data = []

        for model_name, model_metrics in self.results.items():
            row = {"model": model_name}

            for metric in metrics:
                if metric in model_metrics:
                    row[metric] = model_metrics[metric]
                else:
                    row[metric] = None

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by F1 score (or first available metric)
        if "f1" in comparison_df.columns:
            comparison_df = comparison_df.sort_values("f1", ascending=False)

        logger.info(f"Model comparison generated for {len(comparison_df)} models")

        return comparison_df

    def get_best_model(self, metric: str = "f1") -> str:
        """
        Get the name of the best-performing model based on a metric.

        Args:
            metric: Metric to use for comparison.

        Returns:
            Name of the best model.
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet")

        best_model = None
        best_score = -np.inf

        for model_name, model_metrics in self.results.items():
            if metric in model_metrics:
                score = model_metrics[metric]
                if score > best_score:
                    best_score = score
                    best_model = model_name

        if best_model is None:
            raise ValueError(f"Metric '{metric}' not found in any model results")

        logger.info(f"Best model by {metric}: {best_model} ({best_score:.4f})")

        return best_model

    def generate_report(
        self,
        model_name: str = None,
    ) -> str:
        """
        Generate a text report of evaluation results.

        Args:
            model_name: Name of the model. If None, generates report for all models.

        Returns:
            Text report string.
        """
        if not self.results:
            return "No models have been evaluated yet."

        report_lines = ["=" * 80, "MODEL EVALUATION REPORT", "=" * 80, ""]

        # Report for specific model
        if model_name:
            if model_name not in self.results:
                return f"Model '{model_name}' not found in results."

            metrics = self.results[model_name]
            report_lines.extend(self._format_model_report(model_name, metrics))

        # Report for all models
        else:
            for model_name, metrics in self.results.items():
                report_lines.extend(self._format_model_report(model_name, metrics))
                report_lines.append("")

            # Add comparison table
            report_lines.append("=" * 80)
            report_lines.append("MODEL COMPARISON")
            report_lines.append("=" * 80)
            report_lines.append("")

            comparison_df = self.compare_models()
            report_lines.append(comparison_df.to_string(index=False))
            report_lines.append("")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def _format_model_report(self, model_name: str, metrics: Dict[str, Any]) -> List[str]:
        """
        Format report for a single model.

        Args:
            model_name: Name of the model.
            metrics: Dictionary of metrics.

        Returns:
            List of formatted report lines.
        """
        lines = [
            f"Model: {model_name}",
            "-" * 80,
            f"Accuracy:  {metrics['accuracy']:.4f}",
            f"Precision: {metrics['precision']:.4f}",
            f"Recall:    {metrics['recall']:.4f}",
            f"F1 Score:  {metrics['f1']:.4f}",
        ]

        if "roc_auc" in metrics:
            lines.append(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

        lines.extend([
            "",
            "Confusion Matrix:",
            f"  TN: {metrics['tn']:6d}  |  FP: {metrics['fp']:6d}",
            f"  FN: {metrics['fn']:6d}  |  TP: {metrics['tp']:6d}",
            "",
            f"Specificity: {metrics['specificity']:.4f}",
            f"False Positive Rate: {metrics['false_positive_rate']:.4f}",
            f"False Negative Rate: {metrics['false_negative_rate']:.4f}",
            "-" * 80,
        ])

        return lines

    def save_results(self, file_path: str, format: str = "json") -> None:
        """
        Save evaluation results to file.

        Args:
            file_path: Path to save results.
            format: Output format ('json' or 'csv').
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            import json

            # Convert numpy arrays to lists for JSON serialization
            results_json = {}
            for model_name, metrics in self.results.items():
                results_json[model_name] = {}
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        results_json[model_name][key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        results_json[model_name][key] = float(value)
                    elif key != "classification_report":  # Skip complex nested dict
                        results_json[model_name][key] = value

            with open(file_path, "w") as f:
                json.dump(results_json, f, indent=2)

        elif format == "csv":
            comparison_df = self.compare_models()
            comparison_df.to_csv(file_path, index=False)

        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Evaluation results saved to {file_path}")

    def save_report(self, file_path: str) -> None:
        """
        Save text report to file.

        Args:
            file_path: Path to save report.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_report()

        with open(file_path, "w") as f:
            f.write(report)

        logger.info(f"Evaluation report saved to {file_path}")

    def get_metrics_summary(self, model_name: str) -> Dict[str, float]:
        """
        Get a summary of key metrics for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary of key metrics.
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found in results")

        metrics = self.results[model_name]

        summary = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }

        if "roc_auc" in metrics:
            summary["roc_auc"] = metrics["roc_auc"]

        return summary
