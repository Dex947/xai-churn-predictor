"""
Main pipeline orchestrator for the Churn Prediction System.

This script coordinates the entire ML pipeline from data loading to model evaluation.
"""

import argparse

import pandas as pd
from loguru import logger

from src.evaluation import ModelEvaluator
from src.explainability import ModelExplainer
from src.ingestion import DataLoader
from src.models import ModelTrainer
from src.preprocessing import DataPreprocessor
from src.utils import get_config, setup_logger
from src.visualization import ChurnVisualizer


def run_pipeline(config_path: str = None, skip_download: bool = False):
    """
    Run the complete churn prediction pipeline.

    Args:
        config_path: Path to configuration file.
        skip_download: Skip downloading dataset if it exists.
    """
    # Load configuration
    config = get_config(config_path)

    # Setup logging
    log_config = config.get("logging", {})
    setup_logger(
        log_file=log_config.get("log_file", "logs/churn_prediction.log"),
        level=log_config.get("level", "INFO"),
    )

    logger.info("=" * 80)
    logger.info("STARTING CHURN PREDICTION PIPELINE")
    logger.info("=" * 80)

    try:
        # ========== STEP 1: DATA INGESTION ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("=" * 80)

        data_config = config.get("data", {})
        raw_data_dir = config.get_data_path("raw_data_dir")

        data_loader = DataLoader(
            data_dir=raw_data_dir,
            dataset_url=data_config.get("dataset_url"),
        )

        # Load dataset
        df_raw = data_loader.load_data(
            filename=data_config.get("dataset_name", "Telco-Customer-Churn.csv"),
            force_download=not skip_download,
        )

        logger.info(f"Dataset loaded | Shape: {df_raw.shape}")

        # Get data info
        data_info = data_loader.get_data_info(df_raw)
        logger.info(f"Missing values: {sum(data_info['missing_values'].values())}")

        # Validate dataset
        target_column = data_config.get("target_column", "Churn")
        is_valid, issues = data_loader.validate_dataset(df_raw, target_column=target_column)

        if not is_valid:
            logger.error(f"Dataset validation failed: {issues}")
            return

        # ========== STEP 2: EXPLORATORY DATA ANALYSIS ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 80)

        plots_dir = config.get_data_path("plots_dir")
        visualizer = ChurnVisualizer()

        # Churn distribution
        visualizer.plot_churn_distribution(
            df_raw,
            target_column=target_column,
            show=False,
            save_path=plots_dir / "churn_distribution.png",
        )

        # Numeric features
        numeric_features = config.get("features", {}).get("numeric_features", [])
        if numeric_features:
            existing_numeric = [col for col in numeric_features if col in df_raw.columns]
            if existing_numeric:
                visualizer.plot_numeric_distributions(
                    df_raw,
                    existing_numeric,
                    target_column=target_column,
                    show=False,
                    save_path=plots_dir / "numeric_distributions.png",
                )

        # Categorical features
        categorical_cols = df_raw.select_dtypes(include=["object"]).columns.tolist()
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)

        # Remove ID columns
        drop_cols = config.get("features", {}).get("drop_columns", [])
        categorical_cols = [col for col in categorical_cols if col not in drop_cols]

        if categorical_cols:
            visualizer.plot_categorical_distributions(
                df_raw,
                categorical_cols[:6],  # Limit to 6 for readability
                target_column=target_column,
                show=False,
                save_path=plots_dir / "categorical_distributions.png",
            )

        logger.info(f"EDA plots saved to {plots_dir}")

        # ========== STEP 3: DATA PREPROCESSING ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: DATA PREPROCESSING")
        logger.info("=" * 80)

        preprocessor = DataPreprocessor(config=config.config)

        # Preprocess data
        processed_data = preprocessor.preprocess_pipeline(
            df_raw,
            target_column=target_column,
            fit=True,
        )

        X = processed_data["X"]
        y = processed_data["y"]
        feature_names = processed_data["feature_names"]

        logger.info(f"Preprocessing completed | Features: {X.shape[1]}")

        # Split data
        preprocessing_config = config.get("preprocessing", {})
        splits = preprocessor.split_data(
            pd.concat([X, pd.Series(y, name=target_column, index=X.index)], axis=1),
            target_column=target_column,
            test_size=preprocessing_config.get("test_size", 0.2),
            val_size=preprocessing_config.get("validation_size", 0.0),
            random_state=preprocessing_config.get("random_state", 42),
        )

        X_train = splits["X_train"]
        y_train = splits["y_train"]
        X_test = splits["X_test"]
        y_test = splits["y_test"]

        # Handle class imbalance (only on training data)
        if preprocessing_config.get("handle_imbalance", False):
            X_train, y_train = preprocessor.handle_class_imbalance(
                X_train,
                y_train,
                method=preprocessing_config.get("imbalance_method", "smote"),
                sampling_strategy=preprocessing_config.get("sampling_strategy", "auto"),
                random_state=preprocessing_config.get("random_state", 42),
            )

        logger.info(f"Train set: {X_train.shape} | Test set: {X_test.shape}")

        # Save preprocessor
        models_dir = config.get_data_path("models_dir")
        preprocessor.save_preprocessor(models_dir / "preprocessor.joblib")

        # ========== STEP 4: MODEL TRAINING ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("=" * 80)

        trainer = ModelTrainer(config=config.config)

        # Train all enabled models
        trained_models = trainer.train_all_models(X_train, y_train)

        logger.info(f"Trained {len(trained_models)} models")

        # Save models
        if config.get("misc", {}).get("save_models", True):
            trainer.save_all_models(models_dir)

        # ========== STEP 5: MODEL EVALUATION ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: MODEL EVALUATION")
        logger.info("=" * 80)

        evaluator = ModelEvaluator()
        roc_data = {}
        pr_data = {}
        calib_data = {}

        for model_name in trained_models.keys():
            logger.info(f"\nEvaluating {model_name}...")

            # Predictions
            y_pred = trainer.predict(model_name, X_test)
            y_pred_proba = trainer.predict_proba(model_name, X_test)

            # Evaluate
            metrics = evaluator.evaluate_model(model_name, y_test, y_pred, y_pred_proba)

            # ROC curve data
            roc_data[model_name] = evaluator.get_roc_curve(y_test, y_pred_proba)

            # Precision-recall curve data
            pr_data[model_name] = evaluator.get_precision_recall_curve(y_test, y_pred_proba)

            # Calibration curve data
            calib_data[model_name] = evaluator.get_calibration_curve(y_test, y_pred_proba)

            # Confusion matrix
            visualizer.plot_confusion_matrix(
                metrics["confusion_matrix"],
                title=f"{model_name} - Confusion Matrix",
                show=False,
                save_path=plots_dir / f"confusion_matrix_{model_name}.png",
            )

        # Model comparison
        comparison_df = evaluator.compare_models()
        logger.info(f"\n{comparison_df.to_string()}")

        # Plot comparisons
        visualizer.plot_roc_curves(roc_data, show=False, save_path=plots_dir / "roc_curves.png")
        visualizer.plot_precision_recall_curve(
            pr_data, show=False, save_path=plots_dir / "precision_recall_curves.png"
        )
        visualizer.plot_calibration_curve(
            calib_data, show=False, save_path=plots_dir / "calibration_curves.png"
        )
        visualizer.plot_model_comparison(
            comparison_df,
            metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
            show=False,
            save_path=plots_dir / "model_comparison.png",
        )

        # Save evaluation results
        results_dir = config.get_data_path("results_dir")
        evaluator.save_results(results_dir / "evaluation_results.json", format="json")
        evaluator.save_report(results_dir / "evaluation_report.txt")

        # ========== STEP 6: EXPLAINABILITY ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: EXPLAINABILITY (SHAP & LIME)")
        logger.info("=" * 80)

        explainability_config = config.get("explainability", {})

        if explainability_config.get("shap", {}).get("enabled", True):
            # Get best model
            best_model_name = evaluator.get_best_model(metric="f1")
            best_model = trained_models[best_model_name]

            logger.info(f"Generating explanations for best model: {best_model_name}")

            # Sample data for SHAP (for performance)
            n_samples_shap = explainability_config.get("shap", {}).get("n_samples_for_shap", 1000)
            X_train_sample = X_train.sample(min(n_samples_shap, len(X_train)), random_state=42)
            X_test_sample = X_test.sample(min(100, len(X_test)), random_state=42)

            # Create explainer
            explainer = ModelExplainer(
                model=best_model,
                X_train=X_train_sample,
                feature_names=feature_names,
                class_names=processed_data.get("target_classes", ["No Churn", "Churn"]),
            )

            # Compute SHAP values
            shap_values = explainer.compute_shap_values(X_test_sample)

            # SHAP plots
            shap_config = explainability_config.get("shap", {})

            if shap_config.get("plot_summary", True):
                explainer.plot_shap_summary(
                    X_test_sample,
                    shap_values,
                    max_display=shap_config.get("max_display", 20),
                    show=False,
                    save_path=plots_dir / "shap_summary.png",
                )

            # Feature importance from SHAP
            importance_df = explainer.get_feature_importance_shap(shap_values, top_n=20)
            visualizer.plot_feature_importance(
                importance_df,
                title=f"{best_model_name} - Feature Importance (SHAP)",
                show=False,
                save_path=plots_dir / "feature_importance_shap.png",
            )

            logger.info("SHAP explanations generated")

        # Feature importance from models
        for model_name in trained_models.keys():
            try:
                importance_df = trainer.get_feature_importance(model_name, feature_names, top_n=20)
                visualizer.plot_feature_importance(
                    importance_df,
                    title=f"{model_name} - Feature Importance",
                    show=False,
                    save_path=plots_dir / f"feature_importance_{model_name}.png",
                )
            except Exception as e:
                logger.warning(f"Could not generate feature importance for {model_name}: {e}")

        # ========== STEP 7: GENERATE SUMMARY ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: GENERATE SUMMARY")
        logger.info("=" * 80)

        summary = generate_summary(
            data_info=data_info,
            comparison_df=comparison_df,
            best_model_name=best_model_name,
            feature_names=feature_names,
        )

        # Save summary
        summary_path = results_dir / "summary.md"
        with open(summary_path, "w") as f:
            f.write(summary)

        logger.info(f"Summary saved to {summary_path}")

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        print("\n" + summary)

    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        raise


def generate_summary(
    data_info: dict,
    comparison_df: pd.DataFrame,
    best_model_name: str,
    feature_names: list,
) -> str:
    """
    Generate a summary report of the pipeline execution.

    Args:
        data_info: Dataset information.
        comparison_df: Model comparison DataFrame.
        best_model_name: Name of the best performing model.
        feature_names: List of feature names.

    Returns:
        Summary report as markdown string.
    """
    summary_lines = [
        "# Customer Churn Prediction - Pipeline Summary",
        "",
        "## Dataset Information",
        f"- **Total Records**: {data_info['num_rows']:,}",
        f"- **Features**: {data_info['num_columns']}",
        f"- **Missing Values**: {sum(data_info['missing_values'].values())}",
        f"- **Duplicate Records**: {data_info['duplicates']}",
        "",
        "## Feature Engineering",
        f"- **Final Features**: {len(feature_names)}",
        "",
        "## Model Performance",
        "",
        comparison_df.to_markdown(index=False),
        "",
        f"## Best Model: **{best_model_name}**",
        "",
        "## Key Findings",
        "",
        "### Top Factors Contributing to Churn:",
        "- Analysis based on SHAP values and feature importance",
        "- See detailed plots in `data/plots/` directory",
        "",
        "## Output Files",
        "",
        "### Models",
        "- Trained models saved in `data/models/`",
        "- Preprocessor saved for inference",
        "",
        "### Visualizations",
        "- EDA plots: `data/plots/`",
        "- Model performance plots: `data/plots/`",
        "- SHAP explanations: `data/plots/`",
        "",
        "### Reports",
        "- Detailed evaluation: `data/results/evaluation_report.txt`",
        "- Metrics (JSON): `data/results/evaluation_results.json`",
        "",
        "## Next Steps",
        "",
        "1. **Deploy Model**: Use the Streamlit dashboard for interactive predictions",
        "2. **Monitor Performance**: Track model performance on new data",
        "3. **Iterate**: Experiment with hyperparameter tuning for improved results",
        "",
        "---",
        "",
        "*Generated by Churn Prediction System*",
    ]

    return "\n".join(summary_lines)


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: config/config.yaml)",
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading dataset if it already exists",
    )

    args = parser.parse_args()

    run_pipeline(config_path=args.config, skip_download=args.skip_download)


if __name__ == "__main__":
    main()
