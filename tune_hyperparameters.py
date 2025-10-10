"""
Hyperparameter tuning script for the Churn Prediction System.

This script performs intelligent hyperparameter optimization following
the Global rules mindset:
1. Measure baseline (already done in BASELINE_ANALYSIS.md)
2. Define search spaces based on domain knowledge
3. Execute systematic grid search with cross-validation
4. Validate improvements statistically
5. Document all learnings

Usage:
    python tune_hyperparameters.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.ingestion import DataLoader
from src.models import HyperparameterTuner, ModelTrainer
from src.preprocessing import DataPreprocessor
from src.utils import get_config, setup_logger


def load_baseline_results(results_path: str) -> dict:
    """Load baseline results for comparison."""
    with open(results_path, "r") as f:
        return json.load(f)


def run_hyperparameter_tuning(config_path: str = None):
    """
    Run hyperparameter tuning pipeline.

    Args:
        config_path: Path to configuration file.
    """
    # Load configuration
    config = get_config(config_path)

    # Setup logging
    log_config = config.get("logging", {})
    setup_logger(
        log_file=log_config.get("log_file", "logs/hyperparameter_tuning.log"),
        level=log_config.get("level", "INFO"),
    )

    logger.info("=" * 80)
    logger.info("HYPERPARAMETER TUNING PIPELINE")
    logger.info("=" * 80)

    try:
        # ========== STEP 1: LOAD DATA ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA LOADING")
        logger.info("=" * 80)

        data_config = config.get("data", {})
        raw_data_dir = config.get_data_path("raw_data_dir")

        data_loader = DataLoader(data_dir=raw_data_dir, dataset_url=data_config.get("dataset_url"))

        df_raw = data_loader.load_data(
            filename=data_config.get("dataset_name", "Telco-Customer-Churn.csv"),
            force_download=False,
        )

        logger.info(f"Dataset loaded | Shape: {df_raw.shape}")

        # ========== STEP 2: PREPROCESSING ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info("=" * 80)

        preprocessor = DataPreprocessor(config=config.config)
        target_column = data_config.get("target_column", "Churn")

        processed_data = preprocessor.preprocess_pipeline(df_raw, target_column=target_column, fit=True)

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

        # Handle class imbalance
        if preprocessing_config.get("handle_imbalance", False):
            X_train, y_train = preprocessor.handle_class_imbalance(
                X_train,
                y_train,
                method=preprocessing_config.get("imbalance_method", "smote"),
                sampling_strategy=preprocessing_config.get("sampling_strategy", "auto"),
                random_state=preprocessing_config.get("random_state", 42),
            )

        logger.info(f"Train set: {X_train.shape} | Test set: {X_test.shape}")

        # ========== STEP 3: LOAD BASELINE RESULTS ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: LOAD BASELINE RESULTS")
        logger.info("=" * 80)

        results_dir = config.get_data_path("results_dir")
        baseline_path = results_dir / "evaluation_results.json"

        if baseline_path.exists():
            baseline_results = load_baseline_results(str(baseline_path))
            logger.info("Baseline results loaded:")
            for model_name, metrics in baseline_results.items():
                logger.info(f"  {model_name}: F1={metrics.get('f1', 0):.4f}, ROC-AUC={metrics.get('roc_auc', 0):.4f}")
        else:
            logger.warning("Baseline results not found. Run main.py first.")
            baseline_results = {}

        # ========== STEP 4: HYPERPARAMETER TUNING ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: HYPERPARAMETER TUNING")
        logger.info("=" * 80)

        tuning_config = config.get("hyperparameter_tuning", {})

        tuner = HyperparameterTuner(
            method=tuning_config.get("method", "grid_search"),
            cv_folds=tuning_config.get("cv_folds", 5),
            scoring="f1",  # Primary metric
            n_jobs=-1,
            random_state=preprocessing_config.get("random_state", 42),
        )

        # Tune all models
        models_to_tune = ["logistic_regression", "random_forest", "xgboost"]
        tuned_models = tuner.tune_all_models(X_train, y_train, models=models_to_tune)

        logger.info(f"Tuned {len(tuned_models)} models")

        # ========== STEP 5: EVALUATE TUNED MODELS ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: EVALUATE TUNED MODELS ON TEST SET")
        logger.info("=" * 80)

        from src.evaluation import ModelEvaluator

        evaluator = ModelEvaluator()
        tuned_results = {}

        for model_name, model in tuned_models.items():
            logger.info(f"\nEvaluating tuned {model_name}...")

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            metrics = evaluator.evaluate_model(f"{model_name}_tuned", y_test, y_pred, y_pred_proba)
            tuned_results[model_name] = metrics

        # ========== STEP 6: COMPARE WITH BASELINE ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: COMPARISON WITH BASELINE")
        logger.info("=" * 80)

        comparison_data = []

        for model_name in tuned_models.keys():
            if model_name in baseline_results:
                baseline_f1 = baseline_results[model_name].get("f1", 0)
                baseline_roc = baseline_results[model_name].get("roc_auc", 0)

                tuned_f1 = tuned_results[model_name].get("f1", 0)
                tuned_roc = tuned_results[model_name].get("roc_auc", 0)

                f1_improvement = tuned_f1 - baseline_f1
                roc_improvement = tuned_roc - baseline_roc

                comparison_data.append(
                    {
                        "model": model_name,
                        "baseline_f1": baseline_f1,
                        "tuned_f1": tuned_f1,
                        "f1_improvement": f1_improvement,
                        "f1_improvement_pct": (f1_improvement / baseline_f1) * 100,
                        "baseline_roc_auc": baseline_roc,
                        "tuned_roc_auc": tuned_roc,
                        "roc_improvement": roc_improvement,
                        "significant": f1_improvement > 0.02,
                    }
                )

                logger.info(f"\n{model_name.upper()}:")
                logger.info(f"  F1: {baseline_f1:.4f} → {tuned_f1:.4f} ({f1_improvement:+.4f}, {(f1_improvement/baseline_f1)*100:+.2f}%)")
                logger.info(f"  ROC-AUC: {baseline_roc:.4f} → {tuned_roc:.4f} ({roc_improvement:+.4f})")
                logger.info(f"  Significant: {'✅ YES' if f1_improvement > 0.02 else '❌ NO'}")

        comparison_df = pd.DataFrame(comparison_data)

        # ========== STEP 7: SAVE RESULTS ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: SAVE RESULTS")
        logger.info("=" * 80)

        models_dir = config.get_data_path("models_dir")
        tuner.save_results(str(models_dir / "tuned"))

        # Save comparison report
        comparison_path = results_dir / "hyperparameter_tuning_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Comparison saved to {comparison_path}")

        # Generate and save report
        report = tuner.generate_report()
        report_path = results_dir / "hyperparameter_tuning_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")

        # ========== STEP 8: DECISION ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: DECISION - KEEP TUNED MODELS?")
        logger.info("=" * 80)

        significant_improvements = comparison_df[comparison_df["significant"] == True]

        if len(significant_improvements) > 0:
            logger.info(f"✅ {len(significant_improvements)} model(s) show significant improvement (>2%)")
            logger.info("Recommendation: KEEP tuned models")

            # Save tuned models as primary models
            for model_name in significant_improvements["model"]:
                tuned_model = tuned_models[model_name]
                model_path = models_dir / f"{model_name}.joblib"
                import joblib

                joblib.dump(tuned_model, model_path)
                logger.info(f"  Saved {model_name} as primary model")

        else:
            logger.info("❌ No significant improvements detected")
            logger.info("Recommendation: KEEP baseline models")
            logger.info("Tuned models saved in models/tuned/ for reference")

        logger.info("\n" + "=" * 80)
        logger.info("HYPERPARAMETER TUNING COMPLETE")
        logger.info("=" * 80)

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)

    except Exception as e:
        logger.exception(f"Hyperparameter tuning failed: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Churn Prediction")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: config/config.yaml)",
    )

    args = parser.parse_args()

    run_hyperparameter_tuning(config_path=args.config)


if __name__ == "__main__":
    main()
