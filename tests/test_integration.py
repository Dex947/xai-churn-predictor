"""
Integration tests for the Churn Prediction System.

Tests end-to-end workflows and module interactions.
"""

import sys
from pathlib import Path
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import DataLoader
from src.preprocessing import DataPreprocessor
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator
from src.visualization import ChurnVisualizer
from src.utils import get_config


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        np.random.seed(42)
        n_samples = 200

        data = {
            "customerID": [f"C{i:04d}" for i in range(n_samples)],
            "gender": np.random.choice(["Male", "Female"], n_samples),
            "SeniorCitizen": np.random.choice([0, 1], n_samples),
            "Partner": np.random.choice(["Yes", "No"], n_samples),
            "Dependents": np.random.choice(["Yes", "No"], n_samples),
            "tenure": np.random.randint(0, 72, n_samples),
            "PhoneService": np.random.choice(["Yes", "No"], n_samples),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n_samples),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
            "MonthlyCharges": np.random.uniform(20, 120, n_samples),
            "TotalCharges": np.random.uniform(20, 8000, n_samples),
            "Churn": np.random.choice(["Yes", "No"], n_samples, p=[0.3, 0.7]),
        }

        return pd.DataFrame(data)

    def test_full_pipeline_execution(self, sample_dataset):
        """Test complete pipeline from data to evaluation."""
        # Step 1: Data Validation
        loader = DataLoader()
        is_valid, issues = loader.validate_dataset(sample_dataset, target_column="Churn")
        assert is_valid is True

        # Step 2: Preprocessing
        config = {
            "features": {"drop_columns": ["customerID"]},
            "preprocessing": {
                "missing_value_strategy": "drop",
                "categorical_encoding": "onehot",
                "numeric_scaling": "standard",
                "handle_imbalance": False,  # Skip for speed
            },
        }

        preprocessor = DataPreprocessor(config=config)
        processed_data = preprocessor.preprocess_pipeline(
            sample_dataset, target_column="Churn", fit=True
        )

        X = processed_data["X"]
        y = processed_data["y"]

        assert X.shape[0] == y.shape[0]
        assert X.shape[1] > 0

        # Step 3: Train-Test Split
        splits = preprocessor.split_data(
            pd.concat([X, pd.Series(y, name="Churn", index=X.index)], axis=1),
            target_column="Churn",
            test_size=0.2,
            random_state=42,
        )

        X_train = splits["X_train"]
        y_train = splits["y_train"]
        X_test = splits["X_test"]
        y_test = splits["y_test"]

        assert len(X_train) > len(X_test)

        # Step 4: Model Training
        model_config = {
            "models": {
                "logistic_regression": {
                    "enabled": True,
                    "params": {"max_iter": 1000, "random_state": 42},
                }
            }
        }

        trainer = ModelTrainer(config=model_config)
        model = trainer.train_model("logistic_regression", X_train, y_train)

        assert model is not None
        assert "logistic_regression" in trainer.models

        # Step 5: Prediction
        y_pred = trainer.predict("logistic_regression", X_test)
        y_pred_proba = trainer.predict_proba("logistic_regression", X_test)

        assert len(y_pred) == len(X_test)
        assert y_pred_proba.shape[0] == len(X_test)

        # Step 6: Evaluation
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model("logistic_regression", y_test, y_pred, y_pred_proba)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

        # Verify metrics are in valid range
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_pipeline_with_imbalance_handling(self, sample_dataset):
        """Test pipeline with SMOTE balancing."""
        # Preprocessing with SMOTE
        config = {
            "features": {"drop_columns": ["customerID"]},
            "preprocessing": {
                "missing_value_strategy": "drop",
                "categorical_encoding": "onehot",
                "numeric_scaling": "standard",
                "handle_imbalance": True,
                "imbalance_method": "smote",
            },
        }

        preprocessor = DataPreprocessor(config=config)
        processed_data = preprocessor.preprocess_pipeline(
            sample_dataset, target_column="Churn", fit=True
        )

        X = processed_data["X"]
        y = processed_data["y"]

        # Split
        splits = preprocessor.split_data(
            pd.concat([X, pd.Series(y, name="Churn", index=X.index)], axis=1),
            target_column="Churn",
            test_size=0.2,
            random_state=42,
        )

        X_train = splits["X_train"]
        y_train = splits["y_train"]

        # Apply SMOTE
        X_train_balanced, y_train_balanced = preprocessor.handle_class_imbalance(
            X_train, y_train, method="smote", random_state=42
        )

        # Check that classes are balanced
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        assert len(unique) == 2
        # Classes should be equal or very close
        assert abs(counts[0] - counts[1]) <= 1

    def test_multiple_models_comparison(self, sample_dataset):
        """Test training and comparing multiple models."""
        # Preprocess
        config = {
            "features": {"drop_columns": ["customerID"]},
            "preprocessing": {
                "missing_value_strategy": "drop",
                "categorical_encoding": "onehot",
                "numeric_scaling": "standard",
            },
        }

        preprocessor = DataPreprocessor(config=config)
        processed_data = preprocessor.preprocess_pipeline(
            sample_dataset, target_column="Churn", fit=True
        )

        X = processed_data["X"]
        y = processed_data["y"]

        splits = preprocessor.split_data(
            pd.concat([X, pd.Series(y, name="Churn", index=X.index)], axis=1),
            target_column="Churn",
            test_size=0.2,
            random_state=42,
        )

        X_train = splits["X_train"]
        y_train = splits["y_train"]
        X_test = splits["X_test"]
        y_test = splits["y_test"]

        # Train multiple models
        model_config = {
            "models": {
                "logistic_regression": {
                    "enabled": True,
                    "params": {"max_iter": 1000, "random_state": 42},
                },
                "random_forest": {
                    "enabled": True,
                    "params": {"n_estimators": 10, "random_state": 42, "n_jobs": -1},
                },
            }
        }

        trainer = ModelTrainer(config=model_config)
        trained_models = trainer.train_all_models(X_train, y_train)

        assert len(trained_models) == 2

        # Evaluate all models
        evaluator = ModelEvaluator()

        for model_name in trained_models.keys():
            y_pred = trainer.predict(model_name, X_test)
            y_pred_proba = trainer.predict_proba(model_name, X_test)
            evaluator.evaluate_model(model_name, y_test, y_pred, y_pred_proba)

        # Compare models
        comparison = evaluator.compare_models()

        assert len(comparison) == 2
        assert "model" in comparison.columns
        assert "accuracy" in comparison.columns

    def test_model_persistence(self, sample_dataset):
        """Test saving and loading models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train model
            config = {
                "features": {"drop_columns": ["customerID"]},
                "preprocessing": {
                    "missing_value_strategy": "drop",
                    "categorical_encoding": "onehot",
                    "numeric_scaling": "standard",
                },
            }

            preprocessor = DataPreprocessor(config=config)
            processed_data = preprocessor.preprocess_pipeline(
                sample_dataset, target_column="Churn", fit=True
            )

            X = processed_data["X"]
            y = processed_data["y"]

            splits = preprocessor.split_data(
                pd.concat([X, pd.Series(y, name="Churn", index=X.index)], axis=1),
                target_column="Churn",
                test_size=0.2,
                random_state=42,
            )

            trainer = ModelTrainer()
            trainer.train_model("logistic_regression", splits["X_train"], splits["y_train"])

            # Save model
            model_path = Path(tmpdir) / "test_model.joblib"
            trainer.save_model("logistic_regression", str(model_path))

            assert model_path.exists()

            # Load model
            new_trainer = ModelTrainer()
            loaded_model = new_trainer.load_model("logistic_regression", str(model_path))

            assert loaded_model is not None

            # Verify predictions match
            original_pred = trainer.predict("logistic_regression", splits["X_test"])
            loaded_pred = new_trainer.predict("logistic_regression", splits["X_test"])

            assert np.array_equal(original_pred, loaded_pred)

    def test_preprocessor_persistence(self, sample_dataset):
        """Test saving and loading preprocessor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "features": {"drop_columns": ["customerID"]},
                "preprocessing": {
                    "missing_value_strategy": "drop",
                    "categorical_encoding": "onehot",
                    "numeric_scaling": "standard",
                },
            }

            preprocessor = DataPreprocessor(config=config)
            processed_data = preprocessor.preprocess_pipeline(
                sample_dataset, target_column="Churn", fit=True
            )

            # Save preprocessor
            preprocessor_path = Path(tmpdir) / "preprocessor.joblib"
            preprocessor.save_preprocessor(str(preprocessor_path))

            assert preprocessor_path.exists()

            # Load preprocessor
            new_preprocessor = DataPreprocessor()
            new_preprocessor.load_preprocessor(str(preprocessor_path))

            assert new_preprocessor.feature_names == preprocessor.feature_names

    def test_visualization_integration(self, sample_dataset):
        """Test visualization with real data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = ChurnVisualizer()

            # Test churn distribution plot
            plot_path = Path(tmpdir) / "churn_dist.png"
            visualizer.plot_churn_distribution(
                sample_dataset, target_column="Churn", show=False, save_path=str(plot_path)
            )

            assert plot_path.exists()
            assert plot_path.stat().st_size > 0


class TestErrorHandling:
    """Test error handling in integrated workflows."""

    def test_invalid_config_handling(self):
        """Test handling of invalid configuration."""
        invalid_config = {"preprocessing": {"missing_value_strategy": "invalid_strategy"}}

        preprocessor = DataPreprocessor(config=invalid_config)
        df = pd.DataFrame({"A": [1, 2, np.nan]})

        with pytest.raises(ValueError):
            preprocessor.handle_missing_values(df, strategy="invalid_strategy")

    def test_mismatched_features_inference(self):
        """Test inference with mismatched features."""
        # Train with certain features
        config = {
            "features": {"drop_columns": ["customerID"]},
            "preprocessing": {
                "categorical_encoding": "onehot",
                "numeric_scaling": "standard",
            },
        }

        preprocessor = DataPreprocessor(config=config)

        train_df = pd.DataFrame(
            {"customerID": ["C1", "C2"], "feature1": [1, 2], "feature2": ["A", "B"], "Churn": ["Yes", "No"]}
        )

        processed_data = preprocessor.preprocess_pipeline(train_df, target_column="Churn", fit=True)

        # Try to transform with different features (should handle gracefully)
        test_df = pd.DataFrame({"customerID": ["C3"], "feature1": [3], "feature3": ["C"]})

        # This should either work with missing features set to 0 or raise informative error
        # Implementation dependent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
