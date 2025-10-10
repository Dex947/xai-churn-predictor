"""
Edge case tests for the Churn Prediction System.

Tests boundary conditions, invalid inputs, and error handling.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import DataLoader
from src.preprocessing import DataPreprocessor
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator


class TestEdgeCasesDataLoader:
    """Test edge cases for data loading."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        loader = DataLoader()
        df = pd.DataFrame()

        is_valid, issues = loader.validate_dataset(df, target_column="Churn")

        assert is_valid is False
        assert "DataFrame is empty" in issues

    def test_missing_target_column(self):
        """Test validation with missing target column."""
        loader = DataLoader()
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": ["a", "b", "c"]})

        is_valid, issues = loader.validate_dataset(df, target_column="Churn")

        assert is_valid is False
        assert any("Target column" in issue for issue in issues)

    def test_all_missing_values(self):
        """Test DataFrame with all missing values."""
        loader = DataLoader()
        df = pd.DataFrame({"A": [np.nan, np.nan, np.nan], "B": [np.nan, np.nan, np.nan]})

        info = loader.get_data_info(df)

        assert info["missing_values"]["A"] == 3
        assert info["missing_values"]["B"] == 3


class TestEdgeCasesPreprocessor:
    """Test edge cases for preprocessing."""

    def test_single_row_dataframe(self):
        """Test preprocessing with single row."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({"A": [1], "B": ["x"]})

        df_encoded = preprocessor.encode_categorical(df, method="onehot", fit=True)

        assert len(df_encoded) == 1

    def test_all_same_values(self):
        """Test with all identical values."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({"A": [1, 1, 1, 1], "B": ["x", "x", "x", "x"]})

        df_encoded = preprocessor.encode_categorical(df, method="onehot", fit=True)

        # Should still work, just with less variance
        assert len(df_encoded) == 4

    def test_extreme_class_imbalance(self):
        """Test SMOTE with extreme imbalance (90:10)."""
        preprocessor = DataPreprocessor()
        X = pd.DataFrame(np.random.rand(100, 5))
        y = np.array([0] * 90 + [1] * 10)  # 90:10 ratio (need at least 6 samples for SMOTE)

        X_resampled, y_resampled = preprocessor.handle_class_imbalance(
            X, y, method="smote", random_state=42
        )

        # Should balance the classes
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == counts[1]  # Balanced

    def test_missing_values_all_strategies(self):
        """Test all missing value strategies."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({"A": [1, 2, np.nan, 4, 5], "B": [10, np.nan, 30, 40, 50]})

        # Test drop strategy
        df_drop = preprocessor.handle_missing_values(df.copy(), strategy="drop")
        assert len(df_drop) == 3

        # Test mean strategy
        df_mean = preprocessor.handle_missing_values(df.copy(), strategy="mean")
        assert df_mean.isnull().sum().sum() == 0

        # Test median strategy
        df_median = preprocessor.handle_missing_values(df.copy(), strategy="median")
        assert df_median.isnull().sum().sum() == 0

    def test_invalid_strategy(self):
        """Test invalid missing value strategy."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({"A": [1, 2, np.nan]})

        with pytest.raises(ValueError, match="Unknown strategy"):
            preprocessor.handle_missing_values(df, strategy="invalid_strategy")


class TestEdgeCasesModelTrainer:
    """Test edge cases for model training."""

    def test_single_class_data(self):
        """Test training with single class (should fail gracefully)."""
        X = pd.DataFrame(np.random.rand(10, 5))
        y = np.array([0] * 10)  # All same class

        trainer = ModelTrainer()

        # This should raise an error or handle gracefully
        with pytest.raises(Exception):
            trainer.train_model("logistic_regression", X, y)

    def test_more_features_than_samples(self):
        """Test with more features than samples."""
        X = pd.DataFrame(np.random.rand(5, 100))  # 5 samples, 100 features
        y = np.array([0, 0, 1, 1, 0])

        trainer = ModelTrainer()
        model = trainer.train_model("logistic_regression", X, y)

        assert model is not None

    def test_predict_without_training(self):
        """Test prediction without training."""
        X = pd.DataFrame(np.random.rand(10, 5))

        trainer = ModelTrainer()

        with pytest.raises(ValueError, match="not trained"):
            trainer.predict("logistic_regression", X)

    def test_invalid_model_name(self):
        """Test with invalid model name."""
        trainer = ModelTrainer()

        with pytest.raises(ValueError, match="Unknown model"):
            trainer.get_model("invalid_model")


class TestEdgeCasesEvaluator:
    """Test edge cases for evaluation."""

    def test_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])  # Perfect match
        y_pred_proba = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        )

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model("test_model", y_true, y_pred, y_pred_proba)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_worst_predictions(self):
        """Test evaluation with worst possible predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1, 0])  # All wrong
        y_pred_proba = np.array(
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        )

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model("test_model", y_true, y_pred, y_pred_proba)

        assert metrics["accuracy"] == 0.0

    def test_empty_results(self):
        """Test comparison with no models evaluated."""
        evaluator = ModelEvaluator()
        comparison = evaluator.compare_models()

        assert len(comparison) == 0

    def test_get_best_model_no_results(self):
        """Test getting best model with no results."""
        evaluator = ModelEvaluator()

        with pytest.raises(ValueError, match="No models have been evaluated"):
            evaluator.get_best_model()


class TestBoundaryConditions:
    """Test boundary conditions."""

    def test_zero_variance_feature(self):
        """Test with zero variance feature."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({"constant": [5, 5, 5, 5], "variable": [1, 2, 3, 4]})

        # Should handle without error
        df_scaled = preprocessor.scale_features(df, method="standard", fit=True)
        assert df_scaled is not None

    def test_very_large_dataset_simulation(self):
        """Test with simulated large dataset."""
        # Simulate large dataset (not actually large to keep test fast)
        loader = DataLoader()
        df = pd.DataFrame({"A": range(10000), "B": range(10000), "Churn": ["Yes", "No"] * 5000})

        info = loader.get_data_info(df)

        assert info["num_rows"] == 10000

    def test_unicode_column_names(self):
        """Test with unicode characters in column names."""
        df = pd.DataFrame({"特徴1": [1, 2, 3], "feature_2": [4, 5, 6]})

        loader = DataLoader()
        info = loader.get_data_info(df)

        assert info["num_columns"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
