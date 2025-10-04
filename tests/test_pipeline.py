"""
Unit tests for the Churn Prediction System.
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
from src.visualization import ChurnVisualizer


class TestDataLoader:
    """Test data loading functionality."""

    def test_get_data_info(self):
        """Test data info extraction."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, np.nan, 4.4, 5.5]
        })

        loader = DataLoader()
        info = loader.get_data_info(df)

        assert info['num_rows'] == 5
        assert info['num_columns'] == 3
        assert info['missing_values']['C'] == 1

    def test_validate_dataset(self):
        """Test dataset validation."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['a', 'b', 'c'],
            'Churn': ['Yes', 'No', 'Yes']
        })

        loader = DataLoader()
        is_valid, issues = loader.validate_dataset(df, target_column='Churn')

        assert is_valid is True
        assert len(issues) == 0


class TestDataPreprocessor:
    """Test data preprocessing functionality."""

    def test_handle_missing_values_drop(self):
        """Test dropping missing values."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })

        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(df, strategy='drop')

        assert len(df_clean) == 4
        assert df_clean.isnull().sum().sum() == 0

    def test_encode_categorical(self):
        """Test categorical encoding."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'C'],
            'cat2': ['X', 'Y', 'X', 'Y']
        })

        preprocessor = DataPreprocessor()
        df_encoded = preprocessor.encode_categorical(df, method='onehot', fit=True)

        # One-hot encoding with drop_first=True
        assert df_encoded.shape[1] > 2

    def test_split_data(self):
        """Test train-test split."""
        df = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'target': [0] * 50 + [1] * 50
        })

        preprocessor = DataPreprocessor()
        splits = preprocessor.split_data(
            df,
            target_column='target',
            test_size=0.2,
            random_state=42
        )

        assert 'X_train' in splits
        assert 'X_test' in splits
        assert 'y_train' in splits
        assert 'y_test' in splits
        assert len(splits['X_test']) == 20


class TestModelTrainer:
    """Test model training functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_get_model(self, sample_data):
        """Test model initialization."""
        trainer = ModelTrainer()
        model = trainer.get_model('logistic_regression')
        assert model is not None

    def test_train_model(self, sample_data):
        """Test model training."""
        X, y = sample_data
        trainer = ModelTrainer()
        model = trainer.train_model('logistic_regression', X, y)

        assert model is not None
        assert 'logistic_regression' in trainer.models

    def test_predict(self, sample_data):
        """Test predictions."""
        X, y = sample_data
        trainer = ModelTrainer()
        trainer.train_model('logistic_regression', X, y)

        predictions = trainer.predict('logistic_regression', X)
        assert len(predictions) == len(X)


class TestModelEvaluator:
    """Test model evaluation functionality."""

    def test_evaluate_model(self):
        """Test model evaluation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0])
        y_pred_proba = np.array([
            [0.8, 0.2], [0.9, 0.1], [0.3, 0.7], [0.2, 0.8],
            [0.7, 0.3], [0.6, 0.4], [0.1, 0.9], [0.95, 0.05]
        ])

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model('test_model', y_true, y_pred, y_pred_proba)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics

    def test_compare_models(self):
        """Test model comparison."""
        evaluator = ModelEvaluator()

        # Add mock results
        evaluator.results = {
            'model1': {'accuracy': 0.8, 'f1': 0.75, 'precision': 0.7, 'recall': 0.8, 'roc_auc': 0.85},
            'model2': {'accuracy': 0.85, 'f1': 0.8, 'precision': 0.75, 'recall': 0.85, 'roc_auc': 0.9}
        }

        comparison = evaluator.compare_models()

        assert len(comparison) == 2
        assert 'model' in comparison.columns


class TestChurnVisualizer:
    """Test visualization functionality."""

    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        visualizer = ChurnVisualizer()
        assert visualizer.figsize == (12, 8)

    def test_plot_creation(self):
        """Test that plots can be created without errors."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'Churn': ['Yes', 'No', 'Yes', 'No', 'Yes']
        })

        visualizer = ChurnVisualizer()

        # This should not raise an error
        try:
            visualizer.plot_churn_distribution(df, show=False)
            success = True
        except Exception:
            success = False

        assert success is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
