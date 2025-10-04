"""
Data preprocessing module for the Churn Prediction System.

This module handles data cleaning, encoding, scaling, and train-test splitting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler


class DataPreprocessor:
    """Preprocess customer churn data for machine learning."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the DataPreprocessor.

        Args:
            config: Configuration dictionary containing preprocessing parameters.
        """
        self.config = config or {}
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None
        self.target_encoder = None

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "drop",
        fill_value: Any = None,
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: DataFrame with potential missing values.
            strategy: Strategy to handle missing values ('drop', 'mean', 'median', 'mode', 'constant').
            fill_value: Value to use when strategy is 'constant'.

        Returns:
            DataFrame with missing values handled.
        """
        logger.info(f"Handling missing values using strategy: {strategy}")

        initial_missing = df.isnull().sum().sum()
        logger.info(f"Initial missing values: {initial_missing}")

        if initial_missing == 0:
            logger.info("No missing values found")
            return df

        df_clean = df.copy()

        if strategy == "drop":
            df_clean = df_clean.dropna()
            logger.info(f"Dropped rows with missing values | New shape: {df_clean.shape}")

        elif strategy == "mean":
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())

        elif strategy == "median":
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

        elif strategy == "mode":
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        elif strategy == "constant":
            df_clean = df_clean.fillna(fill_value)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        final_missing = df_clean.isnull().sum().sum()
        logger.info(f"Final missing values: {final_missing}")

        return df_clean

    def clean_data(self, df: pd.DataFrame, drop_columns: List[str] = None) -> pd.DataFrame:
        """
        Perform initial data cleaning.

        Args:
            df: Raw DataFrame.
            drop_columns: List of columns to drop.

        Returns:
            Cleaned DataFrame.
        """
        logger.info("Starting data cleaning")
        df_clean = df.copy()

        # Drop specified columns
        if drop_columns:
            existing_cols = [col for col in drop_columns if col in df_clean.columns]
            if existing_cols:
                df_clean = df_clean.drop(columns=existing_cols)
                logger.info(f"Dropped columns: {existing_cols}")

        # Handle TotalCharges (known issue in Telco dataset - spaces instead of numbers)
        if "TotalCharges" in df_clean.columns:
            # Convert to numeric, replacing errors with NaN
            df_clean["TotalCharges"] = pd.to_numeric(
                df_clean["TotalCharges"], errors="coerce"
            )
            logger.info("Converted TotalCharges to numeric")

        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_rows - len(df_clean)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")

        logger.info(f"Data cleaning completed | Final shape: {df_clean.shape}")

        return df_clean

    def encode_target(self, y: pd.Series, fit: bool = True) -> np.ndarray:
        """
        Encode the target variable.

        Args:
            y: Target variable.
            fit: Whether to fit the encoder (True for training, False for inference).

        Returns:
            Encoded target variable.
        """
        if fit:
            self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(y)
            logger.info(
                f"Target encoded | Classes: {self.target_encoder.classes_} | "
                f"Mapping: {dict(zip(self.target_encoder.classes_, self.target_encoder.transform(self.target_encoder.classes_)))}"
            )
        else:
            if self.target_encoder is None:
                raise ValueError("Target encoder not fitted. Call with fit=True first.")
            y_encoded = self.target_encoder.transform(y)

        return y_encoded

    def encode_categorical(
        self,
        df: pd.DataFrame,
        method: str = "onehot",
        categorical_cols: List[str] = None,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            df: DataFrame with categorical variables.
            method: Encoding method ('onehot' or 'label').
            categorical_cols: List of categorical columns. If None, auto-detect.
            fit: Whether to fit encoders (True for training, False for inference).

        Returns:
            DataFrame with encoded categorical variables.
        """
        df_encoded = df.copy()

        # Auto-detect categorical columns if not provided
        if categorical_cols is None:
            categorical_cols = df_encoded.select_dtypes(include=["object", "category"]).columns.tolist()

        logger.info(f"Encoding {len(categorical_cols)} categorical columns using {method}")

        if method == "onehot":
            if fit:
                df_encoded = pd.get_dummies(
                    df_encoded,
                    columns=categorical_cols,
                    drop_first=True,  # Avoid multicollinearity
                    dtype=int,
                )
                logger.info(f"One-hot encoding completed | New shape: {df_encoded.shape}")
            else:
                # For inference, we need to maintain the same columns
                df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True, dtype=int)

        elif method == "label":
            for col in categorical_cols:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                    logger.debug(f"Label encoded: {col} | Classes: {le.classes_}")
                else:
                    if col not in self.label_encoders:
                        raise ValueError(f"No encoder found for column {col}")
                    le = self.label_encoders[col]
                    df_encoded[col] = le.transform(df_encoded[col].astype(str))

        else:
            raise ValueError(f"Unknown encoding method: {method}")

        return df_encoded

    def scale_features(
        self,
        X: pd.DataFrame,
        method: str = "standard",
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Scale numerical features.

        Args:
            X: Features DataFrame.
            method: Scaling method ('standard', 'minmax', 'robust', 'none').
            fit: Whether to fit the scaler (True for training, False for inference).

        Returns:
            DataFrame with scaled features.
        """
        if method == "none":
            logger.info("No scaling applied")
            return X

        logger.info(f"Scaling features using {method} scaler")

        if fit:
            if method == "standard":
                self.scaler = StandardScaler()
            elif method == "minmax":
                self.scaler = MinMaxScaler()
            elif method == "robust":
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")

            X_scaled = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)

        # Convert back to DataFrame
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        logger.info(f"Scaling completed | Shape: {X_scaled.shape}")

        return X_scaled

    def handle_class_imbalance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        method: str = "smote",
        sampling_strategy: str = "auto",
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Handle class imbalance using oversampling techniques.

        Args:
            X: Features.
            y: Target variable.
            method: Sampling method ('smote', 'adasyn', 'random_oversample', 'none').
            sampling_strategy: Ratio of minority to majority class.
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (resampled_X, resampled_y).
        """
        if method == "none":
            logger.info("No class imbalance handling applied")
            return X, y

        logger.info(f"Handling class imbalance using {method}")

        # Log class distribution before resampling
        unique, counts = np.unique(y, return_counts=True)
        class_dist_before = dict(zip(unique, counts))
        logger.info(f"Class distribution before resampling: {class_dist_before}")

        if method == "smote":
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == "adasyn":
            sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == "random_oversample":
            sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Convert back to DataFrame
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

        # Log class distribution after resampling
        unique, counts = np.unique(y_resampled, return_counts=True)
        class_dist_after = dict(zip(unique, counts))
        logger.info(f"Class distribution after resampling: {class_dist_after}")
        logger.info(f"Resampling completed | New shape: {X_resampled.shape}")

        return X_resampled, y_resampled

    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        val_size: float = 0.0,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Dict[str, Any]:
        """
        Split data into train, validation (optional), and test sets.

        Args:
            df: Complete DataFrame.
            target_column: Name of the target column.
            test_size: Proportion of data for test set.
            val_size: Proportion of data for validation set (from training data).
            random_state: Random seed for reproducibility.
            stratify: Whether to stratify the split based on target.

        Returns:
            Dictionary containing train, val (if applicable), and test splits.
        """
        logger.info("Splitting data into train/validation/test sets")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        stratify_param = y if stratify else None

        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )

        splits = {
            "X_test": X_test,
            "y_test": y_test,
        }

        # Further split train into train and validation if needed
        if val_size > 0:
            # Adjust val_size relative to train_val size
            val_size_adjusted = val_size / (1 - test_size)
            stratify_param = y_train_val if stratify else None

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val,
                y_train_val,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=stratify_param,
            )

            splits["X_train"] = X_train
            splits["y_train"] = y_train
            splits["X_val"] = X_val
            splits["y_val"] = y_val

            logger.info(
                f"Data split completed | Train: {len(X_train)} | "
                f"Val: {len(X_val)} | Test: {len(X_test)}"
            )
        else:
            splits["X_train"] = X_train_val
            splits["y_train"] = y_train_val

            logger.info(
                f"Data split completed | Train: {len(X_train_val)} | Test: {len(X_test)}"
            )

        return splits

    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str = "Churn",
        fit: bool = True,
        config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.

        Args:
            df: Raw DataFrame.
            target_column: Name of the target column.
            fit: Whether to fit transformers (True for training, False for inference).
            config: Configuration dictionary (uses self.config if None).

        Returns:
            Dictionary containing processed data and metadata.
        """
        if config is None:
            config = self.config

        logger.info("Starting preprocessing pipeline")

        # 1. Clean data
        drop_columns = config.get("features", {}).get("drop_columns", ["customerID"])
        df_clean = self.clean_data(df, drop_columns)

        # 2. Handle missing values
        missing_strategy = config.get("preprocessing", {}).get("missing_value_strategy", "drop")
        df_clean = self.handle_missing_values(df_clean, strategy=missing_strategy)

        # 3. Split features and target
        if target_column not in df_clean.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        y = df_clean[target_column]
        X = df_clean.drop(columns=[target_column])

        # 4. Encode categorical features
        encoding_method = config.get("preprocessing", {}).get("categorical_encoding", "onehot")
        X_encoded = self.encode_categorical(X, method=encoding_method, fit=fit)

        # 5. Encode target
        y_encoded = self.encode_target(y, fit=fit)

        # 6. Scale features
        scaling_method = config.get("preprocessing", {}).get("numeric_scaling", "standard")
        X_scaled = self.scale_features(X_encoded, method=scaling_method, fit=fit)

        # Store feature names
        if fit:
            self.feature_names = X_scaled.columns.tolist()

        result = {
            "X": X_scaled,
            "y": y_encoded,
            "feature_names": self.feature_names,
            "target_classes": self.target_encoder.classes_ if self.target_encoder else None,
        }

        logger.info(f"Preprocessing pipeline completed | Features: {X_scaled.shape[1]}")

        return result

    def save_preprocessor(self, file_path: str) -> None:
        """
        Save the preprocessor state.

        Args:
            file_path: Path to save the preprocessor.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "label_encoders": self.label_encoders,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "target_encoder": self.target_encoder,
            "config": self.config,
        }

        joblib.dump(state, file_path)
        logger.info(f"Preprocessor saved to {file_path}")

    def load_preprocessor(self, file_path: str) -> None:
        """
        Load the preprocessor state.

        Args:
            file_path: Path to load the preprocessor from.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {file_path}")

        state = joblib.load(file_path)

        self.label_encoders = state["label_encoders"]
        self.scaler = state["scaler"]
        self.feature_names = state["feature_names"]
        self.target_encoder = state["target_encoder"]
        self.config = state["config"]

        logger.info(f"Preprocessor loaded from {file_path}")
