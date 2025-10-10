"""
Data ingestion module for the Churn Prediction System.

This module handles downloading and loading the customer churn dataset.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm


class DataLoader:
    """Load and download customer churn datasets."""

    def __init__(self, data_dir: str = "data/raw", dataset_url: str = None):
        """
        Initialize the DataLoader.

        Args:
            data_dir: Directory to store raw data.
            dataset_url: URL to download the dataset from.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_url = dataset_url

    def download_dataset(
        self,
        url: str = None,
        filename: str = "Telco-Customer-Churn.csv",
        force_download: bool = False,
    ) -> Path:
        """
        Download the dataset from a URL.

        Args:
            url: URL to download from. If None, uses the instance's dataset_url.
            filename: Name to save the file as.
            force_download: If True, download even if file exists.

        Returns:
            Path to the downloaded file.

        Raises:
            ValueError: If no URL is provided.
            requests.RequestException: If download fails.
        """
        if url is None:
            url = self.dataset_url

        if url is None:
            raise ValueError("No URL provided for dataset download")

        file_path = self.data_dir / filename

        # Check if file already exists
        if file_path.exists() and not force_download:
            logger.info(f"Dataset already exists at {file_path}")
            return file_path

        logger.info(f"Downloading dataset from {url}")

        try:
            # Stream download with progress bar
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(file_path, "wb") as f, tqdm(
                desc=filename,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"Dataset downloaded successfully to {file_path}")
            return file_path

        except requests.RequestException as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

    def load_csv(
        self,
        file_path: str = None,
        filename: str = "Telco-Customer-Churn.csv",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            file_path: Full path to the CSV file. If None, uses data_dir/filename.
            filename: Filename to load (used if file_path is None).
            **kwargs: Additional arguments to pass to pd.read_csv.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        if file_path is None:
            file_path = self.data_dir / filename

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading data from {file_path}")

        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info(
                f"Data loaded successfully | Shape: {df.shape} | Columns: {len(df.columns)}"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise

    def load_data(
        self,
        url: str = None,
        filename: str = "Telco-Customer-Churn.csv",
        force_download: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Download (if needed) and load the dataset.

        Args:
            url: URL to download from.
            filename: Name of the file.
            force_download: If True, download even if file exists.
            **kwargs: Additional arguments to pass to pd.read_csv.

        Returns:
            DataFrame containing the loaded data.
        """
        file_path = self.data_dir / filename

        # Download if file doesn't exist or force_download is True
        if not file_path.exists() or force_download:
            if url or self.dataset_url:
                self.download_dataset(url, filename, force_download)
            else:
                raise FileNotFoundError(
                    f"File {file_path} not found and no URL provided for download"
                )

        return self.load_csv(file_path, **kwargs)

    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset.

        Args:
            df: DataFrame to analyze.

        Returns:
            Dictionary containing dataset information.
        """
        info = {
            "shape": df.shape,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicates": df.duplicated().sum(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        }

        logger.debug(f"Dataset info: {info['num_rows']} rows, {info['num_columns']} columns")

        return info

    def save_data(
        self,
        df: pd.DataFrame,
        filename: str,
        output_dir: str = None,
        **kwargs,
    ) -> Path:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame to save.
            filename: Name of the output file.
            output_dir: Directory to save to. If None, uses data_dir.
            **kwargs: Additional arguments to pass to df.to_csv.

        Returns:
            Path to the saved file.
        """
        if output_dir is None:
            output_dir = self.data_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / filename

        logger.info(f"Saving data to {file_path}")

        df.to_csv(file_path, index=False, **kwargs)

        logger.info(f"Data saved successfully | Shape: {df.shape}")

        return file_path

    def load_from_local(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a local file path.

        Args:
            file_path: Path to the local file.

        Returns:
            DataFrame containing the loaded data.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading data from local file: {file_path}")

        # Determine file type from extension
        extension = file_path.suffix.lower()

        if extension == ".csv":
            df = pd.read_csv(file_path)
        elif extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif extension == ".json":
            df = pd.read_json(file_path)
        elif extension == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

        logger.info(f"Data loaded successfully | Shape: {df.shape}")

        return df

    def validate_dataset(
        self,
        df: pd.DataFrame,
        required_columns: list = None,
        target_column: str = "Churn",
    ) -> Tuple[bool, list]:
        """
        Validate that the dataset has the required structure.

        Args:
            df: DataFrame to validate.
            required_columns: List of required column names.
            target_column: Name of the target column.

        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        issues = []

        # Check if DataFrame is empty
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues

        # Check for target column
        if target_column and target_column not in df.columns:
            issues.append(f"Target column '{target_column}' not found")

        # Check for required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                issues.append(f"Missing required columns: {missing_columns}")

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Dataset validation passed")
        else:
            logger.warning(f"Dataset validation failed: {issues}")

        return is_valid, issues


if __name__ == "__main__":
    # Example usage
    loader = DataLoader(
        dataset_url="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    )

    # Load data
    df = loader.load_data()

    # Get info
    info = loader.get_data_info(df)
    print(f"Dataset shape: {info['shape']}")
    print(f"Missing values: {info['missing_values']}")

    # Validate
    is_valid, issues = loader.validate_dataset(df, target_column="Churn")
    print(f"Valid: {is_valid}, Issues: {issues}")
