"""
Configuration loader utility for the Churn Prediction System.

This module handles loading and validating configuration from YAML files.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger


class ConfigLoader:
    """Load and manage configuration settings."""

    def __init__(self, config_path: str = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the configuration YAML file.
                        If None, uses default config/config.yaml
        """
        if config_path is None:
            # Get project root directory
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_directories()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Dictionary containing configuration settings.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            yaml.YAMLError: If YAML file is invalid.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise

    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        # Get project root
        project_root = self.config_path.parent.parent

        # Create data directories
        data_config = self.config.get("data", {})
        directories = [
            data_config.get("raw_data_dir", "data/raw"),
            data_config.get("processed_data_dir", "data/processed"),
            data_config.get("models_dir", "data/models"),
            data_config.get("results_dir", "data/results"),
            data_config.get("plots_dir", "data/plots"),
        ]

        # Create logs directory
        log_file = self.config.get("logging", {}).get("log_file", "logs/churn_prediction.log")
        log_dir = Path(log_file).parent
        directories.append(str(log_dir))

        # Create all directories
        for directory in directories:
            dir_path = project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.debug("Required directories created/verified")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Supports nested keys using dot notation (e.g., 'data.raw_data_dir').

        Args:
            key: Configuration key (supports dot notation for nested keys).
            default: Default value if key doesn't exist.

        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_data_path(self, data_type: str) -> Path:
        """
        Get the absolute path for a data directory.

        Args:
            data_type: Type of data directory (e.g., 'raw_data_dir', 'models_dir').

        Returns:
            Absolute path to the data directory.
        """
        project_root = self.config_path.parent.parent
        relative_path = self.config["data"].get(f"{data_type}", f"data/{data_type}")
        return project_root / relative_path

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.

        Args:
            model_name: Name of the model (e.g., 'logistic_regression', 'xgboost').

        Returns:
            Dictionary containing model configuration.
        """
        models_config = self.config.get("models", {})
        model_config = models_config.get(model_name, {})

        if not model_config:
            logger.warning(f"No configuration found for model: {model_name}")

        return model_config

    def is_model_enabled(self, model_name: str) -> bool:
        """
        Check if a model is enabled in the configuration.

        Args:
            model_name: Name of the model.

        Returns:
            True if model is enabled, False otherwise.
        """
        model_config = self.get_model_config(model_name)
        return model_config.get("enabled", False)

    def get_enabled_models(self) -> list:
        """
        Get list of enabled models.

        Returns:
            List of model names that are enabled.
        """
        models_config = self.config.get("models", {})
        return [
            name for name, config in models_config.items()
            if config.get("enabled", False)
        ]

    def update_config(self, key: str, value: Any) -> None:
        """
        Update a configuration value.

        Args:
            key: Configuration key (supports dot notation).
            value: New value.
        """
        keys = key.split(".")
        config = self.config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Update the value
        config[keys[-1]] = value
        logger.debug(f"Configuration updated: {key} = {value}")

    def save_config(self, output_path: str = None) -> None:
        """
        Save configuration to YAML file.

        Args:
            output_path: Path to save the configuration.
                        If None, overwrites the original file.
        """
        if output_path is None:
            output_path = self.config_path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {output_path}")

    def __repr__(self) -> str:
        """String representation of the ConfigLoader."""
        return f"ConfigLoader(config_path={self.config_path})"


# Global config instance (lazy-loaded)
_config_instance = None


def get_config(config_path: str = None) -> ConfigLoader:
    """
    Get the global configuration instance.

    Args:
        config_path: Path to configuration file (only used on first call).

    Returns:
        ConfigLoader instance.
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)

    return _config_instance


def reload_config(config_path: str = None) -> ConfigLoader:
    """
    Reload the global configuration instance.

    Args:
        config_path: Path to configuration file.

    Returns:
        New ConfigLoader instance.
    """
    global _config_instance
    _config_instance = ConfigLoader(config_path)
    return _config_instance
