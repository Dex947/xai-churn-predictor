"""
Constants and configuration values for the Churn Prediction System.

This module centralizes all magic numbers and hard-coded values.
"""

# SHAP Explainer Configuration
SHAP_BACKGROUND_SAMPLES_DEFAULT = 100
SHAP_BACKGROUND_SAMPLES_FALLBACK = 50
SHAP_MAX_DISPLAY_FEATURES = 10

# Visualization Configuration
PLOT_TEXT_OFFSET = 50
PLOT_HISTOGRAM_BINS = 30
PLOT_MAX_CATEGORIES = 10
PLOT_DPI = 300
PLOT_BBOX_INCHES = "tight"

# Model Configuration
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_JOBS = -1

# Dashboard Configuration
DASHBOARD_PORT = 8501
DASHBOARD_MAX_FILE_SIZE_MB = 200

# Data Validation
MIN_DATASET_ROWS = 100
MAX_MISSING_PERCENTAGE = 50.0

# File Paths
DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_LOG_PATH = "logs/churn_prediction.log"
DEFAULT_MODELS_DIR = "data/models"
DEFAULT_PLOTS_DIR = "data/plots"
DEFAULT_RESULTS_DIR = "data/results"

# Class Names
CHURN_CLASS_NAMES = ["No Churn", "Churn"]
CHURN_COLORS = ["#2ecc71", "#e74c3c"]  # Green, Red

# Logging
LOG_ROTATION_SIZE = "10 MB"
LOG_RETENTION_PERIOD = "1 week"
