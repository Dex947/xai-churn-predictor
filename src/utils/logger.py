"""
Logging utility for the Churn Prediction System.

This module sets up centralized logging using loguru.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    log_file: str = "logs/churn_prediction.log",
    level: str = "INFO",
    log_format: str = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Set up the logger with file and console handlers.

    Args:
        log_file: Path to the log file.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Custom log format string.
        rotation: When to rotate log file (e.g., "500 MB", "1 week").
        retention: How long to keep old log files.
    """
    # Remove default handler
    logger.remove()

    # Default format if not provided
    if log_format is None:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console handler
    logger.add(
        sys.stdout,
        format=log_format,
        level=level,
        colorize=True,
    )

    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Add file handler with rotation
    logger.add(
        log_file,
        format=log_format,
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        encoding="utf-8",
    )

    logger.info(f"Logger initialized | Level: {level} | Log file: {log_file}")


def get_logger():
    """
    Get the configured logger instance.

    Returns:
        Logger instance.
    """
    return logger
