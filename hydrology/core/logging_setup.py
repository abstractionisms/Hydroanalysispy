"""
Logging setup for hydrology package.

Provides consistent logging configuration across all scripts, eliminating
duplicate setup_logging() functions.
"""

import logging
from pathlib import Path
from typing import Optional
from .paths import get_log_path


def setup_logging(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_level: Optional[int] = None
) -> logging.Logger:
    """
    Configure logging for hydrology scripts.

    Sets up both file and console logging with consistent formatting.
    Removes any existing handlers to avoid duplicates.

    Args:
        name: Logger name (typically __name__ from calling module)
        log_file: Optional log file name (saved to outputs/logs/)
        level: Minimum logging level for file handler
        console_level: Optional separate level for console (defaults to level)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(__name__, 'analysis.log')
        >>> logger.info("Starting analysis...")
    """
    logger = logging.getLogger(name)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level if console_level is not None else level)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        try:
            log_path = get_log_path(log_file)
            file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
            logger.debug(f"Logging to file: {log_path}")
        except Exception as e:
            logger.warning(f"Could not set up file logging to {log_file}: {e}")

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a simple logger without file output (console only).

    Useful for library modules that shouldn't create log files.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data...")
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
