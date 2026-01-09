"""Core utilities for hydrology package."""

from .paths import (
    PROJECT_ROOT,
    DATA_DIR,
    CONFIG_DIR,
    OUTPUT_DIR,
    PLOT_DIR,
    LOG_DIR,
    CACHE_DIR,
    EXPORT_DIR,
    ensure_dir,
    get_site_plot_dir,
)
from .logging_setup import setup_logging
from .config import load_config, get_site_config, ConfigurationError

__all__ = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'CONFIG_DIR',
    'OUTPUT_DIR',
    'PLOT_DIR',
    'LOG_DIR',
    'CACHE_DIR',
    'EXPORT_DIR',
    'ensure_dir',
    'get_site_plot_dir',
    'setup_logging',
    'load_config',
    'get_site_config',
    'ConfigurationError',
]
