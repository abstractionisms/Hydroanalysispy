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
from .timezone import normalize_index_timezone, ensure_utc, remove_timezone
from .parameters import (
    DISCHARGE,
    GAGE_HEIGHT,
    WATER_TEMP,
    DEFAULT_DISCHARGE_CODE,
    DEFAULT_STAGE_CODE,
    DEFAULT_TEMP_CODE,
    STAT_MEAN,
    STAT_MIN,
    STAT_MAX,
    get_parameter_info,
    get_parameter_name,
    get_parameter_unit,
)

__all__ = [
    # Paths
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
    # Logging
    'setup_logging',
    # Config
    'load_config',
    'get_site_config',
    'ConfigurationError',
    # Timezone
    'normalize_index_timezone',
    'ensure_utc',
    'remove_timezone',
    # Parameters
    'DISCHARGE',
    'GAGE_HEIGHT',
    'WATER_TEMP',
    'DEFAULT_DISCHARGE_CODE',
    'DEFAULT_STAGE_CODE',
    'DEFAULT_TEMP_CODE',
    'STAT_MEAN',
    'STAT_MIN',
    'STAT_MAX',
    'get_parameter_info',
    'get_parameter_name',
    'get_parameter_unit',
]
