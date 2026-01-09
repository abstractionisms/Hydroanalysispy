"""
Path management for hydrology package.

This module defines all project paths relative to the project root, eliminating
hardcoded absolute paths and making the code portable across machines.
"""

from pathlib import Path
from typing import Optional

# Project root is 3 levels up from this file:
# this file -> core/ -> hydrology/ -> Hydrology/ (project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Output subdirectories
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"
CACHE_DIR = OUTPUT_DIR / "cache"
EXPORT_DIR = OUTPUT_DIR / "exports"

# Archive directories
ARCHIVE_DIR = PROJECT_ROOT / "archive"
DEPRECATED_DIR = ARCHIVE_DIR / "deprecated"
VARIANTS_DIR = ARCHIVE_DIR / "variants"


def ensure_dir(path: Path) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        path: Path to directory

    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_site_plot_dir(site_id: str, subdir: Optional[str] = None) -> Path:
    """
    Get plot directory for a specific site, creating it if needed.

    Args:
        site_id: USGS site identifier
        subdir: Optional subdirectory within site directory

    Returns:
        Path to site plot directory

    Example:
        >>> plot_dir = get_site_plot_dir('12422500', 'stage_discharge')
        >>> # Returns: outputs/plots/12422500/stage_discharge/
    """
    base = PLOT_DIR / site_id
    if subdir:
        base = base / subdir
    return ensure_dir(base)


def get_config_path(config_name: str, config_type: Optional[str] = None) -> Path:
    """
    Get configuration file path.

    Args:
        config_name: Configuration name (with or without extension)
        config_type: Optional config type ('sites' or 'analysis')

    Returns:
        Path to configuration file

    Example:
        >>> config_path = get_config_path('spokane_watershed', 'sites')
        >>> # Returns: configs/sites/spokane_watershed.yaml
    """
    base = CONFIG_DIR
    if config_type:
        base = base / config_type

    # Add extension if not present
    if not config_name.endswith(('.yaml', '.yml', '.json')):
        # Try YAML first, then JSON
        yaml_path = base / f"{config_name}.yaml"
        yml_path = base / f"{config_name}.yml"
        json_path = base / f"{config_name}.json"

        if yaml_path.exists():
            return yaml_path
        elif yml_path.exists():
            return yml_path
        elif json_path.exists():
            return json_path
        else:
            # Default to YAML for new configs
            return yaml_path
    else:
        return base / config_name


def get_log_path(log_name: str) -> Path:
    """
    Get log file path in the logs directory.

    Args:
        log_name: Log file name (with or without .log extension)

    Returns:
        Path to log file
    """
    if not log_name.endswith('.log'):
        log_name = f"{log_name}.log"
    return ensure_dir(LOG_DIR) / log_name


# Ensure critical directories exist
ensure_dir(DATA_DIR)
ensure_dir(CONFIG_DIR)
ensure_dir(OUTPUT_DIR)
ensure_dir(PLOT_DIR)
ensure_dir(LOG_DIR)
ensure_dir(CACHE_DIR)
ensure_dir(EXPORT_DIR)
