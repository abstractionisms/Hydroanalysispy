"""
Configuration management for hydrology package.

Supports loading JSON and YAML configuration files with validation
and site-specific config extraction.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .paths import get_config_path, CONFIG_DIR
from .logging_setup import get_logger

logger = get_logger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


def load_config(
    config_name: str,
    config_type: Optional[str] = None,
    required: bool = True
) -> Dict[str, Any]:
    """
    Load configuration file (JSON or YAML).

    Automatically detects file format based on extension. Searches in
    configs/ directory and optional config_type subdirectory.

    Args:
        config_name: Name of config file (with or without extension), or a relative/absolute path
        config_type: Optional config subdirectory ('sites' or 'analysis')
        required: If True, raise error if config not found; if False, return empty dict

    Returns:
        Configuration dictionary

    Raises:
        ConfigurationError: If config file not found or invalid (when required=True)

    Examples:
        >>> config = load_config('spokane_watershed', 'sites')
        >>> config = load_config('climate_correlation.yaml', 'analysis')
        >>> config = load_config('default_config')
        >>> config = load_config('configs/analysis/example.yaml')  # Relative to project root
    """
    # Handle different path formats
    config_path = Path(config_name)

    # If it's an absolute path or contains path separators, use it as-is (relative to PROJECT_ROOT)
    if config_path.is_absolute():
        # Absolute path - use as-is
        pass
    elif '/' in config_name or '\\' in config_name:
        # Relative path with separators - resolve from PROJECT_ROOT
        from .paths import PROJECT_ROOT
        config_path = PROJECT_ROOT / config_name
    else:
        # Just a filename - use get_config_path to search in configs/
        config_path = get_config_path(config_name, config_type)

    if not config_path.exists():
        if required:
            raise ConfigurationError(
                f"Config file not found: {config_name} "
                f"(searched in: {config_path.parent})"
            )
        else:
            logger.warning(f"Config file not found: {config_path}, using empty config")
            return {}

    logger.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ConfigurationError(
                        "YAML config found but PyYAML not installed. "
                        "Install with: pip install pyyaml"
                    )
                config = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                config = json.load(f)
            else:
                raise ConfigurationError(
                    f"Unsupported config format: {config_path.suffix}"
                )

        logger.info(f"Configuration loaded successfully")
        return config if config is not None else {}

    except yaml.YAMLError as e:
        raise ConfigurationError(f"YAML parse error in {config_path}: {e}")
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"JSON parse error in {config_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading config {config_path}: {e}")


def get_site_config(
    site_id: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract configuration for a specific site from a config dict.

    Searches for site in 'sites_to_process', 'sites', or top-level keys.

    Args:
        site_id: Site identifier (e.g., '12422500')
        config: Full configuration dictionary

    Returns:
        Site-specific configuration dictionary

    Raises:
        ConfigurationError: If site not found in configuration

    Example:
        >>> config = load_config('spokane_watershed', 'sites')
        >>> site = get_site_config('12422500', config)
        >>> print(site['description'])
        'Spokane River at Spokane, WA'
    """
    # Try 'sites_to_process' (legacy format)
    sites = config.get('sites_to_process', [])
    if not sites:
        # Try 'sites' (new format)
        sites = config.get('sites', [])

    for site in sites:
        if isinstance(site, dict):
            # Check various possible ID keys
            site_id_value = (
                site.get('site_id') or
                site.get('id') or
                site.get('usgs_id') or
                site.get('station_id') or
                site.get('site')
            )
            if str(site_id_value) == str(site_id):
                return site

    raise ConfigurationError(
        f"Site {site_id} not found in configuration. "
        f"Available sites: {[s.get('site_id', s.get('id', 'unknown')) for s in sites]}"
    )


def get_enabled_sites(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get list of enabled sites from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of enabled site configurations

    Example:
        >>> config = load_config('spokane_watershed', 'sites')
        >>> enabled = get_enabled_sites(config)
        >>> for site in enabled:
        ...     print(site['site_id'], site['description'])
    """
    sites = config.get('sites_to_process', config.get('sites', []))

    enabled_sites = []
    for site in sites:
        if isinstance(site, dict):
            # Default to enabled if not specified
            if site.get('enabled', True):
                enabled_sites.append(site)

    return enabled_sites


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries (override takes precedence).

    Useful for combining default configs with specific configs.

    Args:
        base: Base configuration
        override: Override configuration (takes precedence)

    Returns:
        Merged configuration dictionary

    Example:
        >>> defaults = load_config('default_config')
        >>> specific = load_config('climate_correlation', 'analysis')
        >>> config = merge_configs(defaults, specific)
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_site_config(site: Dict[str, Any]) -> bool:
    """
    Validate that a site configuration has required fields.

    Args:
        site: Site configuration dictionary

    Returns:
        True if valid

    Raises:
        ConfigurationError: If required fields missing
    """
    required_fields = ['site_id']
    optional_fields = ['description', 'latitude', 'longitude', 'param_cd',
                      'start_date', 'end_date', 'enabled']

    for field in required_fields:
        if field not in site and 'id' not in site:
            raise ConfigurationError(
                f"Site config missing required field: {field}. Site: {site}"
            )

    return True
