"""Data fetching and parsing modules for hydrology package."""

from .usgs import (
    fetch_waterml_data,
    parse_waterml,
    fetch_discharge_data,
    BASE_URL_DV,
    BASE_URL_IV,
    DEFAULT_PARAM_DISCHARGE,
    DEFAULT_PARAM_STAGE,
)
from .climate import fetch_climate_data
from .inventory import (
    load_inventory,
    get_site_info,
    get_multiple_sites,
    search_sites,
)
from .nwm import NWMClient, NWMForecast, compare_nwm_usgs, get_forecast_skill

__all__ = [
    'fetch_waterml_data',
    'parse_waterml',
    'fetch_discharge_data',
    'fetch_climate_data',
    'load_inventory',
    'get_site_info',
    'get_multiple_sites',
    'search_sites',
    'BASE_URL_DV',
    'BASE_URL_IV',
    'DEFAULT_PARAM_DISCHARGE',
    'DEFAULT_PARAM_STAGE',
    # National Water Model
    'NWMClient',
    'NWMForecast',
    'compare_nwm_usgs',
    'get_forecast_skill',
]
