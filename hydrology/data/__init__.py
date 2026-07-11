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
from .groundwater import (
    fetch_usgs_groundwater_measurements,
    normalize_usgs_groundwater_measurements,
)

# HyRiver imports are optional (require conda-forge packages)
try:
    from .hyriver import (
        get_watershed_boundary, get_flowlines, get_navigation_flowlines, get_basin_characteristics,
        get_daymet_climate, get_nid_dams, get_elevation_profile,
    )
    _HYRIVER_AVAILABLE = True
except ImportError:
    _HYRIVER_AVAILABLE = False

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
    'fetch_usgs_groundwater_measurements',
    'normalize_usgs_groundwater_measurements',
    # HyRiver (optional)
    'get_watershed_boundary',
    'get_flowlines',
    'get_navigation_flowlines',
    'get_basin_characteristics',
    'get_daymet_climate',
    'get_nid_dams',
    'get_elevation_profile',
]
