"""
USGS Parameter Code definitions for hydrology package.

This module centralizes all USGS NWIS parameter codes used throughout
the package. Using these constants ensures consistency and makes it
easy to add support for new parameters.

Reference: https://help.waterdata.usgs.gov/codes-and-parameters/parameters
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterInfo:
    """Information about a USGS parameter code."""
    code: str
    name: str
    description: str
    unit: str
    unit_abbrev: str


# =============================================================================
# Primary Parameter Codes
# =============================================================================

# Discharge (Streamflow)
DISCHARGE = ParameterInfo(
    code="00060",
    name="Discharge",
    description="Discharge, cubic feet per second",
    unit="cubic feet per second",
    unit_abbrev="cfs"
)

# Gage Height (Stage)
GAGE_HEIGHT = ParameterInfo(
    code="00065",
    name="Gage Height",
    description="Gage height, feet",
    unit="feet",
    unit_abbrev="ft"
)

# Water Temperature
WATER_TEMP = ParameterInfo(
    code="00010",
    name="Water Temperature",
    description="Temperature, water, degrees Celsius",
    unit="degrees Celsius",
    unit_abbrev="°C"
)


# =============================================================================
# Additional Parameter Codes (for future expansion)
# =============================================================================

# Dissolved Oxygen
DISSOLVED_OXYGEN = ParameterInfo(
    code="00300",
    name="Dissolved Oxygen",
    description="Dissolved oxygen, water, unfiltered, mg/L",
    unit="milligrams per liter",
    unit_abbrev="mg/L"
)

# pH
PH = ParameterInfo(
    code="00400",
    name="pH",
    description="pH, water, unfiltered, field, standard units",
    unit="standard units",
    unit_abbrev="pH"
)

# Specific Conductance
SPECIFIC_CONDUCTANCE = ParameterInfo(
    code="00095",
    name="Specific Conductance",
    description="Specific conductance, water, unfiltered, µS/cm @ 25°C",
    unit="microsiemens per centimeter at 25°C",
    unit_abbrev="µS/cm"
)

# Turbidity
TURBIDITY = ParameterInfo(
    code="63680",
    name="Turbidity",
    description="Turbidity, water, unfiltered, broad band light source (400-680 nm)",
    unit="formazin nephelometric units",
    unit_abbrev="FNU"
)


# =============================================================================
# Statistic Codes for Daily Values
# =============================================================================

STAT_MEAN = "00003"    # Mean
STAT_MIN = "00002"     # Minimum
STAT_MAX = "00001"     # Maximum
STAT_MEDIAN = "00008"  # Median


# =============================================================================
# Convenience Mappings
# =============================================================================

# Map parameter codes to ParameterInfo objects
PARAMETER_REGISTRY: Dict[str, ParameterInfo] = {
    "00060": DISCHARGE,
    "00065": GAGE_HEIGHT,
    "00010": WATER_TEMP,
    "00300": DISSOLVED_OXYGEN,
    "00400": PH,
    "00095": SPECIFIC_CONDUCTANCE,
    "63680": TURBIDITY,
}

# Default parameters for common operations
DEFAULT_DISCHARGE_CODE = DISCHARGE.code
DEFAULT_STAGE_CODE = GAGE_HEIGHT.code
DEFAULT_TEMP_CODE = WATER_TEMP.code


def get_parameter_info(param_code: str) -> ParameterInfo:
    """
    Get parameter information by code.

    Args:
        param_code: USGS parameter code (e.g., '00060')

    Returns:
        ParameterInfo object with name, description, units

    Raises:
        KeyError: If parameter code is not in registry

    Example:
        >>> info = get_parameter_info('00060')
        >>> print(f"{info.name}: {info.description}")
        Discharge: Discharge, cubic feet per second
    """
    if param_code in PARAMETER_REGISTRY:
        return PARAMETER_REGISTRY[param_code]
    raise KeyError(f"Unknown parameter code: {param_code}")


def get_parameter_name(param_code: str) -> str:
    """
    Get human-readable name for a parameter code.

    Args:
        param_code: USGS parameter code

    Returns:
        Parameter name or the code itself if unknown
    """
    try:
        return PARAMETER_REGISTRY[param_code].name
    except KeyError:
        return param_code


def get_parameter_unit(param_code: str) -> str:
    """
    Get the unit abbreviation for a parameter code.

    Args:
        param_code: USGS parameter code

    Returns:
        Unit abbreviation or empty string if unknown
    """
    try:
        return PARAMETER_REGISTRY[param_code].unit_abbrev
    except KeyError:
        return ""
