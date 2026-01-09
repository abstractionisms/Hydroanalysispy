"""
Plot configuration and selector components for the Hydrology Dashboard.
Centralizes plot categorization and provides reusable selector widgets.
"""

import streamlit as st
from typing import List, Dict, Any

# =============================================================================
# PLOT CATEGORIES
# =============================================================================

# Climate-dependent plots (require merged discharge + climate data)
CLIMATE_PLOTS = [
    'anomaly',
    'hexbin_temp',
    'lagged_precip',
    'correlation_matrix',
    'precip_discharge',
    'seasonal_scatter',
    'double_mass_curve',
    'lag_correlation'
]

# Discharge-only plots (work with just discharge data)
DISCHARGE_PLOTS = [
    'timeseries',
    'flow_duration',
    'monthly_boxplot',
    'discharge_heatmap',
    'temporal_heatmap',
    'low_flow_trend',
    'annual_trend',
    'baseflow_separation',
    'recession_curves',
    'flood_frequency',
    '7q10_analysis',
    'anomaly_detection',
    'cumulative_departure',
    'spectral_analysis'
]

# Stage-dependent plots (require gage height data)
STAGE_PLOTS = [
    'rating_curve'
]

# All plots combined
ALL_PLOTS = DISCHARGE_PLOTS + CLIMATE_PLOTS + STAGE_PLOTS


# =============================================================================
# SELECTOR WIDGETS
# =============================================================================

def multi_plot_selector(available_plots: Dict[str, Any], key_prefix: str = "") -> List[str]:
    """
    Multi-select plot selector with categorized sections.
    Used in Single Analysis mode for selecting multiple plots.

    Args:
        available_plots: Dict of available plot configurations (AVAILABLE_PLOTS)
        key_prefix: Prefix for widget keys to avoid duplicates

    Returns:
        List of selected plot names
    """
    plot_names = list(available_plots.keys())

    # Climate plots section
    st.caption("🌡️ Climate-dependent plots (need merged data):")
    climate_available = [p for p in plot_names if p in CLIMATE_PLOTS]
    selected_climate = st.multiselect(
        "Climate plots",
        climate_available,
        default=['anomaly'] if 'anomaly' in climate_available else [],
        key=f"{key_prefix}climate_plots"
    )

    # Discharge plots section
    st.caption("📊 Discharge-only plots:")
    discharge_available = [p for p in plot_names if p in DISCHARGE_PLOTS]
    selected_discharge = st.multiselect(
        "Discharge plots",
        discharge_available,
        default=['timeseries'] if 'timeseries' in discharge_available else [],
        key=f"{key_prefix}discharge_plots"
    )

    # Stage plots section
    st.caption("📏 Stage-dependent plots (need gage height):")
    stage_available = [p for p in plot_names if p in STAGE_PLOTS]
    selected_stage = st.multiselect(
        "Stage plots",
        stage_available,
        default=[],
        key=f"{key_prefix}stage_plots"
    )

    return selected_climate + selected_discharge + selected_stage


def single_plot_selector(available_plots: Dict[str, Any], key_suffix: str = "") -> str:
    """
    Single-select plot selector with categorized display.
    Used in comparison modes for selecting one plot at a time.

    Args:
        available_plots: Dict of available plot configurations (AVAILABLE_PLOTS)
        key_suffix: Suffix for widget key to avoid duplicates

    Returns:
        Selected plot name
    """
    plot_names = list(available_plots.keys())

    # Build options list ordered by category (discharge first as most common)
    options = []
    labels = {}

    # Discharge plots first (most commonly used)
    for p in DISCHARGE_PLOTS:
        if p in plot_names:
            options.append(p)
            labels[p] = f"📊 {p}"

    # Climate plots
    for p in CLIMATE_PLOTS:
        if p in plot_names:
            options.append(p)
            labels[p] = f"🌡️ {p}"

    # Stage plots
    for p in STAGE_PLOTS:
        if p in plot_names:
            options.append(p)
            labels[p] = f"📏 {p}"

    selected = st.selectbox(
        "Select plot type",
        options,
        format_func=lambda x: labels.get(x, x),
        key=f"compare_plot{key_suffix}"
    )

    # Show plot description below selector
    if selected and selected in available_plots:
        info = available_plots[selected]
        if isinstance(info, dict):
            desc = info.get('description', '')
            if desc:
                st.caption(f"_{desc}_")

    return selected


def get_plot_category(plot_name: str) -> str:
    """
    Get the category of a plot.

    Returns:
        'climate', 'discharge', 'stage', or 'unknown'
    """
    if plot_name in CLIMATE_PLOTS:
        return 'climate'
    elif plot_name in DISCHARGE_PLOTS:
        return 'discharge'
    elif plot_name in STAGE_PLOTS:
        return 'stage'
    return 'unknown'


def get_plot_requirements(plot_name: str) -> List[str]:
    """
    Get data requirements for a plot.

    Returns:
        List of required data types: 'discharge', 'climate', 'stage'
    """
    category = get_plot_category(plot_name)

    if category == 'climate':
        return ['discharge', 'climate']
    elif category == 'stage':
        return ['discharge', 'stage']
    else:
        return ['discharge']
