"""
Plot configuration and selector components for the Hydrology Dashboard.
Centralizes plot categorization and provides reusable selector widgets.
"""

import streamlit as st
from typing import List, Dict, Any

# =============================================================================
# PLOT DISPLAY NAMES - Human readable names for UI
# =============================================================================

PLOT_DISPLAY_NAMES = {
    # Climate plots
    'anomaly': 'Monthly Anomaly Analysis',
    'hexbin_temp': 'Discharge vs Temperature',
    'lagged_precip': 'Lagged Precipitation Scatter',
    'correlation_matrix': 'Correlation Matrix',
    'precip_discharge': 'Precipitation & Discharge Overlay',
    'seasonal_scatter': 'Seasonal Discharge Pattern',
    'double_mass_curve': 'Double Mass Curve',
    'lag_correlation': 'Precipitation Lag Correlation',

    # Discharge plots
    'timeseries': 'Recent Time Series',
    'flow_duration': 'Flow Duration Curve',
    'monthly_boxplot': 'Monthly Distribution',
    'discharge_heatmap': 'Discharge Density Heatmap',
    'temporal_heatmap': 'Multi-Period Heatmaps',
    'low_flow_trend': '7-Day Low Flow Trend',
    'annual_trend': 'Annual Mean Trend',
    'baseflow_separation': 'Baseflow Separation',
    'recession_curves': 'Recession Curve Analysis',
    'flood_frequency': 'Flood Frequency (Log-Pearson III)',
    '7q10_analysis': '7Q10 Low Flow Analysis',
    'anomaly_detection': 'Anomaly Detection',
    'cumulative_departure': 'Cumulative Departure',
    'spectral_analysis': 'Spectral Analysis (FFT)',

    # Stage plots
    'rating_curve': 'Stage-Discharge Rating Curve',
}

# =============================================================================
# PLOT CATEGORIES
# =============================================================================

CLIMATE_PLOTS = [
    'anomaly', 'hexbin_temp', 'lagged_precip', 'correlation_matrix',
    'precip_discharge', 'seasonal_scatter', 'double_mass_curve', 'lag_correlation'
]

DISCHARGE_PLOTS = [
    'timeseries', 'flow_duration', 'monthly_boxplot', 'discharge_heatmap',
    'temporal_heatmap', 'low_flow_trend', 'annual_trend', 'baseflow_separation',
    'recession_curves', 'flood_frequency', '7q10_analysis', 'anomaly_detection',
    'cumulative_departure', 'spectral_analysis'
]

STAGE_PLOTS = ['rating_curve']

ALL_PLOTS = DISCHARGE_PLOTS + CLIMATE_PLOTS + STAGE_PLOTS


def get_display_name(plot_key: str) -> str:
    """Get human-readable display name for a plot."""
    return PLOT_DISPLAY_NAMES.get(plot_key, plot_key.replace('_', ' ').title())


# =============================================================================
# SELECTOR WIDGETS
# =============================================================================

def multi_plot_selector(available_plots: Dict[str, Any], key_prefix: str = "") -> List[str]:
    """
    Multi-select plot selector with categorized sections in expanders.
    Used in Single Analysis mode for selecting multiple plots.
    """
    plot_names = list(available_plots.keys())

    with st.expander("Climate Plots (need weather data)", expanded=False):
        climate_available = [p for p in CLIMATE_PLOTS if p in plot_names]
        climate_options = {get_display_name(p): p for p in climate_available}
        selected_climate_names = st.multiselect(
            "Select climate plots",
            list(climate_options.keys()),
            default=[get_display_name('anomaly')] if 'anomaly' in climate_available else [],
            key=f"{key_prefix}climate_plots",
            label_visibility="collapsed"
        )
        selected_climate = [climate_options[n] for n in selected_climate_names]

    with st.expander("Discharge Plots", expanded=True):
        discharge_available = [p for p in DISCHARGE_PLOTS if p in plot_names]
        discharge_options = {get_display_name(p): p for p in discharge_available}
        selected_discharge_names = st.multiselect(
            "Select discharge plots",
            list(discharge_options.keys()),
            default=[get_display_name('timeseries')] if 'timeseries' in discharge_available else [],
            key=f"{key_prefix}discharge_plots",
            label_visibility="collapsed"
        )
        selected_discharge = [discharge_options[n] for n in selected_discharge_names]

    with st.expander("Stage Plots (need gage height)", expanded=False):
        stage_available = [p for p in STAGE_PLOTS if p in plot_names]
        stage_options = {get_display_name(p): p for p in stage_available}
        selected_stage_names = st.multiselect(
            "Select stage plots",
            list(stage_options.keys()),
            default=[],
            key=f"{key_prefix}stage_plots",
            label_visibility="collapsed"
        )
        selected_stage = [stage_options[n] for n in selected_stage_names]

    return selected_climate + selected_discharge + selected_stage


def single_plot_selector(available_plots: Dict[str, Any], key_suffix: str = "") -> str:
    """
    Single-select plot selector with categorized display.
    Used in comparison modes for selecting one plot at a time.
    """
    plot_names = list(available_plots.keys())

    # Build options with display names
    options = []
    name_to_key = {}

    # Discharge plots first (most common)
    for p in DISCHARGE_PLOTS:
        if p in plot_names:
            display = f"[Discharge] {get_display_name(p)}"
            options.append(display)
            name_to_key[display] = p

    # Climate plots
    for p in CLIMATE_PLOTS:
        if p in plot_names:
            display = f"[Climate] {get_display_name(p)}"
            options.append(display)
            name_to_key[display] = p

    # Stage plots
    for p in STAGE_PLOTS:
        if p in plot_names:
            display = f"[Stage] {get_display_name(p)}"
            options.append(display)
            name_to_key[display] = p

    selected_display = st.selectbox(
        "Select plot type",
        options,
        key=f"compare_plot{key_suffix}"
    )

    selected_key = name_to_key.get(selected_display, 'timeseries')

    # Show description
    if selected_key in available_plots:
        info = available_plots[selected_key]
        if isinstance(info, dict):
            desc = info.get('description', '')
            if desc:
                st.caption(f"_{desc}_")

    return selected_key


def get_plot_category(plot_name: str) -> str:
    """Get the category of a plot."""
    if plot_name in CLIMATE_PLOTS:
        return 'climate'
    elif plot_name in DISCHARGE_PLOTS:
        return 'discharge'
    elif plot_name in STAGE_PLOTS:
        return 'stage'
    return 'unknown'


def get_plot_requirements(plot_name: str) -> List[str]:
    """Get data requirements for a plot."""
    category = get_plot_category(plot_name)
    if category == 'climate':
        return ['discharge', 'climate']
    elif category == 'stage':
        return ['discharge', 'stage']
    return ['discharge']
