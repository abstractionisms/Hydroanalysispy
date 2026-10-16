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

    # Reach analysis plots
    'reach_comparison': 'Reach Comparison',
    'summer_low_flow_trend': 'Summer 7-Day Low Flow Trend',
    'reach_index': 'Aquifer Contribution Index',
    'paired_annual_lows': 'Paired Annual Lows (Low-Flow Window)',
    'avista_window_comparison': 'Low-Flow Window Hydrograph Overlay',
    'threshold_exceedance': 'Days Below Critical Thresholds',
    'precip_response_comparison': 'Precipitation Response Comparison',
    'summer_climate_context': 'Summer Climate Context',
    'seasonal_gain_loss': 'Seasonal Reach Gain/Loss',
    'seasonal_gain_loss_annual': 'Seasonal Gain/Loss by Water-Year Period',
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
    'temporal_heatmap', 'low_flow_trend', 'summer_low_flow_trend', 'annual_trend',
    'baseflow_separation', 'recession_curves', 'flood_frequency', '7q10_analysis',
    'anomaly_detection', 'cumulative_departure', 'spectral_analysis'
]

STAGE_PLOTS = ['rating_curve']

REACH_PLOTS = ['reach_comparison', 'reach_index', 'paired_annual_lows',
               'avista_window_comparison', 'threshold_exceedance', 'precip_response_comparison',
               'summer_climate_context', 'seasonal_gain_loss', 'seasonal_gain_loss_annual']

ALL_PLOTS = DISCHARGE_PLOTS + CLIMATE_PLOTS + STAGE_PLOTS + REACH_PLOTS
SINGLE_SITE_PLOTS = DISCHARGE_PLOTS + CLIMATE_PLOTS + STAGE_PLOTS

PLOT_PRESETS = {
    "Manual selection": {
        "intent": "Power user",
        "description": "Start empty and choose any plots below.",
        "plots": [],
    },
    "Quick site summary": {
        "intent": "First-pass read",
        "description": "A compact read on recent behavior, distribution, seasonality, and trend.",
        "plots": ["timeseries", "flow_duration", "monthly_boxplot", "annual_trend"],
    },
    "Flood frequency": {
        "intent": "High-flow risk",
        "description": "High-flow context and return-period analysis.",
        "plots": ["timeseries", "flow_duration", "flood_frequency", "anomaly_detection"],
    },
    "Drought / low-flow": {
        "intent": "Low-flow risk",
        "description": "Low-flow trend, 7Q10, duration, and threshold context.",
        "plots": ["timeseries", "flow_duration", "low_flow_trend", "7q10_analysis"],
    },
    "Climate relationship": {
        "intent": "Weather response",
        "description": "Precipitation, temperature, lag, and correlation views.",
        "plots": ["precip_discharge", "lag_correlation", "lagged_precip", "hexbin_temp"],
    },
    "Compare sites": {
        "intent": "Cross-gage context",
        "description": "Plots commonly useful before deeper multi-site comparison.",
        "plots": ["timeseries", "flow_duration", "monthly_boxplot", "cumulative_departure"],
    },
}

PURPOSE_GROUPS = [
    {
        "label": "Flow behavior",
        "help": "Core hydrograph, distribution, seasonal shape, and long-term flow behavior.",
        "plots": [
            "timeseries", "flow_duration", "monthly_boxplot", "discharge_heatmap",
            "temporal_heatmap", "annual_trend", "cumulative_departure",
        ],
    },
    {
        "label": "Extremes / frequency",
        "help": "Floods, low flows, exceedance behavior, anomalies, and recurrence context.",
        "plots": [
            "flood_frequency", "7q10_analysis", "low_flow_trend",
            "summer_low_flow_trend", "anomaly_detection", "threshold_exceedance",
        ],
    },
    {
        "label": "Seasonality",
        "help": "Monthly, seasonal, and water-year patterns.",
        "plots": [
            "seasonal_scatter", "monthly_boxplot", "temporal_heatmap",
            "seasonal_gain_loss", "seasonal_gain_loss_annual",
        ],
    },
    {
        "label": "Climate linkage",
        "help": "Weather-linked flow response, precipitation lag, and climate correlation.",
        "plots": [
            "anomaly", "precip_discharge", "lagged_precip", "lag_correlation",
            "hexbin_temp", "correlation_matrix", "double_mass_curve",
            "summer_climate_context", "precip_response_comparison",
        ],
    },
    {
        "label": "Stage / rating",
        "help": "Gage-height and stage-discharge relationship plots.",
        "plots": ["rating_curve"],
    },
    {
        "label": "Multi-site / reach",
        "help": "Reach, paired-site, aquifer contribution, and comparison plots.",
        "plots": [
            "reach_comparison", "reach_index", "paired_annual_lows",
            "avista_window_comparison",
        ],
    },
    {
        "label": "Advanced diagnostics",
        "help": "Specialized signal and hydrograph decomposition views.",
        "plots": ["baseflow_separation", "recession_curves", "spectral_analysis"],
    },
]


def get_display_name(plot_key: str) -> str:
    """Get human-readable display name for a plot."""
    return PLOT_DISPLAY_NAMES.get(plot_key, plot_key.replace('_', ' ').title())


def resolve_plot_preset(preset_name: str, available_plots: Dict[str, Any]) -> List[str]:
    """Return available plot keys for a named preset."""
    preset = PLOT_PRESETS.get(preset_name, PLOT_PRESETS["Manual selection"])
    available = set(available_plots.keys())
    return [plot for plot in preset["plots"] if plot in available]


def get_grouped_plot_options(available_plots: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return purpose-based plot groups filtered to available plots."""
    available = set(available_plots.keys())
    grouped = []
    assigned = set()

    for group in PURPOSE_GROUPS:
        plots = [plot for plot in group["plots"] if plot in available and plot not in assigned]
        if plots:
            grouped.append({**group, "plots": plots})
            assigned.update(plots)

    leftovers = [plot for plot in available_plots.keys() if plot not in assigned]
    if leftovers:
        grouped.append({
            "label": "All other plots",
            "help": "Additional available plots not assigned to a purpose group.",
            "plots": leftovers,
        })

    return grouped


def describe_selected_plots(selected_plots: List[str]) -> str:
    """Summarize the current plot mix by count and analysis purpose."""
    if not selected_plots:
        return "No plots selected"

    labels = []
    for group in PURPOSE_GROUPS:
        if any(plot in selected_plots for plot in group["plots"]):
            labels.append(group["label"])

    if not labels:
        return f"{len(selected_plots)} selected"
    return f"{len(selected_plots)} selected: " + ", ".join(labels)


def catalog_plot_options(available_plots: Dict[str, Any]) -> List[tuple]:
    """Return one ``(label, plot_key)`` catalog covering every available plot once."""
    options = []
    for group in get_grouped_plot_options(available_plots):
        for plot in group["plots"]:
            options.append((f"{group['label']} · {get_display_name(plot)}", plot))
    return options


def resolve_generated_plots(selected_plots: List[str]) -> List[str]:
    """Return plot keys that should be generated from the user's selection."""
    return list(selected_plots)


# =============================================================================
# SELECTOR WIDGETS
# =============================================================================

def multi_plot_selector(available_plots: Dict[str, Any], key_prefix: str = "") -> List[str]:
    """
    Preset plus one plot catalog. Used by Site Analysis export figures.
    """
    st.caption("Pick a preset, then add or remove figures in the catalog. Every single-site plot is in that one list.")
    preset_names = list(PLOT_PRESETS.keys())
    selected_preset = st.selectbox(
        "Preset",
        preset_names,
        index=preset_names.index("Quick site summary"),
        key=f"{key_prefix}plot_preset_choice",
        help="Guided starting set. Changing it resets the catalog to that set.",
    )
    preset = PLOT_PRESETS[selected_preset]
    st.caption(f"{preset['intent']} — {preset['description']}")
    catalog = catalog_plot_options(available_plots)
    label_to_plot = {label: plot for label, plot in catalog}
    plot_to_label = {plot: label for label, plot in catalog}
    default_labels = [
        plot_to_label[plot]
        for plot in resolve_plot_preset(selected_preset, available_plots)
        if plot in plot_to_label
    ]
    preset_key = selected_preset.lower().replace(" / ", "_").replace(" ", "_")
    selected_labels = st.multiselect(
        "Plots",
        list(label_to_plot.keys()),
        default=default_labels,
        key=f"{key_prefix}plot_catalog_{preset_key}",
        help="One catalog. Labels are grouped by purpose, and nothing is tucked into a second picker.",
    )
    selected = [label_to_plot[label] for label in selected_labels if label in label_to_plot]

    st.caption(describe_selected_plots(selected))
    return selected


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

    # Reach analysis plots
    for p in REACH_PLOTS:
        if p in plot_names:
            display = f"[Reach] {get_display_name(p)}"
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
    elif plot_name in REACH_PLOTS:
        return 'reach'
    return 'unknown'


def get_plot_requirements(plot_name: str) -> List[str]:
    """Get data requirements for a plot."""
    category = get_plot_category(plot_name)
    if category == 'climate':
        return ['discharge', 'climate']
    elif category == 'stage':
        return ['discharge', 'stage']
    elif category == 'reach':
        return ['discharge_upstream', 'discharge_downstream']
    return ['discharge']
