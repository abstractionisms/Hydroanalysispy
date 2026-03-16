"""Visualization modules for hydrology package."""

from .plots import (
    plot_anomaly,
    plot_hexbin,
    plot_monthly_lagged_scatter,
    plot_timeseries,
    plot_flow_duration,
    plot_correlation_matrix,
    plot_monthly_boxplot,
    plot_discharge_heatmap,
    plot_temporal_heatmap,
    plot_precip_discharge_overlay,
    plot_reach_comparison,
    plot_summer_low_flow_trend,
    plot_reach_index,
    plot_paired_annual_lows,
    plot_avista_window_comparison,
    plot_threshold_exceedance,
    plot_precip_response_comparison,
    plot_summer_climate_context,
    plot_seasonal_gain_loss,
    plot_seasonal_gain_loss_annual,
    AVAILABLE_PLOTS
)

from .composer import create_multi_plot, PlotLayout
from .map_utils import create_watershed_map, get_condition_color, get_condition_label

__all__ = [
    'plot_anomaly',
    'plot_hexbin',
    'plot_monthly_lagged_scatter',
    'plot_timeseries',
    'plot_flow_duration',
    'plot_correlation_matrix',
    'plot_monthly_boxplot',
    'plot_discharge_heatmap',
    'plot_temporal_heatmap',
    'plot_precip_discharge_overlay',
    'plot_reach_comparison',
    'plot_summer_low_flow_trend',
    'plot_reach_index',
    'plot_paired_annual_lows',
    'plot_avista_window_comparison',
    'plot_threshold_exceedance',
    'plot_precip_response_comparison',
    'plot_summer_climate_context',
    'plot_seasonal_gain_loss',
    'plot_seasonal_gain_loss_annual',
    'create_multi_plot',
    'PlotLayout',
    'AVAILABLE_PLOTS',
    # Map utilities
    'create_watershed_map',
    'get_condition_color',
    'get_condition_label',
]
