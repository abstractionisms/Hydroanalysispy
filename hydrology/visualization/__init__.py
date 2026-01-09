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
    AVAILABLE_PLOTS
)

from .composer import create_multi_plot, PlotLayout

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
    'create_multi_plot',
    'PlotLayout',
    'AVAILABLE_PLOTS',
]
