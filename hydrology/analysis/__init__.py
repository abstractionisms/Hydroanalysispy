"""Analysis functions for hydrology package."""

from .trends import calculate_annual_means, analyze_trend
from .stage_discharge import fit_powerlaw_rating_curve
from .alerts import AlertMonitor, AlertThreshold, Alert, create_flood_alert, create_low_flow_alert
from .multisite import MultiSiteAnalyzer, quick_correlation_check, CorrelationResult, LagAnalysisResult
from .frequency import fit_flood_frequency, estimate_return_periods, low_flow_frequency, get_plotting_positions
from .indicators import calculate_spi, calculate_sri, classify_drought, calculate_baseflow_index_timeseries
from .baseflow import BaseflowResult, compare_baseflow_methods, eckhardt_filter, lyne_hollick_filter
from .signatures import compute_hydrologic_signatures
from .changepoints import pettitt_test

__all__ = [
    'calculate_annual_means',
    'analyze_trend',
    'fit_powerlaw_rating_curve',
    # Alert monitoring
    'AlertMonitor',
    'AlertThreshold',
    'Alert',
    'create_flood_alert',
    'create_low_flow_alert',
    # Multi-site analysis
    'MultiSiteAnalyzer',
    'quick_correlation_check',
    'CorrelationResult',
    'LagAnalysisResult',
    # Frequency analysis
    'fit_flood_frequency',
    'estimate_return_periods',
    'low_flow_frequency',
    'get_plotting_positions',
    # Drought indicators
    'calculate_spi',
    'calculate_sri',
    'classify_drought',
    'calculate_baseflow_index_timeseries',
    # Baseflow separation
    'BaseflowResult',
    'compare_baseflow_methods',
    'eckhardt_filter',
    'lyne_hollick_filter',
    # Hydrologic signatures
    'compute_hydrologic_signatures',
    # Changepoints
    'pettitt_test',
]
