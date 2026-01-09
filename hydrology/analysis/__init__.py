"""Analysis functions for hydrology package."""

from .trends import calculate_annual_means, analyze_trend
from .stage_discharge import fit_powerlaw_rating_curve

__all__ = [
    'calculate_annual_means',
    'analyze_trend',
    'fit_powerlaw_rating_curve',
]
