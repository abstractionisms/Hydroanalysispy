"""
Trend analysis functions for hydrology package.

Provides statistical trend analysis including Mann-Kendall test,
linear regression, and time series aggregation functions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, Tuple

try:
    import pymannkendall as mk
    MK_AVAILABLE = True
except ImportError:
    MK_AVAILABLE = False

from ..core.logging_setup import get_logger

logger = get_logger(__name__)


def calculate_annual_means(
    df: pd.DataFrame,
    column: str
) -> Optional[pd.Series]:
    """
    Calculate annual mean values from daily data.

    Args:
        df: DataFrame with datetime index
        column: Column name to analyze

    Returns:
        Series of annual means with year as index, or None if failed

    Example:
        >>> discharge = fetch_discharge_data('12422500', '2000-01-01', '2023-12-31')
        >>> annual = calculate_annual_means(discharge, 'Discharge_cfs')
        >>> print(annual)
        2000    1234.5
        2001    1456.7
        ...
    """
    if df is None or df.empty:
        logger.warning(f"Cannot calculate annual means: dataframe is None or empty")
        return None

    if column not in df.columns:
        logger.error(f"Column '{column}' not found in dataframe")
        return None

    try:
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame must have datetime index")
            return None

        # Resample to annual (year start) and calculate mean
        annual = df[column].resample('YS').mean().dropna()

        logger.info(f"Calculated annual means: {column} ({len(annual)} years)")
        return annual

    except Exception as e:
        logger.error(f"Error calculating annual means: {e}")
        return None


def calculate_monthly_means(
    df: pd.DataFrame,
    column: str
) -> Optional[pd.DataFrame]:
    """
    Calculate monthly mean values grouped by calendar month.

    Args:
        df: DataFrame with datetime index
        column: Column name to analyze

    Returns:
        DataFrame with month (1-12) as index and mean values, or None if failed

    Example:
        >>> discharge = fetch_discharge_data('12422500', '2000-01-01', '2023-12-31')
        >>> monthly = calculate_monthly_means(discharge, 'Discharge_cfs')
        >>> print(monthly)
             mean    std    count
        1    500.0   120.0  24
        2    600.0   150.0  24
        ...
    """
    if df is None or df.empty or column not in df.columns:
        logger.warning(f"Cannot calculate monthly means: invalid input")
        return None

    try:
        df['month'] = df.index.month
        monthly_stats = df.groupby('month')[column].agg(['mean', 'std', 'count'])

        logger.info(f"Calculated monthly statistics: {column}")
        return monthly_stats

    except Exception as e:
        logger.error(f"Error calculating monthly means: {e}")
        return None


def linear_regression_trend(
    series: pd.Series,
    series_name: str = "series"
) -> Optional[Dict[str, Any]]:
    """
    Perform linear regression trend analysis on time series.

    Args:
        series: Time series data (typically annual means)
        series_name: Name for logging

    Returns:
        Dictionary with regression results or None if failed

    Result keys:
        - slope: Trend slope (units per year)
        - intercept: Y-intercept
        - r_value: Correlation coefficient
        - p_value: P-value for hypothesis test
        - r_squared: R² (coefficient of determination)
        - std_err: Standard error of the slope
        - trend_direction: 'increasing', 'decreasing', or 'no significant trend'

    Example:
        >>> annual = calculate_annual_means(discharge, 'Discharge_cfs')
        >>> trend = linear_regression_trend(annual, 'Annual Discharge')
        >>> print(f"Trend: {trend['slope']:.2f} cfs/year, p={trend['p_value']:.4f}")
    """
    if series is None or series.empty:
        logger.warning(f"Cannot analyze trend: {series_name} is None or empty")
        return None

    if len(series) < 3:
        logger.warning(f"Need at least 3 points for trend analysis, got {len(series)}")
        return None

    try:
        # Extract years and values
        if isinstance(series.index, pd.DatetimeIndex):
            years = series.index.year.astype(float)
        else:
            years = series.index.astype(float)

        values = series.values

        # Remove NaN values
        mask = ~np.isnan(years) & ~np.isnan(values)
        if np.sum(mask) < 3:
            logger.warning(f"Less than 3 valid points after removing NaNs")
            return None

        years_clean = years[mask]
        values_clean = values[mask]

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            years_clean, values_clean
        )

        # Determine trend direction
        alpha = 0.05  # Significance level
        if p_value < alpha:
            if slope > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'
        else:
            trend_direction = 'no significant trend'

        results = {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'r_squared': r_value ** 2,
            'std_err': std_err,
            'trend_direction': trend_direction,
            'n_points': len(years_clean),
            'years': years_clean
        }

        logger.info(f"{series_name} linear trend: slope={slope:.4f}, "
                   f"p={p_value:.4f}, trend={trend_direction}")

        return results

    except Exception as e:
        logger.error(f"Error in linear regression for {series_name}: {e}")
        return None


def mann_kendall_test(
    series: pd.Series,
    series_name: str = "series",
    alpha: float = 0.05
) -> Optional[Dict[str, Any]]:
    """
    Perform Mann-Kendall trend test (non-parametric).

    The Mann-Kendall test is more robust to outliers and non-normal distributions
    than linear regression.

    Args:
        series: Time series data
        series_name: Name for logging
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with Mann-Kendall test results or None

    Result keys:
        - trend: 'increasing', 'decreasing', or 'no trend'
        - p_value: P-value
        - tau: Kendall's tau statistic
        - s: Mann-Kendall S statistic
        - z: Normalized test statistic

    Requires:
        pymannkendall package (pip install pymannkendall)

    Example:
        >>> annual = calculate_annual_means(discharge, 'Discharge_cfs')
        >>> mk_result = mann_kendall_test(annual, 'Annual Discharge')
        >>> print(f"Trend: {mk_result['trend']}, p={mk_result['p_value']:.4f}")
    """
    if not MK_AVAILABLE:
        logger.warning("pymannkendall not installed. Install with: pip install pymannkendall")
        return None

    if series is None or series.empty:
        logger.warning(f"Cannot perform Mann-Kendall: {series_name} is None or empty")
        return None

    if len(series) < 3:
        logger.warning(f"Need at least 3 points for Mann-Kendall, got {len(series)}")
        return None

    try:
        # Remove NaN values and sort by index
        series_clean = series.dropna().sort_index()

        if len(series_clean) < 3:
            logger.warning(f"Less than 3 valid points after removing NaNs")
            return None

        # Perform Mann-Kendall test
        mk_result = mk.original_test(series_clean.values, alpha=alpha)

        results = {
            'trend': mk_result.trend,
            'p_value': mk_result.p,
            'tau': mk_result.tau,
            's': mk_result.s,
            'z': mk_result.z,
            'sens_slope': getattr(mk_result, 'slope', np.nan),
            'sens_intercept': getattr(mk_result, 'intercept', np.nan),
            'n_points': len(series_clean),
            'alpha': alpha
        }

        logger.info(f"{series_name} Mann-Kendall: trend={mk_result.trend}, "
                   f"p={mk_result.p:.4f}, tau={mk_result.tau:.4f}")

        return results

    except Exception as e:
        logger.error(f"Error in Mann-Kendall test for {series_name}: {e}")
        return None


def analyze_trend(
    series: pd.Series,
    series_name: str = "series",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Comprehensive trend analysis (linear regression + Mann-Kendall).

    Performs both parametric (linear regression) and non-parametric (Mann-Kendall)
    trend tests.

    Args:
        series: Time series data (typically annual means)
        series_name: Name for logging
        alpha: Significance level for Mann-Kendall

    Returns:
        Dictionary with both 'linear_regression' and 'mann_kendall' results

    Example:
        >>> annual = calculate_annual_means(discharge, 'Discharge_cfs')
        >>> trends = analyze_trend(annual, 'Annual Discharge')
        >>> print(f"Linear: {trends['linear_regression']['slope']:.2f} cfs/year")
        >>> print(f"Mann-Kendall: {trends['mann_kendall']['trend']}")
    """
    results = {
        'linear_regression': None,
        'mann_kendall': None
    }

    # Linear regression
    results['linear_regression'] = linear_regression_trend(series, series_name)

    # Mann-Kendall test
    results['mann_kendall'] = mann_kendall_test(series, series_name, alpha)

    return results


def calculate_correlation(
    df: pd.DataFrame,
    col1: str,
    col2: str
) -> Optional[Tuple[float, float]]:
    """
    Calculate Pearson correlation coefficient and p-value.

    Args:
        df: DataFrame containing both columns
        col1: First column name
        col2: Second column name

    Returns:
        Tuple of (correlation_coefficient, p_value) or None

    Example:
        >>> merged = merge_discharge_climate(discharge, climate)
        >>> corr, p = calculate_correlation(merged, 'Discharge_cfs', 'Temp_C')
        >>> print(f"Correlation: {corr:.3f}, p-value: {p:.4f}")
    """
    if df is None or df.empty:
        logger.warning("Cannot calculate correlation: dataframe is empty")
        return None

    if col1 not in df.columns or col2 not in df.columns:
        logger.error(f"Columns not found: {col1}, {col2}")
        return None

    try:
        # Drop rows with NaN in either column
        df_clean = df[[col1, col2]].dropna()

        if len(df_clean) < 3:
            logger.warning(f"Need at least 3 points for correlation, got {len(df_clean)}")
            return None

        # Calculate Pearson correlation
        corr_coef, p_value = stats.pearsonr(df_clean[col1], df_clean[col2])

        logger.info(f"Correlation ({col1} vs {col2}): r={corr_coef:.3f}, p={p_value:.4f}")

        return (corr_coef, p_value)

    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        return None
