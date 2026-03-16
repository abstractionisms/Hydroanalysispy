"""
Standardized hydrological and climate indicators for drought monitoring.

Provides:
- SPI (Standardized Precipitation Index): Gamma-fitted precipitation anomalies
- SRI (Standardized Runoff Index): Gamma-fitted streamflow anomalies
- Drought classification (D0-D4 severity per US Drought Monitor)
- Rolling Baseflow Index (BFI) for groundwater contribution tracking

These are standard indicators used in drought monitoring networks (NIDIS, USDM).
The SRI is particularly relevant for the Spokane dry reach thesis.

Example:
    >>> from hydrology.analysis.indicators import calculate_spi, classify_drought
    >>> spi = calculate_spi(monthly_precip, windows=[1, 3, 6, 12])
    >>> current_status = classify_drought(spi['SPI_3'].iloc[-1])
    >>> print(f"Current drought status: {current_status}")
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.logging_setup import get_logger

logger = get_logger(__name__)


# US Drought Monitor classification thresholds
DROUGHT_CLASSES = {
    'D4': ('Exceptional Drought', -2.0),
    'D3': ('Extreme Drought', -1.6),
    'D2': ('Severe Drought', -1.3),
    'D1': ('Moderate Drought', -0.8),
    'D0': ('Abnormally Dry', -0.5),
    'Normal': ('Normal', 0.5),
    'W0': ('Abnormally Wet', 0.8),
    'W1': ('Moderately Wet', 1.3),
    'W2': ('Very Wet', 1.6),
    'W3': ('Extremely Wet', 2.0),
    'W4': ('Exceptionally Wet', float('inf')),
}


def classify_drought(index_value: float) -> Dict[str, str]:
    """
    Classify drought severity based on a standardized index value.

    Uses US Drought Monitor thresholds (D0-D4 for dry, W0-W4 for wet).

    Args:
        index_value: SPI or SRI value

    Returns:
        Dict with 'class', 'label', and 'color'
    """
    if np.isnan(index_value):
        return {'class': 'N/A', 'label': 'No Data', 'color': '#808080'}

    colors = {
        'D4': '#730000', 'D3': '#E60000', 'D2': '#FFAA00',
        'D1': '#FCD37F', 'D0': '#FFFF00',
        'Normal': '#FFFFFF',
        'W0': '#C6EFCE', 'W1': '#70AD47', 'W2': '#2E75B6',
        'W3': '#203864', 'W4': '#0D1B3E',
    }

    if index_value <= -2.0:
        cls = 'D4'
    elif index_value <= -1.6:
        cls = 'D3'
    elif index_value <= -1.3:
        cls = 'D2'
    elif index_value <= -0.8:
        cls = 'D1'
    elif index_value <= -0.5:
        cls = 'D0'
    elif index_value <= 0.5:
        cls = 'Normal'
    elif index_value <= 0.8:
        cls = 'W0'
    elif index_value <= 1.3:
        cls = 'W1'
    elif index_value <= 1.6:
        cls = 'W2'
    elif index_value <= 2.0:
        cls = 'W3'
    else:
        cls = 'W4'

    label = DROUGHT_CLASSES[cls][0]
    return {'class': cls, 'label': label, 'color': colors[cls]}


def _fit_gamma_and_standardize(values: np.ndarray) -> np.ndarray:
    """
    Fit a gamma distribution and transform to standard normal (the SPI method).

    Handles zeros by separating them from the gamma fit (mixed distribution).
    """
    result = np.full_like(values, np.nan, dtype=float)
    valid_mask = np.isfinite(values)
    valid_values = values[valid_mask]

    if len(valid_values) < 10:
        return result

    # Probability of zero (for mixed distribution)
    n_zero = np.sum(valid_values <= 0)
    q = n_zero / len(valid_values)  # empirical probability of zero

    # Fit gamma to positive values only
    positive = valid_values[valid_values > 0]
    if len(positive) < 5:
        return result

    try:
        alpha, loc, beta = sp_stats.gamma.fit(positive, floc=0)
    except Exception:
        # Fallback: method of moments
        mean_p = np.mean(positive)
        var_p = np.var(positive)
        if var_p <= 0:
            return result
        beta = var_p / mean_p
        alpha = mean_p / beta
        loc = 0

    # Transform each value
    for i in range(len(values)):
        if not np.isfinite(values[i]):
            continue
        if values[i] <= 0:
            # H(x) = q for zeros
            cdf = q
        else:
            # H(x) = q + (1-q) * G(x) for positive values
            cdf = q + (1 - q) * sp_stats.gamma.cdf(values[i], alpha, loc=loc, scale=beta)

        # Clip CDF to avoid infinities
        cdf = np.clip(cdf, 0.001, 0.999)
        result[i] = sp_stats.norm.ppf(cdf)

    return result


def calculate_spi(
    precip: pd.Series,
    windows: List[int] = None,
) -> pd.DataFrame:
    """
    Calculate Standardized Precipitation Index (SPI).

    Fits a gamma distribution to rolling-window accumulated precipitation
    and transforms to standard normal deviates.

    Args:
        precip: Daily or monthly precipitation series (datetime index)
        windows: Accumulation windows in months. Default: [1, 3, 6, 12]

    Returns:
        DataFrame with SPI columns (e.g., SPI_1, SPI_3, SPI_6, SPI_12)
    """
    if windows is None:
        windows = [1, 3, 6, 12]

    precip = precip.dropna()
    if precip.empty:
        return pd.DataFrame()

    # Resample to monthly totals if daily
    freq = pd.infer_freq(precip.index)
    if freq and freq.startswith('D'):
        monthly = precip.resample('ME').sum()
    elif freq and (freq.startswith('M') or freq.startswith('ME')):
        monthly = precip
    else:
        # Assume daily
        monthly = precip.resample('ME').sum()

    results = {}

    for window in windows:
        # Rolling accumulation
        accumulated = monthly.rolling(window=window, min_periods=window).sum()

        # Standardize by calendar month (fit gamma per month)
        spi = pd.Series(np.nan, index=accumulated.index)

        months_fitted = 0
        for month in range(1, 13):
            month_mask = accumulated.index.month == month
            month_values = accumulated[month_mask].dropna().values

            if len(month_values) >= 10:
                standardized = _fit_gamma_and_standardize(month_values)
                valid_positions = accumulated[month_mask].dropna().index
                spi[valid_positions] = standardized
                months_fitted += 1

        # Fallback: pooled fit if per-month data is sparse
        if months_fitted < 6:
            all_values = accumulated.dropna().values
            if len(all_values) >= 10:
                standardized = _fit_gamma_and_standardize(all_values)
                spi[accumulated.dropna().index] = standardized

        results[f'SPI_{window}'] = spi

    return pd.DataFrame(results)


def calculate_sri(
    streamflow: pd.Series,
    windows: List[int] = None,
) -> pd.DataFrame:
    """
    Calculate Standardized Runoff Index (SRI).

    Same methodology as SPI but applied to streamflow. Useful for
    detecting hydrological droughts which may lag meteorological droughts.

    Args:
        streamflow: Daily discharge series (datetime index, values in cfs)
        windows: Accumulation windows in months. Default: [1, 3, 6]

    Returns:
        DataFrame with SRI columns (e.g., SRI_1, SRI_3, SRI_6)
    """
    if windows is None:
        windows = [1, 3, 6]

    streamflow = streamflow.dropna()
    if streamflow.empty:
        return pd.DataFrame()

    # Monthly mean discharge
    monthly = streamflow.resample('ME').mean()

    results = {}

    for window in windows:
        accumulated = monthly.rolling(window=window, min_periods=window).mean()

        sri = pd.Series(np.nan, index=accumulated.index)

        # Try per-month fitting first (more accurate)
        months_fitted = 0
        for month in range(1, 13):
            month_mask = accumulated.index.month == month
            month_values = accumulated[month_mask].dropna().values

            if len(month_values) >= 10:
                standardized = _fit_gamma_and_standardize(month_values)
                valid_positions = accumulated[month_mask].dropna().index
                sri[valid_positions] = standardized
                months_fitted += 1

        # Fallback: pooled fit if fewer than 6 months had enough data
        if months_fitted < 6:
            all_values = accumulated.dropna().values
            if len(all_values) >= 10:
                standardized = _fit_gamma_and_standardize(all_values)
                sri[accumulated.dropna().index] = standardized

        results[f'SRI_{window}'] = sri

    return pd.DataFrame(results)


def calculate_baseflow_index_timeseries(
    daily_q: pd.Series,
    alpha: float = 0.925,
    window_days: int = 90,
) -> pd.DataFrame:
    """
    Calculate rolling Baseflow Index (BFI) for tracking groundwater contribution.

    Uses the Lyne-Hollick recursive digital filter for baseflow separation,
    then computes BFI as the ratio of baseflow to total flow over a rolling window.

    Relevant to Spokane dry reach analysis: declining BFI indicates reduced
    groundwater contribution (aquifer depletion or diversion effects).

    Args:
        daily_q: Daily discharge series (datetime index, values in cfs)
        alpha: Lyne-Hollick filter parameter (default 0.925)
        window_days: Rolling window for BFI computation (default 90 days)

    Returns:
        DataFrame with columns: total_flow, baseflow, quickflow, bfi
    """
    daily_q = daily_q.dropna()
    if len(daily_q) < window_days * 2:
        logger.warning("Insufficient data for BFI timeseries")
        return pd.DataFrame()

    Q = daily_q.values.astype(float)

    # Lyne-Hollick filter
    Q_f = np.zeros_like(Q)
    for t in range(1, len(Q)):
        Q_f[t] = alpha * Q_f[t-1] + (1 + alpha) / 2 * (Q[t] - Q[t-1])
        Q_f[t] = max(0, Q_f[t])

    baseflow = np.clip(Q - Q_f, 0, Q)
    quickflow = Q - baseflow

    df = pd.DataFrame({
        'total_flow': Q,
        'baseflow': baseflow,
        'quickflow': quickflow,
    }, index=daily_q.index)

    # Rolling BFI
    rolling_total = df['total_flow'].rolling(window=window_days, min_periods=window_days//2).sum()
    rolling_base = df['baseflow'].rolling(window=window_days, min_periods=window_days//2).sum()
    df['bfi'] = (rolling_base / rolling_total).clip(0, 1)

    return df


def get_seasonal_anomaly(
    daily_q: pd.Series,
    reference_years: int = 10,
) -> pd.DataFrame:
    """
    Compute seasonal flow anomaly comparing recent year to historical norms.

    Args:
        daily_q: Daily discharge series (multiple years)
        reference_years: Number of years for baseline computation

    Returns:
        DataFrame with columns: doy, current_year_flow, historical_median,
        anomaly_pct, percentile
    """
    daily_q = daily_q.dropna()
    if len(daily_q) < 365 * 3:
        return pd.DataFrame()

    df = pd.DataFrame({
        'q': daily_q.values,
        'doy': daily_q.index.dayofyear,
        'year': daily_q.index.year,
    }, index=daily_q.index)

    current_year = df['year'].max()

    # Historical reference (excluding current year)
    hist = df[df['year'] != current_year]
    current = df[df['year'] == current_year]

    if hist.empty or current.empty:
        return pd.DataFrame()

    # Historical stats by day of year
    hist_stats = hist.groupby('doy')['q'].agg(['median', 'mean', 'std']).rename(
        columns={'median': 'historical_median', 'mean': 'historical_mean', 'std': 'historical_std'}
    )

    # Current year values
    current_daily = current.groupby('doy')['q'].mean().rename('current_year_flow')

    result = pd.merge(hist_stats, current_daily, left_index=True, right_index=True, how='inner')

    # Anomaly as percent of median
    result['anomaly_pct'] = (
        (result['current_year_flow'] - result['historical_median'])
        / result['historical_median'] * 100
    ).replace([np.inf, -np.inf], np.nan)

    # Percentile of current flow in historical distribution
    percentiles = []
    for doy in result.index:
        hist_vals = hist[hist['doy'] == doy]['q'].values
        curr_val = result.loc[doy, 'current_year_flow']
        if len(hist_vals) > 5:
            pctile = sp_stats.percentileofscore(hist_vals, curr_val)
            percentiles.append(pctile)
        else:
            percentiles.append(np.nan)

    result['percentile'] = percentiles
    result.index.name = 'doy'

    return result.reset_index()
