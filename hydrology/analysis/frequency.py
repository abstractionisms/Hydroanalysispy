"""
Flood and low-flow frequency analysis with multi-distribution fitting.

Provides multi-distribution fitting (GEV, Gumbel, LP3), return period estimation
with confidence intervals, and low-flow frequency analysis (7Q10). Builds on
the existing LP3 fit in plots.py with proper model selection via AIC/BIC.

References:
    - Bulletin 17C (USGS): Guidelines for flood frequency analysis
    - Hosking & Wallis (1997): Regional Frequency Analysis

Example:
    >>> from hydrology.analysis.frequency import fit_flood_frequency, estimate_return_periods
    >>> from hydrology.data.usgs import fetch_peak_streamflow
    >>> peaks = fetch_peak_streamflow('12422500')
    >>> results = fit_flood_frequency(peaks['peak_va'].dropna().values)
    >>> rp = estimate_return_periods(peaks['peak_va'].dropna().values)
    >>> print(rp)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from dataclasses import dataclass

from ..core.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class DistributionFit:
    """Results from fitting a single distribution to peak flow data."""
    name: str
    display_name: str
    params: Tuple
    aic: float
    bic: float
    ks_statistic: float
    ks_pvalue: float
    quantiles: Dict[float, float]  # return period -> estimated flow


def _log_pearson_type3_fit(data: np.ndarray) -> Optional[Tuple]:
    """
    Fit Log-Pearson Type III distribution per Bulletin 17C.

    LP3 fits a Pearson Type III (gamma) distribution to log-transformed data.
    """
    log_data = np.log10(data[data > 0])
    if len(log_data) < 10:
        return None

    mean_log = np.mean(log_data)
    std_log = np.std(log_data, ddof=1)
    skew_log = sp_stats.skew(log_data, bias=False)

    return (skew_log, mean_log, std_log)


def _lp3_quantile(params: Tuple, exceedance_prob: float) -> float:
    """Compute LP3 quantile for a given exceedance probability."""
    skew, mean_log, std_log = params

    if abs(skew) < 0.001:
        # Normal approximation for near-zero skew
        z = sp_stats.norm.ppf(1 - exceedance_prob)
        log_q = mean_log + z * std_log
    else:
        # Wilson-Hilferty approximation for Pearson III
        k = skew / 6.0
        z = sp_stats.norm.ppf(1 - exceedance_prob)
        kz = k * z
        freq_factor = z + (kz * z - 1) * k + (1/3) * (kz**2 - 6*kz + 1) * k**2 - (kz**2 + 1) * k**3 + kz * k**4 + (1/3) * k**5
        log_q = mean_log + freq_factor * std_log

    return 10 ** log_q


def fit_flood_frequency(
    peaks: np.ndarray,
    distributions: List[str] = None,
    return_periods: List[float] = None,
) -> Dict[str, DistributionFit]:
    """
    Fit multiple distributions to annual peak flow data with model selection.

    Args:
        peaks: Array of annual peak discharge values (cfs)
        distributions: List of distributions to fit.
            Options: 'gev', 'gumbel', 'lp3', 'pearson3', 'lognormal'
            Default: ['gev', 'gumbel', 'lp3']
        return_periods: Return periods (years) for quantile estimation.
            Default: [2, 5, 10, 25, 50, 100]

    Returns:
        Dict mapping distribution name to DistributionFit results,
        ordered by AIC (best first)
    """
    if distributions is None:
        distributions = ['gev', 'gumbel', 'lp3']
    if return_periods is None:
        return_periods = [2, 5, 10, 25, 50, 100]

    peaks = np.asarray(peaks, dtype=float)
    peaks = peaks[np.isfinite(peaks) & (peaks > 0)]

    if len(peaks) < 10:
        logger.warning(f"Insufficient peaks for frequency analysis ({len(peaks)} < 10)")
        return {}

    n = len(peaks)
    results = {}

    for dist_name in distributions:
        try:
            fit = _fit_single_distribution(peaks, dist_name, return_periods, n)
            if fit is not None:
                results[dist_name] = fit
        except Exception as e:
            logger.warning(f"Failed to fit {dist_name}: {e}")

    # Sort by AIC
    results = dict(sorted(results.items(), key=lambda x: x[1].aic))

    return results


def _fit_single_distribution(
    peaks: np.ndarray,
    dist_name: str,
    return_periods: List[float],
    n: int,
) -> Optional[DistributionFit]:
    """Fit a single distribution and compute diagnostics."""

    display_names = {
        'gev': 'Generalized Extreme Value',
        'gumbel': 'Gumbel (EV Type I)',
        'lp3': 'Log-Pearson Type III',
        'pearson3': 'Pearson Type III',
        'lognormal': 'Log-Normal',
    }

    if dist_name == 'gev':
        params = sp_stats.genextreme.fit(peaks)
        log_likelihood = np.sum(sp_stats.genextreme.logpdf(peaks, *params))
        k = 3
        ks_stat, ks_p = sp_stats.kstest(peaks, 'genextreme', args=params)
        quantiles = {}
        for rp in return_periods:
            p = 1 - 1/rp
            quantiles[rp] = float(sp_stats.genextreme.ppf(p, *params))

    elif dist_name == 'gumbel':
        params = sp_stats.gumbel_r.fit(peaks)
        log_likelihood = np.sum(sp_stats.gumbel_r.logpdf(peaks, *params))
        k = 2
        ks_stat, ks_p = sp_stats.kstest(peaks, 'gumbel_r', args=params)
        quantiles = {}
        for rp in return_periods:
            p = 1 - 1/rp
            quantiles[rp] = float(sp_stats.gumbel_r.ppf(p, *params))

    elif dist_name == 'lp3':
        lp3_params = _log_pearson_type3_fit(peaks)
        if lp3_params is None:
            return None
        params = lp3_params

        # Log-likelihood approximation for LP3
        log_peaks = np.log10(peaks[peaks > 0])
        skew, mean_log, std_log = params
        log_likelihood = np.sum(sp_stats.norm.logpdf(log_peaks, mean_log, std_log))
        k = 3

        # KS test on log-transformed data
        ks_stat, ks_p = sp_stats.kstest(log_peaks, 'norm', args=(mean_log, std_log))

        quantiles = {}
        for rp in return_periods:
            p = 1 / rp  # exceedance probability
            quantiles[rp] = _lp3_quantile(params, p)

    elif dist_name == 'pearson3':
        params = sp_stats.pearson3.fit(peaks)
        log_likelihood = np.sum(sp_stats.pearson3.logpdf(peaks, *params))
        k = 3
        ks_stat, ks_p = sp_stats.kstest(peaks, 'pearson3', args=params)
        quantiles = {}
        for rp in return_periods:
            p = 1 - 1/rp
            quantiles[rp] = float(sp_stats.pearson3.ppf(p, *params))

    elif dist_name == 'lognormal':
        params = sp_stats.lognorm.fit(peaks, floc=0)
        log_likelihood = np.sum(sp_stats.lognorm.logpdf(peaks, *params))
        k = 2  # shape + scale (loc fixed at 0)
        ks_stat, ks_p = sp_stats.kstest(peaks, 'lognorm', args=params)
        quantiles = {}
        for rp in return_periods:
            p = 1 - 1/rp
            quantiles[rp] = float(sp_stats.lognorm.ppf(p, *params))

    else:
        logger.warning(f"Unknown distribution: {dist_name}")
        return None

    # AIC and BIC
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    return DistributionFit(
        name=dist_name,
        display_name=display_names.get(dist_name, dist_name),
        params=params,
        aic=aic,
        bic=bic,
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        quantiles=quantiles,
    )


def estimate_return_periods(
    peaks: np.ndarray,
    periods: List[float] = None,
    distribution: str = 'best',
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Estimate return period flows with confidence intervals.

    Args:
        peaks: Array of annual peak discharge values
        periods: Return periods to estimate (years). Default: [2, 5, 10, 25, 50, 100]
        distribution: Distribution to use. 'best' selects by AIC.
        confidence_level: Confidence level for intervals (default 0.95)

    Returns:
        DataFrame with columns: return_period, flow_cfs, lower_ci, upper_ci, distribution
    """
    if periods is None:
        periods = [2, 5, 10, 25, 50, 100]

    peaks = np.asarray(peaks, dtype=float)
    peaks = peaks[np.isfinite(peaks) & (peaks > 0)]
    n = len(peaks)

    if n < 10:
        return pd.DataFrame()

    # Fit distributions
    if distribution == 'best':
        fits = fit_flood_frequency(peaks, return_periods=periods)
        if not fits:
            return pd.DataFrame()
        best_name = next(iter(fits))
        best_fit = fits[best_name]
    else:
        fits = fit_flood_frequency(peaks, distributions=[distribution], return_periods=periods)
        if not fits:
            return pd.DataFrame()
        best_name = distribution
        best_fit = fits[best_name]

    # Bootstrap confidence intervals
    n_bootstrap = 500
    alpha = 1 - confidence_level
    bootstrap_quantiles = {rp: [] for rp in periods}

    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(peaks, size=n, replace=True)
        boot_fits = fit_flood_frequency(boot_sample, distributions=[best_name], return_periods=periods)
        if best_name in boot_fits:
            for rp in periods:
                if rp in boot_fits[best_name].quantiles:
                    bootstrap_quantiles[rp].append(boot_fits[best_name].quantiles[rp])

    rows = []
    for rp in periods:
        flow = best_fit.quantiles.get(rp, np.nan)
        boot_vals = bootstrap_quantiles[rp]
        if len(boot_vals) > 10:
            lower = np.percentile(boot_vals, 100 * alpha / 2)
            upper = np.percentile(boot_vals, 100 * (1 - alpha / 2))
        else:
            lower = np.nan
            upper = np.nan

        rows.append({
            'return_period': rp,
            'flow_cfs': flow,
            'lower_ci': lower,
            'upper_ci': upper,
            'distribution': best_fit.display_name,
        })

    return pd.DataFrame(rows)


def low_flow_frequency(
    daily_q: pd.Series,
    durations: List[int] = None,
    return_periods: List[float] = None,
) -> Dict[str, Any]:
    """
    Low-flow frequency analysis (7Q10 and similar).

    Computes N-day minimum flows for each year and fits a frequency distribution
    to estimate low-flow return periods.

    Args:
        daily_q: Daily discharge series (datetime index, values in cfs)
        durations: Averaging durations in days. Default: [1, 7, 30]
        return_periods: Return periods for estimation. Default: [2, 5, 10, 20, 50]

    Returns:
        Dict with:
        - annual_mins: DataFrame of annual minimum flows by duration
        - estimates: DataFrame of low-flow estimates by duration and return period
        - 7q10: The 7-day, 10-year low flow (most commonly used)
    """
    if durations is None:
        durations = [1, 7, 30]
    if return_periods is None:
        return_periods = [2, 5, 10, 20, 50]

    daily_q = daily_q.dropna()
    if len(daily_q) < 365 * 3:
        logger.warning("Insufficient data for low-flow frequency (need 3+ years)")
        return {}

    # Calculate N-day rolling minimums by water year (Oct-Sep)
    annual_mins = {}
    for dur in durations:
        rolling_mean = daily_q.rolling(window=dur, min_periods=dur).mean()

        # Group by water year
        water_year = rolling_mean.index.year.where(
            rolling_mean.index.month >= 10,
            rolling_mean.index.year - 1
        )
        yearly_min = rolling_mean.groupby(water_year).min().dropna()

        if len(yearly_min) >= 5:
            annual_mins[f'{dur}d_min'] = yearly_min

    if not annual_mins:
        return {}

    annual_mins_df = pd.DataFrame(annual_mins)

    # Fit LP3 to each duration and estimate return period flows
    estimates_rows = []
    q7_10 = None

    for dur in durations:
        col = f'{dur}d_min'
        if col not in annual_mins_df.columns:
            continue

        vals = annual_mins_df[col].dropna().values
        if len(vals) < 5:
            continue

        # For low-flow, we fit to the annual minimums directly
        # Use LP3 (log-Pearson Type III) as standard
        log_vals = np.log10(vals[vals > 0])
        if len(log_vals) < 5:
            continue

        mean_log = np.mean(log_vals)
        std_log = np.std(log_vals, ddof=1)
        skew_log = sp_stats.skew(log_vals, bias=False)

        for rp in return_periods:
            # For low-flow, we want the flow that is NOT exceeded
            # P(Q < q) = 1/T, where T is return period
            exceedance = 1 - 1/rp  # probability of exceeding (i.e., being above)

            lp3_params = (skew_log, mean_log, std_log)
            q_est = _lp3_quantile(lp3_params, exceedance)

            estimates_rows.append({
                'duration_days': dur,
                'return_period': rp,
                'flow_cfs': max(0, q_est),
            })

            if dur == 7 and rp == 10:
                q7_10 = max(0, q_est)

    return {
        'annual_mins': annual_mins_df,
        'estimates': pd.DataFrame(estimates_rows) if estimates_rows else pd.DataFrame(),
        '7q10': q7_10,
    }


def get_plotting_positions(peaks: np.ndarray) -> pd.DataFrame:
    """
    Compute Weibull plotting positions for observed peaks.

    Args:
        peaks: Annual peak discharge values

    Returns:
        DataFrame with return_period and flow_cfs for plotting observed data
    """
    peaks = np.sort(peaks)[::-1]  # Descending
    n = len(peaks)
    ranks = np.arange(1, n + 1)

    # Weibull plotting position: P = m / (n + 1)
    exceedance_prob = ranks / (n + 1)
    return_period = 1 / exceedance_prob

    return pd.DataFrame({
        'return_period': return_period,
        'flow_cfs': peaks,
        'exceedance_prob': exceedance_prob,
        'rank': ranks,
    })
