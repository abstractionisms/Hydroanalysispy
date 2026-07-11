"""
Stage-discharge rating curve analysis for hydrology package.

Provides functions for fitting and analyzing stage-discharge relationships
using power-law and other rating curve models.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from typing import Tuple, Optional, Dict, Any

from ..core.logging_setup import get_logger

logger = get_logger(__name__)


def fit_powerlaw_rating_curve(
    stage: pd.Series,
    discharge: pd.Series,
    min_points: int = 10,
    positive_only: bool = True
) -> Tuple[float, float, float, pd.Series]:
    """
    Fit power-law rating curve: Q = A * H^B

    Uses log-log linear regression to fit the relationship between
    stage (H) and discharge (Q).

    Args:
        stage: Stage/gage height measurements (feet or meters)
        discharge: Discharge measurements (cfs or cms)
        min_points: Minimum number of points required for fitting
        positive_only: Only use positive stage and discharge values

    Returns:
        Tuple of (A, B, R², Q_predicted)
        - A: Coefficient
        - B: Exponent
        - R²: Coefficient of determination
        - Q_predicted: Predicted discharge for all input stage values

    Example:
        >>> A, B, R2, Q_pred = fit_powerlaw_rating_curve(stage, discharge)
        >>> print(f"Rating curve: Q = {A:.3f} * H^{B:.3f} (R² = {R2:.3f})")
    """
    # Combine into dataframe and clean
    df = pd.DataFrame({'H': stage, 'Q': discharge})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if positive_only:
        df = df[(df['H'] > 0) & (df['Q'] > 0)]

    if len(df) < min_points:
        logger.warning(f"Insufficient data for rating curve: {len(df)} points "
                      f"(minimum: {min_points})")
        return (np.nan, np.nan, np.nan,
                pd.Series(index=stage.index, dtype=float))

    try:
        # Log-log linear regression: log(Q) = log(A) + B * log(H)
        X = np.log10(df['H'].values)
        Y = np.log10(df['Q'].values)

        # Fit linear regression on log-log data
        B, logA = np.polyfit(X, Y, 1)
        A = 10 ** logA

        # Calculate R²
        Q_hat_fit = A * (df['H'] ** B)
        ss_res = np.sum((df['Q'] - Q_hat_fit) ** 2)
        ss_tot = np.sum((df['Q'] - df['Q'].mean()) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Predict for all input points
        Q_hat_full = pd.Series(index=stage.index, dtype=float)
        valid_stage = stage[(stage > 0) & stage.notna()]
        Q_hat_full.loc[valid_stage.index] = A * (valid_stage ** B)

        logger.info(f"Power-law rating curve fitted: Q = {A:.3g} * H^{B:.3f}, "
                   f"R² = {R2:.3f}, n = {len(df)}")

        return (A, B, R2, Q_hat_full)

    except Exception as e:
        logger.error(f"Error fitting rating curve: {e}")
        return (np.nan, np.nan, np.nan,
                pd.Series(index=stage.index, dtype=float))


def fit_offset_powerlaw(
    stage: pd.Series,
    discharge: pd.Series,
    offset_guess: float = 0.0,
    min_points: int = 10
) -> Tuple[float, float, float, float, pd.Series]:
    """
    Fit offset power-law rating curve: Q = A * (H - H0)^B

    The offset H0 accounts for a stage at which discharge is zero,
    which is more physically realistic for many streams.

    Args:
        stage: Stage/gage height measurements
        discharge: Discharge measurements
        offset_guess: Initial guess for H0 offset
        min_points: Minimum number of points required

    Returns:
        Tuple of (A, B, H0, R², Q_predicted)
        - A: Coefficient
        - B: Exponent
        - H0: Stage offset (zero-flow stage)
        - R²: Coefficient of determination
        - Q_predicted: Predicted discharge values

    Example:
        >>> A, B, H0, R2, Q_pred = fit_offset_powerlaw(stage, discharge)
        >>> print(f"Rating curve: Q = {A:.3f} * (H - {H0:.3f})^{B:.3f}")
    """
    # Combine and clean data
    df = pd.DataFrame({'H': stage, 'Q': discharge})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df['H'] > 0) & (df['Q'] > 0)]

    if len(df) < min_points:
        logger.warning(f"Insufficient data for offset rating curve: {len(df)} points")
        return (np.nan, np.nan, np.nan, np.nan,
                pd.Series(index=stage.index, dtype=float))

    try:
        # Define offset power-law function
        def offset_powerlaw(H, A, B, H0):
            """Q = A * (H - H0)^B, with protection against negative values."""
            H_eff = np.maximum(H - H0, 1e-10)  # Avoid negative/zero
            return A * (H_eff ** B)

        # Initial parameter guesses
        # Use simple power-law as starting point
        A_simple, B_simple, _, _ = fit_powerlaw_rating_curve(
            df['H'], df['Q'], min_points=min_points
        )
        p0 = [A_simple if not np.isnan(A_simple) else 1.0,
              B_simple if not np.isnan(B_simple) else 1.5,
              offset_guess]

        # Fit using curve_fit
        bounds = ([0, 0, -np.inf], [np.inf, 5, df['H'].min()])
        popt, pcov = curve_fit(
            offset_powerlaw,
            df['H'].values,
            df['Q'].values,
            p0=p0,
            bounds=bounds,
            maxfev=5000
        )

        A, B, H0 = popt

        # Calculate R²
        Q_hat_fit = offset_powerlaw(df['H'].values, A, B, H0)
        ss_res = np.sum((df['Q'].values - Q_hat_fit) ** 2)
        ss_tot = np.sum((df['Q'].values - df['Q'].mean()) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Predict for all points
        Q_hat_full = pd.Series(index=stage.index, dtype=float)
        valid_stage = stage[(stage > H0) & stage.notna()]
        Q_hat_full.loc[valid_stage.index] = offset_powerlaw(
            valid_stage.values, A, B, H0
        )

        logger.info(f"Offset power-law fitted: Q = {A:.3g} * (H - {H0:.3f})^{B:.3f}, "
                   f"R² = {R2:.3f}")

        return (A, B, H0, R2, Q_hat_full)

    except Exception as e:
        logger.error(f"Error fitting offset rating curve: {e}")
        # Fall back to simple power-law
        logger.info("Falling back to simple power-law")
        A, B, R2, Q_pred = fit_powerlaw_rating_curve(stage, discharge, min_points)
        return (A, B, 0.0, R2, Q_pred)


def fit_best_rating_curve(
    stage: pd.Series,
    discharge: pd.Series,
    min_points: int = 10,
) -> Dict[str, Any]:
    """
    Fit simple and offset power-law ratings; return the better model by R².

    Many USGS gages (e.g. Walla Walla nr Touchet 14018500) have a non-zero
    control stage H0. Fitting Q = A * H^B on raw stage then produces garbage
    (even negative R²). Prefer Q = A * (H - H0)^B when it improves fit.

    Returns dict with keys:
      model: 'offset_powerlaw' | 'powerlaw' | 'none'
      A, B, H0, R2, Q_pred, n_points, equation
    """
    df = pd.DataFrame({"H": stage, "Q": discharge})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["H"] > 0) & (df["Q"] > 0)]
    n = len(df)

    empty = {
        "model": "none",
        "A": np.nan,
        "B": np.nan,
        "H0": 0.0,
        "R2": np.nan,
        "Q_pred": pd.Series(index=stage.index, dtype=float),
        "n_points": n,
        "equation": "n/a",
    }
    if n < min_points:
        logger.warning("Insufficient pairs for rating curve: %s", n)
        return empty

    A_s, B_s, R2_s, Q_s = fit_powerlaw_rating_curve(
        df["H"], df["Q"], min_points=min_points
    )
    A_o, B_o, H0, R2_o, Q_o = fit_offset_powerlaw(
        df["H"], df["Q"], min_points=min_points
    )

    # Prefer offset when R² is clearly better (or simple fit is invalid)
    use_offset = (
        np.isfinite(R2_o)
        and (not np.isfinite(R2_s) or R2_o >= R2_s + 0.02 or R2_s < 0.5)
        and np.isfinite(A_o)
        and np.isfinite(B_o)
    )

    if use_offset:
        A_f, B_f, H0_f, R2_f, Q_full = fit_offset_powerlaw(
            stage, discharge, min_points=min_points
        )
        eq = f"Q = {A_f:.4g} · (H − {H0_f:.3f})^{B_f:.3f}"
        return {
            "model": "offset_powerlaw",
            "A": float(A_f),
            "B": float(B_f),
            "H0": float(H0_f),
            "R2": float(R2_f) if np.isfinite(R2_f) else np.nan,
            "Q_pred": Q_full,
            "n_points": n,
            "equation": eq,
        }

    A_f, B_f, R2_f, Q_full = fit_powerlaw_rating_curve(
        stage, discharge, min_points=min_points
    )
    eq = f"Q = {A_f:.4g} · H^{B_f:.3f}"
    return {
        "model": "powerlaw",
        "A": float(A_f) if np.isfinite(A_f) else np.nan,
        "B": float(B_f) if np.isfinite(B_f) else np.nan,
        "H0": 0.0,
        "R2": float(R2_f) if np.isfinite(R2_f) else np.nan,
        "Q_pred": Q_full,
        "n_points": n,
        "equation": eq,
    }


def season_labels(index: pd.DatetimeIndex) -> pd.Series:
    """Map datetime index → meteorological season labels (DJF/MAM/JJA/SON)."""
    month = pd.DatetimeIndex(index).month
    labels = np.full(len(month), "UNK", dtype=object)
    labels[np.isin(month, [12, 1, 2])] = "DJF"
    labels[np.isin(month, [3, 4, 5])] = "MAM"
    labels[np.isin(month, [6, 7, 8])] = "JJA"
    labels[np.isin(month, [9, 10, 11])] = "SON"
    return pd.Series(labels, index=index, name="season")


def calculate_residuals(
    observed: pd.Series,
    predicted: pd.Series
) -> Dict[str, float]:
    """
    Calculate residual statistics for rating curve fit.

    Args:
        observed: Observed discharge values
        predicted: Predicted discharge values

    Returns:
        Dictionary with residual statistics:
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - mape: Mean absolute percentage error
        - bias: Mean bias (predicted - observed)
        - nash_sutcliffe: Nash-Sutcliffe efficiency

    Example:
        >>> A, B, R2, Q_pred = fit_powerlaw_rating_curve(stage, discharge)
        >>> residuals = calculate_residuals(discharge, Q_pred)
        >>> print(f"RMSE: {residuals['rmse']:.2f} cfs")
    """
    # Align series and remove NaN
    df = pd.DataFrame({'obs': observed, 'pred': predicted}).dropna()

    if df.empty:
        logger.warning("No valid data for residual calculation")
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'bias': np.nan,
            'nash_sutcliffe': np.nan,
            'n_points': 0
        }

    obs = df['obs'].values
    pred = df['pred'].values

    # Calculate metrics
    errors = pred - obs
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    rmse = np.sqrt(np.mean(squared_errors))
    mae = np.mean(abs_errors)

    # MAPE (avoid division by zero)
    mape = np.mean(np.abs(errors[obs != 0] / obs[obs != 0])) * 100 if np.any(obs != 0) else np.nan

    bias = np.mean(errors)

    # Nash-Sutcliffe efficiency
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    nash_sutcliffe = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'bias': bias,
        'nash_sutcliffe': nash_sutcliffe,
        'n_points': len(df)
    }


def flow_duration_curve(
    discharge: pd.Series,
    n_bins: int = 100
) -> pd.DataFrame:
    """
    Calculate flow duration curve (exceedance probability).

    Args:
        discharge: Discharge time series
        n_bins: Number of bins for the curve

    Returns:
        DataFrame with 'exceedance_pct' and 'discharge' columns

    Example:
        >>> discharge = fetch_discharge_data('12422500', '2000-01-01', '2023-12-31')
        >>> fdc = flow_duration_curve(discharge['Discharge_cfs'])
        >>> print(fdc.head())
           exceedance_pct  discharge
        0             0.0    5000.0
        1             1.0    4500.0
        ...
    """
    # Remove NaN and sort in descending order
    discharge_clean = discharge.dropna().sort_values(ascending=False)

    if discharge_clean.empty:
        logger.warning("No valid data for flow duration curve")
        return pd.DataFrame(columns=['exceedance_pct', 'discharge'])

    # Calculate exceedance probabilities
    n = len(discharge_clean)
    exceedance_pct = np.linspace(0, 100, min(n, n_bins))

    # Interpolate discharge values at exceedance percentiles
    ranks = np.arange(1, n + 1)
    ranks_pct = (ranks / n) * 100

    discharge_values = np.interp(exceedance_pct, ranks_pct, discharge_clean.values)

    fdc_df = pd.DataFrame({
        'exceedance_pct': exceedance_pct,
        'discharge': discharge_values
    })

    logger.info(f"Flow duration curve calculated: {len(fdc_df)} points")

    return fdc_df


def classify_flow_regime(fdc: pd.DataFrame) -> Dict[str, float]:
    """
    Classify flow regime based on flow duration curve characteristics.

    Args:
        fdc: Flow duration curve DataFrame from flow_duration_curve()

    Returns:
        Dictionary with regime characteristics:
        - q1: 1% exceedance flow (high flow)
        - q10: 10% exceedance flow
        - q50: 50% exceedance flow (median)
        - q90: 90% exceedance flow (base flow)
        - q99: 99% exceedance flow (low flow)
        - flashiness: Q10/Q90 ratio (measure of variability)

    Example:
        >>> fdc = flow_duration_curve(discharge['Discharge_cfs'])
        >>> regime = classify_flow_regime(fdc)
        >>> print(f"Median flow: {regime['q50']:.1f} cfs")
        >>> print(f"Flashiness index: {regime['flashiness']:.2f}")
    """
    if fdc.empty:
        return {k: np.nan for k in ['q1', 'q10', 'q50', 'q90', 'q99', 'flashiness']}

    def get_flow_at_exceedance(pct):
        """Interpolate flow at specific exceedance percentage."""
        return np.interp(pct, fdc['exceedance_pct'], fdc['discharge'])

    q1 = get_flow_at_exceedance(1)
    q10 = get_flow_at_exceedance(10)
    q50 = get_flow_at_exceedance(50)
    q90 = get_flow_at_exceedance(90)
    q99 = get_flow_at_exceedance(99)

    flashiness = q10 / q90 if q90 > 0 else np.nan

    return {
        'q1': q1,
        'q10': q10,
        'q50': q50,
        'q90': q90,
        'q99': q99,
        'flashiness': flashiness
    }
