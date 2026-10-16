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


def rating_stage_window(stage: pd.Series | np.ndarray) -> Dict[str, float]:
    """Stage axis and H₀ slider bounds that keep zero-flow on screen.

    The axis is fixed from the record, not the current H₀, so dragging zero-flow
    moves the line instead of rescaling the chart. The left gutter is only wide
    enough for that line. It is not a second copy of the stage range.

    ``h0_hi`` reaches into the lower part of the cloud so the line can cross
    real points. ``h0_fit_hi`` stays just below the lowest point so an automatic
    fit still uses every pair.
    """
    h = pd.Series(stage, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    h = h[h > 0]
    if h.empty:
        return {
            "h_min": np.nan,
            "h_max": np.nan,
            "h_ref": np.nan,
            "h0_lo": -1.0,
            "h0_hi": 0.0,
            "h0_fit_lo": -1.0,
            "h0_fit_hi": -0.01,
            "x_lo": -1.2,
            "x_hi": 1.0,
        }

    h_min = float(h.min())
    h_max = float(h.max())
    span = max(h_max - h_min, 0.5)
    eps = max(0.01, 0.01 * span)
    h0_fit_hi = h_min - eps
    # A full span of empty stage left the cloud in the right half of the chart.
    below = max(2.0, 0.35 * span)
    h0_lo = h_min - below
    if h0_lo >= h0_fit_hi:
        h0_lo = h0_fit_hi - 0.5
    # Into the lower cloud, so the zero-flow line can cross real points.
    h0_hi = float(np.quantile(h.to_numpy(), 0.35))
    if h0_hi < h0_lo + 0.2:
        h0_hi = h0_lo + 0.2
    pad = max(0.15, 0.04 * span)
    return {
        "h_min": h_min,
        "h_max": h_max,
        "h_ref": float(np.median(h.to_numpy())),
        "h0_lo": float(h0_lo),
        "h0_hi": float(h0_hi),
        "h0_fit_lo": float(h0_lo),
        "h0_fit_hi": float(h0_fit_hi),
        "x_lo": float(h0_lo - pad),
        "x_hi": float(h_max + pad),
    }


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
        # Log-space fit: ratings are judged in percent, not raw cfs.
        # B and H₀ are bounded so the solver cannot run off to a nonsense exponent.
        def offset_powerlaw_log(H, log10_A, B, H0):
            H_eff = np.maximum(H - H0, 1e-6)
            return log10_A + B * np.log10(H_eff)

        A_simple, B_simple, _, _ = fit_powerlaw_rating_curve(
            df['H'], df['Q'], min_points=min_points
        )
        window = rating_stage_window(df['H'])
        h0_lo = window["h0_fit_lo"]
        h0_hi = window["h0_fit_hi"]
        guess_h0 = offset_guess if np.isfinite(offset_guess) else window["h_min"] - 0.2
        guess_h0 = float(np.clip(guess_h0, h0_lo + 1e-4, h0_hi - 1e-4))
        guess_b = B_simple if np.isfinite(B_simple) else 1.67
        guess_b = float(np.clip(guess_b, 0.5, 3.5))
        guess_a = A_simple if np.isfinite(A_simple) and A_simple > 0 else 1.0
        guess_log_a = float(np.clip(np.log10(guess_a), -2.0, 8.0))

        popt, _pcov = curve_fit(
            offset_powerlaw_log,
            df['H'].values,
            np.log10(df['Q'].values),
            p0=[guess_log_a, guess_b, guess_h0],
            bounds=([-2.0, 0.5, h0_lo], [8.0, 3.5, h0_hi]),
            maxfev=8000,
        )

        log10_A, B, H0 = popt
        A = 10 ** log10_A

        def offset_powerlaw(H, A, B, H0):
            H_eff = np.maximum(np.asarray(H, dtype=float) - H0, 1e-6)
            return A * (H_eff ** B)

        # Arithmetic R² is still reported so existing callers keep their meaning.
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

    # Prefer offset when it explains percent error better. Arithmetic R² lets
    # floods crown a curve that is wrong at low stage.
    log_s = evaluate_rating_params(df["H"], df["Q"], A_s, B_s, 0.0).get("r2_log", np.nan)
    log_o = evaluate_rating_params(df["H"], df["Q"], A_o, B_o, H0).get("r2_log", np.nan)
    use_offset = (
        np.isfinite(log_o)
        and np.isfinite(A_o)
        and np.isfinite(B_o)
        and (not np.isfinite(log_s) or log_o >= log_s + 0.01 or (np.isfinite(R2_s) and R2_s < 0.5))
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


def predict_rating_discharge(
    stage: np.ndarray | pd.Series,
    A: float,
    B: float,
    H0: float = 0.0,
) -> np.ndarray:
    """Q = A * (H - H0)^B with safe floor on effective stage."""
    H = np.asarray(stage, dtype=float)
    H_eff = np.maximum(H - float(H0), 1e-10)
    A = float(A)
    B = float(B)
    if not np.isfinite(A) or not np.isfinite(B) or A <= 0:
        return np.full_like(H, np.nan, dtype=float)
    return A * (H_eff ** B)


def evaluate_rating_params(
    stage: pd.Series | np.ndarray,
    discharge: pd.Series | np.ndarray,
    A: float,
    B: float,
    H0: float = 0.0,
) -> Dict[str, Any]:
    """
    Score a user- or auto-specified rating Q = A*(H-H0)^B against paired data.

    Returns R², residual stats, equation string, and predicted Q on the clean pairs.
    """
    raw = pd.DataFrame({"H": stage, "Q": discharge})
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    raw = raw[(raw["H"] > 0) & (raw["Q"] > 0)]
    h_ref = float(raw["H"].median()) if not raw.empty else np.nan
    n_excluded = int((raw["H"] <= float(H0) + 1e-9).sum()) if not raw.empty else 0
    q_ref = np.nan
    if np.isfinite(h_ref) and h_ref > float(H0) and np.isfinite(A) and A > 0 and np.isfinite(B):
        q_ref = float(predict_rating_discharge(np.array([h_ref]), A, B, H0)[0])

    df = raw[raw["H"] > float(H0) + 1e-9]

    empty = {
        "model": "offset_powerlaw" if abs(float(H0)) > 1e-9 else "powerlaw",
        "A": float(A),
        "B": float(B),
        "H0": float(H0),
        "R2": np.nan,
        "r2_log": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "mape": np.nan,
        "mdape": np.nan,
        "bias": np.nan,
        "nash_sutcliffe": np.nan,
        "n_points": 0,
        "n_excluded": n_excluded,
        "h_ref": h_ref,
        "q_ref": q_ref,
        "equation": "n/a",
        "Q_pred": np.array([], dtype=float),
        "H_clean": np.array([], dtype=float),
        "Q_clean": np.array([], dtype=float),
        "residuals": np.array([], dtype=float),
    }
    if len(df) < 3 or not np.isfinite(A) or not np.isfinite(B) or A <= 0:
        return empty

    H = df["H"].values
    Q = df["Q"].values
    Q_hat = predict_rating_discharge(H, A, B, H0)
    valid = np.isfinite(Q_hat)
    H, Q, Q_hat = H[valid], Q[valid], Q_hat[valid]
    if len(Q) < 3:
        return empty

    ss_res = float(np.sum((Q - Q_hat) ** 2))
    ss_tot = float(np.sum((Q - Q.mean()) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    residuals = Q_hat - Q
    abs_err = np.abs(residuals)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(abs_err))
    nz = Q != 0
    mape = float(np.mean(np.abs(residuals[nz] / Q[nz])) * 100) if np.any(nz) else np.nan
    bias = float(np.mean(residuals))
    nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    log_q = np.log10(Q)
    log_hat = np.log10(np.maximum(Q_hat, 1e-12))
    ss_res_l = float(np.sum((log_q - log_hat) ** 2))
    ss_tot_l = float(np.sum((log_q - log_q.mean()) ** 2))
    r2_log = 1.0 - ss_res_l / ss_tot_l if ss_tot_l > 0 else np.nan
    mdape = float(np.median(np.abs(residuals / Q) * 100.0))

    if abs(float(H0)) > 1e-6:
        eq = f"Q = {A:.4g} · (H − {H0:.3f})^{B:.3f}"
        model = "offset_powerlaw"
    else:
        eq = f"Q = {A:.4g} · H^{B:.3f}"
        model = "powerlaw"

    return {
        "model": model,
        "A": float(A),
        "B": float(B),
        "H0": float(H0),
        "R2": float(R2) if np.isfinite(R2) else np.nan,
        "r2_log": float(r2_log) if np.isfinite(r2_log) else np.nan,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "mdape": mdape,
        "bias": bias,
        "nash_sutcliffe": float(nse) if np.isfinite(nse) else np.nan,
        "n_points": int(len(Q)),
        "n_excluded": n_excluded,
        "h_ref": h_ref,
        "q_ref": q_ref,
        "equation": eq,
        "Q_pred": Q_hat,
        "H_clean": H,
        "Q_clean": Q,
        "residuals": residuals,
    }


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
