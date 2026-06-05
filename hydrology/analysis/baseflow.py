"""Baseflow separation methods for streamflow hydrographs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaseflowResult:
    """Baseflow separation result with daily components and summary BFI."""

    method: str
    components: pd.DataFrame
    bfi: float
    parameters: Dict[str, float]


def _clean_daily_flow(daily_q: pd.Series) -> pd.Series:
    series = pd.Series(daily_q).dropna().astype(float)
    series = series[np.isfinite(series)]
    series = series[series >= 0]
    return series.sort_index()


def _build_result(
    method: str,
    total: pd.Series,
    baseflow: np.ndarray,
    parameters: Dict[str, float],
) -> BaseflowResult:
    total_values = total.values.astype(float)
    base = np.clip(baseflow.astype(float), 0, total_values)
    quick = total_values - base
    components = pd.DataFrame(
        {
            "total_flow": total_values,
            "baseflow": base,
            "quickflow": quick,
        },
        index=total.index,
    )
    total_sum = float(components["total_flow"].sum())
    bfi = float(components["baseflow"].sum() / total_sum) if total_sum > 0 else float("nan")
    return BaseflowResult(method=method, components=components, bfi=bfi, parameters=parameters)


def lyne_hollick_filter(
    daily_q: pd.Series,
    alpha: float = 0.925,
    passes: int = 3,
) -> BaseflowResult:
    """Separate baseflow with the Lyne-Hollick recursive digital filter."""
    flow = _clean_daily_flow(daily_q)
    if flow.empty:
        return _build_result(
            "lyne_hollick",
            flow,
            np.array([], dtype=float),
            {"alpha": alpha, "passes": float(passes)},
        )
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if passes < 1:
        raise ValueError("passes must be >= 1")

    q = flow.values.astype(float)
    quick = q.copy()
    direction = 1
    for _ in range(passes):
        work = quick if direction == 1 else quick[::-1]
        filtered = np.zeros_like(work)
        for i in range(1, len(work)):
            filtered[i] = alpha * filtered[i - 1] + ((1 + alpha) / 2.0) * (work[i] - work[i - 1])
            filtered[i] = min(max(filtered[i], 0.0), work[i])
        quick = filtered if direction == 1 else filtered[::-1]
        direction *= -1

    return _build_result(
        "lyne_hollick",
        flow,
        q - quick,
        {"alpha": alpha, "passes": float(passes)},
    )


def eckhardt_filter(
    daily_q: pd.Series,
    alpha: float = 0.98,
    bfi_max: float = 0.80,
) -> BaseflowResult:
    """Separate baseflow with the Eckhardt two-parameter recursive filter."""
    flow = _clean_daily_flow(daily_q)
    if flow.empty:
        return _build_result(
            "eckhardt",
            flow,
            np.array([], dtype=float),
            {"alpha": alpha, "bfi_max": bfi_max},
        )
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 < bfi_max < 1:
        raise ValueError("bfi_max must be between 0 and 1")

    q = flow.values.astype(float)
    base = np.zeros_like(q)
    base[0] = min(q[0], q[0] * bfi_max)
    denominator = 1 - alpha * bfi_max
    for i in range(1, len(q)):
        numerator = (1 - bfi_max) * alpha * base[i - 1] + (1 - alpha) * bfi_max * q[i]
        base[i] = min(q[i], max(0.0, numerator / denominator))

    return _build_result(
        "eckhardt",
        flow,
        base,
        {"alpha": alpha, "bfi_max": bfi_max},
    )


def compare_baseflow_methods(daily_q: pd.Series) -> Dict[str, float | str]:
    """Compare Lyne-Hollick and Eckhardt BFI estimates for the same hydrograph."""
    lyne_hollick = lyne_hollick_filter(daily_q)
    eckhardt = eckhardt_filter(daily_q)
    difference = abs(lyne_hollick.bfi - eckhardt.bfi)

    if difference <= 0.05:
        agreement = "strong"
    elif difference <= 0.10:
        agreement = "moderate"
    else:
        agreement = "weak"

    return {
        "lyne_hollick_bfi": lyne_hollick.bfi,
        "eckhardt_bfi": eckhardt.bfi,
        "bfi_difference": difference,
        "agreement": agreement,
    }
