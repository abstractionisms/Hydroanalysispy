"""Compact hydrologic signature metrics for daily streamflow."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .baseflow import lyne_hollick_filter


def compute_hydrologic_signatures(daily_q: pd.Series) -> Dict[str, float]:
    """Compute a small, dashboard-friendly set of daily-flow signatures."""
    flow = pd.Series(daily_q).dropna().astype(float)
    flow = flow[np.isfinite(flow)]
    flow = flow[flow >= 0].sort_index()
    if flow.empty:
        return {}

    total_flow = float(flow.sum())
    flashiness = float(flow.diff().abs().dropna().sum() / total_flow) if total_flow > 0 else float("nan")
    if isinstance(flow.index, pd.DatetimeIndex):
        monthly = flow.groupby(flow.index.month).mean()
        peak_month = int(monthly.idxmax())
        low_month = int(monthly.idxmin())
    else:
        peak_month = 0
        low_month = 0

    baseflow = lyne_hollick_filter(flow)
    mean_flow = float(flow.mean())

    return {
        "n_days": float(len(flow)),
        "mean_flow": mean_flow,
        "median_flow": float(flow.median()),
        "min_flow": float(flow.min()),
        "max_flow": float(flow.max()),
        "q05": float(flow.quantile(0.95)),
        "q10": float(flow.quantile(0.90)),
        "q50": float(flow.quantile(0.50)),
        "q90": float(flow.quantile(0.10)),
        "q95": float(flow.quantile(0.05)),
        "coefficient_of_variation": float(flow.std(ddof=1) / mean_flow) if mean_flow > 0 else float("nan"),
        "richards_baker_flashiness": flashiness,
        "baseflow_index_lh": float(baseflow.bfi),
        "high_flow_frequency": float((flow > flow.quantile(0.90)).sum() / len(flow)),
        "low_flow_frequency": float((flow < flow.quantile(0.10)).sum() / len(flow)),
        "peak_month": float(peak_month),
        "low_month": float(low_month),
    }
