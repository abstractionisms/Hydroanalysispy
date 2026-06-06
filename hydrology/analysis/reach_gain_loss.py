"""Reach-scale gain/loss summaries for paired streamflow gages."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def classify_reach_gain_loss(median_gain_cfs: float, deadband_cfs: float = 1.0) -> str:
    """Classify reach gain/loss using a small deadband around zero."""
    if not np.isfinite(median_gain_cfs):
        return "insufficient_data"
    if median_gain_cfs > deadband_cfs:
        return "gaining"
    if median_gain_cfs < -deadband_cfs:
        return "losing"
    return "neutral"


def summarize_reach_gain_loss(
    upstream_q: pd.Series,
    downstream_q: pd.Series,
    reach_km: float | None = None,
    drainage_area_sqmi: float | None = None,
    low_flow_quantile: float = 0.25,
    deadband_cfs: float = 1.0,
) -> Dict[str, float | str]:
    """Summarize paired upstream/downstream flow differences for one reach."""
    upstream = pd.Series(upstream_q).dropna().astype(float)
    downstream = pd.Series(downstream_q).dropna().astype(float)
    paired = pd.concat({"upstream": upstream, "downstream": downstream}, axis=1).dropna()
    if paired.empty:
        return {"classification": "insufficient_data", "confidence": "none", "n_days": 0.0}

    paired["gain_cfs"] = paired["downstream"] - paired["upstream"]
    low_threshold = paired["upstream"].quantile(low_flow_quantile)
    low_flow = paired[paired["upstream"] <= low_threshold]

    median_gain = float(paired["gain_cfs"].median())
    low_flow_gain = float(low_flow["gain_cfs"].median()) if not low_flow.empty else float("nan")
    classification = classify_reach_gain_loss(median_gain, deadband_cfs=deadband_cfs)
    variability = float(paired["gain_cfs"].std(ddof=1)) if len(paired) > 1 else 0.0

    if len(paired) < 7:
        confidence = "low"
    elif variability <= max(abs(median_gain), deadband_cfs):
        confidence = "high"
    else:
        confidence = "moderate"

    result: Dict[str, float | str] = {
        "n_days": float(len(paired)),
        "median_gain_cfs": median_gain,
        "mean_gain_cfs": float(paired["gain_cfs"].mean()),
        "low_flow_median_gain_cfs": low_flow_gain,
        "classification": classification,
        "confidence": confidence,
    }
    if reach_km and reach_km > 0:
        result["median_gain_cfs_per_km"] = median_gain / reach_km
    if drainage_area_sqmi and drainage_area_sqmi > 0:
        result["median_gain_cfs_per_sqmi"] = median_gain / drainage_area_sqmi

    return result
