"""Lightweight reach thermal-sensitivity screening context."""

from __future__ import annotations

from typing import Dict, List


def classify_thermal_sensitivity(
    summer_flow_cfs: float | None,
    channel_width_m: float | None,
    canopy_cover_pct: float | None,
    reach_gain_cfs: float | None,
) -> Dict[str, object]:
    """Classify simple thermal-sensitivity drivers for a reach."""
    score = 0
    drivers: List[str] = []

    if summer_flow_cfs is not None and summer_flow_cfs < 50:
        score += 1
        drivers.append("low summer flow")
    if channel_width_m is not None and channel_width_m >= 10:
        score += 1
        drivers.append("wide channel")
    if canopy_cover_pct is not None and canopy_cover_pct < 40:
        score += 2
        drivers.append("low canopy cover")
    elif canopy_cover_pct is not None and canopy_cover_pct >= 75:
        score -= 1
        drivers.append("high canopy cover")
    if reach_gain_cfs is not None and reach_gain_cfs < -1:
        score += 1
        drivers.append("losing reach")
    elif reach_gain_cfs is not None and reach_gain_cfs > 1:
        score -= 1
        drivers.append("gaining reach")

    if score >= 3:
        label = "high"
    elif score >= 1:
        label = "moderate"
    else:
        label = "low"

    return {
        "class": label,
        "score": score,
        "drivers": drivers,
        "note": "Screening context only; this is not a QUAL2K, Heat Source, Shade, or TTools model.",
    }
