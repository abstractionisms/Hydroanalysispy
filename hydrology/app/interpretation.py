"""Small interpretation helpers for dashboard chart context."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class InsightCard:
    title: str
    value: str
    body: str
    state: str = "ready"


def _flow_column(df: pd.DataFrame) -> str | None:
    for col in ("Discharge_cfs", "value", "streamflow_cfs"):
        if col in df.columns:
            return col
    return df.columns[0] if len(df.columns) else None


def _status_from_percentile(percentile: float | None) -> tuple[str, str]:
    if percentile is None:
        return "Unknown", "limited"
    if percentile >= 90:
        return "Much above normal", "limited"
    if percentile >= 75:
        return "Above normal", "ready"
    if percentile >= 25:
        return "Near normal", "ready"
    if percentile >= 10:
        return "Below normal", "limited"
    return "Much below normal", "blocked"


def summarize_flow_context(df: pd.DataFrame | None) -> list[InsightCard]:
    """Summarize selected-site flow record into plain-language cards."""
    if df is None or df.empty:
        return [
            InsightCard(
                "Flow Context",
                "No data",
                "Daily discharge was not available for this site and date range.",
                "blocked",
            )
        ]

    col = _flow_column(df)
    if not col:
        return []

    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return [
            InsightCard(
                "Flow Context",
                "No values",
                "The selected discharge column contains no numeric values.",
                "blocked",
            )
        ]

    latest_date = series.index.max()
    latest_value = float(series.loc[latest_date])
    record_years = (series.index.max() - series.index.min()).days / 365.25

    doy = latest_date.dayofyear
    seasonal = series[(series.index.dayofyear >= doy - 15) & (series.index.dayofyear <= doy + 15)]
    baseline = seasonal if len(seasonal) >= 30 else series
    percentile = float((baseline < latest_value).mean() * 100) if len(baseline) else None
    status, status_state = _status_from_percentile(percentile)

    recent = series.tail(30).mean() if len(series) >= 30 else series.mean()
    previous = series.iloc[-60:-30].mean() if len(series) >= 60 else None
    if previous and previous > 0:
        trend_pct = (recent - previous) / previous * 100
        trend_label = "Rising" if trend_pct > 5 else "Falling" if trend_pct < -5 else "Stable"
        trend_body = f"The last 30 days average {abs(trend_pct):.0f}% {'above' if trend_pct >= 0 else 'below'} the prior 30 days."
        trend_state = "limited" if abs(trend_pct) > 25 else "ready"
    else:
        trend_label = "Not enough context"
        trend_body = "At least 60 days are needed for a recent-vs-prior trend comparison."
        trend_state = "limited"

    return [
        InsightCard(
            "Current Flow",
            f"{latest_value:,.0f} cfs",
            f"Latest daily value on {latest_date:%Y-%m-%d}.",
            "ready",
        ),
        InsightCard(
            "Seasonal Context",
            f"{percentile:.0f}th pct." if percentile is not None else "Unknown",
            f"{status} for this time of year based on the selected historical record.",
            status_state,
        ),
        InsightCard(
            "Recent Direction",
            trend_label,
            trend_body,
            trend_state,
        ),
        InsightCard(
            "Record Depth",
            f"{record_years:.1f} yrs",
            "Enough for trend and duration views." if record_years >= 10 else "Use caution for climate normals and frequency-style interpretation.",
            "ready" if record_years >= 10 else "limited",
        ),
    ]


def summarize_recommendations(has_stage: bool, has_climate: bool, record_years: float) -> list[InsightCard]:
    """Recommend next analysis actions based on available data."""
    cards = [
        InsightCard(
            "Recommended Next",
            "Flow duration",
            "Start with flow duration and seasonal anomaly to understand normal vs unusual flow states.",
            "ready",
        )
    ]
    if has_climate:
        cards.append(
            InsightCard(
                "Climate Link",
                "Use SPI",
                "Climate data is available, so precipitation and drought context are worth reviewing.",
                "ready",
            )
        )
    else:
        cards.append(
            InsightCard(
                "Climate Link",
                "Limited",
                "Climate-linked plots may be incomplete until weather data can be merged.",
                "limited",
            )
        )
    if has_stage:
        cards.append(
            InsightCard(
                "Stage Review",
                "Available",
                "Stage overlay and rating-curve checks are valid for this selected period.",
                "ready",
            )
        )
    if record_years >= 10:
        cards.append(
            InsightCard(
                "Frequency",
                "On demand",
                "Peak-flow frequency can be run separately without slowing the page load.",
                "limited",
            )
        )
    return cards


def describe_standardized_index(value: float | None) -> tuple[str, str]:
    """Return label and plain-English meaning for SRI/SPI values."""
    if value is None or pd.isna(value):
        return "Unknown", "No current standardized value was available."
    if value <= -2:
        return "Exceptional dry signal", "Values below -2 are rare and indicate severe dry conditions."
    if value <= -1.3:
        return "Severe dry signal", "Values below -1.3 suggest materially drier-than-normal conditions."
    if value <= -0.8:
        return "Moderate dry signal", "Values below -0.8 suggest emerging dry conditions."
    if value < 0.8:
        return "Near normal", "Values between about -0.8 and +0.8 are close to the historical middle range."
    if value < 1.3:
        return "Moderately wet signal", "Positive values indicate wetter or higher-flow conditions than normal."
    if value < 2:
        return "Very wet signal", "Values above 1.3 are materially wetter than normal."
    return "Exceptional wet signal", "Values above 2 are rare and indicate very wet or high-flow conditions."
