"""Small interpretation helpers for dashboard chart context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InsightCard:
    title: str
    value: str
    body: str
    state: str = "ready"
    help: str = ""


def _flow_column(df: pd.DataFrame) -> str | None:
    for col in ("Discharge_cfs", "value", "streamflow_cfs"):
        if col in df.columns:
            return col
    return df.columns[0] if len(df.columns) else None


def _series(df: pd.DataFrame | None, col: str | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    use = col or _flow_column(df)
    if not use or use not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[use], errors="coerce").dropna()
    return s[s >= 0]


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
                help="Widen the date range or choose a gage with continuous daily discharge (00060).",
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
                help="The table loaded but discharge cells are empty or non-numeric for this period.",
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
            help=(
                "Most recent daily mean discharge in the selected window. "
                "Compare with Seasonal Context — the same number can be high or low "
                "depending on the time of year."
            ),
        ),
        InsightCard(
            "Seasonal Context",
            f"{percentile:.0f}th pct." if percentile is not None else "Unknown",
            f"{status} for this time of year based on the selected historical record.",
            status_state,
            help=(
                "Percentile of current flow among same calendar days (±15 days) in this "
                "record. 50th ≈ typical for the season; 90th ≈ much wetter than normal."
            ),
        ),
        InsightCard(
            "Recent Direction",
            trend_label,
            trend_body,
            trend_state,
            help=(
                "Compares the last 30-day mean to the prior 30 days. Short-term only — "
                "not a multi-year trend. Rising after storms is normal."
            ),
        ),
        InsightCard(
            "Record Depth",
            f"{record_years:.1f} yrs",
            "Enough for trend and duration views." if record_years >= 10 else "Use caution for climate normals and frequency-style interpretation.",
            "ready" if record_years >= 10 else "limited",
            help=(
                "Years of daily data in the selected range. Flood frequency and climate "
                "normals usually want 10–30+ years; short windows are fine for screening."
            ),
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
            help=(
                "The FDC and Q10/Q50/Q90 strip show how often high vs low flows occur. "
                "Scroll to Interactive Charts → Duration statistics."
            ),
        )
    ]
    if has_climate:
        cards.append(
            InsightCard(
                "Climate Link",
                "Use SPI",
                "Climate data is available, so precipitation and drought context are worth reviewing.",
                "ready",
                help=(
                    "SPI tracks meteorological drought; SRI tracks runoff drought. "
                    "Open Drought & Baseflow Indicators below the charts when ready."
                ),
            )
        )
    else:
        cards.append(
            InsightCard(
                "Climate Link",
                "Limited",
                "Climate-linked plots may be incomplete until weather data can be merged.",
                "limited",
                help=(
                    "No reliable precip/temp merge for this period. SPI may still fetch "
                    "Open-Meteo, but co-plotted climate overlays will be thin."
                ),
            )
        )
    if has_stage:
        cards.append(
            InsightCard(
                "Stage Review",
                "Available",
                "Stage overlay and rating-curve checks are valid for this selected period.",
                "ready",
                help=(
                    "Gage height is present. Use dual-axis stage on the hydrograph and the "
                    "Rating curve workshop to tune A, B, and H₀."
                ),
            )
        )
    if record_years >= 10:
        cards.append(
            InsightCard(
                "Frequency",
                "On demand",
                "Peak-flow frequency can be run separately without slowing the page load.",
                "limited",
                help=(
                    "Annual-max flood frequency is opt-in so page load stays fast. "
                    "Open Frequency Analysis under Advanced when you need return periods."
                ),
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


def _percentile_label(p: float) -> str:
    if p >= 90:
        return "much higher than usual"
    if p >= 75:
        return "higher than usual"
    if p >= 25:
        return "near the middle of the record"
    if p >= 10:
        return "lower than usual"
    return "much lower than usual"


def summarize_flow_duration(df_q: pd.DataFrame | None) -> str:
    """Plain-language flow-duration / regime summary from daily Q."""
    s = _series(df_q)
    if s.empty or len(s) < 30:
        return "Not enough discharge values to characterize flow duration."

    q10 = float(np.percentile(s, 90))  # high flow exceeded ~10% of time
    q50 = float(np.percentile(s, 50))
    q90 = float(np.percentile(s, 10))  # low flow exceeded ~90% of time
    mean = float(s.mean())
    # Flashiness-ish: high/low ratio
    ratio = q10 / max(q90, 1e-6)
    if ratio >= 50:
        regime = "flashy / high-contrast"
        regime_note = (
            "High flows dwarf low flows (Q10/Q90 is large), so the stream swings "
            "between wet-season peaks and sustained low baseflow."
        )
    elif ratio >= 15:
        regime = "moderately variable"
        regime_note = (
            "There is a clear high-flow season and a quieter low-flow season, "
            "but extremes are not extreme for all rivers."
        )
    else:
        regime = "relatively steady"
        regime_note = (
            "High and low ends of the duration curve are closer together, "
            "suggesting more regulated or baseflow-supported behavior."
        )

    return (
        f"**Flow duration / regime ({regime}):** "
        f"median daily flow is about **{q50:,.0f} cfs**, with high-flow (Q10) near "
        f"**{q10:,.0f} cfs** and low-flow (Q90) near **{q90:,.0f} cfs** "
        f"(mean **{mean:,.0f} cfs**). {regime_note}"
    )


def summarize_seasonal_pattern(df_q: pd.DataFrame | None) -> str:
    """Seasonal mean-flow contrast from daily Q."""
    s = _series(df_q)
    if s.empty or len(s) < 90:
        return "Not enough data for a seasonal pattern summary."

    months = s.index.month
    buckets = {
        "winter (DJF)": s[np.isin(months, [12, 1, 2])],
        "spring (MAM)": s[np.isin(months, [3, 4, 5])],
        "summer (JJA)": s[np.isin(months, [6, 7, 8])],
        "fall (SON)": s[np.isin(months, [9, 10, 11])],
    }
    means = {k: float(v.mean()) for k, v in buckets.items() if len(v) >= 10}
    if len(means) < 2:
        return "Seasonal contrast could not be estimated for this period."

    wettest = max(means, key=means.get)
    driest = min(means, key=means.get)
    wet_v, dry_v = means[wettest], means[driest]
    factor = wet_v / max(dry_v, 1e-6)
    return (
        f"**Seasonal pattern:** average daily flow is highest in **{wettest}** "
        f"(~{wet_v:,.0f} cfs) and lowest in **{driest}** (~{dry_v:,.0f} cfs) — "
        f"about **{factor:.1f}×** higher in the wetter season. "
        "Use monthly boxplots / heatmaps to see timing of the high-flow pulse."
    )


def summarize_climate_merged(df_merged: pd.DataFrame | None) -> str:
    """Climate linkage when precip/temp are merged to discharge."""
    if df_merged is None or df_merged.empty:
        return (
            "**Climate:** no merged temperature/precipitation for this site and period "
            "(Open-Meteo/Meteostat may still work for SPI separately)."
        )

    parts = []
    if "Precip_mm" in df_merged.columns:
        p = pd.to_numeric(df_merged["Precip_mm"], errors="coerce").dropna()
        if not p.empty:
            annual = p.resample("YE").sum()
            mean_ann = float(annual.mean()) if len(annual) else float(p.sum() / max(len(p) / 365.25, 1))
            wet_days = float((p > 1.0).mean() * 100)
            parts.append(
                f"mean annual precip ~**{mean_ann:,.0f} mm**, "
                f"with precip >1 mm on about **{wet_days:.0f}%** of days"
            )
    if "Temp_C" in df_merged.columns:
        t = pd.to_numeric(df_merged["Temp_C"], errors="coerce").dropna()
        if not t.empty:
            parts.append(
                f"mean temp **{float(t.mean()):.1f}°C** "
                f"(range {float(t.min()):.1f}–{float(t.max()):.1f}°C)"
            )
    if not parts:
        return "**Climate:** merged frame present but precip/temp columns were empty."
    return "**Climate (merged to streamflow days):** " + "; ".join(parts) + "."


def summarize_rating_curve(df_q: pd.DataFrame | None) -> str:
    """Text summary of stage–discharge fit when gage height is present."""
    if df_q is None or df_q.empty:
        return ""
    if "Gage_Height_ft" not in df_q.columns or "Discharge_cfs" not in df_q.columns:
        return ""
    try:
        from hydrology.analysis.stage_discharge import fit_best_rating_curve
    except Exception:
        return ""

    stage = pd.to_numeric(df_q["Gage_Height_ft"], errors="coerce")
    q = pd.to_numeric(df_q["Discharge_cfs"], errors="coerce")
    fit = fit_best_rating_curve(stage, q, min_points=10)
    if fit.get("model") == "none" or not np.isfinite(fit.get("R2", np.nan)):
        return (
            "**Rating curve:** stage is present but a stable Q–H fit could not be "
            "estimated (need enough positive pairs)."
        )
    r2 = fit["R2"]
    quality = "strong" if r2 >= 0.9 else "moderate" if r2 >= 0.7 else "weak"
    note = ""
    if fit["model"] == "offset_powerlaw":
        note = (
            f" An offset stage H₀≈{fit['H0']:.2f} ft was needed — common when gage "
            "zero is not the zero-flow control elevation."
        )
    return (
        f"**Stage–discharge rating:** best model is `{fit['equation']}` "
        f"with **R²={r2:.3f}** ({quality} fit, n={fit['n_points']:,}).{note} "
        "Points often show seasonal loops (rising vs falling limb / vegetation / ice)."
    )


def summarize_selected_plots(plot_keys: Iterable[str]) -> str:
    """Explain what the selected plot set is good for."""
    keys = [str(k) for k in plot_keys]
    if not keys:
        return "No static plots were selected."

    themes = []
    if any(k in keys for k in ("timeseries", "anomaly", "monthly_boxplot", "discharge_heatmap")):
        themes.append("time patterns and seasonality")
    if any(k in keys for k in ("flow_duration", "low_flow_trend", "7q10_analysis")):
        themes.append("low-flow / duration behavior")
    if any(k in keys for k in ("flood_frequency", "annual_trend")):
        themes.append("peaks and long-term change")
    if any(k in keys for k in ("rating_curve",)):
        themes.append("stage–discharge relationship")
    if any("precip" in k or "temp" in k or "climate" in k or "lag" in k or "hexbin" in k for k in keys):
        themes.append("climate–streamflow links")
    if not themes:
        themes.append("general site diagnostics")

    pretty = ", ".join(keys[:8]) + ("…" if len(keys) > 8 else "")
    return (
        f"**Plots generated ({len(keys)}):** {pretty}. "
        f"Together they emphasize {', '.join(themes)}."
    )


def build_plot_analysis_report(
    *,
    site_id: str,
    site_desc: str,
    plot_keys: list[str] | tuple[str, ...],
    df_q: pd.DataFrame | None,
    df_merged: pd.DataFrame | None = None,
) -> str:
    """
    Full markdown narrative after static plot generation.

    Combines record stats, duration/regime, seasons, climate, and rating notes
    so the user gets a text explanation next to the figure grid.
    """
    s = _series(df_q)
    lines: list[str] = [
        f"### Automated read — {site_desc}",
        f"USGS **{site_id}** · generated from the selected period and plot set.",
        "",
    ]

    if s.empty:
        lines.append("No usable discharge series was available for this explanation.")
        return "\n".join(lines)

    years = (s.index.max() - s.index.min()).days / 365.25
    latest = float(s.iloc[-1])
    latest_date = s.index.max()
    # Seasonal percentile for latest
    doy = latest_date.dayofyear
    seasonal = s[(s.index.dayofyear >= doy - 15) & (s.index.dayofyear <= doy + 15)]
    baseline = seasonal if len(seasonal) >= 30 else s
    pct = float((baseline < latest).mean() * 100) if len(baseline) else None

    lines.append(
        f"**Record:** {len(s):,} daily values spanning **{years:.1f} years** "
        f"({s.index.min():%Y-%m-%d} → {s.index.max():%Y-%m-%d})."
    )
    if pct is not None:
        lines.append(
            f"**Latest flow:** **{latest:,.0f} cfs** on {latest_date:%Y-%m-%d} — "
            f"about the **{pct:.0f}th percentile** for this time of year "
            f"({_percentile_label(pct)})."
        )
    lines.append("")
    lines.append(summarize_selected_plots(plot_keys))
    lines.append("")
    lines.append(summarize_flow_duration(df_q))
    lines.append("")
    lines.append(summarize_seasonal_pattern(df_q))
    lines.append("")
    lines.append(summarize_climate_merged(df_merged))
    rating = summarize_rating_curve(df_q)
    if rating:
        lines.append("")
        lines.append(rating)

    # Computed metric strip + relevance
    if not s.empty:
        q10 = float(np.percentile(s, 90))
        q50 = float(np.percentile(s, 50))
        q90 = float(np.percentile(s, 10))
        mean = float(s.mean())
        peak = float(s.max())
        lines.append("")
        lines.append("#### Key metrics (this period)")
        lines.append(
            f"- Mean **{mean:,.0f} cfs** · Median (Q50) **{q50:,.0f} cfs** · "
            f"Q10 **{q10:,.0f} cfs** · Q90 **{q90:,.0f} cfs** · Peak day **{peak:,.0f} cfs**"
        )
        if mean > q50 * 1.35:
            lines.append(
                "- **Mean ≫ median** — floods pull the average up; prefer median/Q50 for “typical” conditions."
            )
        lines.append("")
        lines.append(
            format_metric_relevance_markdown(
                metric_keys_for_plots(plot_keys),
                df_q=df_q,
                df_merged=df_merged,
                plot_keys=plot_keys,
            )
        )

    lines.append("")
    lines.append(
        "_This is an automated screening narrative from the plotted data — not a formal "
        "hydrologic design report. Verify peaks, ice, regulation, and rating shifts "
        "before engineering use._"
    )
    return "\n".join(lines)


def build_interactive_chart_brief(
    df_q: pd.DataFrame | None,
    df_merged: pd.DataFrame | None = None,
) -> str:
    """Shorter narrative under auto-loaded interactive hydrograph + FDC."""
    s = _series(df_q)
    if s.empty:
        return "Interactive charts loaded without discharge values to summarize."
    parts = [
        summarize_flow_duration(df_q),
        summarize_seasonal_pattern(df_q),
        "",
        format_metric_relevance_markdown(
            ["q10", "q50", "q90", "mean_flow"],
            df_q=df_q,
            df_merged=df_merged,
            plot_keys=["flow_duration", "timeseries"],
        ),
    ]
    return "\n\n".join(parts)


# Plain-English "why this metric matters" for dashboard cards and reports
METRIC_RELEVANCE: dict[str, dict[str, str]] = {
    "record_length": {
        "label": "Record length",
        "meaning": "How many years of daily data are in the selected window.",
        "relevance": (
            "Longer records support trends, frequency analysis, and climate normals. "
            "Under ~10 years, treat flood frequency and long-term SPI carefully."
        ),
        "use_when": "Judging whether trends/frequency/SPI are defensible.",
    },
    "data_points": {
        "label": "Data points",
        "meaning": "Count of daily observations after quality filtering.",
        "relevance": (
            "More points reduce noise in percentiles and duration curves. "
            "Large gaps matter more than total count for seasonal plots."
        ),
        "use_when": "Checking completeness before trusting summary stats.",
    },
    "mean_flow": {
        "label": "Mean flow",
        "meaning": "Average daily discharge over the selected period.",
        "relevance": (
            "Anchors water-supply style questions, but is pulled up by floods. "
            "Compare to median (Q50) — if mean ≫ median, the record is peak-dominated."
        ),
        "use_when": "Rough water yield; always pair with median for skewed rivers.",
    },
    "peak_flow": {
        "label": "Peak flow",
        "meaning": "Highest daily mean discharge in the selected period.",
        "relevance": (
            "Flags flood magnitude in the window, not the official annual peak series. "
            "Use Frequency Analysis (peak-flow table) for design return periods."
        ),
        "use_when": "Screening large events; not a substitute for LP3 frequency design.",
    },
    "q10": {
        "label": "Q10 (high flow)",
        "meaning": "Discharge exceeded about 10% of days (upper duration curve).",
        "relevance": "Describes wet-season / high-flow habitat and floodplain connectivity.",
        "use_when": "High-flow ecology, channel maintenance, flood context.",
    },
    "q50": {
        "label": "Q50 (median)",
        "meaning": "Discharge exceeded about half the days.",
        "relevance": "Robust central tendency; less skewed by floods than the mean.",
        "use_when": "Typical conditions and year-to-year comparisons.",
    },
    "q90": {
        "label": "Q90 (low flow)",
        "meaning": "Discharge exceeded about 90% of days (lower duration curve).",
        "relevance": "Drought, baseflow, and aquatic-habitat stress indicator.",
        "use_when": "Low-flow management, summer shortages, baseflow screening.",
    },
    "rating_r2": {
        "label": "Rating R²",
        "meaning": "How tightly stage and discharge follow the fitted Q–H model.",
        "relevance": (
            "High R² means a stable control; scatter/seasonal color can flag hysteresis, "
            "rating shifts, ice, or vegetation."
        ),
        "use_when": "Interpreting stage–discharge plots and gage reliability.",
    },
    "spi_sri": {
        "label": "SPI / SRI",
        "meaning": "Standardized precip (SPI) or runoff (SRI) anomalies (γ → normal).",
        "relevance": (
            "Negative = drier than normal for that accumulation window. "
            "SRI lags SPI when soil/groundwater buffer the response."
        ),
        "use_when": "Drought monitoring and comparing meteorological vs hydrologic drought.",
    },
}


def metric_keys_for_plots(plot_keys: Iterable[str] | None = None) -> list[str]:
    """Pick relevance keys that match the generated plot set."""
    keys = set(plot_keys) if plot_keys is not None else set()
    out = ["record_length", "data_points", "mean_flow", "peak_flow", "q50", "q10", "q90"]
    if any(k in keys for k in ("flow_duration", "low_flow_trend", "7q10_analysis")):
        out.extend(["q10", "q90"])
    if "rating_curve" in keys:
        out.append("rating_r2")
    if any("spi" in k or "drought" in k or "precip" in k for k in keys):
        out.append("spi_sri")
    if not keys:
        out = ["record_length", "data_points", "mean_flow", "peak_flow", "q10", "q50", "q90"]
    seen: set[str] = set()
    ordered: list[str] = []
    for k in out:
        if k not in seen and k in METRIC_RELEVANCE:
            seen.add(k)
            ordered.append(k)
    return ordered


def compute_hydrologic_profile(
    df_q: pd.DataFrame | None,
    df_merged: pd.DataFrame | None = None,
    discharge_col: str | None = None,
) -> dict:
    """Site/period stats that drive dynamic metric relevance."""
    s = _series(df_q, discharge_col)
    profile: dict = {"ok": False}
    if s.empty or len(s) < 5:
        return profile

    q10 = float(np.percentile(s, 90))
    q50 = float(np.percentile(s, 50))
    q90 = float(np.percentile(s, 10))
    mean = float(s.mean())
    peak = float(s.max())
    years = max((s.index.max() - s.index.min()).days / 365.25, 0.01)
    n = int(len(s))
    cv = float(s.std() / mean) if mean > 0 else float("nan")
    ratio = q10 / max(q90, 1e-6)
    skew = mean / max(q50, 1e-6)

    months = s.index.month
    season_means: dict[str, float] = {}
    for name, mlist in (
        ("DJF", [12, 1, 2]),
        ("MAM", [3, 4, 5]),
        ("JJA", [6, 7, 8]),
        ("SON", [9, 10, 11]),
    ):
        sub = s[np.isin(months, mlist)]
        if len(sub) >= 10:
            season_means[name] = float(sub.mean())

    wet_season = max(season_means, key=season_means.get) if season_means else None
    dry_season = min(season_means, key=season_means.get) if season_means else None
    season_factor = None
    if wet_season and dry_season and season_means[dry_season] > 0:
        season_factor = season_means[wet_season] / season_means[dry_season]

    if ratio >= 50 or (np.isfinite(cv) and cv >= 1.5):
        regime, regime_plain = "flashy", "flashy / peak-dominated"
    elif ratio >= 15 or (np.isfinite(cv) and cv >= 0.8):
        regime, regime_plain = "seasonal", "seasonally variable"
    else:
        regime, regime_plain = "steady", "relatively steady / baseflow-supported"

    zero_frac = float((s <= max(q90 * 0.05, 0.01)).mean())

    has_stage = (
        df_q is not None
        and "Gage_Height_ft" in df_q.columns
        and pd.to_numeric(df_q["Gage_Height_ft"], errors="coerce").notna().sum() >= 10
    )
    rating = None
    if has_stage:
        try:
            from hydrology.analysis.stage_discharge import fit_best_rating_curve

            stage = pd.to_numeric(df_q["Gage_Height_ft"], errors="coerce")
            q = pd.to_numeric(df_q["Discharge_cfs"], errors="coerce")
            fit = fit_best_rating_curve(stage, q, min_points=10)
            if fit.get("model") != "none" and np.isfinite(fit.get("R2", np.nan)):
                rating = fit
        except Exception:
            rating = None

    climate: dict = {"has_precip": False, "has_temp": False}
    if df_merged is not None and not df_merged.empty:
        if "Precip_mm" in df_merged.columns:
            p = pd.to_numeric(df_merged["Precip_mm"], errors="coerce").dropna()
            if not p.empty:
                climate["has_precip"] = True
                if len(p) > 60:
                    climate["mean_annual_mm"] = float(p.resample("YE").sum().mean())
                else:
                    climate["mean_annual_mm"] = float(p.mean() * 365.25)
                climate["wet_day_pct"] = float((p > 1.0).mean() * 100)
        if "Temp_C" in df_merged.columns:
            t = pd.to_numeric(df_merged["Temp_C"], errors="coerce").dropna()
            if not t.empty:
                climate["has_temp"] = True
                climate["mean_temp_c"] = float(t.mean())

    latest = float(s.iloc[-1])
    latest_date = s.index.max()
    doy = latest_date.dayofyear
    seasonal = s[(s.index.dayofyear >= doy - 15) & (s.index.dayofyear <= doy + 15)]
    baseline = seasonal if len(seasonal) >= 30 else s
    latest_pct = float((baseline < latest).mean() * 100) if len(baseline) else None

    profile.update(
        {
            "ok": True,
            "n": n,
            "years": years,
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "mean": mean,
            "peak": peak,
            "cv": cv,
            "q10_q90": ratio,
            "mean_median": skew,
            "regime": regime,
            "regime_plain": regime_plain,
            "season_means": season_means,
            "wet_season": wet_season,
            "dry_season": dry_season,
            "season_factor": season_factor,
            "zero_frac": zero_frac,
            "has_stage": has_stage,
            "rating": rating,
            "climate": climate,
            "latest": latest,
            "latest_date": latest_date,
            "latest_pct": latest_pct,
            "start": s.index.min(),
            "end": s.index.max(),
        }
    )
    return profile


def dynamic_metric_relevance(
    df_q: pd.DataFrame | None,
    df_merged: pd.DataFrame | None = None,
    plot_keys: Iterable[str] | None = None,
    discharge_col: str | None = None,
    metric_keys: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    """Metric relevance rows computed from this gage/period's hydrology."""
    profile = compute_hydrologic_profile(df_q, df_merged, discharge_col)
    if metric_keys is not None:
        keys = [k for k in metric_keys if k in METRIC_RELEVANCE]
    else:
        keys = metric_keys_for_plots(plot_keys or [])

    if not profile.get("ok"):
        return metric_relevance_table(keys)

    years, n = profile["years"], profile["n"]
    q10, q50, q90 = profile["q10"], profile["q50"], profile["q90"]
    mean, peak = profile["mean"], profile["peak"]
    ratio, skew = profile["q10_q90"], profile["mean_median"]
    regime = profile["regime_plain"]
    wet, dry = profile.get("wet_season"), profile.get("dry_season")
    sfactor = profile.get("season_factor")

    here = {
        "record_length": (
            f"This selection covers **{years:.1f} years**. "
            + (
                "Long enough for rough frequency/SPI screening."
                if years >= 10
                else "Short for design frequency — treat extremes as exploratory only."
            )
        ),
        "data_points": (
            f"**{n:,}** daily values in window. "
            + (
                "Dense enough for stable percentiles."
                if n >= 365 * 3
                else "Thin sample — duration quantiles can jump if you change dates."
            )
        ),
        "mean_flow": (
            f"Mean **{mean:,.0f} cfs** vs median **{q50:,.0f} cfs** "
            f"(mean is **{skew:.2f}×** the median). "
            + (
                "Floods dominate the average — lean on Q50 for “typical” conditions."
                if skew >= 1.35
                else "Mean and median are close — the series is not heavily peak-skewed."
            )
        ),
        "peak_flow": (
            f"Highest daily mean in this window is **{peak:,.0f} cfs** "
            f"(**{peak / max(q50, 1e-6):.1f}×** the median). "
            "Window peak only — not a formal annual-max design flood."
        ),
        "q10": (
            f"Q10 ≈ **{q10:,.0f} cfs** (exceeded ~10% of days). "
            + (
                f"Q10/Q90 ≈ **{ratio:.0f}** → high flows sit far above baseflow ({regime})."
                if ratio >= 15
                else f"Q10/Q90 ≈ {ratio:.1f} → high-flow contrast is mild for this period."
            )
        ),
        "q50": (
            f"Median **{q50:,.0f} cfs** is the best single “normal day” summary here. "
            + (
                f"Wet season **{wet}** vs dry **{dry}** (~{sfactor:.1f}× contrast)."
                if wet and dry and sfactor
                else "Seasonal contrast is weak or not resolved in this window."
            )
        ),
        "q90": (
            f"Q90 ≈ **{q90:,.0f} cfs** (about 90% of days are at least this wet). "
            + (
                f"Near-dry days are common (~{profile['zero_frac']*100:.0f}% near-zero)."
                if profile["zero_frac"] >= 0.05
                else "Near-zero flow days are uncommon in this window."
            )
            + f" Low-flow relevance is high for a **{regime}** river."
        ),
        "rating_r2": (
            (
                f"Best rating fit R²=**{profile['rating']['R2']:.3f}** "
                f"(`{profile['rating']['equation']}`). "
                + (
                    "Offset H₀ correction active — gage zero ≠ zero-flow control."
                    if profile["rating"]["model"] == "offset_powerlaw"
                    else "Simple power-law is adequate for this pair set."
                )
            )
            if profile.get("rating")
            else (
                "Stage present but rating fit unavailable."
                if profile.get("has_stage")
                else "No gage-height in this window — rating metrics do not apply."
            )
        ),
        "spi_sri": (
            (
                "Climate is merged for this period"
                + (
                    f" (~{profile['climate'].get('mean_annual_mm', 0):.0f} mm/yr precip)"
                    if profile["climate"].get("has_precip")
                    else ""
                )
                + f" — SPI/SRI should be read against this **{regime}** flow regime."
            )
            if profile.get("climate", {}).get("has_precip")
            or profile.get("climate", {}).get("has_temp")
            else (
                "Climate is not merged into discharge for this period; "
                "SPI may still run via Open-Meteo, but co-plotted climate metrics are limited."
            )
        ),
    }

    rows: list[dict[str, str]] = []
    for key in keys:
        meta = METRIC_RELEVANCE.get(key)
        if not meta:
            continue
        observed = {
            "record_length": f"{years:.1f} yrs",
            "data_points": f"{n:,}",
            "mean_flow": f"{mean:,.0f} cfs",
            "peak_flow": f"{peak:,.0f} cfs",
            "q10": f"{q10:,.0f} cfs",
            "q50": f"{q50:,.0f} cfs",
            "q90": f"{q90:,.0f} cfs",
            "rating_r2": (
                f"R²={profile['rating']['R2']:.3f}" if profile.get("rating") else "n/a"
            ),
            "spi_sri": (
                "climate linked"
                if profile.get("climate", {}).get("has_precip")
                else "climate limited"
            ),
        }.get(key, "—")

        rows.append(
            {
                "Metric": meta["label"],
                "This site/period": observed,
                "What it is": meta["meaning"],
                "Why it matters here": here.get(key, meta["relevance"]),
                "Use it when…": meta["use_when"],
                "Regime": regime,
            }
        )
    return rows


def metric_relevance_table(
    keys: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    """Static catalog rows (fallback when no discharge series)."""
    use_keys = list(keys) if keys is not None else list(METRIC_RELEVANCE.keys())
    rows = []
    for key in use_keys:
        meta = METRIC_RELEVANCE.get(key)
        if not meta:
            continue
        rows.append(
            {
                "Metric": meta["label"],
                "This site/period": "—",
                "What it is": meta["meaning"],
                "Why it matters here": meta["relevance"],
                "Use it when…": meta["use_when"],
                "Regime": "—",
            }
        )
    return rows


def format_metric_relevance_markdown(
    keys: Iterable[str] | None = None,
    df_q: pd.DataFrame | None = None,
    df_merged: pd.DataFrame | None = None,
    plot_keys: Iterable[str] | None = None,
) -> str:
    """Markdown metric relevance — dynamic when discharge data is provided."""
    use_keys = list(keys) if keys is not None else None
    if df_q is not None:
        # Prefer explicit metric keys; else derive from plot set
        if use_keys is not None and all(k in METRIC_RELEVANCE for k in use_keys):
            rows = dynamic_metric_relevance(
                df_q, df_merged, metric_keys=use_keys
            )
        else:
            rows = dynamic_metric_relevance(
                df_q, df_merged, plot_keys=plot_keys or use_keys
            )
        profile = compute_hydrologic_profile(df_q, df_merged)
    else:
        rows = metric_relevance_table(use_keys)
        profile = {}

    if not rows:
        return ""

    if profile.get("ok"):
        header = (
            f"#### Metric relevance — **{profile['regime_plain']}** regime "
            f"(Q10/Q90≈{profile['q10_q90']:.1f})"
        )
        blurb = (
            "Each line uses **this site’s numbers** for the selected period "
            "(not generic textbook text)."
        )
    else:
        header = "#### Metric relevance"
        blurb = "What each number means and when to trust it:"

    lines = [header, blurb, ""]
    for row in rows:
        site_bit = row.get("This site/period") or "—"
        lines.append(
            f"- **{row['Metric']}** (**{site_bit}**) — {row['What it is']} "
            f"*{row['Why it matters here']}*"
        )
    return "\n".join(lines)


def metric_help_text(
    key: str,
    df_q: pd.DataFrame | None = None,
    df_merged: pd.DataFrame | None = None,
    discharge_col: str | None = None,
) -> str:
    """Short tooltip for st.metric — site-specific when data is available."""
    meta = METRIC_RELEVANCE.get(key, {})
    base = " ".join(
        p for p in (meta.get("meaning", ""), meta.get("use_when", "")) if p
    )
    if df_q is None:
        rel = meta.get("relevance", "")
        return " ".join(p for p in (base, rel) if p)

    rows = dynamic_metric_relevance(
        df_q, df_merged, discharge_col=discharge_col, metric_keys=[key]
    )
    if not rows:
        rel = meta.get("relevance", "")
        return " ".join(p for p in (base, rel) if p)

    row = rows[0]
    site = row.get("This site/period", "")
    why = row.get("Why it matters here", "")
    # Strip markdown for Streamlit tooltips
    why_plain = why.replace("**", "").replace("`", "")
    site_plain = site.replace("**", "").replace("`", "")
    bits = [base]
    if site_plain and site_plain != "—":
        bits.append(f"This site/period: {site_plain}.")
    if why_plain:
        bits.append(why_plain)
    return " ".join(bits)
