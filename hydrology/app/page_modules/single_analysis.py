"""
Single Analysis page — one gage and one plot workspace.

Hydrograph, duration, raster, percentiles, frequency, rating, indicators,
and static export figures are views in that workspace. Heavy views load
only when they are selected.
"""

import math

import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

from hydrology.app.shared import (
    get_inventory, get_cached_site_info, get_weather_station_info,
    extract_site_id, process_site_data,
    date_range_selector, plot_selector, render_export_buttons,
    render_data_download, site_picker, logger, AVAILABLE_PLOTS)
from hydrology.app.styles import (
    render_site_header, render_availability_badges, render_metric_cards,
    render_plot_capability_board, render_insight_board
)
from hydrology.app.interpretation import (
    summarize_flow_context,
    summarize_recommendations,
    build_plot_analysis_report,
    summarize_seasonal_pattern,
)
from hydrology.app.plot_config import SINGLE_SITE_PLOTS, resolve_generated_plots
from hydrology.app.page_modules.indicators import render_site_indicators
from hydrology.visualization import create_multi_plot, PlotLayout
from hydrology.visualization.interactive import (
    interactive_hydrograph, interactive_fdc, interactive_rating_curve,
    raster_hydrograph, percentile_bands_hydrograph)
from hydrology.core import DEFAULT_DISCHARGE_CODE
from hydrology.analysis.stage_discharge import (
    fit_best_rating_curve,
    evaluate_rating_params,
    rating_stage_window,
)


def _show_rating_chart(fig) -> None:
    """Draw the workshop chart in an iframe.

    Streamlit 1.37 mounts ``st.plotly_chart`` once and then only recolors that
    first figure. An iframe reads the new HTML on every rerun.
    """
    height = int(getattr(fig.layout, "height", None) or 560)
    snippet = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={"responsive": True, "displaylogo": False},
        default_width="100%",
        default_height=f"{height}px",
    )
    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
        "<body style='margin:0;background:#0b1524;'>"
        f"{snippet}</body></html>"
    )
    components.html(page, height=height + 12, scrolling=False)


def _snap_slider(value: float, lo: float, hi: float, step: float) -> float:
    """Place a value on the slider grid.

    Streamlit 1.37 builds the slider identity from ``value`` and drops a drag
    when ``(value - min) / step`` is not a whole number. The thumb moves and
    then snaps back, so the curve and the equation never see the new number.
    """
    lo = float(lo)
    hi = float(hi)
    step = float(step)
    if not math.isfinite(lo) or not math.isfinite(hi) or not math.isfinite(step):
        return float(value)
    if hi < lo:
        lo, hi = hi, lo
    if step <= 0 or hi == lo:
        return lo
    span_steps = int(math.floor((hi - lo) / step + 1e-8))
    span_steps = max(span_steps, 1)
    n = int(round((float(value) - lo) / step))
    n = max(0, min(span_steps, n))
    return lo + n * step


@st.cache_data(ttl=3600, show_spinner=False)
def _load_site_data(site_id, lat, lon, start_str, end_str):
    """Cached wrapper around process_site_data."""
    return process_site_data(site_id, lat, lon, start_str, end_str)


def _render_rating_curve_workshop(
    df_q: pd.DataFrame,
    *,
    site_id: str,
    site_desc: str,
) -> None:
    """Sleek interactive stage–discharge fit: adjust A, B, H0 and see the curve update."""
    if (
        df_q is None
        or df_q.empty
        or "Gage_Height_ft" not in df_q.columns
        or "Discharge_cfs" not in df_q.columns
    ):
        st.info("This period does not have stage and discharge for a rating curve.")
        return

    pairs = df_q[["Gage_Height_ft", "Discharge_cfs"]].dropna()
    pairs = pairs[(pairs["Gage_Height_ft"] > 0) & (pairs["Discharge_cfs"] > 0)]
    if len(pairs) < 10:
        st.info("Need at least 10 positive stage–discharge pairs in this period.")
        return

    st.subheader("Rating curve workshop")
    st.caption(
        "Red is your curve. Teal is the auto-fit. The dotted line is H₀, the stage "
        "where this equation says discharge is zero. Dragging H₀ moves that line "
        "inside a fixed stage axis. Daily discharge is often already computed from "
        "stage, so this is a consistency check, not an independent measurement rating."
    )

    stage = pairs["Gage_Height_ft"]
    discharge = pairs["Discharge_cfs"]
    auto = fit_best_rating_curve(stage, discharge, min_points=10)
    if auto.get("model") == "none" or not pd.notna(auto.get("A")):
        st.info("Could not auto-fit a rating curve for this period.")
        return

    try:
        period = (
            f"{pd.Timestamp(pairs.index.min()).date()}_"
            f"{pd.Timestamp(pairs.index.max()).date()}_{len(pairs)}"
        )
    except Exception:
        period = str(len(pairs))
    state_key = f"rating_ws_{site_id}_{period}"
    window = rating_stage_window(stage)
    auto_A = float(auto["A"])
    auto_B = float(auto["B"])
    auto_H0 = float(auto.get("H0") or 0.0)
    auto_eq = auto.get("equation", "")
    use_offset_auto = abs(auto_H0) > 1e-6 or auto.get("model") == "offset_powerlaw"

    # Bounds are fixed for the period. Values are seeded once, then the widgets
    # own them. Passing value= on every rerun changes the Streamlit 1.37 widget
    # id (it hashes value) and throws away the drag.
    log_step = 0.01
    log_lo = math.log10(max(auto_A, 1e-9)) - 2.0
    log_hi = log_lo + 4.0
    b_lo, b_hi, b_step = 0.15, 4.0, 0.01
    h0_lo = float(window["h0_lo"])
    h0_hi = float(window["h0_hi"])
    h0_step = 0.01
    h0_steps = max(1, int(math.floor((h0_hi - h0_lo) / h0_step + 1e-8)))
    h0_hi = h0_lo + h0_steps * h0_step

    log_key = f"{state_key}_logA"
    b_key = f"{state_key}_B"
    h0_key = f"{state_key}_H0"
    offset_key = f"{state_key}_offset"

    if state_key not in st.session_state:
        st.session_state[state_key] = {}
    ws = st.session_state[state_key]
    # Auto-fit sets this flag and reruns. Apply it before the widgets exist.
    reset = bool(ws.pop("pending_reset", False))
    if reset or log_key not in st.session_state:
        st.session_state[log_key] = _snap_slider(
            math.log10(max(auto_A, 1e-9)), log_lo, log_hi, log_step
        )
    if reset or b_key not in st.session_state:
        st.session_state[b_key] = _snap_slider(auto_B, b_lo, b_hi, b_step)
    if reset or h0_key not in st.session_state:
        st.session_state[h0_key] = _snap_slider(auto_H0, h0_lo, h0_hi, h0_step)
    if reset or offset_key not in st.session_state:
        st.session_state[offset_key] = bool(use_offset_auto)

    c_left, c_right = st.columns([1, 1.55], gap="large")

    with c_left:
        st.markdown("##### Parameters")
        use_offset = st.toggle(
            "Offset model  Q = A·(H − H₀)ᴮ",
            key=offset_key,
            help="H₀ is the effective zero-flow stage when gage zero ≠ control elevation.",
        )

        log_A = st.slider(
            "A  (scale, log₁₀)",
            min_value=float(log_lo),
            max_value=float(log_hi),
            step=log_step,
            key=log_key,
            help="Coefficient in front of the power law (log scale for wide range).",
        )
        A = 10.0 ** float(log_A)
        st.caption(f"A = **{A:.4g}**")

        B = float(st.slider(
            "B  (exponent)",
            min_value=b_lo,
            max_value=b_hi,
            step=b_step,
            key=b_key,
            help="Control shape. ~1.5–2.5 is common for open channels.",
        ))

        H0 = 0.0
        if use_offset:
            H0 = float(st.slider(
                "H₀  (zero-flow stage, ft)",
                min_value=float(h0_lo),
                max_value=float(h0_hi),
                step=h0_step,
                key=h0_key,
                help=(
                    "Moves the dotted zero-flow line. The stage axis stays put, so the "
                    "line and the red curve actually travel. Raising H₀ lowers discharge, "
                    "most of all at low stage."
                ),
            ))
            head = float(window["h_ref"]) - H0
            st.caption(
                f"H₀ = **{H0:.2f} ft**. Median stage is **{window['h_ref']:.2f} ft**, "
                f"so the head there is **{head:.2f} ft**."
            )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("↩ Auto-fit", key=f"{state_key}_reset", use_container_width=True):
                ws["pending_reset"] = True
                st.rerun()
        with b2:
            st.caption(f"Auto: `{auto_eq}`")

        scored = evaluate_rating_params(stage, discharge, A, B, H0)
        auto_scored = evaluate_rating_params(stage, discharge, auto_A, auto_B, auto_H0)
        mdape = scored.get("mdape")
        r2_log = scored.get("r2_log")
        q_ref = scored.get("q_ref")
        n_excluded = int(scored.get("n_excluded") or 0)
        mdape_txt = f"{mdape:.1f}%" if pd.notna(mdape) else "—"
        r2_txt = f"{r2_log:.3f}" if pd.notna(r2_log) else "—"
        q_txt = f"{q_ref:,.0f}" if pd.notna(q_ref) else "—"
        st.markdown(
            "<div class='rating-scoreboard'>"
            f"<span><b>% error</b>{mdape_txt}</span>"
            f"<span><b>Log R²</b>{r2_txt}</span>"
            f"<span><b>Q at mid stage</b>{q_txt}</span>"
            f"<span><b>Left out</b>{n_excluded:,}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Percent error and log R² score your curve. "
            "Q at mid stage is the equation at the median gage height. "
            "Left out is days at or below H₀."
        )

        auto_mdape = auto_scored.get("mdape")
        auto_excluded = int(auto_scored.get("n_excluded") or 0)
        if pd.notna(mdape) and pd.notna(auto_mdape):
            delta = auto_mdape - mdape
            dropped = n_excluded > auto_excluded + 5
            if dropped and delta >= 0:
                st.warning(
                    f"Percent error looks better (Δ = {delta:+.1f} points) because "
                    f"{n_excluded - auto_excluded:,} more low-stage pairs were left out."
                )
            elif delta >= 0.5:
                st.success(f"Your fit beats auto-fit by **{delta:.1f}** points of % error.")
            elif delta <= -1.0:
                st.warning(f"Auto-fit is tighter (Δ % error = {-delta:.1f}). Try ↩ Auto-fit.")
            else:
                st.info(f"Close to auto-fit (Δ % error = {delta:+.1f} points).")

        st.markdown(f"**Equation:** `{scored.get('equation', '—')}`")
        st.caption(
            f"{scored.get('n_points', 0):,} pairs · "
            "Daily mean Q (00060) vs stage (00065) — empirical consistency check, "
            "not a survey-grade rating table."
        )

    with c_right:
        color_by = st.radio(
            "Color points",
            options=["season", "year"],
            format_func=lambda x: "Season (DJF/MAM/JJA/SON)" if x == "season" else "Year",
            horizontal=True,
            key=f"{state_key}_color",
        )
        show_resid = st.checkbox("Show residuals", value=True, key=f"{state_key}_resid")
        log_q = st.checkbox(
            "Log discharge axis",
            value=True,
            key=f"{state_key}_logq",
            help="Percent error is easiest to see on a log axis. The range stays on the observations.",
        )

        fig = interactive_rating_curve(
            df_q,
            A=A,
            B=B,
            H0=H0,
            title=f"{site_desc} — rating workshop",
            show_auto_fit=True,
            auto_fit=auto,
            show_residuals=show_resid,
            log_q=bool(log_q),
            color_by=color_by,
        )
        _show_rating_chart(fig)


def _render_analysis_readiness(data, has_stage: bool, climate_info):
    """Show visible plot readiness before users open deeper controls."""
    df_q = data.get('df_q') if data else None
    df_merged = data.get('df_merged') if data else None
    discharge_days = len(df_q) if df_q is not None else 0
    has_climate = df_merged is not None and not df_merged.empty and (
        'Precip_mm' in df_merged.columns or 'Temp_C' in df_merged.columns
    )

    climate_status = "Ready" if has_climate else "Limited"
    if climate_info and climate_info.get('distance_km') is not None and climate_info['distance_km'] > 50:
        climate_status = "Distant station"

    cards = [
        {
            "title": "Core Flow",
            "body": f"{discharge_days:,} daily observations for hydrographs, duration curves, anomalies, and seasonal context.",
            "status": "Ready" if discharge_days else "Unavailable",
            "state": "ready" if discharge_days else "blocked",
            "help": (
                "Unlocks interactive hydrograph, flow-duration curve, Q10/Q50/Q90 stats, "
                "and most static plots. If blocked, widen the date range or pick another gage."
            ),
        },
        {
            "title": "Climate Links",
            "body": "Temperature and precipitation overlays, lag response, SPI, and correlation views.",
            "status": climate_status,
            "state": "ready" if has_climate else "limited",
            "help": (
                "Merged precip/temp near this gage. Ready = co-plotted climate and SPI context "
                "are usable. Distant station = signals may lag or smear local storms."
            ),
        },
        {
            "title": "Stage + Rating",
            "body": "Dual-axis stage overlay and stage-discharge rating curve when gage height exists.",
            "status": "Ready" if has_stage else "Needs gage height",
            "state": "ready" if has_stage else "blocked",
            "help": (
                "Needs daily gage height (00065). Open Hydrograph for the stage axis, "
                "or Rating to tune A, B, and H₀."
            ),
        },
        {
            "title": "Frequency",
            "body": "Flood frequency runs on demand from annual peak records so it does not slow page load.",
            "status": "On demand",
            "state": "limited",
            "help": (
                "Open Frequency in Plots. Peak records load for that view only. "
                "Best with longer records; short periods are exploratory only."
            ),
        },
    ]

    st.caption("What this site and period can support.")
    render_plot_capability_board(cards)


def _render_hydrologic_summary(data, has_stage: bool, climate_info):
    """Keep flow context and readiness one click away from the plot control."""
    df_q = data.get('df_q') if data else None
    df_merged = data.get('df_merged') if data else None
    has_climate = df_merged is not None and not df_merged.empty and (
        'Precip_mm' in df_merged.columns or 'Temp_C' in df_merged.columns
    )
    record_years = 0
    if df_q is not None and not df_q.empty:
        record_years = (df_q.index.max() - df_q.index.min()).days / 365.25

    with st.expander("Recommended next views", expanded=False):
        render_insight_board(summarize_flow_context(df_q))
        render_insight_board(summarize_recommendations(has_stage, has_climate, record_years))
        _render_analysis_readiness(data, has_stage, climate_info)
        if climate_info:
            station = climate_info.get('name', 'nearest weather station')
            distance = climate_info.get('distance_km')
            if distance is not None:
                st.caption(f"Climate-linked recommendations use {station}, about {distance:.1f} km from this gage.")
            else:
                st.caption(f"Climate-linked recommendations use {station}.")


def _format_frequency_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Format flood-frequency diagnostics for dashboard display."""
    display = diagnostics[["observed_flow_cfs", "fitted_flow_cfs", "return_period"]].copy()
    display = display.rename(columns={
        "observed_flow_cfs": "Observed flow",
        "fitted_flow_cfs": "Fitted flow",
        "return_period": "Return period",
    })
    for col in ["Observed flow", "Fitted flow"]:
        display[col] = display[col].apply(lambda value: f"{value:,.0f}" if pd.notna(value) else "N/A")
    display["Return period"] = display["Return period"].apply(lambda value: f"{value:.1f} yr")
    return display


@st.cache_data(ttl=3600, show_spinner=False)
def _load_peak_streamflow(site_id: str):
    """Cache annual peaks so Frequency does not refetch on every rerun."""
    from hydrology.data.usgs import fetch_peak_streamflow
    return fetch_peak_streamflow(site_id)


def _plot_views(has_stage: bool) -> list[str]:
    """Plot workspace views, in control order."""
    views = ["Hydrograph", "Duration", "Raster", "Percentiles", "Frequency"]
    if has_stage:
        views.append("Rating")
    views.extend(["Indicators", "Export"])
    return views


def _render_hydrograph_note(data) -> None:
    note = summarize_seasonal_pattern(data.get("df_q"))
    if note:
        st.caption(note)


def _render_hydrograph_view(data, desc: str, has_stage: bool, site_id: str) -> None:
    """Interactive hydrograph with aggregation, NWM, and optional stage."""
    col_agg, col_nwm, col_stage = st.columns(3)
    with col_agg:
        agg = st.radio(
            "Aggregation",
            ["daily", "weekly", "monthly"],
            horizontal=True,
            key="hydrograph_agg",
            help="Changes the hydrograph time scale without refetching the selected site data.",
        )
    with col_nwm:
        show_nwm = st.checkbox(
            "NWM recent analysis overlay",
            value=False,
            key="nwm_overlay",
            help="Adds recent NOAA National Water Model analysis/assimilation streamflow when a matching reach is available.",
        )
    with col_stage:
        show_stage = st.checkbox(
            "Show stage (dual axis)",
            value=False,
            disabled=not has_stage,
            key="stage_overlay",
            help="Requires gage-height observations at this site for the selected period.",
        )

    fig_hydro = interactive_hydrograph(
        data['df_q'], discharge_col='Discharge_cfs',
        title=f"{desc} - Hydrograph", aggregation=agg,
        show_percentile_bands=True)

    if show_nwm:
        try:
            from hydrology.data.nwm import compare_nwm_usgs
            from datetime import timedelta
            import plotly.graph_objects as go

            end_dt = data['df_q'].index.max()
            start_dt = end_dt - timedelta(days=30)
            nwm_result = compare_nwm_usgs(
                site_id,
                start_dt.strftime('%Y-%m-%d'),
                end_dt.strftime('%Y-%m-%d')
            )
            if nwm_result and hasattr(nwm_result, 'nwm_data') and nwm_result.nwm_data is not None:
                fig_hydro.add_trace(go.Scatter(
                    x=nwm_result.nwm_data.index,
                    y=nwm_result.nwm_data['streamflow_cfs'],
                    mode='lines', name='NWM Forecast',
                    line=dict(color='#ff7f0e', width=2, dash='dot')))
                st.caption(f"NWM: NSE={nwm_result.nash_sutcliffe:.2f}, RMSE={nwm_result.rmse:.0f} cfs")
            else:
                st.caption("NWM recent analysis was not available for this site/period")
        except Exception as e:
            st.caption(f"NWM overlay unavailable: {str(e)[:60]}")

    if show_stage and has_stage:
        import plotly.graph_objects as go
        stage = data['df_q']['Gage_Height_ft'].dropna()
        if not stage.empty:
            fig_hydro.add_trace(go.Scatter(
                x=stage.index, y=stage.values,
                mode='lines', name='Gage Height (ft)',
                line=dict(color='#2ca02c', width=1.5),
                yaxis='y2',
                hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2f} ft<extra>Stage</extra>'
            ))
            fig_hydro.update_layout(
                yaxis2=dict(
                    title="Gage Height (ft)",
                    overlaying='y', side='right',
                    showgrid=False)
            )

    st.plotly_chart(fig_hydro, width="stretch", key="plotly_hydro")


def _render_duration_view(data, desc: str) -> None:
    koehler = st.checkbox(
        "Koehler (2025) dQ/dt coloring",
        value=True,
        key="fdc_koehler",
        help="Colors the flow-duration curve by rate-of-change to separate stable flow states from flashier transitions.",
    )
    fig_fdc = interactive_fdc(
        data['df_q'], discharge_col='Discharge_cfs',
        title=f"{desc} - Flow Duration Curve",
        color_by_dqdt=koehler)
    st.plotly_chart(fig_fdc, width="stretch", key="plotly_fdc")


def _render_frequency_view(site_id: str, desc: str) -> None:
    """Flood frequency for the selected gage. Runs only while this view is open."""
    from hydrology.analysis.frequency import (
        fit_flood_frequency, estimate_return_periods,
        flood_frequency_diagnostics, get_plotting_positions)
    from hydrology.visualization.interactive import interactive_return_period

    st.caption("Multi-distribution flood frequency from annual peak records.")
    with st.spinner("Fetching peak streamflow data..."):
        peaks_df = _load_peak_streamflow(site_id)

    if peaks_df is None or peaks_df.empty:
        st.warning("No peak streamflow data available for this site")
        return

    peak_col = 'peak_va' if 'peak_va' in peaks_df.columns else peaks_df.columns[0]
    peak_values = peaks_df[peak_col].dropna().values.astype(float)
    peak_values = peak_values[peak_values > 0]

    if len(peak_values) < 10:
        st.warning(f"Insufficient peaks for analysis ({len(peak_values)} < 10)")
        return

    st.metric("Peak Records", len(peak_values))
    fits = fit_flood_frequency(peak_values)
    observed = get_plotting_positions(peak_values)
    rp_table = estimate_return_periods(peak_values)

    if not fits:
        st.error("Could not fit any distributions")
        return

    fig_rp = interactive_return_period(
        observed, fits, rp_table,
        title=f"{desc} - Flood Frequency Analysis")
    st.plotly_chart(fig_rp, width="stretch", key="plotly_rp")

    if not rp_table.empty:
        st.subheader("Return Period Estimates")
        display_rp = rp_table.copy()
        for col in ['flow_cfs', 'lower_ci', 'upper_ci']:
            if col in display_rp.columns:
                display_rp[col] = display_rp[col].apply(
                    lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
                )
        st.dataframe(display_rp, width="stretch", hide_index=True)

    diagnostics = flood_frequency_diagnostics(peak_values, distribution="lp3")
    if not diagnostics.empty:
        st.subheader("LP3 Diagnostic Table")
        st.dataframe(
            _format_frequency_diagnostics(diagnostics).head(25),
            width="stretch",
            hide_index=True,
        )
        st.caption("Observed plotting positions compared with fitted LP3 quantiles.")

    st.subheader("Distribution Comparison")
    model_rows = []
    for _name, fit in fits.items():
        model_rows.append({
            'Distribution': fit.display_name,
            'AIC': f"{fit.aic:.1f}",
            'BIC': f"{fit.bic:.1f}",
            'KS Statistic': f"{fit.ks_statistic:.4f}",
            'KS p-value': f"{fit.ks_pvalue:.4f}",
        })
    st.dataframe(pd.DataFrame(model_rows), width="stretch", hide_index=True)
    st.caption("Lower AIC/BIC = better fit. Best distribution listed first.")


def _render_export_view(data, site_id: str, desc: str) -> None:
    """On-demand static figures. The catalog is the full single-site plot set."""
    st.caption("Static figures for export. Presets fill the catalog; generation stays on demand.")
    single_site_available = {
        plot: info
        for plot, info in AVAILABLE_PLOTS.items()
        if plot in SINGLE_SITE_PLOTS
    }
    selected_plots = plot_selector(single_site_available)
    static_plots = resolve_generated_plots(selected_plots)

    layout_options = {
        'Auto': PlotLayout.AUTO,
        'Vertical': PlotLayout.VERTICAL,
        'Quad (2x2)': PlotLayout.QUAD,
        'Grid 2x3': PlotLayout.GRID_2x3,
        'Grid 3x2': PlotLayout.GRID_3x2,
        'Grid 2x5': PlotLayout.GRID_2x5,
    }

    col_layout, col_dpi, col_btn = st.columns([2, 1, 2])
    with col_layout:
        layout = st.selectbox("Layout", list(layout_options.keys()), key="single_export_layout")
    with col_dpi:
        dpi = st.number_input("DPI", min_value=72, max_value=300, value=150, key="single_export_dpi")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        generate = st.button("Generate Plots", type="primary", width="stretch", key="gen_static_plots")

    if not generate:
        return
    if not static_plots:
        st.warning("Select at least one plot type")
        return

    plot_data = {
        'df_q': data['df_q'],
        'df_merged': data['df_merged'],
        'analysis_results': data['analysis_results']
    }
    with st.spinner("Generating plots..."):
        fig = create_multi_plot(
            plots=static_plots,
            layout=layout_options[layout],
            data=plot_data,
            site_id=site_id,
            title=desc,
            dpi=dpi
        )
    if not fig:
        return

    st.pyplot(fig)
    render_export_buttons(fig, site_id, dpi)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Automated plot analysis")
    st.caption("Plain-language screening narrative from the selected period and plot set.")
    report = build_plot_analysis_report(
        site_id=site_id,
        site_desc=desc,
        plot_keys=static_plots,
        df_q=data.get("df_q"),
        df_merged=data.get("df_merged"),
    )
    st.markdown(report)
    st.download_button(
        "Download analysis as Markdown",
        data=report,
        file_name=f"{site_id}_plot_analysis.md",
        mime="text/markdown",
        key=f"dl_plot_analysis_{site_id}",
    )


def _render_plot_workspace(data, site_id: str, desc: str, has_stage: bool, site_info: dict) -> None:
    """One control for every site-analysis chart, indicator, and export figure."""
    st.markdown("---")
    st.subheader("Plots")
    st.caption("One view at a time. Live charts, indicators, and export figures are all in this list.")
    views = _plot_views(has_stage)
    if st.session_state.get("site_plot_view") not in views:
        st.session_state["site_plot_view"] = views[0]
    view = st.radio(
        "Plot view",
        views,
        horizontal=True,
        key="site_plot_view",
    )

    if view == "Hydrograph":
        _render_hydrograph_view(data, desc, has_stage, site_id)
        _render_hydrograph_note(data)
    elif view == "Duration":
        render_metric_cards(data["df_q"], data.get("df_merged"), section="duration")
        _render_duration_view(data, desc)
    elif view == "Raster":
        fig_raster = raster_hydrograph(
            data['df_q'], discharge_col='Discharge_cfs',
            title=f"{desc} - Raster Hydrograph")
        st.plotly_chart(fig_raster, width="stretch", key="plotly_raster")
    elif view == "Percentiles":
        fig_pctile = percentile_bands_hydrograph(
            data['df_q'], discharge_col='Discharge_cfs',
            title=f"{desc} - Percentile Bands")
        st.plotly_chart(fig_pctile, width="stretch", key="plotly_pctile")
    elif view == "Frequency":
        _render_frequency_view(site_id, desc)
    elif view == "Rating":
        _render_rating_curve_workshop(data["df_q"], site_id=site_id, site_desc=desc)
    elif view == "Indicators":
        st.caption(
            "SRI, precipitation SPI, baseflow proxy, and seasonal anomaly. "
            "This view uses its own lookback, not the date range above."
        )
        render_site_indicators(site_id, site_info)
    elif view == "Export":
        _render_export_view(data, site_id, desc)


def show():
    """Render the Single Analysis page."""
    inventory_df = get_inventory()
    if inventory_df.empty:
        st.error("Could not load site inventory")
        return

    site_id = site_picker(inventory_df, key="single", label="Select Site", location="main")

    site_info = get_cached_site_info(site_id)
    if not site_info:
        st.error(f"Site {site_id} not found")
        return

    lat = site_info.get('latitude')
    lon = site_info.get('longitude')
    desc = site_info.get('description', site_id)

    start_date, end_date = date_range_selector("single")

    if not lat or not lon:
        st.error("Site missing coordinates")
        return

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Auto-load data whenever station or dates change
    with st.spinner("Loading discharge data..."):
        data = _load_site_data(site_id, float(lat), float(lon), start_str, end_str)

    if data is None:
        st.error("No discharge data available for this site/period")
        return

    # Site header and metrics
    render_site_header(site_id, desc, float(lat), float(lon))

    has_stage = 'Gage_Height_ft' in data['df_q'].columns if data['df_q'] is not None else False
    climate_info = get_weather_station_info(float(lat), float(lon)) if lat and lon else None
    render_availability_badges(True, has_stage, climate_info)
    # Record coverage only at top — duration quantiles live on the Duration view
    render_metric_cards(data['df_q'], data['df_merged'], section="record")
    _render_hydrologic_summary(data, has_stage, climate_info)
    _render_plot_workspace(data, site_id, desc, has_stage, site_info)
    render_data_download(data['df_q'], filename_prefix=site_id)
