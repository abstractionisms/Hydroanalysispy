"""Site Analysis for an Ecology gage, using the local DataStream cache."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from hydrology.data.ecology import load_ecology_stations
from hydrology.data.ecology_datastream import (
    DISCHARGE_SUFFIX,
    QUEUE_LIMIT,
    STAGE_SUFFIX,
    EcologyDataError,
    download_queue,
    load_cached_series,
    pending_downloads,
    read_index,
    refresh_published_years,
    water_years_overlapping,
    year_record_status,
)
from hydrology.visualization.interactive import (
    interactive_fdc,
    interactive_hydrograph,
    percentile_bands_hydrograph,
    raster_hydrograph,
)


def show_ecology_station() -> None:
    """Pick a catalog station and plot water years that are already on disk."""
    catalog = load_ecology_stations()
    if catalog.empty:
        st.error("The local Ecology station catalog is missing.")
        return

    station = _select_station(catalog)
    if station is None:
        return

    code = str(station["station_code"])
    name = str(station["station_name"])
    lat = pd.to_numeric(station["latitude"], errors="coerce")
    lon = pd.to_numeric(station["longitude"], errors="coerce")
    st.markdown(f"### {name}")
    st.caption(
        f"Ecology {code} · {station['status']} · "
        f"{float(lat):.4f}, {float(lon):.4f}. "
        "One published-year check, then one daily file at a time. "
        "The next file waits until the previous one is saved."
    )

    today = date.today()
    from hydrology.app.shared import date_range_selector

    start, end = date_range_selector(
        "ecology",
        default_start=date(today.year, 1, 1),
        default_end=today,
    )
    years = water_years_overlapping(start, end)
    index = read_index(code)
    _render_cache_controls(code, years, index)

    frame = load_cached_series(code, start, end)
    if frame.empty or frame["Discharge_cfs"].notna().sum() == 0:
        st.info("No saved discharge in this date range yet.")
        return

    has_stage = "Gage_Height_ft" in frame.columns and frame["Gage_Height_ft"].notna().any()
    _render_record_caption(code, years)
    _render_views(frame, name, code, has_stage)


def _select_station(catalog: pd.DataFrame) -> pd.Series | None:
    status = st.selectbox(
        "Station status",
        ["Active", "All"],
        key="ecology_status",
    )
    subset = catalog if status == "All" else catalog[catalog["status"] == "Active"]
    search = st.text_input(
        "Search Ecology stations",
        placeholder="River name or station code",
        key="ecology_search",
    )
    if search:
        needle = search.lower()
        mask = (
            subset["station_code"].str.lower().str.contains(needle, na=False)
            | subset["station_name"].str.lower().str.contains(needle, na=False)
        )
        subset = subset[mask]
    if subset.empty:
        st.warning("No Ecology stations match that search.")
        return None

    labels = [
        f"{row.station_code} — {row.station_name}"
        for row in subset.itertuples(index=False)
    ]
    choice = st.selectbox("Ecology station", labels, key="ecology_station")
    chosen = choice.split(" — ", 1)[0]
    match = subset[subset["station_code"] == chosen]
    return None if match.empty else match.iloc[0]


def _render_cache_controls(code: str, years: list[int], index: dict | None) -> None:
    check_label = (
        "Check published years again"
        if index is not None
        else "Check which years DataStream has published"
    )
    if st.button(check_label, key=f"ecology_index_{code}"):
        try:
            with st.spinner("Reading the station page for published water years..."):
                refresh_published_years(code)
        except EcologyDataError as exc:
            st.error(str(exc))
            return
        st.rerun()

    if index is None:
        st.caption("The published-year list for this station is not on disk yet.")
        return

    st.caption(f"Published-year list checked {index.get('checked_on', 'on an unknown date')}.")
    pending = pending_downloads(code, years, DISCHARGE_SUFFIX)
    if not pending:
        st.caption("Every published discharge year in this range is on disk.")
    else:
        next_year = pending[0][0]
        st.caption(
            f"Next file is water year {next_year} daily discharge. "
            f"{len(pending)} missing in this range."
        )
        if st.button("Download the next file", key=f"ecology_next_{code}"):
            _run_queue(code, pending[:1], pause=0)
        if len(pending) > 1 and st.button(
            f"Continue through this range ({min(QUEUE_LIMIT, len(pending))} files, one at a time)",
            key=f"ecology_continue_{code}",
        ):
            _run_queue(code, pending, pause=2)

    refresh = [
        year for year in years
        if year_record_status(code, year) in {"provisional", "unchecked"}
    ]
    if refresh:
        status = year_record_status(code, refresh[0])
        if st.button(
            f"Refresh water year {refresh[0]} discharge ({status})",
            key=f"ecology_refresh_{code}_{refresh[0]}",
        ):
            _run_queue(code, [(refresh[0], DISCHARGE_SUFFIX)], pause=0)


def _render_record_caption(code: str, years: list[int]) -> None:
    labels = []
    for year in years:
        status = year_record_status(code, year)
        if status is not None:
            labels.append(f"WY{year} {status}")
    if labels:
        st.caption("Local copies: " + ", ".join(labels) + ".")


def _render_views(frame: pd.DataFrame, name: str, code: str, has_stage: bool) -> None:
    views = ["Hydrograph", "Duration", "Raster", "Percentiles"]
    if has_stage:
        views.append("Rating")
    view = st.radio("Plot", views, horizontal=True, key="ecology_plot_view")
    if view == "Hydrograph":
        aggregation = st.radio(
            "Aggregation",
            ["daily", "weekly", "monthly"],
            horizontal=True,
            key="ecology_hydro_agg",
        )
        figure = interactive_hydrograph(
            frame,
            discharge_col="Discharge_cfs",
            title=f"{name} — Hydrograph",
            aggregation=aggregation,
            show_percentile_bands=True,
        )
        if has_stage:
            _add_stage_axis(figure, frame)
        st.plotly_chart(figure, width="stretch", key="ecology_hydro")
    elif view == "Duration":
        figure = interactive_fdc(
            frame,
            discharge_col="Discharge_cfs",
            title=f"{name} — Flow duration",
            color_by_dqdt=True,
        )
        st.plotly_chart(figure, width="stretch", key="ecology_fdc")
    elif view == "Raster":
        figure = raster_hydrograph(
            frame,
            discharge_col="Discharge_cfs",
            title=f"{name} — Raster hydrograph",
        )
        st.plotly_chart(figure, width="stretch", key="ecology_raster")
    elif view == "Percentiles":
        figure = percentile_bands_hydrograph(
            frame,
            discharge_col="Discharge_cfs",
            title=f"{name} — Percentile bands",
        )
        st.plotly_chart(figure, width="stretch", key="ecology_percentiles")
    elif view == "Rating":
        from hydrology.app.page_modules.single_analysis import _render_rating_curve_workshop

        stage_pending = pending_downloads(code, water_years_overlapping(
            frame.index.min().date(), frame.index.max().date()
        ), STAGE_SUFFIX)
        if stage_pending:
            st.caption(
                f"Stage for water year {stage_pending[0][0]} is a separate file. "
                "It is not downloaded with discharge."
            )
            if st.button("Download the next stage file", key=f"ecology_stage_{code}"):
                _run_queue(code, stage_pending[:1], pause=0)
        _render_rating_curve_workshop(frame, site_id=code, site_desc=name)


def _run_queue(code: str, files: list[tuple[int, str]], pause: float) -> None:
    with st.spinner("Saving the next DataStream file..."):
        result = download_queue(code, files, pause_seconds=pause)
    if result.stopped_on:
        st.error(result.stopped_on)
        if result.saved:
            st.caption(f"Saved {len(result.saved)} file(s) before the queue stopped.")
        return
    if result.saved:
        st.rerun()


def _add_stage_axis(figure, frame: pd.DataFrame) -> None:
    import plotly.graph_objects as go

    stage = frame["Gage_Height_ft"].dropna()
    if stage.empty:
        return
    figure.add_trace(go.Scatter(
        x=stage.index,
        y=stage.values,
        mode="lines",
        name="Stage (ft)",
        line=dict(color="#2ca02c", width=1.5),
        yaxis="y2",
    ))
    figure.update_layout(
        yaxis2=dict(title="Stage (ft)", overlaying="y", side="right", showgrid=False)
    )
