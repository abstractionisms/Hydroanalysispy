"""
Reach Analysis page - surfaces the 9 reach analysis plot functions
that exist in plots.py but were previously unreachable from the UI.
"""

import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

from hydrology.app.shared import (
    get_inventory, get_cached_site_info,
    extract_site_id, fetch_discharge_data,
    date_range_selector, render_export_buttons,
    _filter_inventory, FIPS_TO_STATE,
    logger)
from hydrology.app.plot_config import REACH_PLOTS, get_display_name
from hydrology.visualization.plots import AVAILABLE_PLOTS
from hydrology.core import DEFAULT_DISCHARGE_CODE
from hydrology.app.shared import fetch_climate_cached
from hydrology.visualization.interactive import baseflow_waterfall
from hydrology.analysis.reach_gain_loss import summarize_reach_gain_loss
from hydrology.data.nldi import discover_related_sites

import pandas as pd


def _build_reach_summary_row(upstream_id, downstream_id, upstream_q, downstream_q, reach_km=None):
    """Build one dashboard row for paired reach gain/loss."""
    summary = summarize_reach_gain_loss(upstream_q, downstream_q, reach_km=reach_km)
    median_gain = summary.get("median_gain_cfs")
    low_flow_gain = summary.get("low_flow_median_gain_cfs")
    per_km = summary.get("median_gain_cfs_per_km")
    return {
        "Reach": f"{upstream_id} -> {downstream_id}",
        "Class": summary.get("classification", "insufficient_data"),
        "Median gain/loss": f"{median_gain:,.0f} cfs" if pd.notna(median_gain) else "N/A",
        "Low-flow gain/loss": f"{low_flow_gain:,.0f} cfs" if pd.notna(low_flow_gain) else "N/A",
        "Gain/loss per km": f"{per_km:,.1f} cfs/km" if pd.notna(per_km) else "Add reach length",
        "Confidence": summary.get("confidence", "none"),
    }


def _build_reach_interpretation(reach_row, reach_km=None, length_source="missing"):
    """Convert reach screening metrics into a concise dashboard interpretation."""
    reach_class = str(reach_row.get("Class", "insufficient_data")).lower()
    confidence = str(reach_row.get("Confidence", "none")).lower()
    per_km = reach_row.get("Gain/loss per km", "Add reach length")

    if reach_class == "gaining":
        finding = "Gaining reach"
        direction_text = "downstream flow is higher than upstream flow"
    elif reach_class == "losing":
        finding = "Losing reach"
        direction_text = "downstream flow is lower than upstream flow"
    elif reach_class == "neutral":
        finding = "No clear gain/loss"
        direction_text = "upstream and downstream flow are similar over the paired record"
    else:
        finding = "Not enough paired data"
        direction_text = "paired daily flow overlap is not sufficient for a reach call"

    if reach_km:
        length_label = f"{reach_km:.1f} km, {length_source} inferred" if length_source == "network" else f"{reach_km:.1f} km, manual"
        length_text = f"Normalized gain/loss is {per_km}."
    else:
        length_label = "Not inferred"
        length_text = "cfs/km is unavailable until the network length is inferred or entered."

    review = "Use as a screening result."
    if confidence != "high":
        review = "Screening result; check tributaries, diversions, withdrawals, and data overlap."

    return {
        "Finding": finding,
        "Confidence": confidence,
        "Median gain/loss": reach_row.get("Median gain/loss", "N/A"),
        "Low-flow gain/loss": reach_row.get("Low-flow gain/loss", "N/A"),
        "Reach length": length_label,
        "Interpretation": f"{direction_text}; {length_text}",
        "Review": review,
    }


def _format_reach_chain(site_ids):
    """Format adjacent reaches in selected upstream-to-downstream order."""
    return [
        {"Order": idx, "Reach": f"{upstream} -> {downstream}"}
        for idx, (upstream, downstream) in enumerate(zip(site_ids, site_ids[1:]), start=1)
    ]


def _signed_navigation_distance(site_id, related_sites, origin_site_id):
    """Return signed distance from an NLDI anchor; upstream is negative."""
    if site_id == origin_site_id:
        return 0.0
    for site in related_sites:
        if str(site.get("site_id")) != str(site_id):
            continue
        distance = site.get("distance_km")
        direction = str(site.get("direction", "")).lower()
        if distance is None or direction not in {"upstream", "downstream"}:
            return None
        signed = float(distance)
        return -signed if direction == "upstream" else signed
    return None


def _estimate_reach_km(upstream_id, downstream_id, related_sites, origin_site_id):
    """Estimate reach length from navigation distances relative to one anchor."""
    upstream_distance = _signed_navigation_distance(upstream_id, related_sites, origin_site_id)
    downstream_distance = _signed_navigation_distance(downstream_id, related_sites, origin_site_id)
    if upstream_distance is None or downstream_distance is None:
        return None
    reach_km = abs(downstream_distance - upstream_distance)
    return round(reach_km, 2) if reach_km > 0 else None


def _resolve_reach_km(estimated_reach_km, manual_reach_km):
    """Use inferred network length unless only a manual override is available."""
    if estimated_reach_km and estimated_reach_km > 0:
        return estimated_reach_km
    if manual_reach_km and manual_reach_km > 0:
        return manual_reach_km
    return None


def _flowline_distance_km(reach_km, search_km):
    """Choose a bounded flowline lookup distance for reach mapping."""
    if reach_km and reach_km > 0:
        return min(max(round(reach_km * 1.5, 1), 10.0), 300.0)
    return float(search_km)


def _flowline_style(selected=False):
    """Folium style for reach network flowlines."""
    if selected:
        return {
            "color": "#ffd166",
            "weight": 6,
            "opacity": 0.95,
        }
    return {
        "color": "#4fc3f7",
        "weight": 2,
        "opacity": 0.45,
    }


def _map_bounds_for_reach(
    selected_flowlines,
    context_flowlines,
    upstream_lat,
    upstream_lon,
    downstream_lat,
    downstream_lon,
):
    """Return folium bounds focused on the selected gages."""
    lat_min = min(float(upstream_lat), float(downstream_lat))
    lat_max = max(float(upstream_lat), float(downstream_lat))
    lon_min = min(float(upstream_lon), float(downstream_lon))
    lon_max = max(float(upstream_lon), float(downstream_lon))

    lat_pad = max((lat_max - lat_min) * 0.25, 0.01)
    lon_pad = max((lon_max - lon_min) * 0.25, 0.01)
    return [
        [round(lat_min - lat_pad, 6), round(lon_min - lon_pad, 6)],
        [round(lat_max + lat_pad, 6), round(lon_max + lon_pad, 6)],
    ]


def _reach_map_component_key(upstream_id, downstream_id, bounds):
    """Return a stable key that remounts the map when the selected reach changes."""
    flat_bounds = "_".join(f"{coord:.4f}" for row in bounds for coord in row)
    return f"reach_map_{upstream_id}_{downstream_id}_{flat_bounds}"


def _filter_related_sites_to_inventory(origin_site_id, related_sites, inventory_df):
    """Keep NLDI candidates that HydroPlot can resolve from its inventory."""
    if inventory_df is None or inventory_df.empty or "site_id" not in inventory_df.columns:
        return [], [str(site.get("site_id")) for site in related_sites if site.get("site_id")]

    inventory_ids = set(inventory_df["site_id"].astype(str))
    filtered = []
    omitted = []
    for site in related_sites:
        site_id = str(site.get("site_id", ""))
        if not site_id:
            continue
        if site_id == str(origin_site_id) or site_id in inventory_ids:
            filtered.append(site)
        else:
            omitted.append(site_id)
    return filtered, omitted


def _format_related_site_rows(origin_site_id, related_sites):
    """Make NLDI candidate stations readable for the reach-selection UI."""
    rows = [
        {
            "Station": str(origin_site_id),
            "Position": "Anchor",
            "Distance from anchor": "0.0 km",
            "Name": "Selected anchor station",
        }
    ]
    for site in related_sites:
        direction = str(site.get("direction", "unknown")).lower()
        navigation_mode = str(site.get("navigation_mode", "")).lower()
        if "trib" in navigation_mode:
            position = "Tributary"
        elif direction == "upstream":
            position = "Upstream"
        elif direction == "downstream":
            position = "Downstream"
        else:
            position = "Unknown"
        distance = site.get("distance_km")
        rows.append(
            {
                "Station": str(site.get("site_id", "")),
                "Position": position,
                "Distance from anchor": f"{float(distance):.1f} km" if distance is not None else "Unknown",
                "Name": site.get("name", ""),
            }
        )
    return rows


def _candidate_position(site):
    """Return the user-facing network position for a candidate site."""
    direction = str(site.get("direction", "unknown")).lower()
    navigation_mode = str(site.get("navigation_mode", "")).lower()
    if "trib" in navigation_mode:
        return "Tributary"
    if direction == "upstream":
        return "Upstream"
    if direction == "downstream":
        return "Downstream"
    return "Anchor"


def _build_reach_candidate_options(origin_site_id, origin_name, related_sites):
    """Build labeled gage options for reach selectors."""
    candidates = [
        {
            "site_id": str(origin_site_id),
            "label": f"Anchor | {origin_site_id} | 0.0 km | {origin_name}",
            "position": "Anchor",
            "distance_km": 0.0,
        }
    ]
    for site in related_sites:
        site_id = str(site.get("site_id", ""))
        if not site_id:
            continue
        position = _candidate_position(site)
        distance = site.get("distance_km")
        distance_label = f"{float(distance):.1f} km" if distance is not None else "distance unknown"
        name = site.get("name") or site.get("description") or ""
        candidates.append(
            {
                "site_id": site_id,
                "label": f"{position} | {site_id} | {distance_label} | {name}",
                "position": position,
                "distance_km": float(distance) if distance is not None else None,
            }
        )

    position_order = {"Anchor": 0, "Upstream": 1, "Tributary": 2, "Downstream": 3}
    candidates.sort(
        key=lambda candidate: (
            position_order.get(candidate["position"], 9),
            candidate["distance_km"] if candidate["distance_km"] is not None else 9999.0,
        )
    )
    return candidates


def _pair_key(upstream_id, downstream_id):
    """Return a stable selected-reach key."""
    return f"{upstream_id}__{downstream_id}"


def _build_recommended_reach_pairs(origin_site_id, candidates, max_pairs=8):
    """Build processable upstream/downstream reach pairs for the workspace."""
    origin_site_id = str(origin_site_id)
    by_position = {"Upstream": [], "Downstream": [], "Tributary": []}
    for candidate in candidates:
        site_id = str(candidate.get("site_id", ""))
        if not site_id or site_id == origin_site_id:
            continue
        position = candidate.get("position")
        if position in by_position:
            by_position[position].append(candidate)

    def distance_value(candidate):
        distance = candidate.get("distance_km")
        return float(distance) if distance is not None else 9999.0

    for values in by_position.values():
        values.sort(key=distance_value)

    pairs = []
    for upstream in by_position["Upstream"]:
        pairs.append({
            "key": _pair_key(upstream["site_id"], origin_site_id),
            "upstream_id": str(upstream["site_id"]),
            "downstream_id": origin_site_id,
            "label": f'{upstream["site_id"]} -> {origin_site_id}',
            "kind": "mainstem upstream",
            "distance_km": upstream.get("distance_km"),
        })
    for downstream in by_position["Downstream"]:
        pairs.append({
            "key": _pair_key(origin_site_id, downstream["site_id"]),
            "upstream_id": origin_site_id,
            "downstream_id": str(downstream["site_id"]),
            "label": f'{origin_site_id} -> {downstream["site_id"]}',
            "kind": "mainstem downstream",
            "distance_km": downstream.get("distance_km"),
        })
    for tributary in by_position["Tributary"]:
        pairs.append({
            "key": _pair_key(tributary["site_id"], origin_site_id),
            "upstream_id": str(tributary["site_id"]),
            "downstream_id": origin_site_id,
            "label": f'{tributary["site_id"]} -> {origin_site_id}',
            "kind": "tributary context",
            "distance_km": tributary.get("distance_km"),
        })

    seen = set()
    unique_pairs = []
    for pair in pairs:
        if pair["key"] in seen:
            continue
        seen.add(pair["key"])
        unique_pairs.append(pair)
    return unique_pairs[:max_pairs]


def _candidate_index_for_site(candidates, preferred_site_id, fallback_positions):
    """Return the selector index for a preferred site or role fallback."""
    if preferred_site_id:
        preferred_site_id = str(preferred_site_id)
        for idx, candidate in enumerate(candidates):
            if str(candidate.get("site_id")) == preferred_site_id:
                return idx
    for idx, candidate in enumerate(candidates):
        if candidate.get("position") in fallback_positions:
            return idx
    return 0


def _candidate_label_for_site(candidates, site_id):
    """Return the dropdown label for a candidate site ID."""
    if site_id is None:
        return None
    site_id = str(site_id)
    for candidate in candidates:
        if str(candidate.get("site_id")) == site_id:
            return candidate.get("label")
    return None


def _default_candidate_label(candidates, preferred_site_id, fallback_positions):
    """Return the dropdown label for a preferred site or role fallback."""
    preferred_label = _candidate_label_for_site(candidates, preferred_site_id)
    if preferred_label:
        return preferred_label
    return candidates[_candidate_index_for_site(candidates, None, fallback_positions)]["label"]


def _selected_candidate_site_id(candidate_rows, selection_state):
    """Return the site ID represented by a selected candidate table row."""
    try:
        selected_rows = selection_state.get("selection", {}).get("rows", [])
    except AttributeError:
        selected_rows = getattr(getattr(selection_state, "selection", None), "rows", [])
    if not selected_rows:
        return None
    row_index = selected_rows[0]
    if row_index < 0 or row_index >= len(candidate_rows):
        return None
    return str(candidate_rows[row_index].get("Station"))


def _ensure_widget_value_is_valid(key, options):
    """Clear stale widget values when candidate options change."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def _selectbox_kwargs_for_state(key, default_index, session_state):
    """Avoid passing both explicit index and pre-set Streamlit widget state."""
    kwargs = {"key": key}
    if key not in session_state:
        kwargs["index"] = default_index
    return kwargs


def _get_discharge_series(df):
    """Return the primary discharge series from a fetched dataframe."""
    if "Discharge_cfs" in df.columns:
        return df["Discharge_cfs"]
    if "value" in df.columns:
        return df["value"]
    return df.select_dtypes(include="number").iloc[:, 0]


def show():
    """Render the Reach Analysis page."""
    inventory_df = get_inventory()
    if inventory_df.empty:
        st.error("Could not load site inventory")
        return

    st.header("Reach Analysis")
    st.caption("Select an upstream/downstream reach, then run gain/loss and baseflow analysis.")

    states = ["All States"] + sorted(FIPS_TO_STATE.values())

    with st.expander("Select reach", expanded=True):
        top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns([1.2, 2.2, 3, 1, 1.2])
        with top_col1:
            anchor_state = st.selectbox("State", states, key="reach_anchor_state")
        with top_col2:
            anchor_search = st.text_input(
                "Find river gage",
                placeholder="River, station, or USGS ID...",
                key="reach_anchor_search",
            )

        anchor_filtered = _filter_inventory(inventory_df, anchor_search, anchor_state)
        anchor_options = [
            f"{row['site_id']} - {str(row.get('description', ''))[:80]}"
            for _, row in anchor_filtered.iterrows()
        ]
        if anchor_search or anchor_state != "All States":
            st.caption(f"{len(anchor_options)} matching gages")
        if not anchor_options:
            st.warning("No gages match the current search")
            return

        with top_col3:
            anchor_sel = st.selectbox("Anchor gage", anchor_options, key="reach_anchor")
        anchor_id = extract_site_id(anchor_sel)
        anchor_info = get_cached_site_info(anchor_id)
        with top_col4:
            search_km = st.number_input(
                "Km",
                min_value=10,
                max_value=300,
                value=75,
                step=5,
                key="reach_nldi_search_km",
            )
        with top_col5:
            include_tributaries = st.toggle(
                "Tributaries",
                value=True,
                key="reach_include_tributaries",
            )
            st.markdown("<br>", unsafe_allow_html=True)
            find_gages = st.button("Find Related Gages", width="stretch", key="reach_find_related")
        if anchor_info:
            st.caption(anchor_info.get("description", ""))

    related_key = f"reach_related_sites_{anchor_id}_{search_km}_{include_tributaries}"
    if find_gages:
        with st.spinner("Finding upstream and downstream gages on the river network..."):
            st.session_state[related_key] = discover_related_sites(
                anchor_id,
                direction="both",
                distance_km=float(search_km),
                include_tributaries=include_tributaries,
                max_sites=25,
            )

    discovered_related_sites = st.session_state.get(related_key, [])
    related_sites, omitted_related_site_ids = _filter_related_sites_to_inventory(
        anchor_id,
        discovered_related_sites,
        inventory_df,
    )
    anchor_name = anchor_info.get("description", "Anchor gage") if anchor_info else "Anchor gage"
    candidate_records = _build_reach_candidate_options(anchor_id, anchor_name, related_sites)
    candidate_rows = _format_related_site_rows(anchor_id, related_sites)
    with st.expander("Candidate gages", expanded=not bool(discovered_related_sites)):
        if omitted_related_site_ids:
            st.caption(
                f"{len(omitted_related_site_ids)} NLDI gages were hidden because they are not in the HydroPlot inventory for this app."
            )
        if related_sites:
            candidate_selection = st.dataframe(
                pd.DataFrame(candidate_rows),
                width="stretch",
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="reach_candidate_table",
            )
            selected_candidate_id = _selected_candidate_site_id(candidate_rows, candidate_selection)
            action_col1, action_col2, action_col3 = st.columns([1, 1, 3])
            with action_col1:
                if st.button("Use as Upstream", disabled=selected_candidate_id is None, width="stretch"):
                    selected_label = _candidate_label_for_site(candidate_records, selected_candidate_id)
                    if selected_label:
                        st.session_state["reach_upstream_choice"] = selected_label
            with action_col2:
                if st.button("Use as Downstream", disabled=selected_candidate_id is None, width="stretch"):
                    selected_label = _candidate_label_for_site(candidate_records, selected_candidate_id)
                    if selected_label:
                        st.session_state["reach_downstream_choice"] = selected_label
            with action_col3:
                if selected_candidate_id:
                    st.caption(f"Selected candidate: {selected_candidate_id}")
                else:
                    st.caption("Select one candidate row, then assign it to upstream or downstream.")
        elif discovered_related_sites:
            st.warning("NLDI found related gages, but none are available in the HydroPlot inventory for this dashboard.")
        else:
            st.info("Use Find Related Gages to discover likely upstream/downstream candidates for the selected anchor gage.")

    with st.expander("Selected reach", expanded=True):
        candidate_options = [candidate["label"] for candidate in candidate_records]
        site_by_label = {candidate["label"]: candidate["site_id"] for candidate in candidate_records}
        _ensure_widget_value_is_valid("reach_upstream_choice", candidate_options)
        _ensure_widget_value_is_valid("reach_downstream_choice", candidate_options)
        if "reach_upstream_choice" not in st.session_state:
            st.session_state["reach_upstream_choice"] = _default_candidate_label(candidate_records, None, {"Upstream", "Tributary"})
        if "reach_downstream_choice" not in st.session_state:
            st.session_state["reach_downstream_choice"] = _default_candidate_label(candidate_records, None, {"Downstream", "Anchor"})
        default_upstream_idx = candidate_options.index(st.session_state["reach_upstream_choice"])
        default_downstream_idx = candidate_options.index(st.session_state["reach_downstream_choice"])

        reach_col1, reach_col2 = st.columns(2)
        with reach_col1:
            upstream_sel = st.selectbox(
                "Upstream gage",
                candidate_options,
                **_selectbox_kwargs_for_state("reach_upstream_choice", default_upstream_idx, st.session_state),
            )
        with reach_col2:
            downstream_sel = st.selectbox(
                "Downstream gage",
                candidate_options,
                **_selectbox_kwargs_for_state("reach_downstream_choice", default_downstream_idx, st.session_state),
            )
        upstream_id = site_by_label[upstream_sel]
        downstream_id = site_by_label[downstream_sel]
        up_info = get_cached_site_info(upstream_id)
        dn_info = get_cached_site_info(downstream_id)

        estimated_reach_km = _estimate_reach_km(upstream_id, downstream_id, related_sites, anchor_id)
        config_col1, config_col2 = st.columns([1, 2])
        with config_col1:
            if estimated_reach_km:
                st.metric("Network length", f"{estimated_reach_km:.1f} km")
            else:
                st.metric("Network length", "Not inferred")
        with config_col2:
            if upstream_id == downstream_id:
                st.warning("Choose two different gages for a reach.")
            else:
                st.metric("Selected reach", f"{upstream_id} -> {downstream_id}")

        manual_reach_km = 0.0
        with st.expander("Advanced reach length override", expanded=False):
            manual_reach_km = st.number_input(
                "Manual reach length km",
                min_value=0.0,
                max_value=1000.0,
                value=0.0,
                step=0.1,
                help="Optional. Used only when the network length cannot be inferred.",
                key="reach_length_km",
            )
            if estimated_reach_km:
                st.caption("Network-inferred length is used for cfs/km. Manual value is ignored while an inferred length is available.")
            else:
                st.caption("Optional fallback for cfs/km when related gage distances do not define the selected reach.")
        reach_km = _resolve_reach_km(estimated_reach_km, manual_reach_km)

    reach_plot_options = {get_display_name(p): p for p in REACH_PLOTS if p in AVAILABLE_PLOTS}
    default_plots = ['reach_comparison', 'reach_index', 'seasonal_gain_loss']
    default_display = [get_display_name(p) for p in default_plots if p in reach_plot_options.values()]

    with st.expander("Run analysis", expanded=True):
        settings_col1, settings_col2 = st.columns([1, 1])
        with settings_col1:
            start_date, end_date = date_range_selector("reach", default_start=date(2000, 1, 1))
        with settings_col2:
            selected_display = st.multiselect(
                "Plots",
                list(reach_plot_options.keys()),
                default=default_display,
                key="reach_plot_select"
            )
        selected_plots = [reach_plot_options[d] for d in selected_display]

        with st.expander("Plot descriptions"):
            for display_name, plot_key in reach_plot_options.items():
                info = AVAILABLE_PLOTS.get(plot_key, {})
                desc = info.get('description', '') if isinstance(info, dict) else ''
                st.markdown(f"**{display_name}**: {desc}")

        # Reach map - show network flowlines and upstream/downstream stations
        if up_info and dn_info and up_info.get('latitude') and dn_info.get('latitude'):
            with st.expander("Reach Map", expanded=False):
                import folium
                from streamlit_folium import st_folium
                from hydrology.data.hyriver import get_flowlines, get_navigation_flowlines

                up_lat, up_lon = float(up_info['latitude']), float(up_info['longitude'])
                dn_lat, dn_lon = float(dn_info['latitude']), float(dn_info['longitude'])
                center_lat = (up_lat + dn_lat) / 2
                center_lon = (up_lon + dn_lon) / 2

                m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles=None)
                folium.TileLayer(
                    "CartoDB dark_matter",
                    name="Base map",
                    no_wrap=True,
                ).add_to(m)

                flowline_distance = _flowline_distance_km(reach_km, search_km)
                try:
                    flowlines = get_flowlines(downstream_id, distance_km=flowline_distance)
                    selected_flowlines = None
                    if flowlines is not None and not flowlines.empty:
                        folium.GeoJson(
                            flowlines.to_json(),
                            name="River network context",
                            style_function=lambda feature: _flowline_style(selected=False),
                            tooltip="NHDPlus river/tributary flowline",
                        ).add_to(m)
                        selected_flowlines = get_navigation_flowlines(
                            downstream_id,
                            navigation="upstreamMain",
                            distance_km=flowline_distance,
                        )
                        if selected_flowlines is None or selected_flowlines.empty:
                            selected_flowlines = flowlines
                        folium.GeoJson(
                            selected_flowlines.to_json(),
                            name="Selected reach network",
                            style_function=lambda feature: _flowline_style(selected=True),
                            tooltip=f"Selected reach network: {upstream_id} -> {downstream_id}",
                        ).add_to(m)
                    else:
                        st.caption("River-network geometry was not available for this reach; showing gage locations only.")
                except Exception as e:
                    logger.warning(f"Could not add reach flowlines: {e}")
                    st.caption("River-network geometry was not available for this reach; showing gage locations only.")

                folium.CircleMarker(
                    [up_lat, up_lon], radius=10, color='#2196F3', fill=True,
                    fillColor='#2196F3', fillOpacity=0.8,
                    tooltip=f"Upstream: {up_info.get('description', upstream_id)}"
                ).add_to(m)

                folium.CircleMarker(
                    [dn_lat, dn_lon], radius=10, color='#FF9800', fill=True,
                    fillColor='#FF9800', fillOpacity=0.8,
                    tooltip=f"Downstream: {dn_info.get('description', downstream_id)}"
                ).add_to(m)

                reach_bounds = _map_bounds_for_reach(None, None, up_lat, up_lon, dn_lat, dn_lon)
                m.fit_bounds(reach_bounds)
                st_folium(
                    m,
                    width=None,
                    height=300,
                    returned_objects=[],
                    key=_reach_map_component_key(upstream_id, downstream_id, reach_bounds),
                )
                st.caption("Gold = selected reach network; blue = nearby river/tributary context. Blue marker = upstream gage; orange marker = downstream gage.")

        col_layout, col_dpi, col_btn = st.columns([2, 1, 2])
        with col_layout:
            layout_choice = st.selectbox("Layout", ["Auto", "Vertical", "Grid 2x3"], key="reach_layout")
        with col_dpi:
            dpi = st.number_input("DPI", min_value=72, max_value=300, value=150, key="reach_dpi")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            generate = st.button("Generate Reach Analysis", type="primary", width="stretch", key="gen_reach")

    if generate:
        if not selected_plots:
            st.warning("Select at least one plot")
            return
        if upstream_id == downstream_id:
            st.warning("Choose two different gages before generating reach analysis.")
            return

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # Fetch data for both stations
        with st.spinner("Fetching upstream data..."):
            df_upstream = fetch_discharge_data(upstream_id, DEFAULT_DISCHARGE_CODE, start_str, end_str)

        with st.spinner("Fetching downstream data..."):
            df_downstream = fetch_discharge_data(downstream_id, DEFAULT_DISCHARGE_CODE, start_str, end_str)

        if df_upstream is None or df_upstream.empty:
            st.error(f"No discharge data for upstream station {upstream_id}")
            return
        if df_downstream is None or df_downstream.empty:
            st.error(f"No discharge data for downstream station {downstream_id}")
            return

        # Fetch climate data if needed for precip_response_comparison or summer_climate_context
        df_climate = None
        climate_needed = any(p in selected_plots for p in ['precip_response_comparison', 'summer_climate_context'])
        if climate_needed:
            # Use downstream station coords for climate
            if dn_info and dn_info.get('latitude') and dn_info.get('longitude'):
                with st.spinner("Fetching climate data..."):
                    df_climate = fetch_climate_cached(
                        float(dn_info['latitude']), float(dn_info['longitude']),
                        start_str, end_str, downstream_id
                    )

        # Show data summary
        up_desc = up_info.get('description', upstream_id) if up_info else upstream_id
        dn_desc = dn_info.get('description', downstream_id) if dn_info else downstream_id

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Upstream", f"{len(df_upstream):,} days", delta=up_desc, delta_color="off")
        with col2:
            st.metric("Downstream", f"{len(df_downstream):,} days", delta=dn_desc, delta_color="off")

        upstream_q = _get_discharge_series(df_upstream)
        downstream_q = _get_discharge_series(df_downstream)
        reach_row = _build_reach_summary_row(upstream_id, downstream_id, upstream_q, downstream_q, reach_km=reach_km)
        length_source = "network" if estimated_reach_km else "manual" if reach_km else "missing"
        reach_interpretation = _build_reach_interpretation(reach_row, reach_km=reach_km, length_source=length_source)
        st.subheader("Automated Reach Summary")
        st.dataframe(pd.DataFrame([reach_interpretation]), width="stretch", hide_index=True)
        with st.expander("Reach details", expanded=False):
            st.dataframe(pd.DataFrame(_format_reach_chain([upstream_id, downstream_id])), width="stretch", hide_index=True)
            st.dataframe(pd.DataFrame([reach_row]), width="stretch", hide_index=True)

        st.markdown("---")

        # Generate plots
        n_plots = len(selected_plots)
        if layout_choice == "Vertical" or n_plots <= 2:
            ncols = 1
            nrows = n_plots
        elif layout_choice == "Grid 2x3" or n_plots > 3:
            ncols = 2
            nrows = (n_plots + 1) // 2
        else:
            ncols = min(n_plots, 3)
            nrows = (n_plots + ncols - 1) // ncols

        fig_height = min(max(5 * nrows, 6), 40)  # Cap to prevent image size errors
        fig_width = min(8 * ncols, 20)

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), dpi=dpi, squeeze=False)

        with st.spinner("Generating reach analysis plots..."):
            for idx, plot_name in enumerate(selected_plots):
                row = idx // ncols
                col = idx % ncols
                ax = axes[row, col]

                plot_info = AVAILABLE_PLOTS.get(plot_name)
                if plot_info and isinstance(plot_info, dict):
                    plot_func = plot_info.get('function')
                    if plot_func:
                        try:
                            kwargs = {
                                'ax': ax,
                                'df_upstream': df_upstream,
                                'df_downstream': df_downstream,
                                'df_q': df_downstream,  # Some plots use df_q as primary
                                'config': {
                                    'upstream_name': up_desc[:40],
                                    'downstream_name': dn_desc[:40],
                                },
                            }
                            if df_climate is not None:
                                kwargs['df_climate'] = df_climate
                            plot_func(**kwargs)
                        except Exception as e:
                            ax.text(0.5, 0.5, f"Error: {str(e)[:60]}",
                                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
                            logger.error(f"Reach plot error for {plot_name}: {e}")

            # Hide unused axes
            for idx in range(n_plots, nrows * ncols):
                axes[idx // ncols, idx % ncols].set_visible(False)

        fig.suptitle(f"Reach Analysis: {up_desc[:45]} \u2192 {dn_desc[:45]}", fontsize=14, fontweight='bold')
        fig.tight_layout()

        st.pyplot(fig)
        render_export_buttons(fig, f"reach_{upstream_id}_{downstream_id}", dpi)
        plt.close(fig)

        # Baseflow separation waterfall (interactive Plotly)
        st.markdown("---")
        st.subheader("Baseflow Separation")
        show_waterfall = st.checkbox(
            "Show Baseflow Waterfall", value=True, key="show_bf_waterfall"
        )
        if show_waterfall:
            with st.spinner("Computing baseflow separation..."):
                fig_wf = baseflow_waterfall(
                    df_upstream, df_downstream,
                    upstream_name=up_desc[:40],
                    downstream_name=dn_desc[:40],
                    title=f"Baseflow Waterfall: {up_desc[:35]} → {dn_desc[:35]}")
            st.plotly_chart(fig_wf, width="stretch", key="plotly_bf_waterfall")
    else:
        st.info("Select upstream and downstream stations, choose plots, then click 'Generate Reach Analysis'")
