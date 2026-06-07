"""
Reach Analysis page - surfaces the 9 reach analysis plot functions
that exist in plots.py but were previously unreachable from the UI.
"""

import streamlit as st
import matplotlib.pyplot as plt
from datetime import date
import json

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


def _clip_flowlines_between_gages(flowlines, upstream_lat, upstream_lon, downstream_lat, downstream_lon):
    """Clip a flowline layer to the portion between selected upstream/downstream gages."""
    if flowlines is None or getattr(flowlines, "empty", True):
        return None

    try:
        import geopandas as gpd
        from shapely.geometry import Point

        original_crs = flowlines.crs
        work = flowlines
        upstream_point = Point(float(upstream_lon), float(upstream_lat))
        downstream_point = Point(float(downstream_lon), float(downstream_lat))

        if original_crs is not None and getattr(original_crs, "is_geographic", False):
            work = flowlines.to_crs(epsg=3857)
            points = gpd.GeoSeries([upstream_point, downstream_point], crs=original_crs).to_crs(epsg=3857)
            upstream_point = points.iloc[0]
            downstream_point = points.iloc[1]

        path = _trace_flowline_path_between_points(work.geometry, upstream_point, downstream_point)
        if path is None or path.is_empty:
            return None

        clipped_gdf = gpd.GeoDataFrame({"segment": ["selected reach"]}, geometry=[path], crs=work.crs)
        if original_crs is not None and clipped_gdf.crs != original_crs:
            clipped_gdf = clipped_gdf.to_crs(original_crs)
        return clipped_gdf
    except Exception as e:
        logger.warning(f"Could not clip selected reach flowlines: {e}")
        return None


def _iter_lines(geometries):
    """Yield LineString parts from a geometry collection."""
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            yield geom
        elif geom.geom_type == "MultiLineString":
            yield from geom.geoms


def _coord_key(coord, precision=6):
    """Round map coordinates so shared NHD segment endpoints connect."""
    return (round(float(coord[0]), precision), round(float(coord[1]), precision))


def _nearest_segment_projection(lines, point):
    """Return the nearest segment plus the projected point on that segment."""
    from shapely.geometry import LineString

    best = None
    for line_index, line in enumerate(lines):
        coords = list(line.coords)
        for segment_index, (start, end) in enumerate(zip(coords, coords[1:])):
            segment = LineString([start, end])
            if segment.length == 0:
                continue
            distance_on_segment = segment.project(point)
            projected = segment.interpolate(distance_on_segment)
            distance_to_point = projected.distance(point)
            candidate = (distance_to_point, line_index, segment_index, projected)
            if best is None or candidate[0] < best[0]:
                best = candidate
    return best


def _add_graph_edge(graph, left_key, right_key, left_coord, right_coord):
    """Add an undirected weighted edge to a coordinate graph."""
    from math import hypot

    if left_key == right_key:
        return
    weight = hypot(right_coord[0] - left_coord[0], right_coord[1] - left_coord[1])
    graph.setdefault(left_key, {})[right_key] = weight
    graph.setdefault(right_key, {})[left_key] = weight


def _shortest_coord_path(graph, start_key, end_key):
    """Dijkstra path through the flowline coordinate graph."""
    import heapq

    queue = [(0.0, start_key, [start_key])]
    visited = set()
    while queue:
        distance, node, path = heapq.heappop(queue)
        if node == end_key:
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph.get(node, {}).items():
            if neighbor not in visited:
                heapq.heappush(queue, (distance + weight, neighbor, path + [neighbor]))
    return None


def _trace_flowline_path_between_points(geometries, upstream_point, downstream_point):
    """Trace the connected NHD path between two projected gage points."""
    from shapely.geometry import LineString, Point

    lines = list(_iter_lines(geometries))
    if not lines:
        return None

    upstream_projection = _nearest_segment_projection(lines, upstream_point)
    downstream_projection = _nearest_segment_projection(lines, downstream_point)
    if upstream_projection is None or downstream_projection is None:
        return None

    projection_by_segment = {}
    _, up_line_idx, up_segment_idx, up_projected = upstream_projection
    _, dn_line_idx, dn_segment_idx, dn_projected = downstream_projection
    projection_by_segment.setdefault((up_line_idx, up_segment_idx), []).append(("upstream", up_projected))
    projection_by_segment.setdefault((dn_line_idx, dn_segment_idx), []).append(("downstream", dn_projected))

    graph = {}
    coordinates = {}
    start_key = None
    end_key = None

    for line_index, line in enumerate(lines):
        coords = list(line.coords)
        for segment_index, (segment_start, segment_end) in enumerate(zip(coords, coords[1:])):
            segment = LineString([segment_start, segment_end])
            split_points = [(0.0, None, Point(segment_start))]
            for label, projected in projection_by_segment.get((line_index, segment_index), []):
                split_points.append((segment.project(projected), label, projected))
            split_points.append((segment.length, None, Point(segment_end)))
            split_points = sorted(split_points, key=lambda item: item[0])

            keyed_points = []
            for _, label, point in split_points:
                key = _coord_key(point.coords[0])
                coordinates[key] = point.coords[0]
                if label == "upstream":
                    start_key = key
                elif label == "downstream":
                    end_key = key
                keyed_points.append((key, point.coords[0]))

            for (left_key, left_coord), (right_key, right_coord) in zip(keyed_points, keyed_points[1:]):
                _add_graph_edge(graph, left_key, right_key, left_coord, right_coord)

    if start_key is None or end_key is None:
        return None

    path_keys = _shortest_coord_path(graph, start_key, end_key)
    if not path_keys or len(path_keys) < 2:
        return None

    return LineString([coordinates[key] for key in path_keys])


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


def _leaflet_fit_bounds_script(bounds):
    """Return a Folium-compatible script that forces Leaflet to fit selected reach bounds."""
    bounds_json = json.dumps(bounds)
    return (
        "<script>"
        "setTimeout(function(){"
        "for (const key in window) {"
        "const value = window[key];"
        "if (value && value.fitBounds && value.eachLayer) {"
        f"value.fitBounds({bounds_json}, {{paddingTopLeft:[24,24], paddingBottomRight:[24,24], maxZoom:13}});"
        "}"
        "}"
        "}, 250);"
        "</script>"
    )


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


def _candidate_display_name(candidate):
    """Return the candidate station name without repeating selector metadata."""
    if candidate.get("name"):
        return str(candidate["name"])
    label = str(candidate.get("label", ""))
    parts = [part.strip() for part in label.split("|")]
    return parts[-1] if len(parts) >= 4 else ""


def _format_pair_distance(distance_km):
    """Format pair distance for compact candidate labels."""
    return f"{float(distance_km):.1f} km" if distance_km is not None else "distance unknown"


def _pair_key(upstream_id, downstream_id):
    """Return a stable selected-reach key."""
    return f"{upstream_id}__{downstream_id}"


def _pair_label_for_key(reach_pairs, pair_key):
    """Return a pair label for a selected reach key."""
    for pair in reach_pairs:
        if pair.get("key") == pair_key:
            return pair.get("label")
    return None


def _cycle_pair_key(reach_pairs, current_key, step):
    """Return the previous or next reach pair key, wrapping around the list."""
    if not reach_pairs:
        return None
    keys = [pair["key"] for pair in reach_pairs]
    if current_key not in keys:
        return keys[0]
    current_index = keys.index(current_key)
    return keys[(current_index + step) % len(keys)]


def _resolve_selected_pair_key(reach_pairs, session_state):
    """Keep a selected reach pair if it remains valid; otherwise choose the first available pair."""
    if not reach_pairs:
        return None
    valid_keys = {pair["key"] for pair in reach_pairs}
    current = session_state.get("reach_selected_pair_key")
    if current in valid_keys:
        return current
    return reach_pairs[0]["key"]


def _build_recommended_reach_pairs(origin_site_id, candidates, max_pairs=12):
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
        distance_label = _format_pair_distance(upstream.get("distance_km"))
        name = _candidate_display_name(upstream)
        pairs.append({
            "key": _pair_key(upstream["site_id"], origin_site_id),
            "upstream_id": str(upstream["site_id"]),
            "downstream_id": origin_site_id,
            "label": f'Upstream: {upstream["site_id"]} -> {origin_site_id} | {distance_label} | {name}',
            "kind": "mainstem upstream",
            "distance_km": upstream.get("distance_km"),
            "name": name,
        })
    for downstream in by_position["Downstream"]:
        distance_label = _format_pair_distance(downstream.get("distance_km"))
        name = _candidate_display_name(downstream)
        pairs.append({
            "key": _pair_key(origin_site_id, downstream["site_id"]),
            "upstream_id": origin_site_id,
            "downstream_id": str(downstream["site_id"]),
            "label": f'Downstream: {origin_site_id} -> {downstream["site_id"]} | {distance_label} | {name}',
            "kind": "mainstem downstream",
            "distance_km": downstream.get("distance_km"),
            "name": name,
        })
    for tributary in by_position["Tributary"]:
        distance_label = _format_pair_distance(tributary.get("distance_km"))
        name = _candidate_display_name(tributary)
        pairs.append({
            "key": _pair_key(tributary["site_id"], origin_site_id),
            "upstream_id": str(tributary["site_id"]),
            "downstream_id": origin_site_id,
            "label": f'Tributary: {tributary["site_id"]} -> {origin_site_id} | {distance_label} | {name}',
            "kind": "tributary context",
            "distance_km": tributary.get("distance_km"),
            "name": name,
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


def _render_reach_map(up_info, dn_info, upstream_id, downstream_id, reach_km, search_km):
    """Render selected reach flowlines and gage markers in the current Streamlit container."""
    if not (up_info and dn_info and up_info.get('latitude') and dn_info.get('latitude')):
        st.info("Selected gage coordinates are unavailable for mapping.")
        return

    import folium
    from folium import Element
    from streamlit_folium import st_folium
    from hydrology.data.hyriver import get_flowlines

    up_lat, up_lon = float(up_info['latitude']), float(up_info['longitude'])
    dn_lat, dn_lon = float(dn_info['latitude']), float(dn_info['longitude'])
    center_lat = (up_lat + dn_lat) / 2
    center_lon = (up_lon + dn_lon) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles=None,
        max_bounds=True,
        control_scale=True,
    )
    folium.TileLayer(
        "CartoDB dark_matter",
        name="Base map",
        no_wrap=True,
        detect_retina=True,
    ).add_to(m)

    flowline_distance = _flowline_distance_km(reach_km, search_km)
    try:
        flowlines = get_flowlines(downstream_id, distance_km=flowline_distance)
        if flowlines is not None and not flowlines.empty:
            folium.GeoJson(
                flowlines.to_json(),
                name="River network context",
                style_function=lambda feature: _flowline_style(selected=False),
                tooltip="NHDPlus river/tributary flowline",
            ).add_to(m)
            clipped_flowlines = _clip_flowlines_between_gages(
                flowlines,
                upstream_lat=up_lat,
                upstream_lon=up_lon,
                downstream_lat=dn_lat,
                downstream_lon=dn_lon,
            )
            if clipped_flowlines is not None and not clipped_flowlines.empty:
                folium.GeoJson(
                    clipped_flowlines.to_json(),
                    name="Selected reach network",
                    style_function=lambda feature: _flowline_style(selected=True),
                    tooltip=f"Selected reach network: {upstream_id} -> {downstream_id}",
                ).add_to(m)
            else:
                st.caption("Could not trace a connected NHD path between the selected gage points; showing river-network context only.")
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
    m.fit_bounds(reach_bounds, padding=(24, 24), max_zoom=13)
    m.get_root().html.add_child(Element(_leaflet_fit_bounds_script(reach_bounds)))
    st_folium(
        m,
        width=None,
        height=520,
        returned_objects=[],
        key=_reach_map_component_key(upstream_id, downstream_id, reach_bounds),
    )
    st.caption("Gold = selected reach network; blue = nearby river/tributary context. Blue marker = upstream gage; orange marker = downstream gage.")


def show():
    """Render the Reach Analysis page."""
    inventory_df = get_inventory()
    if inventory_df.empty:
        st.error("Could not load site inventory")
        return

    st.header("Reach Analysis")
    st.caption("Select an upstream/downstream reach, then run gain/loss and baseflow analysis.")

    states = ["All States"] + sorted(FIPS_TO_STATE.values())

    search_col1, search_col2, search_col3, search_col4, search_col5 = st.columns([1.05, 1.9, 3.0, 0.85, 1.15])
    with search_col1:
        anchor_state = st.selectbox("State", states, key="reach_anchor_state")
    with search_col2:
        anchor_search = st.text_input(
            "Find gage",
            placeholder="River, station, or USGS ID...",
            key="reach_anchor_search",
        )

    anchor_filtered = _filter_inventory(inventory_df, anchor_search, anchor_state)
    anchor_options = [
        f"{row['site_id']} - {str(row.get('description', ''))[:80]}"
        for _, row in anchor_filtered.iterrows()
    ]
    if not anchor_options:
        st.warning("No gages match the current search")
        return

    with search_col3:
        anchor_sel = st.selectbox("Anchor gage", anchor_options, key="reach_anchor")
    anchor_id = extract_site_id(anchor_sel)
    anchor_info = get_cached_site_info(anchor_id)
    with search_col4:
        search_km = st.number_input(
            "Km",
            min_value=10,
            max_value=300,
            value=75,
            step=5,
            key="reach_nldi_search_km",
        )
    with search_col5:
        include_tributaries = st.toggle(
            "Tributaries",
            value=True,
            key="reach_include_tributaries",
        )
        find_gages = st.button("Find Reaches", width="stretch", key="reach_find_related")
    if anchor_search or anchor_state != "All States":
        st.caption(f"{len(anchor_options)} matching gages")
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
    reach_pairs = _build_recommended_reach_pairs(anchor_id, candidate_records)
    selected_pair_key = _resolve_selected_pair_key(reach_pairs, st.session_state)
    if selected_pair_key:
        st.session_state["reach_selected_pair_key"] = selected_pair_key
    selected_pair = next((pair for pair in reach_pairs if pair["key"] == selected_pair_key), None)

    if selected_pair:
        upstream_id = selected_pair["upstream_id"]
        downstream_id = selected_pair["downstream_id"]
    else:
        upstream_id = anchor_id
        downstream_id = anchor_id
    up_info = get_cached_site_info(upstream_id)
    dn_info = get_cached_site_info(downstream_id)
    estimated_reach_km = _estimate_reach_km(upstream_id, downstream_id, related_sites, anchor_id)
    manual_reach_km = float(st.session_state.get("reach_length_km", 0.0) or 0.0)
    reach_km = _resolve_reach_km(estimated_reach_km, manual_reach_km)

    reach_plot_options = {get_display_name(p): p for p in REACH_PLOTS if p in AVAILABLE_PLOTS}
    default_plots = ['reach_comparison', 'reach_index', 'seasonal_gain_loss']
    default_display = [get_display_name(p) for p in default_plots if p in reach_plot_options.values()]

    selected_display = default_display
    layout_choice = "Auto"
    dpi = 150

    candidate_col, map_col, summary_col = st.columns([1.05, 2.15, 1.0])
    with candidate_col:
        st.subheader("Candidate Reaches")
        if omitted_related_site_ids:
            st.caption(f"{len(omitted_related_site_ids)} NLDI gages hidden outside HydroPlot inventory.")
        if reach_pairs:
            pair_labels = {pair["label"]: pair["key"] for pair in reach_pairs}
            pair_label_options = list(pair_labels.keys())
            _ensure_widget_value_is_valid("reach_pair_select", pair_label_options)
            selected_key = st.session_state.get("reach_selected_pair_key")

            prev_col, count_col, next_col = st.columns([1, 1.1, 1])
            with prev_col:
                if st.button("Previous", width="stretch", disabled=len(reach_pairs) <= 1, key="reach_pair_previous"):
                    selected_key = _cycle_pair_key(reach_pairs, selected_key, -1)
                    st.session_state["reach_selected_pair_key"] = selected_key
                    st.session_state["reach_pair_select"] = _pair_label_for_key(reach_pairs, selected_key)
            with count_col:
                current_index = [pair["key"] for pair in reach_pairs].index(st.session_state["reach_selected_pair_key"]) + 1
                st.caption(f"Reach {current_index} of {len(reach_pairs)}")
            with next_col:
                if st.button("Next", width="stretch", disabled=len(reach_pairs) <= 1, key="reach_pair_next"):
                    selected_key = _cycle_pair_key(reach_pairs, selected_key, 1)
                    st.session_state["reach_selected_pair_key"] = selected_key
                    st.session_state["reach_pair_select"] = _pair_label_for_key(reach_pairs, selected_key)

            selected_pair_label = (
                st.session_state.get("reach_pair_select")
                or _pair_label_for_key(reach_pairs, st.session_state.get("reach_selected_pair_key"))
            )
            chosen_label = st.selectbox(
                "Selected candidate reach",
                pair_label_options,
                index=pair_label_options.index(selected_pair_label),
                key="reach_pair_select",
            )
            st.session_state["reach_selected_pair_key"] = pair_labels[chosen_label]
            selected_pair = next(pair for pair in reach_pairs if pair["key"] == pair_labels[chosen_label])
            upstream_id = selected_pair["upstream_id"]
            downstream_id = selected_pair["downstream_id"]
            up_info = get_cached_site_info(upstream_id)
            dn_info = get_cached_site_info(downstream_id)
            estimated_reach_km = _estimate_reach_km(upstream_id, downstream_id, related_sites, anchor_id)
            manual_reach_km = float(st.session_state.get("reach_length_km", 0.0) or 0.0)
            reach_km = _resolve_reach_km(estimated_reach_km, manual_reach_km)
            st.caption(f"{selected_pair['kind']} | {selected_pair.get('distance_km') or 'distance unknown'} km from anchor")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Reach": pair["label"],
                        "Kind": pair["kind"],
                    }
                    for pair in reach_pairs
                ]),
                width="stretch",
                hide_index=True,
                height=min(360, 38 + 34 * len(reach_pairs)),
            )
        elif discovered_related_sites:
            st.warning("NLDI found related gages, but none are available in the HydroPlot inventory.")
        else:
            st.info("Click Find Reaches to discover processable upstream/downstream candidates.")

    with map_col:
        st.subheader("Reach Map")
        _render_reach_map(up_info, dn_info, upstream_id, downstream_id, reach_km, search_km)

    with summary_col:
        st.subheader("Selected Reach")
        if upstream_id == downstream_id:
            st.warning("Choose two different gages for a reach.")
        else:
            st.metric("Reach", f"{upstream_id} -> {downstream_id}")
        if estimated_reach_km:
            st.metric("Network length", f"{estimated_reach_km:.1f} km")
        elif manual_reach_km:
            st.metric("Network length", f"{manual_reach_km:.1f} km manual")
        else:
            st.metric("Network length", "Not inferred")
        st.metric("Candidate gages", len(candidate_records))
        generate = st.button(
            "Run Analysis",
            type="primary",
            width="stretch",
            key="gen_reach",
            disabled=upstream_id == downstream_id,
        )

    with st.expander("Analysis settings", expanded=False):
        settings_col1, settings_col2, settings_col3 = st.columns([1.2, 1.7, 0.7])
        with settings_col1:
            start_date, end_date = date_range_selector("reach", default_start=date(2000, 1, 1))
        with settings_col2:
            selected_display = st.multiselect(
                "Plots",
                list(reach_plot_options.keys()),
                default=default_display,
                key="reach_plot_select"
            )
        with settings_col3:
            layout_choice = st.selectbox("Layout", ["Auto", "Vertical", "Grid 2x3"], key="reach_layout")
            dpi = st.number_input("DPI", min_value=72, max_value=300, value=150, key="reach_dpi")
        with st.expander("Plot descriptions", expanded=False):
            for display_name, plot_key in reach_plot_options.items():
                info = AVAILABLE_PLOTS.get(plot_key, {})
                desc = info.get('description', '') if isinstance(info, dict) else ''
                st.markdown(f"**{display_name}**: {desc}")
    selected_plots = [reach_plot_options[d] for d in selected_display]

    with st.expander("Candidate gage details", expanded=False):
        if omitted_related_site_ids:
            st.caption(
                f"{len(omitted_related_site_ids)} NLDI gages were hidden because they are not in the HydroPlot inventory for this app."
            )
        if related_sites:
            st.dataframe(pd.DataFrame(candidate_rows), width="stretch", hide_index=True)
        elif discovered_related_sites:
            st.warning("NLDI found related gages, but none are available in the HydroPlot inventory for this dashboard.")
        else:
            st.info("Use Find Reaches to discover likely upstream/downstream candidates for the selected anchor gage.")

    with st.expander("Advanced reach length override", expanded=False):
        manual_reach_km = st.number_input(
            "Manual reach length km",
            min_value=0.0,
            max_value=1000.0,
            value=manual_reach_km,
            step=0.1,
            help="Optional. Used only when the network length cannot be inferred.",
            key="reach_length_km",
        )
        if estimated_reach_km:
            st.caption("Network-inferred length is used for cfs/km. Manual value is ignored while an inferred length is available.")
        else:
            st.caption("Optional fallback for cfs/km when related gage distances do not define the selected reach.")

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
