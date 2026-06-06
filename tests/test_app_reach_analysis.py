import pandas as pd

from hydrology.app.page_modules.reach_analysis import (
    _build_reach_interpretation,
    _build_recommended_reach_pairs,
    _build_reach_summary_row,
    _build_reach_candidate_options,
    _candidate_index_for_site,
    _candidate_label_for_site,
    _default_candidate_label,
    _estimate_reach_km,
    _filter_related_sites_to_inventory,
    _format_reach_chain,
    _format_related_site_rows,
    _flowline_distance_km,
    _flowline_style,
    _map_bounds_for_reach,
    _pair_key,
    _reach_map_component_key,
    _resolve_reach_km,
    _selectbox_kwargs_for_state,
    _selected_candidate_site_id,
)


def test_build_reach_summary_row_formats_gain_loss_for_dashboard():
    index = pd.date_range("2021-08-01", periods=10, freq="D")
    upstream = pd.Series([100] * 10, index=index)
    downstream = pd.Series([85] * 10, index=index)

    row = _build_reach_summary_row("up", "down", upstream, downstream, reach_km=5)

    assert row["Reach"] == "up -> down"
    assert row["Class"] == "losing"
    assert row["Median gain/loss"] == "-15 cfs"
    assert row["Gain/loss per km"] == "-3.0 cfs/km"
    assert row["Confidence"] == "high"


def test_build_reach_summary_row_explains_missing_reach_length():
    index = pd.date_range("2021-08-01", periods=10, freq="D")
    upstream = pd.Series([100] * 10, index=index)
    downstream = pd.Series([115] * 10, index=index)

    row = _build_reach_summary_row("up", "down", upstream, downstream)

    assert row["Gain/loss per km"] == "Add reach length"


def test_format_reach_chain_shows_upstream_to_downstream_order():
    rows = _format_reach_chain(["headwater", "mid", "lower"])

    assert rows == [
        {"Reach": "headwater -> mid", "Order": 1},
        {"Reach": "mid -> lower", "Order": 2},
    ]


def test_estimate_reach_km_from_navigation_distance_difference():
    related_sites = [
        {"site_id": "up", "direction": "upstream", "distance_km": 12.0},
        {"site_id": "down", "direction": "downstream", "distance_km": 8.0},
    ]

    assert _estimate_reach_km("up", "origin", related_sites, "origin") == 12.0
    assert _estimate_reach_km("origin", "down", related_sites, "origin") == 8.0
    assert _estimate_reach_km("up", "down", related_sites, "origin") == 20.0


def test_resolve_reach_km_prefers_estimated_network_length():
    assert _resolve_reach_km(12.3, 20.0) == 12.3


def test_resolve_reach_km_uses_manual_override_when_estimate_missing():
    assert _resolve_reach_km(None, 20.0) == 20.0


def test_resolve_reach_km_allows_missing_length():
    assert _resolve_reach_km(None, 0.0) is None


def test_flowline_distance_uses_reach_length_with_buffer():
    assert _flowline_distance_km(12.0, 75.0) == 18.0


def test_flowline_distance_falls_back_to_search_distance():
    assert _flowline_distance_km(None, 75.0) == 75.0


def test_flowline_style_highlights_selected_network():
    assert _flowline_style(selected=True)["weight"] > _flowline_style(selected=False)["weight"]


def test_map_bounds_focuses_on_selected_gage_markers_not_flowline_extent():
    class FakeFlowlines:
        total_bounds = [-125.0, 40.0, -110.0, 50.0]
        empty = False

    bounds = _map_bounds_for_reach(
        FakeFlowlines(),
        context_flowlines=object(),
        upstream_lat=10.0,
        upstream_lon=20.0,
        downstream_lat=11.0,
        downstream_lon=21.0,
    )

    assert bounds == [[9.75, 19.75], [11.25, 21.25]]


def test_map_bounds_pads_close_gage_markers():
    bounds = _map_bounds_for_reach(
        selected_flowlines=None,
        context_flowlines=None,
        upstream_lat=10.0,
        upstream_lon=20.0,
        downstream_lat=10.001,
        downstream_lon=20.001,
    )

    assert bounds == [[9.99, 19.99], [10.011, 20.011]]


def test_build_reach_interpretation_summarizes_gaining_reach():
    row = {
        "Reach": "up -> down",
        "Class": "gaining",
        "Median gain/loss": "342 cfs",
        "Low-flow gain/loss": "339 cfs",
        "Gain/loss per km": "12.0 cfs/km",
        "Confidence": "moderate",
    }

    summary = _build_reach_interpretation(row, reach_km=28.5, length_source="network")

    assert summary["Finding"] == "Gaining reach"
    assert "downstream flow is higher" in summary["Interpretation"]
    assert summary["Reach length"] == "28.5 km, network inferred"
    assert summary["Review"] == "Screening result; check tributaries, diversions, withdrawals, and data overlap."


def test_build_reach_interpretation_explains_missing_length():
    row = {
        "Reach": "up -> down",
        "Class": "insufficient_data",
        "Median gain/loss": "N/A",
        "Low-flow gain/loss": "N/A",
        "Gain/loss per km": "Add reach length",
        "Confidence": "none",
    }

    summary = _build_reach_interpretation(row, reach_km=None, length_source="missing")

    assert summary["Finding"] == "Not enough paired data"
    assert summary["Reach length"] == "Not inferred"
    assert "cfs/km is unavailable" in summary["Interpretation"]


def test_format_related_site_rows_makes_station_choices_legible():
    rows = _format_related_site_rows(
        "anchor",
        [
            {
                "site_id": "up",
                "name": "Upstream mainstem gage",
                "direction": "upstream",
                "distance_km": 29.4,
            },
            {
                "site_id": "trib",
                "name": "Tributary gage",
                "direction": "upstream",
                "navigation_mode": "upstream_trib",
                "distance_km": 12.0,
            },
            {
                "site_id": "down",
                "name": "Downstream gage",
                "direction": "downstream",
                "distance_km": 18.1,
            },
        ],
    )

    assert rows[0]["Station"] == "anchor"
    assert rows[0]["Position"] == "Anchor"
    assert rows[1]["Position"] == "Upstream"
    assert rows[1]["Distance from anchor"] == "29.4 km"
    assert rows[2]["Position"] == "Tributary"
    assert rows[3]["Position"] == "Downstream"


def test_build_reach_candidate_options_groups_both_directions():
    candidates = _build_reach_candidate_options(
        "anchor",
        "Anchor gage",
        [
            {
                "site_id": "up",
                "name": "Upstream gage",
                "direction": "upstream",
                "distance_km": 5.2,
            },
            {
                "site_id": "down",
                "name": "Downstream gage",
                "direction": "downstream",
                "distance_km": 7.8,
            },
        ],
    )

    labels = [candidate["label"] for candidate in candidates]

    assert labels[0].startswith("Anchor | anchor")
    assert labels[1].startswith("Upstream | up")
    assert labels[2].startswith("Downstream | down")
    assert candidates[1]["site_id"] == "up"
    assert candidates[2]["site_id"] == "down"


def test_filter_related_sites_to_inventory_keeps_only_processable_gages():
    inventory = pd.DataFrame(
        {
            "site_id": ["anchor", "usable"],
            "description": ["Anchor gage", "Usable related gage"],
        }
    )
    related_sites = [
        {"site_id": "usable", "direction": "upstream", "distance_km": 4.0},
        {"site_id": "outside", "direction": "downstream", "distance_km": 8.0},
    ]

    filtered, omitted = _filter_related_sites_to_inventory("anchor", related_sites, inventory)

    assert [site["site_id"] for site in filtered] == ["usable"]
    assert omitted == ["outside"]


def test_filter_related_sites_to_inventory_keeps_anchor_even_if_not_related():
    inventory = pd.DataFrame({"site_id": ["anchor"]})

    filtered, omitted = _filter_related_sites_to_inventory("anchor", [], inventory)

    assert filtered == []
    assert omitted == []


def test_reach_map_component_key_changes_with_selected_pair_and_bounds():
    first = _reach_map_component_key("up", "down", [[10.0, 20.0], [11.0, 21.0]])
    second = _reach_map_component_key("up", "other", [[10.0, 20.0], [11.0, 21.0]])

    assert first != second
    assert first.startswith("reach_map_up_down_")


def test_selectbox_kwargs_omit_index_when_session_state_has_widget_key():
    kwargs = _selectbox_kwargs_for_state("reach_upstream_choice", 2, {"reach_upstream_choice": "label"})

    assert kwargs == {"key": "reach_upstream_choice"}


def test_selectbox_kwargs_include_index_for_initial_render():
    kwargs = _selectbox_kwargs_for_state("reach_upstream_choice", 2, {})

    assert kwargs == {"key": "reach_upstream_choice", "index": 2}


def test_candidate_index_for_site_prefers_selected_site():
    candidates = [
        {"site_id": "anchor", "position": "Anchor"},
        {"site_id": "up", "position": "Upstream"},
        {"site_id": "down", "position": "Downstream"},
    ]

    assert _candidate_index_for_site(candidates, "down", {"Upstream"}) == 2


def test_candidate_label_for_site_returns_matching_dropdown_label():
    candidates = [
        {"site_id": "anchor", "label": "Anchor | anchor | 0.0 km | Anchor"},
        {"site_id": "up", "label": "Upstream | up | 5.0 km | Upstream"},
    ]

    assert _candidate_label_for_site(candidates, "up") == "Upstream | up | 5.0 km | Upstream"


def test_default_candidate_label_uses_preferred_site_before_role_fallback():
    candidates = [
        {"site_id": "anchor", "label": "Anchor label", "position": "Anchor"},
        {"site_id": "up", "label": "Upstream label", "position": "Upstream"},
        {"site_id": "down", "label": "Downstream label", "position": "Downstream"},
    ]

    assert _default_candidate_label(candidates, "down", {"Upstream"}) == "Downstream label"


def test_default_candidate_label_falls_back_to_role_label():
    candidates = [
        {"site_id": "anchor", "label": "Anchor label", "position": "Anchor"},
        {"site_id": "up", "label": "Upstream label", "position": "Upstream"},
    ]

    assert _default_candidate_label(candidates, None, {"Upstream"}) == "Upstream label"


def test_candidate_index_for_site_falls_back_to_role():
    candidates = [
        {"site_id": "anchor", "position": "Anchor"},
        {"site_id": "up", "position": "Upstream"},
        {"site_id": "down", "position": "Downstream"},
    ]

    assert _candidate_index_for_site(candidates, None, {"Upstream"}) == 1


def test_selected_candidate_site_id_reads_single_selected_table_row():
    candidate_rows = [
        {"Station": "anchor"},
        {"Station": "up"},
        {"Station": "down"},
    ]
    selection_state = {"selection": {"rows": [2]}}

    assert _selected_candidate_site_id(candidate_rows, selection_state) == "down"


def test_build_recommended_reach_pairs_prefers_mainstem_pairs():
    candidates = [
        {"site_id": "anchor", "position": "Anchor", "distance_km": 0.0, "label": "Anchor | anchor | 0.0 km | Anchor"},
        {"site_id": "up", "position": "Upstream", "distance_km": 5.0, "label": "Upstream | up | 5.0 km | Upstream"},
        {"site_id": "trib", "position": "Tributary", "distance_km": 3.0, "label": "Tributary | trib | 3.0 km | Tributary"},
        {"site_id": "down", "position": "Downstream", "distance_km": 8.0, "label": "Downstream | down | 8.0 km | Downstream"},
    ]

    pairs = _build_recommended_reach_pairs("anchor", candidates, max_pairs=5)

    assert pairs[0]["upstream_id"] == "up"
    assert pairs[0]["downstream_id"] == "anchor"
    assert pairs[1]["upstream_id"] == "anchor"
    assert pairs[1]["downstream_id"] == "down"
    assert all(pair["upstream_id"] != pair["downstream_id"] for pair in pairs)
    assert any(pair["kind"] == "tributary context" for pair in pairs)


def test_pair_key_is_stable_and_readable():
    assert _pair_key("12419000", "12422000") == "12419000__12422000"
