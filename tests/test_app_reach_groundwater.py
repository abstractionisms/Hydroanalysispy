import pandas as pd

from hydrology.app.page_modules.reach_analysis import (
    _build_reach_summary_row,
    _build_reach_candidate_options,
    _estimate_reach_km,
    _format_reach_chain,
    _format_related_site_rows,
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


def test_format_related_site_rows_makes_station_choices_legible():
    rows = _format_related_site_rows(
        "anchor",
        [
            {
                "site_id": "up",
                "name": "Upstream mainstem gauge",
                "direction": "upstream",
                "distance_km": 29.4,
            },
            {
                "site_id": "trib",
                "name": "Tributary gauge",
                "direction": "upstream",
                "navigation_mode": "upstream_trib",
                "distance_km": 12.0,
            },
            {
                "site_id": "down",
                "name": "Downstream gauge",
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
        "Anchor gauge",
        [
            {
                "site_id": "up",
                "name": "Upstream gauge",
                "direction": "upstream",
                "distance_km": 5.2,
            },
            {
                "site_id": "down",
                "name": "Downstream gauge",
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
