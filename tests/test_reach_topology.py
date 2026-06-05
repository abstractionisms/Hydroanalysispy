from hydrology.analysis.reach_topology import (
    ReachPair,
    build_reach_chain,
    classify_pair_direction,
    derive_adjacent_reaches,
    validate_reach_pair,
)


def test_classify_pair_direction_accepts_downstream_metadata():
    sites = [
        {"site_id": "up", "direction": "upstream", "distance_km": 8.0},
        {"site_id": "down", "direction": "downstream", "distance_km": 12.0},
    ]

    assert classify_pair_direction("up", "down", sites, origin_site_id="origin") == "ordered"


def test_validate_reach_pair_flags_same_station():
    pair = validate_reach_pair("12422500", "12422500", related_sites=[])

    assert pair.status == "invalid"
    assert "same station" in pair.notes[0]


def test_validate_reach_pair_flags_unverified_direction():
    pair = validate_reach_pair(
        "12422500",
        "12424000",
        related_sites=[{"site_id": "12424000", "direction": "upstream"}],
    )

    assert isinstance(pair, ReachPair)
    assert pair.status == "unverified"


def test_build_reach_chain_orders_sites_from_upstream_to_downstream():
    selected = ["A", "B", "C"]
    navigation_sites = [
        {"site_id": "C", "direction": "downstream", "distance_km": 20.0},
        {"site_id": "A", "direction": "upstream", "distance_km": 15.0},
        {"site_id": "B", "direction": "upstream", "distance_km": 5.0},
    ]

    chain = build_reach_chain(selected, navigation_sites, origin_site_id="origin")

    assert [station.site_id for station in chain.stations] == ["A", "B", "C"]
    assert chain.status == "verified"


def test_derive_adjacent_reaches_returns_continuum_pairs():
    selected = ["A", "B", "C"]
    navigation_sites = [
        {"site_id": "A", "direction": "upstream", "distance_km": 15.0},
        {"site_id": "B", "direction": "upstream", "distance_km": 5.0},
        {"site_id": "C", "direction": "downstream", "distance_km": 20.0},
    ]
    chain = build_reach_chain(selected, navigation_sites, origin_site_id="origin")

    reaches = derive_adjacent_reaches(chain)

    assert [(reach.upstream_site_id, reach.downstream_site_id) for reach in reaches] == [("A", "B"), ("B", "C")]
