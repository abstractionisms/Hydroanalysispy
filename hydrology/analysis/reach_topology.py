"""Reach station ordering helpers built from navigation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class ReachStation:
    """A gage positioned relative to a reach-chain origin."""

    site_id: str
    direction: str
    distance_km: float | None
    order_key: float


@dataclass(frozen=True)
class ReachChain:
    """Ordered upstream-to-downstream gage chain."""

    stations: List[ReachStation]
    status: str
    notes: List[str]


@dataclass(frozen=True)
class ReachPair:
    """Adjacent upstream/downstream gage pair."""

    upstream_site_id: str
    downstream_site_id: str
    status: str
    notes: List[str]


def classify_pair_direction(
    upstream_site_id: str,
    downstream_site_id: str,
    related_sites: Iterable[Dict],
    origin_site_id: str | None = None,
) -> str:
    """Classify whether navigation metadata supports a proposed pair order."""
    by_id = {str(site.get("site_id")): site for site in related_sites}
    upstream_meta = by_id.get(str(upstream_site_id))
    downstream_meta = by_id.get(str(downstream_site_id))

    if downstream_meta and downstream_meta.get("direction") == "downstream":
        return "ordered"
    if upstream_meta and upstream_meta.get("direction") == "upstream":
        return "ordered"
    if downstream_meta and downstream_meta.get("direction") == "upstream":
        return "reversed_or_tributary"
    if upstream_meta and upstream_meta.get("direction") == "downstream":
        return "reversed_or_tributary"
    return "unknown"


def build_reach_chain(
    selected_site_ids: Iterable[str],
    navigation_sites: Iterable[Dict],
    origin_site_id: str | None = None,
) -> ReachChain:
    """Order selected sites from upstream to downstream using navigation metadata."""
    selected = [str(site_id) for site_id in selected_site_ids]
    by_id = {str(site.get("site_id")): site for site in navigation_sites}
    stations: List[ReachStation] = []
    notes: List[str] = []

    for site_id in selected:
        if site_id == origin_site_id:
            stations.append(ReachStation(site_id, "origin", 0.0, 0.0))
            continue

        meta = by_id.get(site_id)
        if not meta:
            notes.append(f"{site_id} not found in navigation metadata")
            stations.append(ReachStation(site_id, "unknown", None, float("inf")))
            continue

        direction = str(meta.get("direction", "unknown"))
        distance = meta.get("distance_km")
        signed_distance = float(distance) if distance is not None else float("inf")
        if direction == "upstream":
            signed_distance *= -1
        elif direction != "downstream":
            signed_distance = float("inf")

        stations.append(ReachStation(site_id, direction, distance, signed_distance))

    stations.sort(key=lambda station: station.order_key)
    verified_directions = {"upstream", "origin", "downstream"}
    status = (
        "verified"
        if stations and all(station.direction in verified_directions for station in stations)
        else "unverified"
    )
    if len(stations) < 2:
        status = "invalid"
        notes.append("at least two stations are required to define a reach chain")

    return ReachChain(stations, status, notes)


def derive_adjacent_reaches(chain: ReachChain) -> List[ReachPair]:
    """Convert an ordered gage chain into adjacent reach segments."""
    reaches: List[ReachPair] = []
    for upstream, downstream in zip(chain.stations, chain.stations[1:]):
        reaches.append(
            ReachPair(
                upstream.site_id,
                downstream.site_id,
                chain.status,
                chain.notes.copy(),
            )
        )
    return reaches


def validate_reach_pair(
    upstream_site_id: str,
    downstream_site_id: str,
    related_sites: Iterable[Dict],
) -> ReachPair:
    """Validate a proposed upstream/downstream gage pair."""
    if upstream_site_id == downstream_site_id:
        return ReachPair(
            upstream_site_id,
            downstream_site_id,
            "invalid",
            ["same station selected twice"],
        )

    direction = classify_pair_direction(upstream_site_id, downstream_site_id, related_sites)
    if direction == "ordered":
        return ReachPair(
            upstream_site_id,
            downstream_site_id,
            "verified",
            ["NLDI/navigation metadata supports station order"],
        )
    if direction == "reversed_or_tributary":
        return ReachPair(
            upstream_site_id,
            downstream_site_id,
            "unverified",
            ["metadata suggests reversed order or tributary/diversion relationship"],
        )
    return ReachPair(
        upstream_site_id,
        downstream_site_id,
        "unverified",
        ["station order not verified by navigation metadata"],
    )
