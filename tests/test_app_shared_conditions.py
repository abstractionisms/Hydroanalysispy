from hydrology.app import shared


def test_get_site_conditions_falls_back_to_live_flow_rank(monkeypatch):
    monkeypatch.setattr(
        shared,
        "fetch_current_conditions",
        lambda site_ids: {"1": 10.0, "2": 30.0, "3": 20.0},
    )
    monkeypatch.setattr(shared, "fetch_daily_percentiles", lambda site_ids: {})

    conditions = shared.get_site_conditions(["1", "2", "3"])

    assert conditions == {"1": 0.0, "3": 50.0, "2": 100.0}


def test_get_site_conditions_prefers_seasonal_percentiles(monkeypatch):
    monkeypatch.setattr(shared, "fetch_current_conditions", lambda site_ids: {"1": 35.0})
    monkeypatch.setattr(
        shared,
        "fetch_daily_percentiles",
        lambda site_ids: {"1": {"p10": 10.0, "p25": 20.0, "p50": 30.0, "p75": 40.0, "p90": 50.0}},
    )

    conditions = shared.get_site_conditions(["1"])

    assert conditions == {"1": 50.0}
