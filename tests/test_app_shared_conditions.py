import pandas as pd

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


def test_get_site_condition_details_include_flow_and_rank_source(monkeypatch):
    monkeypatch.setattr(
        shared,
        "fetch_current_conditions",
        lambda site_ids: {"1": 10.0, "2": 30.0},
    )
    monkeypatch.setattr(shared, "fetch_daily_percentiles", lambda site_ids: {})

    details = shared.get_site_condition_details(["1", "2"])

    assert details["1"]["flow_cfs"] == 10.0
    assert details["1"]["percentile"] == 0.0
    assert details["1"]["source"] == "Relative live-flow rank among mapped sites"
    assert details["2"]["flow_cfs"] == 30.0
    assert details["2"]["percentile"] == 100.0


def test_get_site_conditions_prefers_seasonal_percentiles(monkeypatch):
    monkeypatch.setattr(shared, "fetch_current_conditions", lambda site_ids: {"1": 35.0})
    monkeypatch.setattr(
        shared,
        "fetch_daily_percentiles",
        lambda site_ids: {"1": {"p10": 10.0, "p25": 20.0, "p50": 30.0, "p75": 40.0, "p90": 50.0}},
    )

    conditions = shared.get_site_conditions(["1"])

    assert conditions == {"1": 50.0}


def test_get_site_condition_details_prefer_seasonal_percentiles(monkeypatch):
    monkeypatch.setattr(shared, "fetch_current_conditions", lambda site_ids: {"1": 35.0})
    monkeypatch.setattr(
        shared,
        "fetch_daily_percentiles",
        lambda site_ids: {"1": {"p10": 10.0, "p25": 20.0, "p50": 30.0, "p75": 40.0}},
    )

    details = shared.get_site_condition_details(["1"])

    assert details["1"]["flow_cfs"] == 35.0
    assert details["1"]["percentile"] == 50.0
    assert details["1"]["source"] == "USGS seasonal percentile"


def test_normalize_climate_columns_converts_daymet_names():
    raw = pd.DataFrame(
        {
            "precip_mm": [1.0, None],
            "tmin_c": [2.0, 4.0],
            "tmax_c": [8.0, 10.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )

    climate = shared.normalize_climate_columns(raw)

    assert list(climate.columns) == ["Temp_C", "Precip_mm"]
    assert climate["Temp_C"].tolist() == [5.0, 7.0]
    assert climate["Precip_mm"].tolist() == [1.0, 0.0]
    assert str(climate.index.tz) == "UTC"


def test_fetch_climate_cached_prefers_daymet_for_site_climate(monkeypatch):
    shared.fetch_climate_cached.clear()
    shared.fetch_climate_cached_result.clear()
    calls = []

    station = pd.DataFrame(
        {"Temp_C": [5.0], "Precip_mm": [0.2]},
        index=pd.date_range("2024-01-01", periods=1, freq="D"),
    )

    def fake_station(*args, **kwargs):
        calls.append("station")
        return station

    def fake_daymet(*args, **kwargs):
        calls.append("daymet")
        return pd.DataFrame(
            {"precip_mm": [1.0], "tmin_c": [2.0], "tmax_c": [8.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="D"),
        )

    monkeypatch.setattr(shared, "fetch_climate_data", fake_station)
    monkeypatch.setattr("hydrology.data.hyriver.get_daymet_climate", fake_daymet)

    climate = shared.fetch_climate_cached(47.0, -117.0, "2024-01-01", "2024-01-01", "12422500")

    assert climate["Temp_C"].tolist() == [5.0]
    assert climate["Precip_mm"].tolist() == [1.0]
    assert calls == ["daymet"]


def test_fetch_climate_cached_falls_back_to_station_data(monkeypatch):
    shared.fetch_climate_cached.clear()
    shared.fetch_climate_cached_result.clear()
    calls = []

    station = pd.DataFrame(
        {"Temp_C": [6.0], "Precip_mm": [0.4]},
        index=pd.date_range("2024-01-01", periods=1, freq="D"),
    )

    def fake_station(*args, **kwargs):
        calls.append("station")
        return station

    def fake_daymet(*args, **kwargs):
        calls.append("daymet")
        return None

    monkeypatch.setattr(shared, "fetch_climate_data", fake_station)
    monkeypatch.setattr("hydrology.data.hyriver.get_daymet_climate", fake_daymet)

    result = shared.fetch_climate_cached_result(47.0, -117.0, "2024-01-01", "2024-01-01", "12422500")

    assert result["source"] == "Meteostat"
    assert result["data"]["Temp_C"].tolist() == [6.0]
    assert result["data"]["Precip_mm"].tolist() == [0.4]
    assert calls == ["daymet", "station"]
