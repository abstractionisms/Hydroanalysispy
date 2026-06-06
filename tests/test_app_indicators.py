import pandas as pd

from hydrology.app.page_modules import indicators


def test_fetch_precip_data_uses_meteostat_precip_mm_fallback(monkeypatch):
    calls = []

    def no_daymet(site_id, start_date, end_date, variables):
        return None

    def fake_fetch_climate_data(latitude, longitude, start_date, end_date, include_temp, include_precip):
        calls.append(
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "include_temp": include_temp,
                "include_precip": include_precip,
            }
        )
        return pd.DataFrame(
            {"Precip_mm": [1.2, 0.0, 4.5]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

    monkeypatch.setattr("hydrology.data.hyriver.get_daymet_climate", no_daymet)
    monkeypatch.setattr("hydrology.data.climate.fetch_climate_data", fake_fetch_climate_data)

    precip = indicators._fetch_precip_data(
        "12422500",
        "47.6593",
        "-117.4491",
        "2024-01-01",
        "2024-01-03",
    )

    assert precip.tolist() == [1.2, 0.0, 4.5]
    assert calls == [
        {
            "latitude": 47.6593,
            "longitude": -117.4491,
            "start_date": pd.Timestamp("2024-01-01"),
            "end_date": pd.Timestamp("2024-01-03"),
            "include_temp": False,
            "include_precip": True,
        }
    ]


def test_fetch_precip_data_result_reports_daymet_source(monkeypatch):
    def fake_daymet(site_id, start_date, end_date, variables):
        return pd.DataFrame(
            {"precip_mm": [2.0, 0.0, 1.5]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

    monkeypatch.setattr("hydrology.data.hyriver.get_daymet_climate", fake_daymet)

    result = indicators._fetch_precip_data_result(
        "12422500",
        "47.6593",
        "-117.4491",
        "2024-01-01",
        "2024-01-03",
    )

    assert result["source"] == "Daymet"
    assert result["n_days"] == 3
    assert result["precip"].tolist() == [2.0, 0.0, 1.5]


def test_fetch_precip_data_result_reports_unavailable_without_coordinates(monkeypatch):
    def no_daymet(site_id, start_date, end_date, variables):
        return None

    monkeypatch.setattr("hydrology.data.hyriver.get_daymet_climate", no_daymet)

    result = indicators._fetch_precip_data_result(
        "12422500",
        None,
        None,
        "2024-01-01",
        "2024-01-03",
    )

    assert result["precip"] is None
    assert result["source"] == "Unavailable"
    assert "missing site coordinates" in result["message"]


def test_spi_readiness_rows_summarize_source_and_record_length():
    rows = indicators._spi_readiness_rows(
        {"source": "Daymet", "n_days": 3650, "message": "Loaded precipitation from Daymet."}
    )

    assert rows == [
        {"Item": "Precipitation source", "Value": "Daymet"},
        {"Item": "Daily precipitation records", "Value": "3,650"},
        {"Item": "Status", "Value": "Loaded precipitation from Daymet."},
    ]
import pandas as pd


def test_summarize_baseflow_methods_returns_dashboard_fields():
    from hydrology.app.page_modules.indicators import _summarize_baseflow_methods

    flow = pd.Series([100, 110, 130, 120, 105, 95, 90, 100], index=pd.date_range("2024-01-01", periods=8))

    summary = _summarize_baseflow_methods(flow)

    assert {"Lyne-Hollick BFI", "Eckhardt BFI", "Difference", "Agreement"}.issubset(summary)
