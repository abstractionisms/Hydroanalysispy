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
import pandas as pd


def test_summarize_baseflow_methods_returns_dashboard_fields():
    from hydrology.app.page_modules.indicators import _summarize_baseflow_methods

    flow = pd.Series([100, 110, 130, 120, 105, 95, 90, 100], index=pd.date_range("2024-01-01", periods=8))

    summary = _summarize_baseflow_methods(flow)

    assert {"Lyne-Hollick BFI", "Eckhardt BFI", "Difference", "Agreement"}.issubset(summary)
