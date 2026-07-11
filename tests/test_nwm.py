import pandas as pd

from hydrology.data import nwm


def test_compare_nwm_usgs_returns_plot_data(monkeypatch):
    nwm_index = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    nwm_df = pd.DataFrame({"streamflow_cfs": [100.0, 120.0, 130.0, 140.0]}, index=nwm_index)
    usgs_df = pd.DataFrame(
        {"value": [95.0, 125.0, 128.0, 145.0]},
        index=pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
    )

    class FakeClient:
        def get_analysis(self, site_id, start_date, end_date):
            return nwm_df

    monkeypatch.setattr(nwm, "NWMClient", FakeClient)
    monkeypatch.setattr(nwm, "fetch_daily_values", lambda *args, **kwargs: usgs_df)

    result = nwm.compare_nwm_usgs("12422500", "2024-01-01", "2024-01-04")

    assert result is not None
    assert result.n_observations == 4
    assert result.nwm_data is not None
    assert result.nwm_data["streamflow_cfs"].tolist() == [100.0, 120.0, 130.0, 140.0]
