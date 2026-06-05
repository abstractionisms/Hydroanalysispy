import pandas as pd

from hydrology.app.page_modules.reach_analysis import _build_reach_summary_row


def test_build_reach_summary_row_formats_gain_loss_for_dashboard():
    index = pd.date_range("2021-08-01", periods=10, freq="D")
    upstream = pd.Series([100] * 10, index=index)
    downstream = pd.Series([85] * 10, index=index)

    row = _build_reach_summary_row("12419000", "12422500", upstream, downstream, reach_km=5)

    assert row["Reach"] == "12419000 -> 12422500"
    assert row["Class"] == "losing"
    assert row["Median gain/loss"] == "-15 cfs"
    assert row["Confidence"] == "high"
