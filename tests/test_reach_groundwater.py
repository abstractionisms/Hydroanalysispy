import pandas as pd

from hydrology.analysis.reach_groundwater import classify_reach_gain_loss, summarize_reach_gain_loss


def test_summarize_reach_gain_loss_classifies_gaining_reach():
    index = pd.date_range("2021-08-01", periods=10, freq="D")
    upstream = pd.Series([100] * 10, index=index)
    downstream = pd.Series([130] * 10, index=index)

    result = summarize_reach_gain_loss(upstream, downstream, reach_km=10, drainage_area_sqmi=100)

    assert result["median_gain_cfs"] == 30
    assert result["median_gain_cfs_per_km"] == 3
    assert result["classification"] == "gaining"
    assert result["confidence"] == "high"


def test_classify_reach_gain_loss_uses_deadband():
    assert classify_reach_gain_loss(0.2, deadband_cfs=1.0) == "neutral"
    assert classify_reach_gain_loss(5.0, deadband_cfs=1.0) == "gaining"
    assert classify_reach_gain_loss(-5.0, deadband_cfs=1.0) == "losing"
