import numpy as np
import pandas as pd

from hydrology.analysis.signatures import compute_hydrologic_signatures


def test_compute_hydrologic_signatures_returns_core_metrics():
    index = pd.date_range("2020-01-01", periods=366, freq="D")
    flow = pd.Series(100 + 30 * np.sin(np.linspace(0, 2 * np.pi, 366)), index=index)

    result = compute_hydrologic_signatures(flow)

    assert result["n_days"] == 366
    assert result["mean_flow"] > 0
    assert result["q05"] > result["q50"] > result["q95"]
    assert 0 <= result["baseflow_index_lh"] <= 1
    assert result["richards_baker_flashiness"] >= 0
    assert 1 <= result["peak_month"] <= 12


def test_compute_hydrologic_signatures_handles_empty_series():
    result = compute_hydrologic_signatures(pd.Series(dtype=float))

    assert result == {}
