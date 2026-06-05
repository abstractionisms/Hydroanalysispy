import pandas as pd

from hydrology.analysis.changepoints import pettitt_test
from hydrology.analysis.trends import mann_kendall_test


def test_pettitt_detects_known_step_change():
    index = pd.Index(range(1900, 1940), name="year")
    values = pd.Series([10] * 20 + [30] * 20, index=index)

    result = pettitt_test(values)

    assert result["change_index"] in {19, 20}
    assert result["change_point"] in {1919, 1920}
    assert result["p_value"] < 0.05
    assert result["mean_before"] < result["mean_after"]


def test_mann_kendall_exposes_sens_slope_when_available():
    values = pd.Series([1, 2, 3, 4, 5, 6])

    result = mann_kendall_test(values)

    if result is not None:
        assert "sens_slope" in result
        assert result["sens_slope"] > 0
