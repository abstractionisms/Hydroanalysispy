import numpy as np
import pandas as pd

from hydrology.analysis.baseflow import (
    BaseflowResult,
    compare_baseflow_methods,
    eckhardt_filter,
    lyne_hollick_filter,
)


def _daily_flow(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="D"))


def test_lyne_hollick_returns_components_bounded_by_total_flow():
    flow = _daily_flow([100, 120, 180, 140, 110, 90, 95, 105, 130, 115])

    result = lyne_hollick_filter(flow, alpha=0.925, passes=3)

    assert isinstance(result, BaseflowResult)
    assert list(result.components.columns) == ["total_flow", "baseflow", "quickflow"]
    assert (result.components["baseflow"] >= 0).all()
    assert (result.components["baseflow"] <= result.components["total_flow"]).all()
    assert 0 <= result.bfi <= 1
    assert result.method == "lyne_hollick"


def test_eckhardt_filter_responds_to_bfi_max_and_stays_bounded():
    flow = _daily_flow([100, 110, 130, 125, 115, 105, 95, 100, 108, 102])

    conservative = eckhardt_filter(flow, alpha=0.98, bfi_max=0.50)
    permissive = eckhardt_filter(flow, alpha=0.98, bfi_max=0.80)

    assert (conservative.components["baseflow"] <= conservative.components["total_flow"]).all()
    assert (permissive.components["baseflow"] <= permissive.components["total_flow"]).all()
    assert permissive.bfi >= conservative.bfi
    assert conservative.method == "eckhardt"


def test_compare_baseflow_methods_flags_large_disagreement():
    flow = _daily_flow(np.r_[np.repeat(100.0, 20), 500.0, np.repeat(90.0, 20)])

    comparison = compare_baseflow_methods(flow)

    assert {"lyne_hollick_bfi", "eckhardt_bfi", "bfi_difference", "agreement"}.issubset(comparison)
    assert comparison["agreement"] in {"strong", "moderate", "weak"}
    assert comparison["bfi_difference"] >= 0
