from hydrology.analysis.temperature_context import classify_thermal_sensitivity


def test_classify_thermal_sensitivity_high_for_wide_unshaded_low_flow_reach():
    result = classify_thermal_sensitivity(
        summer_flow_cfs=20,
        channel_width_m=12,
        canopy_cover_pct=15,
        groundwater_gain_cfs=-3,
    )

    assert result["class"] == "high"
    assert "low canopy cover" in result["drivers"]
    assert "losing reach" in result["drivers"]


def test_classify_thermal_sensitivity_lower_for_shaded_gaining_reach():
    result = classify_thermal_sensitivity(
        summer_flow_cfs=80,
        channel_width_m=4,
        canopy_cover_pct=85,
        groundwater_gain_cfs=10,
    )

    assert result["class"] in {"low", "moderate"}
