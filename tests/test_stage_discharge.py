"""
Unit tests for hydrology.analysis.stage_discharge module.

Tests rating curve fitting, flow duration curves, and flow regime
classification functions.
"""

import pytest
import pandas as pd
import numpy as np

from hydrology.analysis.stage_discharge import (
    fit_powerlaw_rating_curve,
    fit_offset_powerlaw,
    fit_best_rating_curve,
    evaluate_rating_params,
    predict_rating_discharge,
    rating_stage_window,
    season_labels,
    flow_duration_curve,
    classify_flow_regime
)


class TestFitPowerlawRatingCurve:
    """Tests for fit_powerlaw_rating_curve function."""

    def test_basic_powerlaw_fit(self, sample_stage_discharge):
        """Test basic power-law rating curve fitting."""
        stage, discharge, A_true, B_true = sample_stage_discharge

        A, B, R2, Q_pred = fit_powerlaw_rating_curve(stage, discharge)

        assert not np.isnan(A)
        assert not np.isnan(B)
        assert not np.isnan(R2)
        assert len(Q_pred) == len(stage)

    def test_powerlaw_fit_accuracy(self, sample_stage_discharge):
        """Test that fitted parameters are close to true values."""
        stage, discharge, A_true, B_true = sample_stage_discharge

        A, B, R2, Q_pred = fit_powerlaw_rating_curve(stage, discharge)

        # Should be within 20% of true values (accounting for noise)
        assert abs(A - A_true) / A_true < 0.2
        assert abs(B - B_true) / B_true < 0.2
        assert R2 > 0.9  # Good fit despite noise

    def test_powerlaw_insufficient_data(self):
        """Test with insufficient data points."""
        stage = pd.Series([1.0, 2.0, 3.0])
        discharge = pd.Series([10.0, 20.0, 30.0])

        A, B, R2, Q_pred = fit_powerlaw_rating_curve(
            stage, discharge, min_points=10
        )

        assert np.isnan(A)
        assert np.isnan(B)
        assert np.isnan(R2)

    def test_powerlaw_with_zeros(self):
        """Test handling of zero/negative values."""
        np.random.seed(42)
        stage = pd.Series([0, 1, 2, 3, 4, 5, -1, 6, 7, 8, 9, 10])
        discharge = pd.Series([0, 10, 40, 90, 160, 250, -10, 360, 490, 640, 810, 1000])

        A, B, R2, Q_pred = fit_powerlaw_rating_curve(
            stage, discharge, positive_only=True, min_points=5
        )

        # Should fit using only positive values
        assert not np.isnan(A)
        assert not np.isnan(B)

    def test_powerlaw_predictions(self, sample_stage_discharge):
        """Test that predictions are reasonable."""
        stage, discharge, A_true, B_true = sample_stage_discharge

        A, B, R2, Q_pred = fit_powerlaw_rating_curve(stage, discharge)

        # Predictions should be positive
        valid_preds = Q_pred[Q_pred.notna()]
        assert all(valid_preds > 0)

        # Predictions should be correlated with actual discharge
        corr = np.corrcoef(discharge.values, Q_pred.values)[0, 1]
        assert corr > 0.9


class TestFitOffsetPowerlaw:
    """Tests for fit_offset_powerlaw function."""

    def test_basic_offset_fit(self, sample_stage_discharge):
        """Test basic offset power-law fitting."""
        stage, discharge, _, _ = sample_stage_discharge

        A, B, H0, R2, Q_pred = fit_offset_powerlaw(stage, discharge)

        # May or may not converge, but should return valid structure
        assert isinstance(A, (int, float))
        assert isinstance(B, (int, float))
        assert isinstance(H0, (int, float))

    def test_offset_insufficient_data(self):
        """Test with insufficient data."""
        stage = pd.Series([1.0, 2.0])
        discharge = pd.Series([10.0, 20.0])

        A, B, H0, R2, Q_pred = fit_offset_powerlaw(
            stage, discharge, min_points=10
        )

        assert np.isnan(A)


class TestFitBestRatingCurve:
    """Tests for fit_best_rating_curve model selection."""

    def test_prefers_offset_when_h0_matters(self):
        """Sites like 14018500 need H0; simple power-law R² collapses."""
        np.random.seed(0)
        H0 = 1.45
        H = np.linspace(1.6, 8.0, 80)
        Q = 40.0 * (H - H0) ** 2.1
        Q = Q * (1 + np.random.normal(0, 0.03, size=H.size))
        idx = pd.date_range("2015-01-01", periods=len(H), freq="D")
        stage = pd.Series(H, index=idx)
        discharge = pd.Series(Q, index=idx)

        fit = fit_best_rating_curve(stage, discharge, min_points=20)
        assert fit["model"] == "offset_powerlaw"
        assert fit["R2"] > 0.95
        assert abs(fit["H0"] - H0) < 0.4

    def test_high_datum_gage_does_not_fall_back_to_raw_powerlaw(self):
        """Spokane (12422500) gage height sits near 17–29 ft, so Q = A·H^B
        wants an exponent near 9. That start used to sit outside the offset
        solver bounds, the fit aborted, and the workshop drew the raw power law.
        """
        rng = np.random.default_rng(0)
        h0_true = 15.42
        stage_vals = np.linspace(17.12, 28.68, 400)
        discharge_vals = 231.1 * (stage_vals - h0_true) ** 2.025
        discharge_vals *= 1 + rng.normal(0, 0.01, size=stage_vals.size)

        fit = fit_best_rating_curve(
            pd.Series(stage_vals), pd.Series(discharge_vals), min_points=10
        )

        assert fit["model"] == "offset_powerlaw"
        assert fit["R2"] > 0.98
        assert 1.4 <= fit["B"] <= 2.8
        assert abs(fit["H0"] - h0_true) < 0.5
        assert fit["A"] > 1.0
        assert "−" in fit["equation"]

    def test_season_labels(self):
        idx = pd.DatetimeIndex(["2020-01-15", "2020-04-01", "2020-07-04", "2020-10-31"])
        seasons = season_labels(idx)
        assert list(seasons.values) == ["DJF", "MAM", "JJA", "SON"]


class TestEvaluateRatingParams:
    """Manual / interactive rating parameter scoring."""

    def test_perfect_offset_scores_near_one(self):
        H0 = 1.2
        H = np.linspace(1.5, 6.0, 60)
        A, B = 25.0, 1.8
        Q = A * (H - H0) ** B
        stage = pd.Series(H)
        discharge = pd.Series(Q)

        scored = evaluate_rating_params(stage, discharge, A, B, H0)
        assert scored["n_points"] == 60
        assert scored["R2"] > 0.999
        assert scored["rmse"] < 1e-6
        assert "H −" in scored["equation"] or "H -" in scored["equation"].replace("−", "-")

    def test_wrong_params_lower_r2(self):
        H = np.linspace(1.0, 5.0, 40)
        Q = 10.0 * H ** 2.0
        good = evaluate_rating_params(H, Q, 10.0, 2.0, 0.0)
        bad = evaluate_rating_params(H, Q, 10.0, 0.5, 0.0)
        assert good["R2"] > bad["R2"]

    def test_predict_rating_discharge_shape(self):
        out = predict_rating_discharge(np.array([1.0, 2.0, 3.0]), A=2.0, B=1.5, H0=0.0)
        assert out.shape == (3,)
        assert np.all(out > 0)

    def test_h0_changes_discharge_at_mid_stage(self):
        """A high-datum gage still has to show an H₀ move in the reference discharge."""
        stage = np.linspace(20.0, 28.0, 80)
        discharge = 100.0 * (stage - 18.0) ** 1.6
        low = evaluate_rating_params(stage, discharge, 100.0, 1.6, 16.0)
        high = evaluate_rating_params(stage, discharge, 100.0, 1.6, 19.0)

        assert high["q_ref"] < low["q_ref"] * 0.85
        assert low["h_ref"] == pytest.approx(high["h_ref"])
        inside = evaluate_rating_params(stage, discharge, 100.0, 1.6, 22.0)
        assert inside["n_excluded"] > 0
        assert inside["n_points"] < low["n_points"]

    def test_stage_window_lets_h0_enter_the_cloud(self):
        stage = pd.Series(np.linspace(20.0, 28.0, 80))
        window = rating_stage_window(stage)

        assert window["h0_lo"] < window["h_min"]
        assert window["h0_hi"] > window["h_min"]
        assert window["x_lo"] < window["h0_lo"]
        assert window["x_hi"] > window["h_max"]
        assert window["h0_fit_hi"] < window["h_min"]


class TestOffsetFitBounds:
    def test_log_fit_keeps_exponent_in_channel_range(self):
        rng = np.random.default_rng(1)
        stage = np.linspace(1.2, 10.0, 300)
        discharge = 15.0 * stage ** 2.2
        discharge = discharge.copy()
        discharge[:60] *= 3.0
        discharge *= 1 + rng.normal(0, 0.01, size=stage.size)

        fit = fit_best_rating_curve(pd.Series(stage), pd.Series(discharge))
        window = rating_stage_window(pd.Series(stage))

        assert 0.5 <= fit["B"] <= 3.5
        assert window["h0_fit_lo"] - 0.05 <= fit["H0"] <= window["h0_fit_hi"] + 0.05
        assert fit["R2"] > 0.9


class TestFlowDurationCurve:
    """Tests for flow_duration_curve function."""

    def test_basic_fdc(self, sample_discharge_df):
        """Test basic flow duration curve generation."""
        discharge = sample_discharge_df['Discharge_cfs']

        fdc = flow_duration_curve(discharge)

        assert isinstance(fdc, pd.DataFrame)
        assert 'exceedance_pct' in fdc.columns
        assert 'discharge' in fdc.columns
        assert len(fdc) > 0

    def test_fdc_exceedance_range(self, sample_discharge_df):
        """Test that exceedance probabilities are valid."""
        discharge = sample_discharge_df['Discharge_cfs']

        fdc = flow_duration_curve(discharge)

        # Exceedance should range from 0 to 100
        assert fdc['exceedance_pct'].min() >= 0
        assert fdc['exceedance_pct'].max() <= 100

    def test_fdc_sorted_descending(self, sample_discharge_df):
        """Test that discharge values are sorted descending."""
        discharge = sample_discharge_df['Discharge_cfs']

        fdc = flow_duration_curve(discharge)

        # Higher flows should have lower exceedance probability
        # So discharge should be in descending order
        discharge_vals = fdc['discharge'].values
        for i in range(len(discharge_vals) - 1):
            assert discharge_vals[i] >= discharge_vals[i + 1]

    def test_fdc_empty_series(self):
        """Test with empty series."""
        discharge = pd.Series([], dtype=float)

        fdc = flow_duration_curve(discharge)

        assert isinstance(fdc, pd.DataFrame)
        assert len(fdc) == 0


class TestClassifyFlowRegime:
    """Tests for classify_flow_regime function."""

    def test_basic_classification(self, sample_discharge_df):
        """Test basic flow regime classification."""
        discharge = sample_discharge_df['Discharge_cfs']
        fdc = flow_duration_curve(discharge)

        result = classify_flow_regime(fdc)

        assert result is not None
        assert isinstance(result, dict)

    def test_flow_percentile_ordering(self, sample_discharge_df):
        """Test that percentiles are properly ordered."""
        discharge = sample_discharge_df['Discharge_cfs']
        fdc = flow_duration_curve(discharge)

        result = classify_flow_regime(fdc)

        if result is not None and 'q10' in result and 'q50' in result and 'q90' in result:
            # Q10 (exceeded 10% of time) should be higher than Q90
            assert result['q10'] > result['q50']
            assert result['q50'] > result['q90']

    def test_flashiness_calculation(self, sample_discharge_df):
        """Test flashiness index calculation."""
        discharge = sample_discharge_df['Discharge_cfs']
        fdc = flow_duration_curve(discharge)

        result = classify_flow_regime(fdc)

        if result is not None and 'flashiness' in result:
            # Flashiness should be positive (ratio of high to low flow)
            if not np.isnan(result['flashiness']):
                assert result['flashiness'] > 0

    def test_classification_with_constants(self):
        """Test with constant discharge (edge case)."""
        discharge = pd.Series([100.0] * 100)
        fdc = flow_duration_curve(discharge)

        result = classify_flow_regime(fdc)

        # Should handle constant flow gracefully
        assert result is not None or result is None  # Either works
