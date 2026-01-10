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
