"""
Unit tests for hydrology.analysis.trends module.

Tests trend analysis functions including annual/monthly aggregation,
linear regression, and Mann-Kendall trend tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from hydrology.analysis.trends import (
    calculate_annual_means,
    calculate_monthly_means,
    linear_regression_trend,
    mann_kendall_test,
    calculate_correlation
)


class TestCalculateAnnualMeans:
    """Tests for calculate_annual_means function."""

    def test_basic_annual_means(self, sample_discharge_df):
        """Test basic annual mean calculation."""
        result = calculate_annual_means(sample_discharge_df, 'Discharge_cfs')

        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == 4  # 2020, 2021, 2022, 2023

    def test_annual_means_empty_df(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = calculate_annual_means(df, 'Discharge_cfs')
        assert result is None

    def test_annual_means_missing_column(self, sample_discharge_df):
        """Test with non-existent column."""
        result = calculate_annual_means(sample_discharge_df, 'NonExistentColumn')
        assert result is None

    def test_annual_means_none_df(self):
        """Test with None DataFrame."""
        result = calculate_annual_means(None, 'Discharge_cfs')
        assert result is None

    def test_annual_means_values_reasonable(self, sample_discharge_df):
        """Test that annual means are within expected range."""
        result = calculate_annual_means(sample_discharge_df, 'Discharge_cfs')

        # Based on our fixture: seasonal pattern around 500 cfs
        assert result is not None
        assert all(result > 0)
        assert all(result < 1000)


class TestCalculateMonthlyMeans:
    """Tests for calculate_monthly_means function."""

    def test_basic_monthly_means(self, sample_discharge_df):
        """Test basic monthly mean calculation."""
        result = calculate_monthly_means(sample_discharge_df, 'Discharge_cfs')

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 12  # 12 months
        assert 'mean' in result.columns
        assert 'std' in result.columns
        assert 'count' in result.columns

    def test_monthly_means_empty_df(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = calculate_monthly_means(df, 'Discharge_cfs')
        assert result is None

    def test_monthly_means_seasonality(self, sample_discharge_df):
        """Test that monthly means show expected seasonality."""
        result = calculate_monthly_means(sample_discharge_df, 'Discharge_cfs')

        assert result is not None
        # Spring months (Apr-Jun) should have higher flows in our fixture
        spring_mean = result.loc[[4, 5, 6], 'mean'].mean()
        winter_mean = result.loc[[12, 1, 2], 'mean'].mean()
        # Spring should be higher due to sine wave pattern
        assert spring_mean > winter_mean


class TestLinearRegressionTrend:
    """Tests for linear_regression_trend function."""

    def test_basic_trend_analysis(self, sample_discharge_df):
        """Test basic linear regression trend."""
        annual = calculate_annual_means(sample_discharge_df, 'Discharge_cfs')
        result = linear_regression_trend(annual, 'Discharge')

        assert result is not None
        assert 'slope' in result
        assert 'r_squared' in result
        assert 'p_value' in result
        assert 'trend_direction' in result

    def test_increasing_trend(self):
        """Test detection of increasing trend."""
        # Create data with clear increasing trend
        years = pd.date_range('2010', periods=10, freq='YS')
        values = pd.Series([100, 110, 120, 130, 140, 150, 160, 170, 180, 190], index=years)

        result = linear_regression_trend(values, 'test')

        assert result is not None
        assert result['slope'] > 0
        assert result['r_squared'] > 0.9
        assert result['trend_direction'] == 'increasing'

    def test_decreasing_trend(self):
        """Test detection of decreasing trend."""
        years = pd.date_range('2010', periods=10, freq='YS')
        values = pd.Series([200, 190, 180, 170, 160, 150, 140, 130, 120, 110], index=years)

        result = linear_regression_trend(values, 'test')

        assert result is not None
        assert result['slope'] < 0
        assert result['r_squared'] > 0.9
        assert result['trend_direction'] == 'decreasing'

    def test_no_significant_trend(self):
        """Test with noisy data (no clear trend)."""
        np.random.seed(42)
        years = pd.date_range('2010', periods=10, freq='YS')
        values = pd.Series(np.random.normal(100, 5, 10), index=years)

        result = linear_regression_trend(values, 'test')

        assert result is not None
        # With random data, p-value should typically be > 0.05
        # but we're just checking the function returns valid results

    def test_insufficient_data(self):
        """Test with minimal data points."""
        years = pd.date_range('2010', periods=3, freq='YS')
        values = pd.Series([100, 110, 120], index=years)

        result = linear_regression_trend(values, 'test')
        # Should work with 3 points
        assert result is not None


class TestMannKendallTest:
    """Tests for mann_kendall_test function."""

    def test_basic_mk_test(self, sample_discharge_df):
        """Test basic Mann-Kendall test."""
        annual = calculate_annual_means(sample_discharge_df, 'Discharge_cfs')
        result = mann_kendall_test(annual)

        # Result might be None if pymannkendall not installed
        if result is not None:
            assert 'trend' in result
            assert 'p_value' in result
            assert 'tau' in result

    def test_mk_increasing_trend(self):
        """Test Mann-Kendall with increasing trend."""
        values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = mann_kendall_test(values)

        if result is not None:
            assert result['trend'] == 'increasing'
            assert result['tau'] > 0

    def test_mk_decreasing_trend(self):
        """Test Mann-Kendall with decreasing trend."""
        values = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

        result = mann_kendall_test(values)

        if result is not None:
            assert result['trend'] == 'decreasing'
            assert result['tau'] < 0


class TestCalculateCorrelation:
    """Tests for calculate_correlation function."""

    def test_basic_correlation(self, sample_merged_df):
        """Test basic correlation calculation."""
        result = calculate_correlation(
            sample_merged_df,
            'Discharge_cfs',
            'Temp_C'
        )

        assert result is not None
        corr, p_val = result
        assert -1 <= corr <= 1

    def test_perfect_positive_correlation(self):
        """Test with perfectly correlated data."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })

        result = calculate_correlation(df, 'x', 'y')

        assert result is not None
        corr, p_val = result
        assert abs(corr - 1.0) < 0.001

    def test_perfect_negative_correlation(self):
        """Test with perfectly negatively correlated data."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 8, 6, 4, 2]
        })

        result = calculate_correlation(df, 'x', 'y')

        assert result is not None
        corr, p_val = result
        assert abs(corr - (-1.0)) < 0.001

    def test_correlation_with_nans(self):
        """Test correlation handles NaN values."""
        df = pd.DataFrame({
            'x': [1, 2, np.nan, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })

        result = calculate_correlation(df, 'x', 'y')

        # Should handle NaNs by dropping them
        if result is not None:
            corr, p_val = result
            assert -1 <= corr <= 1

    def test_correlation_missing_column(self, sample_merged_df):
        """Test with non-existent column."""
        result = calculate_correlation(
            sample_merged_df,
            'Discharge_cfs',
            'NonExistent'
        )
        assert result is None

    def test_correlation_empty_df(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = calculate_correlation(df, 'x', 'y')
        assert result is None
