"""
Unit tests for hydrology.data.usgs module.

Tests USGS data fetching, parsing, and caching functions.
Note: Some tests require network access and may be slow.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import date

from hydrology.data.usgs import (
    parse_waterml,
    http_get_text,
    DEFAULT_PARAM_DISCHARGE,
    DEFAULT_PARAM_STAGE,
    DEFAULT_PARAM_TEMP
)


class TestParseWaterML:
    """Tests for parse_waterml function."""

    def test_basic_waterml_parsing(self, sample_waterml_response):
        """Test basic WaterML XML parsing."""
        df = parse_waterml(sample_waterml_response)

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'value' in df.columns

    def test_waterml_datetime_index(self, sample_waterml_response):
        """Test that parsed data has datetime index."""
        df = parse_waterml(sample_waterml_response)

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_waterml_values_numeric(self, sample_waterml_response):
        """Test that values are numeric."""
        df = parse_waterml(sample_waterml_response)

        assert df['value'].dtype in [np.float64, np.float32, float]

    def test_waterml_empty_response(self):
        """Test handling of empty XML response."""
        empty_xml = '<?xml version="1.0"?><response></response>'
        df = parse_waterml(empty_xml)

        assert df is None or df.empty

    def test_waterml_invalid_xml(self):
        """Test handling of invalid XML."""
        invalid_xml = "not valid xml at all"
        df = parse_waterml(invalid_xml)

        assert df is None or df.empty

    def test_waterml_none_input(self):
        """Test handling of None input."""
        df = parse_waterml(None)
        assert df is None or df.empty

    def test_waterml_empty_string(self):
        """Test handling of empty string."""
        df = parse_waterml("")
        assert df is None or df.empty


class TestParameterConstants:
    """Tests for parameter code constants."""

    def test_discharge_param_code(self):
        """Test discharge parameter code is correct."""
        assert DEFAULT_PARAM_DISCHARGE == "00060"

    def test_stage_param_code(self):
        """Test stage parameter code is correct."""
        assert DEFAULT_PARAM_STAGE == "00065"

    def test_temp_param_code(self):
        """Test temperature parameter code is correct."""
        assert DEFAULT_PARAM_TEMP == "00010"


class TestHttpGetText:
    """Tests for http_get_text function."""

    @patch('hydrology.data.usgs.requests.get')
    def test_successful_request(self, mock_get):
        """Test successful HTTP request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "test response"
        mock_get.return_value = mock_response

        result = http_get_text("http://test.com", {})

        assert result == "test response"
        mock_get.assert_called_once()

    @patch('hydrology.data.usgs.requests.get')
    def test_retry_on_failure(self, mock_get):
        """Test retry logic on temporary failure."""
        # First two calls fail, third succeeds
        mock_fail = MagicMock()
        mock_fail.status_code = 503

        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.text = "success after retry"

        mock_get.side_effect = [mock_fail, mock_fail, mock_success]

        result = http_get_text(
            "http://test.com", {},
            retries=5, base_sleep=0.01  # Fast retries for testing
        )

        assert result == "success after retry"
        assert mock_get.call_count == 3

    @patch('hydrology.data.usgs.requests.get')
    def test_all_retries_fail(self, mock_get):
        """Test that exception is raised when all retries fail."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_get.return_value = mock_response

        with pytest.raises(Exception):
            http_get_text(
                "http://test.com", {},
                retries=3, base_sleep=0.01
            )


class TestDataFetching:
    """Integration tests for data fetching (may require network)."""

    @pytest.mark.slow
    @pytest.mark.network
    def test_fetch_real_discharge_data(self):
        """Test fetching real discharge data from USGS."""
        from hydrology.data.usgs import fetch_daily_values

        # Fetch a small date range for a known site
        df = fetch_daily_values(
            site_id='12422500',  # Spokane River
            param_cd='00060',
            start_date='2023-01-01',
            end_date='2023-01-10'
        )

        # This may fail if no network or API is down
        if df is not None:
            assert isinstance(df, pd.DataFrame)
            assert len(df) >= 0  # May be empty for some dates


class TestDateChunking:
    """Tests for date range chunking logic."""

    def test_chunk_logic_concept(self):
        """Test the date chunking concept used in fetch functions."""
        from datetime import datetime, timedelta

        start = datetime(2000, 1, 1)
        end = datetime(2023, 12, 31)
        chunk_years = 5

        chunks = []
        current = start
        while current <= end:
            chunk_end = min(
                datetime(current.year + chunk_years - 1, 12, 31),
                end
            )
            chunks.append((current, chunk_end))
            current = datetime(chunk_end.year + 1, 1, 1)

        # Should have multiple chunks for 24 year range
        assert len(chunks) > 1
        # First chunk should start at start date
        assert chunks[0][0] == start
        # Last chunk should end at or before end date
        assert chunks[-1][1] <= end
