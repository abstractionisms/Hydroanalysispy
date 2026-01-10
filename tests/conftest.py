"""
Pytest fixtures for hydrology package tests.

Provides sample data and mock objects for testing data fetching,
analysis, and visualization modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


@pytest.fixture
def sample_discharge_df():
    """Create sample discharge DataFrame for testing."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    # Simulate seasonal discharge pattern
    day_of_year = dates.dayofyear
    seasonal = 500 + 300 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    noise = np.random.normal(0, 50, len(dates))
    discharge = seasonal + noise
    discharge = np.maximum(discharge, 10)  # Ensure positive values

    df = pd.DataFrame({
        'Discharge_cfs': discharge
    }, index=dates)
    df.index = df.index.tz_localize('UTC')
    return df


@pytest.fixture
def sample_climate_df():
    """Create sample climate DataFrame for testing."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    # Simulate seasonal temperature pattern
    day_of_year = dates.dayofyear
    temp = 10 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temp += np.random.normal(0, 3, len(dates))

    # Simulate precipitation (random with seasonal tendency)
    precip = np.random.exponential(2, len(dates))
    precip *= (1 + 0.5 * np.sin(2 * np.pi * (day_of_year - 300) / 365))

    df = pd.DataFrame({
        'Temp_C': temp,
        'Precip_mm': precip
    }, index=dates)
    df.index = df.index.tz_localize('UTC')
    return df


@pytest.fixture
def sample_merged_df(sample_discharge_df, sample_climate_df):
    """Create merged discharge + climate DataFrame."""
    return pd.merge(
        sample_discharge_df,
        sample_climate_df,
        left_index=True,
        right_index=True,
        how='inner'
    )


@pytest.fixture
def sample_stage_discharge():
    """Create sample stage-discharge pairs for rating curve testing."""
    np.random.seed(42)
    n = 200

    # Generate stage values
    stage = np.random.uniform(1.0, 10.0, n)

    # Generate discharge using power-law with noise: Q = 50 * H^1.6
    A_true, B_true = 50, 1.6
    discharge = A_true * (stage ** B_true)
    discharge *= np.random.normal(1.0, 0.1, n)  # Add 10% noise

    return pd.Series(stage), pd.Series(discharge), A_true, B_true


@pytest.fixture
def sample_inventory_df():
    """Create sample inventory DataFrame."""
    return pd.DataFrame({
        'site_id': ['12422500', '12424000', '12431000'],
        'description': [
            'Spokane River at Spokane, WA',
            'Hangman Creek at Spokane, WA',
            'Little Spokane River at Dartford, WA'
        ],
        'latitude': [47.6593, 47.6601, 47.8123],
        'longitude': [-117.4491, -117.4047, -117.4156],
        'begin_date': ['1891-01-01', '1948-01-01', '1929-01-01'],
        'end_date': ['2024-01-01', '2024-01-01', '2024-01-01'],
        'huc_code': ['17010305', '17010306', '17010305']
    })


@pytest.fixture
def sample_waterml_response():
    """Sample WaterML XML response for testing parsing."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<ns1:timeSeriesResponse xmlns:ns1="http://www.cuahsi.org/waterML/1.1/">
    <ns1:timeSeries>
        <ns1:values>
            <ns1:value dateTime="2020-01-01T00:00:00.000">100.0</ns1:value>
            <ns1:value dateTime="2020-01-02T00:00:00.000">105.0</ns1:value>
            <ns1:value dateTime="2020-01-03T00:00:00.000">98.0</ns1:value>
            <ns1:value dateTime="2020-01-04T00:00:00.000">110.0</ns1:value>
            <ns1:value dateTime="2020-01-05T00:00:00.000">115.0</ns1:value>
        </ns1:values>
    </ns1:timeSeries>
</ns1:timeSeriesResponse>'''


@pytest.fixture
def sample_json_response():
    """Sample USGS JSON response for testing parsing."""
    return {
        "value": {
            "timeSeries": [{
                "values": [{
                    "value": [
                        {"value": "100.0", "dateTime": "2020-01-01T00:00:00.000"},
                        {"value": "105.0", "dateTime": "2020-01-02T00:00:00.000"},
                        {"value": "98.0", "dateTime": "2020-01-03T00:00:00.000"},
                        {"value": "110.0", "dateTime": "2020-01-04T00:00:00.000"},
                        {"value": "115.0", "dateTime": "2020-01-05T00:00:00.000"}
                    ]
                }]
            }]
        }
    }


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir
