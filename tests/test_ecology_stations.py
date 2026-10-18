"""Tests for the packaged Ecology streamflow station catalog."""

import re

import pandas as pd

from hydrology.data.ecology import (
    CATALOG_COLUMNS,
    get_ecology_station,
    load_ecology_stations,
)


def test_catalog_covers_the_public_gage_layer():
    df = load_ecology_stations()

    assert list(df.columns) == CATALOG_COLUMNS
    assert len(df) >= 300
    assert df["station_code"].is_unique
    assert set(df["status"].unique()) >= {"Active", "Removed"}
    assert df["latitude"].between(45, 50).all()
    assert df["longitude"].between(-125, -116).all()
    assert df["flow_station_url"].str.contains("StationDetails?sta=", regex=False).all()


def test_known_southeast_washington_gages():
    marengo = get_ecology_station("35b150")
    bolles = get_ecology_station("32B100")
    luckenbill = get_ecology_station("32B090")

    assert marengo is not None
    assert "Tucannon" in marengo["station_name"]
    assert "Marengo" in marengo["station_name"]
    assert marengo["status"] == "Active"

    assert bolles is not None
    assert "Touchet" in bolles["station_name"]
    assert "Bolles" in bolles["station_name"]

    assert luckenbill is not None
    assert "Luckenbill" in luckenbill["station_name"]
    assert luckenbill["flow_station_url"].endswith("sta=32B090")


def test_missing_station_returns_none():
    assert get_ecology_station("ZZZ999") is None


def test_catalog_text_has_no_contact_details():
    df = load_ecology_stations().fillna("")
    text = "\n".join(
        value
        for column in ("description", "comments", "station_name")
        for value in df[column].astype(str).tolist()
    )

    assert not re.search(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", text, re.I)
    assert not re.search(r"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]\d{4}", text)
    assert "please contact" not in text.lower()


def test_missing_catalog_returns_empty_frame(tmp_path):
    df = load_ecology_stations(tmp_path / "missing.csv")

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == CATALOG_COLUMNS
