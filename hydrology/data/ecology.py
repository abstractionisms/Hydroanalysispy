"""Washington Department of Ecology streamflow station catalog.

The catalog is a snapshot of Ecology's public EIM map layer
"Ecology streamflow gages" (ADS/eimmap MapServer layer 16), retrieved
2026-10-17. Layers 17 and 18 are the same stations drawn at other map
scales, so they are not included.

This is station metadata: location, status, gage type, and the parameters
Ecology lists for the site. It is not a discharge time series. Daily and
instantaneous values stay on Freshwater DataStream; each row's
``flow_station_url`` opens that station.

Nothing in this module contacts Ecology. ``SOURCE_LAYER`` is a citation for
the snapshot. The dashboard does not import this module, and
``load_ecology_stations`` only reads the local CSV when called.

Published contact names, email addresses, and phone numbers were removed
from ``description`` and ``comments``. Operational notes were kept.

Status counts in this snapshot: 85 Active, 3 Inactive, 3 In-House, and
264 Removed. Removed stations are kept because their historical technical
notes and daily files are still public.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..core.logging_setup import get_logger

logger = get_logger(__name__)

CATALOG_PATH = Path(__file__).parent / "catalogs" / "ecology_streamflow_stations.csv"
SOURCE_LAYER = (
    "https://gis.ecology.wa.gov/serverext/rest/services/ADS/eimmap/MapServer/16"
)
RETRIEVED_ON = "2026-10-17"

CATALOG_COLUMNS = [
    "station_id",
    "station_code",
    "station_name",
    "status",
    "station_type",
    "station_subtype",
    "coop_flag",
    "river_mile",
    "latitude",
    "longitude",
    "parameters",
    "description",
    "comments",
    "flow_station_url",
]


def load_ecology_stations(catalog_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the Ecology streamflow station catalog.

    Args:
        catalog_path: CSV path. Uses the packaged snapshot when omitted.

    Returns:
        One row per station. Empty if the file is missing.
    """
    path = Path(catalog_path) if catalog_path is not None else CATALOG_PATH
    if not path.exists():
        logger.error(f"Ecology station catalog not found: {path}")
        return pd.DataFrame(columns=CATALOG_COLUMNS)

    df = pd.read_csv(
        path,
        dtype={
            "station_id": "Int64",
            "station_code": str,
            "coop_flag": "Int64",
        },
    )
    logger.info(f"Loaded {len(df)} Ecology streamflow stations from {path}")
    return df


def get_ecology_station(
    station_code: str,
    catalog_path: Optional[Path] = None,
) -> Optional[dict]:
    """Return one station by Ecology station code, such as ``35B150``.

    Matching ignores case and surrounding spaces. Returns None when the
    code is not in the catalog.
    """
    df = load_ecology_stations(catalog_path)
    if df.empty:
        return None

    code = str(station_code).strip().upper()
    match = df[df["station_code"].str.strip().str.upper() == code]
    if match.empty:
        logger.warning(f"Ecology station not found: {station_code}")
        return None

    station = match.iloc[0].to_dict()
    logger.info(f"Found Ecology station {code} - {station['station_name']}")
    return station
