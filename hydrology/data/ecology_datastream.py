"""On-demand cache for Ecology Freshwater DataStream daily tables.

Nothing in this module runs unless a caller asks for one station. A published-year
check saves only the list of water years. A water-year download saves that year's
daily discharge file and, when DataStream publishes one, the daily stage file.
Later reads use the local copies. Reviewed years are not downloaded again.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from ..core.logging_setup import get_logger
from ..core.paths import DATA_DIR
from .ecology import load_ecology_stations

logger = get_logger(__name__)

CACHE_DIR = DATA_DIR / "ecology_cache"
DETAILS_URL = "https://apps.ecology.wa.gov/ContinuousFlowAndWQ/StationDetails?sta={code}"
FILE_URL = (
    "https://apps.ecology.wa.gov/ContinuousFlowAndWQ/StationData/Prod/"
    "{code}/{code}_{year}_{suffix}.txt"
)
DISCHARGE_SUFFIX = "DSG_DV"
STAGE_SUFFIX = "STG_DV"
PROVISIONAL_CODES = {2, 3, 140}
MAX_BYTES = 2_000_000
USER_AGENT = "HydroPlot local cache"
QUEUE_LIMIT = 4
QUEUE_PAUSE_SECONDS = 2.0

_DATA_LINE = re.compile(
    r"^(\d{2}/\d{2}/\d{4})"
    r"(?:\s+\d{2}:\d{2})?"
    r"(?:\s+(-?\d+(?:\.\d+)?))?"
    r"\s+(\d+)\s*$"
)
_FILE_TOKEN = re.compile(r"\b([A-Za-z0-9]+)_(\d{4})_(DSG|STG)_DV\b")


class EcologyDataError(Exception):
    """A DataStream request or a saved table could not be used."""


@dataclass(frozen=True)
class DownloadResult:
    """Files saved before the queue stopped."""

    saved: list[Path]
    stopped_on: str | None = None


def water_years_overlapping(start: date, end: date) -> list[int]:
    """Water years that contain any day in the inclusive calendar range.

    Ecology names a water year by the calendar year in which it ends.
    Water year 2024 is 1 October 2023 through 30 September 2024.
    """
    if end < start:
        start, end = end, start
    first = start.year + 1 if start.month >= 10 else start.year
    last = end.year + 1 if end.month >= 10 else end.year
    return list(range(first, last + 1))


def record_status(quality_codes: set[int]) -> str:
    """Label a saved year from the quality codes in its daily table."""
    if 140 in quality_codes:
        return "unchecked"
    if quality_codes & {2, 3}:
        return "provisional"
    return "reviewed"


def parse_daily_table(text: str) -> pd.DataFrame:
    """Parse a DataStream daily text table into date, value, and quality."""
    rows = []
    for line in text.splitlines():
        if line.startswith("Quality Codes"):
            break
        match = _DATA_LINE.match(line.strip())
        if not match:
            continue
        value = float(match.group(2)) if match.group(2) is not None else None
        rows.append((match.group(1), value, int(match.group(3))))
    frame = pd.DataFrame(rows, columns=["date", "value", "quality"])
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y")
    frame["date"] = frame["date"].dt.tz_localize("America/Los_Angeles")
    return frame


def parse_published_years(html: str, station_code: str) -> dict[str, list[int]]:
    """Read DSG_DV and STG_DV water years for one station from its details page."""
    code = station_code.upper()
    discharge: set[int] = set()
    stage: set[int] = set()
    for found, year_text, kind in _FILE_TOKEN.findall(html):
        if found.upper() != code:
            continue
        year = int(year_text)
        if kind == "DSG":
            discharge.add(year)
        else:
            stage.add(year)
    return {
        "discharge_years": sorted(discharge),
        "stage_years": sorted(stage),
    }


def require_catalog_station(station_code: str) -> str:
    """Return the catalog station code, or raise before any network call."""
    code = str(station_code).strip().upper()
    catalog = load_ecology_stations()
    if catalog.empty or code not in set(catalog["station_code"].str.upper()):
        raise EcologyDataError(f"Station {station_code} is not in the local Ecology catalog.")
    return code


def index_path(station_code: str, cache_dir: Path | None = None) -> Path:
    return _station_dir(station_code, cache_dir) / "index.json"


def read_index(station_code: str, cache_dir: Path | None = None) -> dict | None:
    path = index_path(station_code, cache_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cached_table_path(
    station_code: str,
    water_year: int,
    suffix: str,
    cache_dir: Path | None = None,
) -> Path:
    code = str(station_code).strip().upper()
    return _station_dir(code, cache_dir) / f"{code}_{int(water_year)}_{suffix}.txt"


def refresh_published_years(
    station_code: str,
    *,
    fetch=None,
    cache_dir: Path | None = None,
) -> dict:
    """Download the station details page and save only its list of water years."""
    code = require_catalog_station(station_code)
    fetch = fetch or _fetch_bytes
    html = _decode(fetch(_details_url(code)))
    years = parse_published_years(html, code)
    payload = {
        "station_code": code,
        "checked_on": datetime.now(timezone.utc).date().isoformat(),
        **years,
    }
    path = index_path(code, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Saved DataStream year list for %s (%s discharge years)",
        code,
        len(years["discharge_years"]),
    )
    return payload


def pending_downloads(
    station_code: str,
    years: list[int],
    suffix: str,
    cache_dir: Path | None = None,
) -> list[tuple[int, str]]:
    """Published files in ``years`` that are not on disk yet, oldest first.

    Stage waits until that water year's discharge file is already saved, so a
    range never asks for two products at once.
    """
    index = read_index(station_code, cache_dir)
    if index is None:
        return []
    years_key = "discharge_years" if suffix == DISCHARGE_SUFFIX else "stage_years"
    published = set(index.get(years_key, []))
    code = str(station_code).strip().upper()
    pending = []
    for year in years:
        if year not in published:
            continue
        if suffix == STAGE_SUFFIX:
            discharge = cached_table_path(code, year, DISCHARGE_SUFFIX, cache_dir)
            if not discharge.exists():
                continue
        if cached_table_path(code, year, suffix, cache_dir).exists():
            continue
        pending.append((int(year), suffix))
    return pending


def download_queue(
    station_code: str,
    files: list[tuple[int, str]],
    *,
    pause_seconds: float = QUEUE_PAUSE_SECONDS,
    limit: int = QUEUE_LIMIT,
    fetch=None,
    sleep=time.sleep,
    cache_dir: Path | None = None,
) -> DownloadResult:
    """Download files oldest-first on one HTTP session. Stop at the first failure.

    ``limit`` caps one click. The pause sits between files, not before the first.
    """
    code = require_catalog_station(station_code)
    index = read_index(code, cache_dir)
    if index is None:
        raise EcologyDataError("Check published years before downloading a file.")
    if fetch is None:
        fetch = _session_fetch()

    saved: list[Path] = []
    for position, (year, suffix) in enumerate(list(files)[:limit]):
        if position:
            sleep(pause_seconds)
        try:
            saved.append(_download_one(code, int(year), suffix, fetch, cache_dir, index))
        except EcologyDataError as exc:
            logger.info("DataStream queue stopped for %s after %s file(s): %s", code, len(saved), exc)
            return DownloadResult(saved, str(exc))
    return DownloadResult(saved, None)


def load_cached_series(
    station_code: str,
    start: date,
    end: date,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Join cached discharge and stage for the calendar range. No network."""
    code = str(station_code).strip().upper()
    frames = []
    for year in water_years_overlapping(start, end):
        discharge_path = cached_table_path(code, year, DISCHARGE_SUFFIX, cache_dir)
        if not discharge_path.exists():
            continue
        discharge = _read_saved_table(discharge_path)
        discharge = discharge.rename(columns={"value": "Discharge_cfs", "quality": "discharge_quality"})
        stage_path = cached_table_path(code, year, STAGE_SUFFIX, cache_dir)
        if stage_path.exists():
            stage = _read_saved_table(stage_path)
            stage = stage.rename(columns={"value": "Gage_Height_ft", "quality": "stage_quality"})
            discharge = discharge.merge(stage, on="date", how="left")
        frames.append(discharge)
    if not frames:
        return pd.DataFrame(columns=["Discharge_cfs", "discharge_quality", "Gage_Height_ft", "stage_quality"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()
    start_ts = pd.Timestamp(start).tz_localize("America/Los_Angeles")
    end_ts = pd.Timestamp(end).tz_localize("America/Los_Angeles") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    combined = combined.loc[(combined.index >= start_ts) & (combined.index <= end_ts)]
    return combined


def year_record_status(
    station_code: str,
    water_year: int,
    cache_dir: Path | None = None,
) -> str | None:
    """Reviewed, provisional, or unchecked for a cached discharge year."""
    path = cached_table_path(station_code, water_year, DISCHARGE_SUFFIX, cache_dir)
    if not path.exists():
        return None
    table = _read_saved_table(path)
    if table.empty:
        return None
    return record_status(set(table["quality"].dropna().astype(int)))


def _station_dir(station_code: str, cache_dir: Path | None) -> Path:
    root = cache_dir or CACHE_DIR
    return root / str(station_code).strip().upper()


def _details_url(station_code: str) -> str:
    return DETAILS_URL.format(code=urllib.parse.quote(station_code))


def _file_url(station_code: str, water_year: int, suffix: str) -> str:
    return FILE_URL.format(
        code=urllib.parse.quote(station_code),
        year=int(water_year),
        suffix=suffix,
    )


def _download_one(
    station_code: str,
    water_year: int,
    suffix: str,
    fetch,
    cache_dir: Path | None,
    index: dict,
) -> Path:
    years_key = "discharge_years" if suffix == DISCHARGE_SUFFIX else "stage_years"
    if suffix not in {DISCHARGE_SUFFIX, STAGE_SUFFIX}:
        raise EcologyDataError(f"Unknown DataStream table {suffix}.")
    if water_year not in index.get(years_key, []):
        raise EcologyDataError(f"Water year {water_year} has no published {suffix} table.")
    raw = fetch(_file_url(station_code, water_year, suffix))
    table = parse_daily_table(_decode(raw))
    if table.empty:
        raise EcologyDataError(
            f"DataStream returned no daily rows for {station_code} {water_year} {suffix}."
        )
    return _write_table(station_code, water_year, suffix, raw, cache_dir)


def _session_fetch():
    """One connection for a queue. The next file reuses it."""
    session = requests.Session()

    def fetch(url: str) -> bytes:
        try:
            response = session.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            raise EcologyDataError("DataStream could not be reached.") from exc
        if response.status_code >= 400:
            raise EcologyDataError(f"DataStream returned HTTP {response.status_code}.")
        if len(response.content) > MAX_BYTES:
            raise EcologyDataError("DataStream file was larger than a daily table.")
        return response.content

    return fetch


def _fetch_bytes(url: str) -> bytes:
    return _session_fetch()(url)


def _decode(payload: bytes) -> str:
    return payload.decode("utf-8", "replace")


def _write_table(
    station_code: str,
    water_year: int,
    suffix: str,
    payload: bytes,
    cache_dir: Path | None,
) -> Path:
    path = cached_table_path(station_code, water_year, suffix, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path


def _read_saved_table(path: Path) -> pd.DataFrame:
    return parse_daily_table(path.read_text(encoding="utf-8", errors="replace"))
