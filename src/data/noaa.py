"""NOAA Storm Events pipeline — catastrophe loss proxy for property lines.

Downloads bulk CSV detail files from NOAA's Storm Events database, filters
to property-damage-relevant event types, normalizes dollar amounts, and
aggregates to quarterly totals.

NOAA data: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

The DAMAGE_PROPERTY column uses suffixes: K (thousands), M (millions), B (billions).
We convert everything to millions USD for consistency.

Important: NOAA damage figures are *total economic loss*, not insured loss.
We apply an empirical insured-to-economic ratio by event type. These ratios
are rough estimates — the literature suggests 40-60% for most US perils.
"""

from __future__ import annotations

import io
import gzip
import logging
import re
from pathlib import Path

import pandas as pd
import requests

from src.data import cache

logger = logging.getLogger(__name__)

NAMESPACE = "noaa"

# Base URL for NOAA Storm Events bulk CSV files
BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

# Event types relevant to insured property losses
RELEVANT_EVENT_TYPES = {
    "Hurricane",
    "Hurricane (Typhoon)",
    "Tornado",
    "Hail",
    "Flood",
    "Flash Flood",
    "Wildfire",
    "Winter Storm",
    "Ice Storm",
    "Tropical Storm",
    "Strong Wind",
    "High Wind",
    "Thunderstorm Wind",
}

# Empirical insured-to-economic loss ratios by event type (rough estimates)
INSURED_RATIO = {
    "Hurricane": 0.55,
    "Hurricane (Typhoon)": 0.55,
    "Tornado": 0.55,
    "Hail": 0.65,
    "Flood": 0.30,        # NFIP covers some, but large uninsured share
    "Flash Flood": 0.25,
    "Wildfire": 0.50,
    "Winter Storm": 0.45,
    "Ice Storm": 0.45,
    "Tropical Storm": 0.45,
    "Strong Wind": 0.50,
    "High Wind": 0.50,
    "Thunderstorm Wind": 0.50,
}


# ---------------------------------------------------------------------------
# Damage amount parsing
# ---------------------------------------------------------------------------

def parse_damage(value) -> float:
    """Parse NOAA's damage string into millions USD.

    Examples: '25K' → 0.025, '1.5M' → 1.5, '2B' → 2000.0, '0' → 0.0
    """
    if pd.isna(value) or value == "" or value == "0":
        return 0.0

    value = str(value).strip().upper()

    # Try to match number + suffix pattern
    match = re.match(r"^([\d.]+)\s*([KMB]?)$", value)
    if not match:
        # Some older records use plain numbers (in dollars)
        try:
            return float(value) / 1_000_000
        except ValueError:
            return 0.0

    number = float(match.group(1))
    suffix = match.group(2)

    multipliers = {"": 1 / 1_000_000, "K": 0.001, "M": 1.0, "B": 1000.0}
    return number * multipliers.get(suffix, 1 / 1_000_000)


# ---------------------------------------------------------------------------
# File listing and download
# ---------------------------------------------------------------------------

def _list_detail_files(start_year: int, end_year: int) -> list[str]:
    """Generate expected file names for NOAA detail CSVs by year.

    Files are named like: StormEvents_details-ftp_v1.0_dYYYY_c*.csv.gz
    """
    files = []
    for year in range(start_year, end_year + 1):
        # The exact filename includes a creation date suffix we don't know,
        # so we'll discover it from the directory listing
        files.append(year)
    return files


def _discover_files(start_year: int, end_year: int) -> list[str]:
    """Scrape the NOAA FTP listing to find detail CSV filenames."""
    logger.info("Discovering NOAA Storm Events files for %d-%d", start_year, end_year)
    resp = requests.get(BASE_URL, timeout=30)
    resp.raise_for_status()

    # Find all detail file links
    pattern = re.compile(r"(StormEvents_details-ftp_v1\.0_d(\d{4})_c\d+\.csv\.gz)")
    files = []
    for match in pattern.finditer(resp.text):
        filename, year_str = match.group(1), match.group(2)
        year = int(year_str)
        if start_year <= year <= end_year:
            files.append(filename)

    logger.info("Found %d detail files", len(files))
    return files


def _download_and_parse(filename: str) -> pd.DataFrame:
    """Download a single gzipped CSV and parse relevant columns."""
    url = BASE_URL + filename
    logger.info("Downloading %s", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    with gzip.open(io.BytesIO(resp.content), "rt", encoding="latin-1") as f:
        df = pd.read_csv(
            f,
            usecols=[
                "BEGIN_YEARMONTH", "BEGIN_DAY", "EVENT_TYPE",
                "STATE", "DAMAGE_PROPERTY", "DAMAGE_CROPS",
            ],
            dtype=str,
            low_memory=False,
        )
    return df


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _process_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Filter, parse damages, and add date columns."""
    # Filter to relevant event types
    df = df[df["EVENT_TYPE"].isin(RELEVANT_EVENT_TYPES)].copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "date", "year", "month", "EVENT_TYPE", "STATE",
            "property_damage_m", "insured_loss_m",
        ])

    # Parse damage amounts (to millions USD)
    df["property_damage_m"] = df["DAMAGE_PROPERTY"].apply(parse_damage)
    df["crop_damage_m"] = df["DAMAGE_CROPS"].apply(parse_damage)
    df["total_damage_m"] = df["property_damage_m"] + df["crop_damage_m"]

    # Apply insured-to-economic ratio
    df["insured_ratio"] = df["EVENT_TYPE"].map(INSURED_RATIO).fillna(0.45)
    df["insured_loss_m"] = df["total_damage_m"] * df["insured_ratio"]

    # Parse date
    df["year_month"] = pd.to_numeric(df["BEGIN_YEARMONTH"], errors="coerce")
    df = df.dropna(subset=["year_month"])
    df["year_month"] = df["year_month"].astype(int)
    df["year"] = df["year_month"] // 100
    df["month"] = df["year_month"] % 100
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )

    return df[["date", "year", "month", "EVENT_TYPE", "STATE",
               "property_damage_m", "insured_loss_m"]]


def _aggregate_quarterly(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level data to quarterly totals."""
    events = events.set_index("date")
    quarterly = events.resample("QE").agg({
        "property_damage_m": "sum",
        "insured_loss_m": "sum",
    })
    quarterly.columns = ["cat_losses_economic_m", "cat_losses_quarterly"]
    quarterly.index.name = "quarter"

    # Also compute event counts per quarter for diagnostics
    counts = events.resample("QE").size().to_frame("event_count")
    quarterly = quarterly.join(counts)

    return quarterly


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refresh(data_dir: str, start_year: int = 2010, end_year: int = 2025,
            force: bool = False) -> pd.DataFrame:
    """Download, parse, and aggregate NOAA Storm Events data.

    Returns a quarterly DataFrame with columns:
        cat_losses_economic_m  — total economic property damage (millions USD)
        cat_losses_quarterly   — estimated insured losses (millions USD)
        event_count            — number of relevant weather events
    """
    if not force and not cache.is_stale(data_dir, NAMESPACE, "quarterly", max_age_hours=168):
        cached = cache.load(data_dir, NAMESPACE, "quarterly")
        if cached is not None:
            logger.info("NOAA cache is fresh, loading from cache")
            return cached

    filenames = _discover_files(start_year, end_year)

    all_events = []
    for fn in filenames:
        try:
            raw = _download_and_parse(fn)
            processed = _process_raw(raw)
            all_events.append(processed)
        except Exception as e:
            logger.warning("Failed to process %s: %s", fn, e)
            continue

    if not all_events:
        raise RuntimeError("No NOAA Storm Events files were successfully processed")

    events = pd.concat(all_events, ignore_index=True)
    cache.save(events, data_dir, NAMESPACE, "events_detail")

    quarterly = _aggregate_quarterly(events)
    cache.save(quarterly, data_dir, NAMESPACE, "quarterly")

    logger.info(
        "NOAA refresh complete: %d events, %d quarters, $%.0fM total insured losses",
        len(events), len(quarterly), quarterly["cat_losses_quarterly"].sum(),
    )
    return quarterly


def load_quarterly(data_dir: str) -> pd.DataFrame | None:
    """Load cached quarterly NOAA data without refreshing."""
    return cache.load(data_dir, NAMESPACE, "quarterly")
