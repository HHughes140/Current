"""FRED data pipeline — pull macro proxy series and compute transformations.

Series pulled:
    TRFVOLUSM227SFWA  — Vehicle Miles Traveled (monthly, millions of miles)
    PAYEMS             — Nonfarm Payrolls (monthly, thousands)
    CPIMEDSL           — Medical Care CPI (monthly, index)
    CUSR0000SETA02     — Used Cars & Trucks CPI (monthly, index)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from fredapi import Fred

from src.data import cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Series definitions
# ---------------------------------------------------------------------------

SERIES = {
    "vmt": {
        "fred_id": "TRFVOLUSM227SFWA",
        "description": "Vehicle Miles Traveled",
        "yoy_col": "vmt_yoy",
    },
    "payrolls": {
        "fred_id": "PAYEMS",
        "description": "Nonfarm Payrolls",
        "yoy_col": "payrolls_yoy",
    },
    "medical_cpi": {
        "fred_id": "CPIMEDSL",
        "description": "Medical Care CPI",
        "yoy_col": "medical_cpi_yoy",
    },
    "used_car_cpi": {
        "fred_id": "CUSR0000SETA02",
        "description": "Used Cars & Trucks CPI",
        "yoy_col": "used_car_cpi_yoy",
    },
}

NAMESPACE = "fred"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def fetch_series(api_key: str, series_name: str, start: str = "2010-01-01") -> pd.DataFrame:
    """Fetch a single FRED series and return as a DataFrame with date index."""
    info = SERIES[series_name]
    fred = Fred(api_key=api_key)
    logger.info("Fetching FRED series %s (%s)", info["fred_id"], info["description"])
    raw = fred.get_series(info["fred_id"], observation_start=start)
    df = raw.to_frame(name=series_name)
    df.index.name = "date"
    df.index = pd.to_datetime(df.index)
    return df


def compute_yoy(df: pd.DataFrame, col: str) -> pd.Series:
    """Compute year-over-year percent change for a monthly series."""
    return df[col].pct_change(periods=12) * 100


def to_quarterly(df: pd.DataFrame, col: str, agg: str = "mean") -> pd.DataFrame:
    """Resample a monthly series to quarterly frequency.

    Args:
        agg: Aggregation method — 'mean' for indices/rates, 'sum' for volumes.
    """
    quarterly = df[[col]].resample("QE").agg(agg)
    quarterly.index.name = "quarter"
    return quarterly


def refresh_all(api_key: str, data_dir: str, force: bool = False) -> pd.DataFrame:
    """Pull all FRED series, compute YoY changes, and build a quarterly panel.

    Returns a DataFrame indexed by quarter with columns:
        vmt, vmt_yoy, payrolls, payrolls_yoy,
        medical_cpi, medical_cpi_yoy, used_car_cpi, used_car_cpi_yoy
    """
    if not force and not cache.is_stale(data_dir, NAMESPACE, "quarterly_panel"):
        logger.info("FRED cache is fresh, loading from cache")
        cached = cache.load(data_dir, NAMESPACE, "quarterly_panel")
        if cached is not None:
            return cached

    monthly_frames: list[pd.DataFrame] = []

    for name, info in SERIES.items():
        df = fetch_series(api_key, name)
        yoy = compute_yoy(df, name)
        df[info["yoy_col"]] = yoy
        monthly_frames.append(df)

        # Cache raw monthly
        cache.save(df, data_dir, NAMESPACE, f"{name}_monthly")

    # Merge all monthly series on date
    monthly = monthly_frames[0]
    for frame in monthly_frames[1:]:
        monthly = monthly.join(frame, how="outer")

    # Build quarterly panel
    quarterly_parts: list[pd.DataFrame] = []
    for name, info in SERIES.items():
        # Levels: mean for indices, mean for rates
        level_q = to_quarterly(monthly, name, agg="mean")
        # YoY: mean of monthly YoY within quarter
        yoy_q = to_quarterly(monthly, info["yoy_col"], agg="mean")
        quarterly_parts.append(level_q)
        quarterly_parts.append(yoy_q)

    quarterly = quarterly_parts[0]
    for part in quarterly_parts[1:]:
        quarterly = quarterly.join(part, how="outer")

    quarterly = quarterly.dropna(how="all")
    cache.save(quarterly, data_dir, NAMESPACE, "quarterly_panel")
    cache.save(monthly, data_dir, NAMESPACE, "monthly_panel")

    logger.info("FRED refresh complete: %d quarters of data", len(quarterly))
    return quarterly


def load_quarterly(data_dir: str) -> pd.DataFrame | None:
    """Load the cached quarterly FRED panel without refreshing."""
    return cache.load(data_dir, NAMESPACE, "quarterly_panel")


def load_monthly(data_dir: str) -> pd.DataFrame | None:
    """Load the cached monthly FRED panel without refreshing."""
    return cache.load(data_dir, NAMESPACE, "monthly_panel")
