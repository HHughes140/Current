"""SEC EDGAR combined ratio extraction for insurance carriers.

Strategy:
1. Use the XBRL companyfacts API to search for loss/combined ratio facts.
2. Fall back to downloading 10-K/10-Q filings and parsing the text for
   ratio disclosures using regex + table heuristics.
3. Flag low-confidence extractions for manual review.

SEC EDGAR API docs: https://www.sec.gov/edgar/sec-api-documentation
Rate limit: 10 requests/second with User-Agent header required.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.data import cache

logger = logging.getLogger(__name__)

NAMESPACE = "edgar"

# SEC requires a descriptive User-Agent
USER_AGENT = "InsuranceNowcast/0.1 (research tool; contact@example.com)"

HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# XBRL concept names that insurance companies may use for loss/combined ratios.
# These vary by filer — not all companies use the same taxonomy.
XBRL_RATIO_CONCEPTS = [
    # US GAAP concepts
    "LossRatio",
    "CombinedRatio",
    "LossAndLossAdjustmentExpenseRatio",
    "PolicyholderBenefitsAndClaimsIncurredToNetPremiumsEarnedRatio",
    # Common custom extensions
    "us-gaap:LossRatio",
    "us-gaap:CombinedRatio",
]

# Regex patterns to find ratio mentions in filing text
RATIO_PATTERNS = [
    # "combined ratio of 95.2%" or "combined ratio was 95.2%"
    re.compile(
        r"combined\s+ratio\s+(?:of|was|at|equals?)\s+([\d.]+)\s*%",
        re.IGNORECASE,
    ),
    # "loss ratio of 62.3%"
    re.compile(
        r"loss\s+ratio\s+(?:of|was|at|equals?)\s+([\d.]+)\s*%",
        re.IGNORECASE,
    ),
    # "expense ratio of 32.1%"
    re.compile(
        r"expense\s+ratio\s+(?:of|was|at|equals?)\s+([\d.]+)\s*%",
        re.IGNORECASE,
    ),
    # Table cell patterns: "95.2" near "Combined Ratio"
    re.compile(
        r"combined\s+ratio.*?([\d]{2,3}\.[\d]{1,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"loss\s+ratio.*?([\d]{2,3}\.[\d]{1,2})",
        re.IGNORECASE,
    ),
]


@dataclass
class RatioExtraction:
    """A single extracted ratio value with metadata."""
    ticker: str
    period: str          # e.g., "2023-Q4" or "2023-FY"
    ratio_type: str      # "loss_ratio", "expense_ratio", "combined_ratio"
    value: float         # percentage, e.g., 95.2
    source: str          # "xbrl" or "text_parse"
    confidence: float    # 0.0 to 1.0
    filing_url: str = ""


# ---------------------------------------------------------------------------
# Rate-limited request helper
# ---------------------------------------------------------------------------

_last_request_time = 0.0


def _sec_get(url: str, **kwargs) -> requests.Response:
    """Make a rate-limited GET request to SEC EDGAR."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < 0.12:  # ~8 req/s to stay under 10/s limit
        time.sleep(0.12 - elapsed)
    _last_request_time = time.time()

    resp = requests.get(url, headers=HEADERS, timeout=30, **kwargs)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# XBRL API approach
# ---------------------------------------------------------------------------

def _try_xbrl(cik: str, ticker: str) -> list[RatioExtraction]:
    """Try to extract ratio facts from XBRL companyfacts API."""
    # Pad CIK to 10 digits
    cik_padded = cik.lstrip("0").zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"

    try:
        resp = _sec_get(url)
        data = resp.json()
    except Exception as e:
        logger.debug("XBRL fetch failed for %s: %s", ticker, e)
        return []

    extractions = []
    facts = data.get("facts", {})

    # Search across all namespaces (us-gaap, dei, custom)
    for ns_name, namespace in facts.items():
        for concept_name, concept_data in namespace.items():
            # Check if this concept looks like a ratio we want
            concept_lower = concept_name.lower()
            is_loss = "loss" in concept_lower and "ratio" in concept_lower
            is_combined = "combined" in concept_lower and "ratio" in concept_lower
            is_expense = "expense" in concept_lower and "ratio" in concept_lower

            if not (is_loss or is_combined or is_expense):
                continue

            if is_combined:
                ratio_type = "combined_ratio"
            elif is_loss:
                ratio_type = "loss_ratio"
            else:
                ratio_type = "expense_ratio"

            # Extract units — ratios are usually "pure" (decimal) or percent
            for unit_key, unit_data in concept_data.get("units", {}).items():
                for fact in unit_data:
                    period_end = fact.get("end", "")
                    val = fact.get("val")
                    form = fact.get("form", "")

                    if val is None or period_end == "":
                        continue

                    # Only 10-K and 10-Q
                    if form not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                        continue

                    # Convert to percentage if in decimal form
                    if val < 2.0:  # likely a decimal ratio like 0.952
                        val = val * 100

                    # Determine period
                    year = period_end[:4]
                    month = int(period_end[5:7])
                    if form in ("10-K", "10-K/A"):
                        period = f"{year}-FY"
                    else:
                        quarter = (month - 1) // 3 + 1
                        period = f"{year}-Q{quarter}"

                    # Sanity check: ratios should be 40-200%
                    if 40 <= val <= 200:
                        extractions.append(RatioExtraction(
                            ticker=ticker,
                            period=period,
                            ratio_type=ratio_type,
                            value=round(val, 2),
                            source="xbrl",
                            confidence=0.9,
                            filing_url=fact.get("accn", ""),
                        ))

    logger.info("XBRL extracted %d ratio facts for %s", len(extractions), ticker)
    return extractions


# ---------------------------------------------------------------------------
# Filing text parsing approach
# ---------------------------------------------------------------------------

def _get_filing_list(cik: str, form_type: str = "10-K", count: int = 40) -> list[dict]:
    """Get a list of recent filings for a company from EDGAR."""
    cik_padded = cik.lstrip("0").zfill(10)
    url = (
        f"https://efts.sec.gov/LATEST/search-index"
        f"?q=%22{form_type}%22&dateRange=custom"
        f"&startdt=2014-01-01&enddt=2025-12-31"
        f"&forms={form_type}"
    )
    # Use the submissions API instead
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    try:
        resp = _sec_get(url)
        data = resp.json()
    except Exception as e:
        logger.warning("Failed to get filings for CIK %s: %s", cik, e)
        return []

    filings = []
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form in (form_type, f"{form_type}/A"):
            accession_clean = accessions[i].replace("-", "")
            filings.append({
                "form": form,
                "date": dates[i],
                "accession": accessions[i],
                "url": (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_padded}/{accession_clean}/{primary_docs[i]}"
                ),
            })

    return filings[:count]


def _parse_filing_text(url: str, ticker: str) -> list[RatioExtraction]:
    """Download a filing and search for ratio values in the text."""
    try:
        resp = _sec_get(url)
    except Exception as e:
        logger.warning("Failed to download filing %s: %s", url, e)
        return []

    # Parse HTML to text
    soup = BeautifulSoup(resp.content, "lxml")
    text = soup.get_text(separator=" ", strip=True)

    # Determine period from filing date in URL or text
    extractions = []

    for pattern in RATIO_PATTERNS:
        for match in pattern.finditer(text):
            try:
                val = float(match.group(1))
            except (ValueError, IndexError):
                continue

            if not (40 <= val <= 200):
                continue

            # Determine ratio type from the pattern/match context
            context = match.group(0).lower()
            if "combined" in context:
                ratio_type = "combined_ratio"
            elif "loss" in context:
                ratio_type = "loss_ratio"
            elif "expense" in context:
                ratio_type = "expense_ratio"
            else:
                continue

            extractions.append(RatioExtraction(
                ticker=ticker,
                period="",  # Will be filled from filing metadata
                ratio_type=ratio_type,
                value=round(val, 2),
                source="text_parse",
                confidence=0.5,  # Lower confidence for text parsing
                filing_url=url,
            ))

    return extractions


# ---------------------------------------------------------------------------
# Deduplication and consolidation
# ---------------------------------------------------------------------------

def _deduplicate(extractions: list[RatioExtraction]) -> list[RatioExtraction]:
    """Keep the highest-confidence extraction for each (ticker, period, ratio_type)."""
    best: dict[tuple, RatioExtraction] = {}
    for ex in extractions:
        key = (ex.ticker, ex.period, ex.ratio_type)
        if key not in best or ex.confidence > best[key].confidence:
            best[key] = ex
    return list(best.values())


def _to_dataframe(extractions: list[RatioExtraction]) -> pd.DataFrame:
    """Convert extractions to a clean DataFrame."""
    if not extractions:
        return pd.DataFrame(columns=[
            "ticker", "period", "loss_ratio", "expense_ratio",
            "combined_ratio", "confidence", "source",
        ])

    records = []
    for ex in extractions:
        records.append({
            "ticker": ex.ticker,
            "period": ex.period,
            "ratio_type": ex.ratio_type,
            "value": ex.value,
            "confidence": ex.confidence,
            "source": ex.source,
        })

    df = pd.DataFrame(records)

    # Pivot so each period has loss_ratio, expense_ratio, combined_ratio columns
    pivot = df.pivot_table(
        index=["ticker", "period"],
        columns="ratio_type",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Add confidence (min across ratio types for that period)
    conf = df.groupby(["ticker", "period"])["confidence"].min().reset_index()
    pivot = pivot.merge(conf, on=["ticker", "period"], how="left")

    # Flatten column names
    pivot.columns = [c if isinstance(c, str) else c for c in pivot.columns]

    return pivot


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_carrier(cik: str, ticker: str) -> list[RatioExtraction]:
    """Extract all available ratio data for a single carrier.

    Tries XBRL first, falls back to text parsing of 10-K and 10-Q filings.
    """
    # Try XBRL
    extractions = _try_xbrl(cik, ticker)

    # If XBRL didn't yield much, try text parsing
    if len(extractions) < 8:  # Less than ~2 years of data
        logger.info("XBRL sparse for %s, trying text parsing of 10-K filings", ticker)
        filing_list = _get_filing_list(cik, "10-K")
        for filing in filing_list:
            text_extractions = _parse_filing_text(filing["url"], ticker)
            # Assign period from filing date
            year = filing["date"][:4]
            for ex in text_extractions:
                if not ex.period:
                    ex.period = f"{year}-FY"
            extractions.extend(text_extractions)

        # Also try 10-Q for quarterly data
        filing_list_q = _get_filing_list(cik, "10-Q")
        for filing in filing_list_q:
            text_extractions = _parse_filing_text(filing["url"], ticker)
            year = filing["date"][:4]
            month = int(filing["date"][5:7])
            quarter = (month - 1) // 3 + 1
            for ex in text_extractions:
                if not ex.period:
                    ex.period = f"{year}-Q{quarter}"
            extractions.extend(text_extractions)

    return _deduplicate(extractions)


def refresh_all(
    carriers: dict[str, str],  # {ticker: cik}
    data_dir: str,
    force: bool = False,
) -> pd.DataFrame:
    """Extract ratio data for all carriers.

    Args:
        carriers: Dict mapping ticker to CIK number.
        data_dir: Path to data directory.
        force: If True, ignore cache.

    Returns:
        DataFrame with columns: ticker, period, loss_ratio, expense_ratio,
        combined_ratio, confidence, source
    """
    if not force and not cache.is_stale(data_dir, NAMESPACE, "ratios", max_age_hours=720):
        cached = cache.load(data_dir, NAMESPACE, "ratios")
        if cached is not None:
            logger.info("EDGAR cache is fresh, loading from cache")
            return cached

    all_extractions: list[RatioExtraction] = []

    for ticker, cik in carriers.items():
        logger.info("Extracting ratios for %s (CIK: %s)", ticker, cik)
        try:
            carrier_extractions = extract_carrier(cik, ticker)
            all_extractions.extend(carrier_extractions)
            logger.info("  → %d ratio values extracted", len(carrier_extractions))
        except Exception as e:
            logger.error("Failed to extract %s: %s", ticker, e)
            continue

    df = _to_dataframe(all_extractions)
    cache.save(df, data_dir, NAMESPACE, "ratios")

    # Flag low-confidence extractions
    low_conf = df[df["confidence"] < 0.7] if "confidence" in df.columns else pd.DataFrame()
    if len(low_conf) > 0:
        logger.warning(
            "%d low-confidence extractions found. Run `nowcast verify-edgar` to review.",
            len(low_conf),
        )

    return df


def load_ratios(data_dir: str) -> pd.DataFrame | None:
    """Load cached ratio data without refreshing."""
    return cache.load(data_dir, NAMESPACE, "ratios")


def save_manual_override(
    data_dir: str,
    ticker: str,
    period: str,
    loss_ratio: float | None = None,
    expense_ratio: float | None = None,
    combined_ratio: float | None = None,
) -> None:
    """Manually set or correct a ratio value.

    Used by the verify-edgar CLI command to fix extraction errors.
    """
    df = load_ratios(data_dir)
    if df is None:
        df = pd.DataFrame(columns=[
            "ticker", "period", "loss_ratio", "expense_ratio",
            "combined_ratio", "confidence", "source",
        ])

    mask = (df["ticker"] == ticker) & (df["period"] == period)

    row = {
        "ticker": ticker,
        "period": period,
        "confidence": 1.0,
        "source": "manual",
    }
    if loss_ratio is not None:
        row["loss_ratio"] = loss_ratio
    if expense_ratio is not None:
        row["expense_ratio"] = expense_ratio
    if combined_ratio is not None:
        row["combined_ratio"] = combined_ratio

    if mask.any():
        for col, val in row.items():
            if col in df.columns:
                df.loc[mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    cache.save(df, data_dir, NAMESPACE, "ratios")
    logger.info("Saved manual override for %s %s", ticker, period)
