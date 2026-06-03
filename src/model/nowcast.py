"""Nowcast engine — produce current-quarter loss ratio estimates.

Takes the latest proxy signal values, applies carrier-specific regression
coefficients, and produces loss ratio estimates with confidence intervals.
Compares to consensus estimates and flags potential earnings surprises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.carriers import Carrier, get_carrier
from src.model.regression import CarrierModel

logger = logging.getLogger(__name__)


@dataclass
class NowcastResult:
    """Nowcast output for a single carrier."""
    ticker: str
    carrier_name: str

    # Estimate
    loss_ratio_change_est: float    # Estimated YoY change in loss ratio (pp)
    prior_year_loss_ratio: float    # Same quarter last year's loss ratio
    loss_ratio_est: float           # = prior_year + change_est
    confidence_interval_1se: tuple[float, float]  # ±1 SE
    confidence_interval_2se: tuple[float, float]  # ±2 SE

    # Consensus comparison
    consensus_loss_ratio: float | None
    delta_vs_consensus: float | None   # estimate - consensus (positive = adverse)
    surprise_flag: str                 # "ADVERSE", "FAVORABLE", or "IN_LINE"

    # Signal detail
    signal_contributions: dict[str, float]  # signal → contribution to change estimate
    residual_se: float
    r_squared: float

    @property
    def surprise_magnitude(self) -> float | None:
        """How many standard errors from consensus."""
        if self.delta_vs_consensus is None:
            return None
        return abs(self.delta_vs_consensus) / self.residual_se if self.residual_se > 0 else None


def _get_current_signals(
    fred_quarterly: pd.DataFrame,
    noaa_quarterly: pd.DataFrame,
    fred_monthly: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Extract the most recent proxy signal values.

    For the current (potentially incomplete) quarter, we use the latest
    available monthly data and compute a partial-quarter average.
    """
    signals: dict[str, float] = {}

    # FRED signals — use the last complete quarter or partial current quarter
    fred_cols = ["vmt_yoy", "payrolls_yoy", "medical_cpi_yoy", "used_car_cpi_yoy"]

    if fred_monthly is not None:
        # Use monthly data for most up-to-date signal
        for col in fred_cols:
            if col in fred_monthly.columns:
                recent = fred_monthly[col].dropna()
                if len(recent) > 0:
                    # Average of the last 3 months (current quarter proxy)
                    signals[col] = recent.iloc[-3:].mean()
    else:
        # Fall back to quarterly
        for col in fred_cols:
            if col in fred_quarterly.columns:
                recent = fred_quarterly[col].dropna()
                if len(recent) > 0:
                    signals[col] = recent.iloc[-1]

    # NOAA cat losses — current quarter
    if "cat_losses_quarterly" in noaa_quarterly.columns:
        recent = noaa_quarterly["cat_losses_quarterly"].dropna()
        if len(recent) > 0:
            signals["cat_losses_quarterly"] = recent.iloc[-1]

    return signals


def nowcast_carrier(
    carrier: Carrier,
    model: CarrierModel,
    current_signals: dict[str, float],
    edgar_ratios: pd.DataFrame,
    consensus: float | None = None,
) -> NowcastResult | None:
    """Produce a nowcast for a single carrier."""

    # Compute signal contributions
    contributions: dict[str, float] = {}
    total_change = model.intercept

    for signal_name in model.signal_names:
        coef = model.coefficients.get(signal_name, 0.0)
        val = current_signals.get(signal_name, 0.0)
        contribution = coef * val
        contributions[signal_name] = contribution
        total_change += contribution

    # Get prior year's loss ratio for the same quarter
    carrier_ratios = edgar_ratios[edgar_ratios["ticker"] == carrier.ticker]
    dep_col = "loss_ratio" if "loss_ratio" in carrier_ratios.columns else "combined_ratio"

    if dep_col not in carrier_ratios.columns or carrier_ratios[dep_col].dropna().empty:
        logger.warning("No historical ratio for %s to compute absolute estimate", carrier.ticker)
        return None

    # Use the most recent available ratio as the base
    latest_ratio = pd.to_numeric(carrier_ratios[dep_col].dropna(), errors="coerce").dropna()
    if latest_ratio.empty:
        return None

    prior_year_lr = float(latest_ratio.iloc[-1])
    lr_estimate = prior_year_lr + total_change

    # Confidence intervals
    se = model.residual_se
    ci_1se = (lr_estimate - se, lr_estimate + se)
    ci_2se = (lr_estimate - 2 * se, lr_estimate + 2 * se)

    # Consensus comparison
    delta = None
    flag = "NO_CONSENSUS"
    if consensus is not None:
        delta = lr_estimate - consensus
        if abs(delta) > se:
            flag = "ADVERSE" if delta > 0 else "FAVORABLE"
        else:
            flag = "IN_LINE"

    return NowcastResult(
        ticker=carrier.ticker,
        carrier_name=carrier.name,
        loss_ratio_change_est=round(total_change, 2),
        prior_year_loss_ratio=round(prior_year_lr, 2),
        loss_ratio_est=round(lr_estimate, 2),
        confidence_interval_1se=(round(ci_1se[0], 2), round(ci_1se[1], 2)),
        confidence_interval_2se=(round(ci_2se[0], 2), round(ci_2se[1], 2)),
        consensus_loss_ratio=consensus,
        delta_vs_consensus=round(delta, 2) if delta is not None else None,
        surprise_flag=flag,
        signal_contributions={k: round(v, 4) for k, v in contributions.items()},
        residual_se=round(se, 2),
        r_squared=round(model.r_squared, 3),
    )


def nowcast_all(
    models: dict[str, CarrierModel],
    fred_quarterly: pd.DataFrame,
    noaa_quarterly: pd.DataFrame,
    edgar_ratios: pd.DataFrame,
    consensus: dict[str, float | None] | None = None,
    fred_monthly: pd.DataFrame | None = None,
) -> list[NowcastResult]:
    """Produce nowcasts for all carriers with fitted models.

    Args:
        models: Dict of {ticker: CarrierModel} from regression.
        consensus: Dict of {ticker: consensus_loss_ratio} or None.
    """
    current_signals = _get_current_signals(fred_quarterly, noaa_quarterly, fred_monthly)

    logger.info("Current quarter signals:")
    for sig, val in current_signals.items():
        logger.info("  %s: %.2f", sig, val)

    results: list[NowcastResult] = []

    for ticker, model in models.items():
        carrier = get_carrier(ticker)
        cons = None
        if consensus and ticker in consensus:
            cons = consensus[ticker]

        result = nowcast_carrier(carrier, model, current_signals, edgar_ratios, cons)
        if result:
            results.append(result)

    # Sort by absolute surprise magnitude (most surprising first)
    results.sort(
        key=lambda r: abs(r.delta_vs_consensus) if r.delta_vs_consensus is not None else 0,
        reverse=True,
    )

    return results
