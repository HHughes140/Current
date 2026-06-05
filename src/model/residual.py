"""Residual demand computation.

Residual = Observed ΔPosition - Expected ΔPosition

Large positive residual → accumulation not explained by factors
Large negative residual → distribution not explained by factors

Aggregated across institutions, normalized to z-scores.
Active institution residuals are weighted more heavily than passive.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_institution_residuals(
    holdings: pd.DataFrame,
    expected_buy_prob: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-institution × stock residuals.

    Args:
        holdings: DataFrame with columns [institution, ticker, quarter, delta_pct, style]
        expected_buy_prob: DataFrame with columns [institution, ticker, quarter, expected_buy_prob]

    Returns:
        DataFrame with residual columns added.
    """
    df = holdings.copy()

    # Normalize observed delta to [0, 1] range for comparison with probability
    # Convert delta_pct to a buy signal: positive delta → 1, negative → 0, weighted by magnitude
    df["observed_signal"] = df["delta_pct"].apply(
        lambda x: min(max((x / 10 + 0.5), 0), 1) if pd.notna(x) else 0.5
    )

    # Merge expected probabilities
    if "expected_buy_prob" in expected_buy_prob.columns:
        df = df.merge(
            expected_buy_prob[["institution", "ticker", "quarter", "expected_buy_prob"]],
            on=["institution", "ticker", "quarter"],
            how="left",
        )
    else:
        df["expected_buy_prob"] = 0.5

    df["expected_buy_prob"] = df["expected_buy_prob"].fillna(0.5)

    # Raw residual
    df["residual"] = df["observed_signal"] - df["expected_buy_prob"]

    return df


def aggregate_residuals(
    residuals: pd.DataFrame,
    active_weight: float = 0.7,
    passive_weight: float = 0.3,
) -> pd.DataFrame:
    """Aggregate institution-level residuals to per-stock scores.

    Active institutions get higher weight since their deviations
    from expected behavior are more informative.

    Returns DataFrame indexed by ticker with:
        residual_active, residual_passive, residual_combined, residual_z
    """
    if residuals.empty:
        return pd.DataFrame()

    # Get the most recent quarter
    latest_quarter = sorted(residuals["quarter"].unique())[-1]
    recent = residuals[residuals["quarter"] == latest_quarter]

    results = []

    for ticker in recent["ticker"].unique():
        ticker_data = recent[recent["ticker"] == ticker]

        active = ticker_data[ticker_data["style"] == "active"]
        passive = ticker_data[ticker_data["style"] == "passive"]

        active_res = active["residual"].mean() if not active.empty else 0
        passive_res = passive["residual"].mean() if not passive.empty else 0

        combined = active_weight * active_res + passive_weight * passive_res
        n_institutions = len(ticker_data["institution"].unique())

        results.append({
            "ticker": ticker,
            "quarter": latest_quarter,
            "residual_active": round(active_res, 4),
            "residual_passive": round(passive_res, 4),
            "residual_combined": round(combined, 4),
            "n_institutions_reporting": n_institutions,
            "n_active_buying": int((active["delta_pct"] > 0).sum()) if not active.empty else 0,
            "n_active_selling": int((active["delta_pct"] < 0).sum()) if not active.empty else 0,
        })

    result_df = pd.DataFrame(results)

    # Z-score the combined residual across the universe
    if len(result_df) > 1:
        mean_r = result_df["residual_combined"].mean()
        std_r = result_df["residual_combined"].std()
        if std_r > 0:
            result_df["residual_z"] = (result_df["residual_combined"] - mean_r) / std_r
        else:
            result_df["residual_z"] = 0.0
    else:
        result_df["residual_z"] = 0.0

    result_df["residual_z"] = result_df["residual_z"].round(3)

    return result_df.sort_values("residual_z", ascending=False).reset_index(drop=True)


def compute_ownership_concentration(holdings: pd.DataFrame) -> pd.DataFrame:
    """Compute ownership concentration changes (HHI delta).

    Rising concentration → fewer institutions holding → potential squeeze.
    Falling concentration → broader ownership → less pressure signal.
    """
    if holdings.empty:
        return pd.DataFrame()

    quarters = sorted(holdings["quarter"].unique())
    if len(quarters) < 2:
        return pd.DataFrame()

    results = []
    for ticker in holdings["ticker"].unique():
        ticker_data = holdings[holdings["ticker"] == ticker]

        for i in range(1, len(quarters)):
            curr = ticker_data[ticker_data["quarter"] == quarters[i]]
            prev = ticker_data[ticker_data["quarter"] == quarters[i - 1]]

            if curr.empty or prev.empty:
                continue

            # HHI: sum of squared portfolio weights
            curr_hhi = (curr["portfolio_weight"] ** 2).sum()
            prev_hhi = (prev["portfolio_weight"] ** 2).sum()

            results.append({
                "ticker": ticker,
                "quarter": quarters[i],
                "hhi_current": round(curr_hhi, 4),
                "hhi_prior": round(prev_hhi, 4),
                "hhi_delta": round(curr_hhi - prev_hhi, 4),
            })

    return pd.DataFrame(results) if results else pd.DataFrame()
