"""Accumulation detection — link 13F position changes to current volume.

The core question: "Which institutions are likely still building positions,
and does current trading activity in the stock support that hypothesis?"

Approach:
1. Identify institutions with multi-quarter accumulation streaks from 13F
2. Estimate each institution's typical accumulation signature (pace, duration)
3. Score current volume/price patterns against those signatures
4. Produce per-stock, per-institution accumulation probability

This bridges the 13F lag: 13F tells you what happened, volume tells you
what's happening now, and this module connects them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AccumulationSignal:
    ticker: str
    institution: str
    institution_name: str
    style: str

    # 13F history
    consecutive_buys: int       # How many quarters in a row they've been buying
    consecutive_sells: int      # How many quarters in a row they've been selling
    avg_quarterly_delta_pct: float  # Average position change per quarter
    total_change_pct: float     # Total position change over the streak
    latest_quarter: str

    # Current volume match
    volume_confirms: bool       # Does current volume support continued accumulation?
    continuation_probability: float  # P(still accumulating this quarter)

    @property
    def direction(self) -> str:
        if self.consecutive_buys >= 2:
            return "ACCUMULATING"
        elif self.consecutive_sells >= 2:
            return "DISTRIBUTING"
        return "UNCLEAR"


def _compute_streaks(holdings: pd.DataFrame) -> pd.DataFrame:
    """Identify consecutive buy/sell streaks per institution × ticker.

    A streak is broken when the direction reverses or the position is flat.
    """
    if holdings.empty or "delta_shares" not in holdings.columns:
        return pd.DataFrame()

    df = holdings.sort_values(["institution", "ticker", "quarter"]).copy()
    df = df[df["delta_shares"].notna()]

    records = []

    for (inst, ticker), group in df.groupby(["institution", "ticker"]):
        group = group.sort_values("quarter")
        if len(group) < 2:
            continue

        # Count consecutive buys/sells from the most recent quarter backward
        deltas = group["delta_shares"].values
        quarters = group["quarter"].values

        consecutive_buys = 0
        consecutive_sells = 0

        for i in range(len(deltas) - 1, -1, -1):
            if deltas[i] > 0:
                if consecutive_sells > 0:
                    break
                consecutive_buys += 1
            elif deltas[i] < 0:
                if consecutive_buys > 0:
                    break
                consecutive_sells += 1
            else:
                break

        # Compute average delta over the streak
        streak_len = max(consecutive_buys, consecutive_sells)
        if streak_len >= 2:
            streak_deltas = group["delta_pct"].iloc[-streak_len:]
            avg_delta = streak_deltas.mean() if not streak_deltas.isna().all() else 0
            total_change = streak_deltas.sum() if not streak_deltas.isna().all() else 0
        else:
            avg_delta = group["delta_pct"].iloc[-1] if pd.notna(group["delta_pct"].iloc[-1]) else 0
            total_change = avg_delta

        latest = group.iloc[-1]

        records.append({
            "institution": inst,
            "institution_name": latest.get("institution_name", inst),
            "style": latest.get("style", "unknown"),
            "ticker": ticker,
            "consecutive_buys": consecutive_buys,
            "consecutive_sells": consecutive_sells,
            "avg_quarterly_delta_pct": round(avg_delta, 2),
            "total_change_pct": round(total_change, 2),
            "latest_quarter": quarters[-1],
            "latest_shares": int(latest["shares"]),
        })

    return pd.DataFrame(records)


def _score_volume_confirmation(
    streaks: pd.DataFrame,
    volume_signals: pd.DataFrame,
) -> pd.DataFrame:
    """Score whether current volume confirms continued accumulation/distribution.

    Logic:
    - If an institution has been buying for 3+ quarters AND current volume
      is elevated (z > 0.5), accumulation is likely continuing.
    - If volume is abnormally high AND the stock has multi-institution
      buying streaks, accumulation probability is higher.
    - If volume is flat/below average, the streak may have ended.
    """
    if streaks.empty:
        return streaks

    df = streaks.copy()
    df["volume_confirms"] = False
    df["continuation_probability"] = 0.5

    for i, row in df.iterrows():
        ticker = row["ticker"]
        vol_row = volume_signals[volume_signals["ticker"] == ticker] if not volume_signals.empty else pd.DataFrame()

        if vol_row.empty:
            continue

        vol_z = vol_row.iloc[0].get("volume_zscore", 0)
        cum_5d = vol_row.iloc[0].get("cum_anomaly_5d", 0)

        if pd.isna(vol_z):
            vol_z = 0
        if pd.isna(cum_5d):
            cum_5d = 0

        is_buying_streak = row["consecutive_buys"] >= 2
        is_selling_streak = row["consecutive_sells"] >= 2

        if is_buying_streak:
            # Elevated volume with positive price action → confirms accumulation
            # Low volume → streak may be ending
            if vol_z > 0.5:
                prob = min(0.5 + 0.1 * row["consecutive_buys"] + 0.05 * vol_z, 0.95)
                confirms = True
            elif vol_z > -0.5:
                # Normal volume — weak continuation signal
                prob = 0.5 + 0.05 * row["consecutive_buys"]
                confirms = False
            else:
                # Below-average volume — accumulation may be pausing
                prob = max(0.3, 0.5 - 0.1 * abs(vol_z))
                confirms = False

        elif is_selling_streak:
            if vol_z > 1.0:
                # High volume during selling streak → confirms distribution
                prob = min(0.5 + 0.1 * row["consecutive_sells"] + 0.05 * vol_z, 0.95)
                confirms = True
            else:
                prob = 0.5 + 0.03 * row["consecutive_sells"]
                confirms = vol_z > 0.5
        else:
            prob = 0.5
            confirms = False

        df.at[i, "volume_confirms"] = confirms
        df.at[i, "continuation_probability"] = round(prob, 3)

    return df


def detect_accumulation(
    holdings: pd.DataFrame,
    volume_signals: pd.DataFrame,
    min_streak: int = 2,
) -> list[AccumulationSignal]:
    """Identify active accumulation/distribution patterns.

    Returns a list of AccumulationSignal objects for institution × stock pairs
    where there's a meaningful streak and volume data to assess continuation.
    """
    streaks = _compute_streaks(holdings)
    if streaks.empty:
        return []

    # Filter to meaningful streaks
    streaks = streaks[
        (streaks["consecutive_buys"] >= min_streak) |
        (streaks["consecutive_sells"] >= min_streak)
    ]

    if streaks.empty:
        return []

    # Score against volume
    scored = _score_volume_confirmation(streaks, volume_signals)

    results = []
    for _, row in scored.iterrows():
        results.append(AccumulationSignal(
            ticker=row["ticker"],
            institution=row["institution"],
            institution_name=row["institution_name"],
            style=row["style"],
            consecutive_buys=int(row["consecutive_buys"]),
            consecutive_sells=int(row["consecutive_sells"]),
            avg_quarterly_delta_pct=row["avg_quarterly_delta_pct"],
            total_change_pct=row["total_change_pct"],
            latest_quarter=row["latest_quarter"],
            volume_confirms=bool(row["volume_confirms"]),
            continuation_probability=row["continuation_probability"],
        ))

    # Sort: highest probability, longest streaks first
    results.sort(key=lambda s: (s.continuation_probability, max(s.consecutive_buys, s.consecutive_sells)), reverse=True)
    return results


def summarize_by_stock(signals: list[AccumulationSignal]) -> pd.DataFrame:
    """Aggregate accumulation signals to per-stock level.

    For each stock, count how many institutions are accumulating vs distributing,
    and compute an average continuation probability.
    """
    if not signals:
        return pd.DataFrame()

    records = []
    by_ticker: dict[str, list[AccumulationSignal]] = {}
    for s in signals:
        by_ticker.setdefault(s.ticker, []).append(s)

    for ticker, sigs in by_ticker.items():
        accumulators = [s for s in sigs if s.direction == "ACCUMULATING"]
        distributors = [s for s in sigs if s.direction == "DISTRIBUTING"]

        avg_prob = np.mean([s.continuation_probability for s in sigs])
        confirmed = sum(1 for s in sigs if s.volume_confirms)

        # Net direction
        net = len(accumulators) - len(distributors)

        records.append({
            "ticker": ticker,
            "n_accumulating": len(accumulators),
            "n_distributing": len(distributors),
            "net_direction": "ACCUMULATE" if net > 0 else ("DISTRIBUTE" if net < 0 else "MIXED"),
            "avg_continuation_prob": round(avg_prob, 3),
            "volume_confirmed_count": confirmed,
            "top_accumulator": accumulators[0].institution_name if accumulators else None,
            "top_distributor": distributors[0].institution_name if distributors else None,
        })

    return pd.DataFrame(records).sort_values("avg_continuation_prob", ascending=False).reset_index(drop=True)
