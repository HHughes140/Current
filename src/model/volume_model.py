"""Volume prediction model — P(volume spike in next 30d).

Uses institutional residual demand as the primary feature, combined with
momentum, volatility, and options activity to predict whether a stock
will experience unusual volume in the near future.

This answers: "Are institutions likely accumulating in a way that will
become visible in volume soon?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

VOLUME_FEATURES = [
    "residual_z",
    "residual_combined",
    "momentum_3m",
    "momentum_6m",
    "volatility_60d",
    "volume_zscore",
    "cum_anomaly_5d",
    "pc_volume_ratio",
    "earnings_revision",
]


@dataclass
class VolumePrediction:
    ticker: str
    spike_probability: float   # P(volume spike in next 30d)
    direction: str             # "UP" or "DOWN" based on residual sign
    confidence: float          # Based on feature availability


def build_volume_features(
    residuals: pd.DataFrame,
    factors: pd.DataFrame,
    volume_signals: pd.DataFrame,
    options_signals: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all signal sources into a feature matrix for volume prediction."""
    df = residuals[["ticker", "residual_z", "residual_combined"]].copy()

    # Merge factor data
    factor_cols = ["ticker", "momentum_3m", "momentum_6m", "volatility_60d", "earnings_revision"]
    available_factor_cols = [c for c in factor_cols if c in factors.columns]
    if available_factor_cols:
        df = df.merge(
            factors[available_factor_cols],
            on="ticker", how="left",
        )

    # Merge volume signals
    vol_cols = ["ticker", "volume_zscore", "cum_anomaly_5d"]
    available_vol_cols = [c for c in vol_cols if c in volume_signals.columns]
    if available_vol_cols:
        df = df.merge(
            volume_signals[available_vol_cols],
            on="ticker", how="left",
        )

    # Merge options signals
    opt_cols = ["ticker", "pc_volume_ratio"]
    available_opt_cols = [c for c in opt_cols if c in options_signals.columns]
    if available_opt_cols:
        df = df.merge(
            options_signals[available_opt_cols],
            on="ticker", how="left",
        )

    return df


def predict_volume_spikes(
    feature_df: pd.DataFrame,
) -> list[VolumePrediction]:
    """Score stocks for volume spike probability.

    Uses a heuristic scoring approach when insufficient training data
    is available for a proper logistic regression. The heuristic combines
    residual magnitude, current volume anomalies, and options skew.
    """
    results = []

    for _, row in feature_df.iterrows():
        ticker = row.get("ticker", "")

        # Heuristic score components (each contributes 0-1)
        components = []
        weights = []

        # Residual magnitude — strongest signal
        res_z = row.get("residual_z", 0)
        if pd.notna(res_z):
            # Sigmoid transform: large residuals → high probability
            res_score = 1 / (1 + np.exp(-abs(res_z)))
            components.append(res_score)
            weights.append(0.35)

        # Current volume anomaly — already elevated volume
        vol_z = row.get("volume_zscore", 0)
        if pd.notna(vol_z):
            vol_score = min(abs(vol_z) / 3, 1.0)
            components.append(vol_score)
            weights.append(0.20)

        # Cumulative anomaly — sustained unusual volume
        cum5 = row.get("cum_anomaly_5d", 0)
        if pd.notna(cum5):
            cum_score = min(abs(cum5) / 5, 1.0)
            components.append(cum_score)
            weights.append(0.15)

        # Options skew — elevated P/C ratio
        pcr = row.get("pc_volume_ratio", 1.0)
        if pd.notna(pcr):
            # Deviation from neutral (1.0)
            opt_score = min(abs(pcr - 1.0) / 0.5, 1.0)
            components.append(opt_score)
            weights.append(0.15)

        # Momentum — stocks with momentum shifts
        mom = row.get("momentum_3m", 0)
        if pd.notna(mom):
            mom_score = min(abs(mom) / 20, 1.0)
            components.append(mom_score)
            weights.append(0.15)

        # Weighted average
        if components:
            total_weight = sum(weights[:len(components)])
            spike_prob = sum(c * w for c, w in zip(components, weights)) / total_weight
        else:
            spike_prob = 0.5

        # Direction from residual sign
        direction = "UP" if (pd.notna(res_z) and res_z > 0) else "DOWN"

        # Confidence based on how many features were available
        confidence = len(components) / len(VOLUME_FEATURES)

        results.append(VolumePrediction(
            ticker=ticker,
            spike_probability=round(spike_prob, 3),
            direction=direction,
            confidence=round(confidence, 2),
        ))

    # Sort by probability descending
    results.sort(key=lambda x: x.spike_probability, reverse=True)
    return results
