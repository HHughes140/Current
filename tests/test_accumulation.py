"""Tests for accumulation detection."""

import pandas as pd
import numpy as np
import pytest

from src.model.accumulation import (
    _compute_streaks,
    _score_volume_confirmation,
    detect_accumulation,
    summarize_by_stock,
)


class TestComputeStreaks:
    def test_buying_streak(self):
        holdings = pd.DataFrame({
            "institution": ["fidelity"] * 4,
            "institution_name": ["Fidelity"] * 4,
            "style": ["active"] * 4,
            "ticker": ["PGR"] * 4,
            "quarter": ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"],
            "shares": [100000, 120000, 140000, 160000],
            "delta_shares": [np.nan, 20000, 20000, 20000],
            "delta_pct": [np.nan, 20.0, 16.7, 14.3],
        })
        streaks = _compute_streaks(holdings)
        assert len(streaks) == 1
        assert streaks.iloc[0]["consecutive_buys"] == 3
        assert streaks.iloc[0]["consecutive_sells"] == 0

    def test_selling_streak(self):
        holdings = pd.DataFrame({
            "institution": ["fidelity"] * 3,
            "institution_name": ["Fidelity"] * 3,
            "style": ["active"] * 3,
            "ticker": ["TRV"] * 3,
            "quarter": ["2024-Q1", "2024-Q2", "2024-Q3"],
            "shares": [100000, 80000, 60000],
            "delta_shares": [np.nan, -20000, -20000],
            "delta_pct": [np.nan, -20.0, -25.0],
        })
        streaks = _compute_streaks(holdings)
        assert len(streaks) == 1
        assert streaks.iloc[0]["consecutive_sells"] == 2
        assert streaks.iloc[0]["consecutive_buys"] == 0

    def test_reversal_breaks_streak(self):
        holdings = pd.DataFrame({
            "institution": ["fidelity"] * 4,
            "institution_name": ["Fidelity"] * 4,
            "style": ["active"] * 4,
            "ticker": ["PGR"] * 4,
            "quarter": ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"],
            "shares": [100000, 120000, 110000, 130000],
            "delta_shares": [np.nan, 20000, -10000, 20000],
            "delta_pct": [np.nan, 20.0, -8.3, 18.2],
        })
        streaks = _compute_streaks(holdings)
        # Should only count the last buy (Q4), streak = 1
        assert streaks.iloc[0]["consecutive_buys"] == 1

    def test_empty_input(self):
        result = _compute_streaks(pd.DataFrame())
        assert result.empty

    def test_single_quarter_no_streak(self):
        holdings = pd.DataFrame({
            "institution": ["fidelity"],
            "institution_name": ["Fidelity"],
            "style": ["active"],
            "ticker": ["PGR"],
            "quarter": ["2024-Q1"],
            "shares": [100000],
            "delta_shares": [np.nan],
            "delta_pct": [np.nan],
        })
        result = _compute_streaks(holdings)
        assert result.empty


class TestVolumeConfirmation:
    def test_elevated_volume_confirms(self):
        streaks = pd.DataFrame({
            "institution": ["fidelity"],
            "institution_name": ["Fidelity"],
            "style": ["active"],
            "ticker": ["PGR"],
            "consecutive_buys": [3],
            "consecutive_sells": [0],
            "avg_quarterly_delta_pct": [15.0],
            "total_change_pct": [45.0],
            "latest_quarter": ["2024-Q4"],
            "latest_shares": [160000],
        })
        volume_signals = pd.DataFrame({
            "ticker": ["PGR"],
            "volume_zscore": [1.5],
            "cum_anomaly_5d": [3.0],
        })
        result = _score_volume_confirmation(streaks, volume_signals)
        assert bool(result.iloc[0]["volume_confirms"]) is True
        assert result.iloc[0]["continuation_probability"] > 0.7

    def test_low_volume_does_not_confirm(self):
        streaks = pd.DataFrame({
            "institution": ["fidelity"],
            "institution_name": ["Fidelity"],
            "style": ["active"],
            "ticker": ["PGR"],
            "consecutive_buys": [3],
            "consecutive_sells": [0],
            "avg_quarterly_delta_pct": [15.0],
            "total_change_pct": [45.0],
            "latest_quarter": ["2024-Q4"],
            "latest_shares": [160000],
        })
        volume_signals = pd.DataFrame({
            "ticker": ["PGR"],
            "volume_zscore": [-1.0],
            "cum_anomaly_5d": [-2.0],
        })
        result = _score_volume_confirmation(streaks, volume_signals)
        assert bool(result.iloc[0]["volume_confirms"]) is False


class TestDetectAccumulation:
    def test_filters_short_streaks(self):
        holdings = pd.DataFrame({
            "institution": ["fidelity"] * 3,
            "institution_name": ["Fidelity"] * 3,
            "style": ["active"] * 3,
            "ticker": ["PGR"] * 3,
            "quarter": ["2024-Q1", "2024-Q2", "2024-Q3"],
            "shares": [100000, 120000, 140000],
            "delta_shares": [np.nan, 20000, 20000],
            "delta_pct": [np.nan, 20.0, 16.7],
        })
        volume_signals = pd.DataFrame({
            "ticker": ["PGR"],
            "volume_zscore": [1.0],
            "cum_anomaly_5d": [2.0],
        })
        # min_streak=2 should include this
        result = detect_accumulation(holdings, volume_signals, min_streak=2)
        assert len(result) == 1
        assert result[0].direction == "ACCUMULATING"

        # min_streak=5 should exclude it
        result = detect_accumulation(holdings, volume_signals, min_streak=5)
        assert len(result) == 0


class TestSummarizeByStock:
    def test_aggregates_correctly(self):
        from src.model.accumulation import AccumulationSignal

        signals = [
            AccumulationSignal("PGR", "fidelity", "Fidelity", "active",
                               3, 0, 15.0, 45.0, "2024-Q4", True, 0.85),
            AccumulationSignal("PGR", "wellington", "Wellington", "active",
                               2, 0, 10.0, 20.0, "2024-Q4", False, 0.65),
            AccumulationSignal("PGR", "blackrock", "BlackRock", "passive",
                               0, 2, -5.0, -10.0, "2024-Q4", False, 0.55),
        ]
        summary = summarize_by_stock(signals)
        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["n_accumulating"] == 2
        assert row["n_distributing"] == 1
        assert row["net_direction"] == "ACCUMULATE"
