"""Tests for accumulation detection and execution fingerprints."""

import pandas as pd
import numpy as np
import pytest

from src.model.accumulation import (
    _compute_streaks,
    compute_volume_profile,
    detect_accumulation,
    summarize_by_stock,
    learn_fingerprints,
    match_current_volume,
    VOLUME_PROFILE_FEATURES,
    AccumulationSignal,
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
        assert streaks.iloc[0]["consecutive_buys"] == 1

    def test_empty_input(self):
        result = _compute_streaks(pd.DataFrame())
        assert result.empty


class TestVolumeProfile:
    def _make_daily(self, n=63, vol_mean=1000000, vol_std=100000, trend=0):
        """Generate synthetic daily data."""
        np.random.seed(42)
        dates = pd.bdate_range("2024-01-01", periods=n)
        volumes = vol_mean + np.random.randn(n) * vol_std + np.arange(n) * trend
        volumes = np.maximum(volumes, 10000)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({"Volume": volumes, "Close": prices}, index=dates)

    def test_returns_all_features(self):
        daily = self._make_daily()
        profile = compute_volume_profile(daily)
        for feat in VOLUME_PROFILE_FEATURES:
            assert feat in profile, f"Missing feature: {feat}"

    def test_empty_data_returns_nan(self):
        profile = compute_volume_profile(pd.DataFrame())
        assert all(pd.isna(v) for v in profile.values())

    def test_trending_volume_detected(self):
        # Volume with strong upward trend
        daily_up = self._make_daily(trend=20000)
        profile_up = compute_volume_profile(daily_up)
        assert profile_up["vol_trend"] > 0

        # Volume with strong downward trend
        daily_down = self._make_daily(trend=-20000)
        profile_down = compute_volume_profile(daily_down)
        assert profile_down["vol_trend"] < 0

    def test_high_vol_fraction(self):
        # Make data where most days are low vol with a few extreme spikes
        np.random.seed(42)
        dates = pd.bdate_range("2024-01-01", periods=60)
        volumes = [500000] * 45 + [5000000] * 15  # 15 days at 10x normal
        prices = [100] * 60
        daily = pd.DataFrame({"Volume": volumes, "Close": prices}, index=dates)
        profile = compute_volume_profile(daily)
        assert profile["high_vol_day_frac"] > 0.1

    def test_autocorrelation_persistent_volume(self):
        # Persistent high volume (autocorrelated) vs random
        np.random.seed(42)
        dates = pd.bdate_range("2024-01-01", periods=60)
        # Persistent: high for 30 days then low for 30 days
        persistent = [2000000] * 30 + [500000] * 30
        daily = pd.DataFrame({"Volume": persistent, "Close": [100] * 60}, index=dates)
        profile = compute_volume_profile(daily)
        assert profile["vol_autocorr"] > 0.5


class TestLearnFingerprints:
    def test_learns_from_sufficient_data(self):
        """With enough buy/sell samples, should produce a fingerprint."""
        np.random.seed(42)
        records = []
        for i in range(30):
            action = "buy" if i % 2 == 0 else "sell"
            # Make buy quarters have higher vol_mean_z
            base = 0.5 if action == "buy" else -0.3
            records.append({
                "institution": "fidelity",
                "ticker": "PGR",
                "quarter": f"20{15 + i // 4}-Q{(i % 4) + 1}",
                "action": action,
                "delta_shares": 10000 if action == "buy" else -10000,
                "delta_pct": 5.0 if action == "buy" else -5.0,
                "vol_mean_z": base + np.random.randn() * 0.3,
                "vol_std_z": np.random.randn() * 0.2,
                "vol_trend": (0.02 if action == "buy" else -0.01) + np.random.randn() * 0.01,
                "vol_autocorr": 0.3 + np.random.randn() * 0.1,
                "high_vol_day_frac": (0.3 if action == "buy" else 0.1) + np.random.rand() * 0.1,
                "low_vol_day_frac": 0.2 + np.random.rand() * 0.1,
                "vol_skew": np.random.randn() * 0.5,
                "vol_kurtosis": np.random.randn(),
                "close_to_close_corr": np.random.randn() * 0.3,
                "up_day_vol_ratio": 1.0 + np.random.randn() * 0.2,
                "end_of_quarter_surge": 1.0 + np.random.randn() * 0.1,
                "intraweek_pattern": np.random.rand() * 0.2,
                "consecutive_high_days": np.random.randint(0, 10),
                "dollar_vol_trend": np.random.randn() * 0.02,
            })

        profiles = pd.DataFrame(records)
        fingerprints = learn_fingerprints(profiles, min_samples=10)
        assert "fidelity" in fingerprints
        assert fingerprints["fidelity"].n_training_samples == 30
        assert fingerprints["fidelity"].auc >= 0.5

    def test_skips_insufficient_data(self):
        profiles = pd.DataFrame({
            "institution": ["fidelity"] * 5,
            "action": ["buy"] * 5,
            **{f: [0.0] * 5 for f in VOLUME_PROFILE_FEATURES},
        })
        fingerprints = learn_fingerprints(profiles, min_samples=10)
        assert "fidelity" not in fingerprints


class TestMatchCurrentVolume:
    def test_returns_probabilities(self):
        # Create a simple trained fingerprint
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from src.model.accumulation import ExecutionFingerprint

        np.random.seed(42)
        X = np.random.randn(50, len(VOLUME_PROFILE_FEATURES))
        y = (X[:, 0] > 0).astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scaler.feature_names_in_ = np.array(VOLUME_PROFILE_FEATURES)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)

        fp = ExecutionFingerprint(
            institution="fidelity",
            model=model,
            scaler=scaler,
            n_training_samples=50,
            auc=0.7,
            feature_importances={},
        )

        # Create current daily data
        dates = pd.bdate_range("2024-10-01", periods=60)
        daily = pd.DataFrame({
            "Volume": np.random.randint(500000, 2000000, 60),
            "Close": 100 + np.cumsum(np.random.randn(60) * 0.5),
        }, index=dates)

        results = match_current_volume("PGR", {"fidelity": fp}, daily)
        assert "fidelity" in results
        assert 0 <= results["fidelity"] <= 1


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
        result = detect_accumulation(holdings, volume_signals, min_streak=2)
        assert len(result) == 1
        assert result[0].direction == "ACCUMULATING"

        result = detect_accumulation(holdings, volume_signals, min_streak=5)
        assert len(result) == 0


class TestSummarizeByStock:
    def test_aggregates_correctly(self):
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
