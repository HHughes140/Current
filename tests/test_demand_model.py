"""Tests for demand model training data construction."""

import pandas as pd
import numpy as np
import pytest

from src.model.demand_model import build_training_data, _prepare_features


class TestBuildTrainingData:
    def test_creates_bought_flag(self):
        holdings = pd.DataFrame({
            "institution": ["a", "a"],
            "ticker": ["PGR", "TRV"],
            "quarter": ["2024-Q1", "2024-Q1"],
            "delta_shares": [1000, -500],
            "style": ["active", "active"],
        })
        factors = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "quarter": ["2024-Q1", "2024-Q1"],
            "market_cap": [1e11, 5e10],
            "pe_trailing": [20.0, 15.0],
        })
        result = build_training_data(holdings, factors)
        pgr = result[result["ticker"] == "PGR"]
        trv = result[result["ticker"] == "TRV"]
        assert pgr["bought"].iloc[0] == 1
        assert trv["bought"].iloc[0] == 0

    def test_empty_input(self):
        result = build_training_data(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_skips_nan_deltas(self):
        holdings = pd.DataFrame({
            "institution": ["a", "a"],
            "ticker": ["PGR", "TRV"],
            "quarter": ["2024-Q1", "2024-Q1"],
            "delta_shares": [1000, np.nan],
            "style": ["active", "active"],
        })
        factors = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "quarter": ["2024-Q1", "2024-Q1"],
            "market_cap": [1e11, 5e10],
        })
        result = build_training_data(holdings, factors)
        assert len(result) == 1


class TestPrepareFeatures:
    def test_fills_nan_with_median(self):
        df = pd.DataFrame({
            "market_cap": [1e11, 2e11, np.nan],
            "pe_trailing": [20.0, np.nan, 15.0],
        })
        X, features = _prepare_features(df)
        assert not X.isna().any().any()
        assert "market_cap" in features

    def test_only_known_features(self):
        df = pd.DataFrame({
            "market_cap": [1e11],
            "random_col": [42],
        })
        X, features = _prepare_features(df)
        assert "market_cap" in features
        assert "random_col" not in features
