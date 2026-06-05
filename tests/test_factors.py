"""Tests for factor computations."""

import pandas as pd
import numpy as np
import pytest

from src.data.factors import _compute_momentum, _compute_volatility


class TestMomentum:
    def test_flat_returns(self):
        dates = pd.date_range("2023-01-01", periods=300, freq="B")
        hist = pd.DataFrame({"Close": [100.0] * 300}, index=dates)
        mom = _compute_momentum(hist)
        assert mom["momentum_3m"] == pytest.approx(0.0, abs=0.01)

    def test_positive_momentum(self):
        dates = pd.date_range("2023-01-01", periods=300, freq="B")
        prices = [100 + i * 0.1 for i in range(300)]
        hist = pd.DataFrame({"Close": prices}, index=dates)
        mom = _compute_momentum(hist)
        assert mom["momentum_3m"] > 0
        assert mom["momentum_12m"] > 0

    def test_insufficient_data(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        hist = pd.DataFrame({"Close": [100.0] * 10}, index=dates)
        mom = _compute_momentum(hist)
        assert mom["momentum_12m"] is None

    def test_empty_df(self):
        mom = _compute_momentum(pd.DataFrame())
        assert mom["momentum_3m"] is None


class TestVolatility:
    def test_zero_vol(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        hist = pd.DataFrame({"Close": [100.0] * 100}, index=dates)
        vol = _compute_volatility(hist)
        assert vol == 0.0

    def test_positive_vol(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100))
        hist = pd.DataFrame({"Close": prices}, index=dates)
        vol = _compute_volatility(hist)
        assert vol > 0

    def test_insufficient_data(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        hist = pd.DataFrame({"Close": [100.0] * 10}, index=dates)
        vol = _compute_volatility(hist)
        assert vol is None
