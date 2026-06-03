"""Tests for FRED data transformations."""

import pandas as pd
import numpy as np
import pytest

from src.data.fred import compute_yoy, to_quarterly


class TestComputeYoY:
    def test_basic_yoy(self):
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        values = [100.0] * 12 + [110.0] * 12
        df = pd.DataFrame({"val": values}, index=dates)
        yoy = compute_yoy(df, "val")
        # First 12 months should be NaN
        assert yoy.iloc[:12].isna().all()
        # Months 13+ should be +10%
        assert yoy.iloc[12] == pytest.approx(10.0, rel=0.01)

    def test_handles_nan(self):
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        values = [100.0] * 12 + [110.0] * 12
        values[5] = np.nan
        df = pd.DataFrame({"val": values}, index=dates)
        yoy = compute_yoy(df, "val")
        # Month 18 (5 + 12) should be NaN because the base month was NaN
        assert pd.isna(yoy.iloc[17])


class TestToQuarterly:
    def test_mean_aggregation(self):
        dates = pd.date_range("2023-01-01", periods=6, freq="MS")
        df = pd.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}, index=dates)
        quarterly = to_quarterly(df, "val", agg="mean")
        assert len(quarterly) == 2
        # Q1 mean = (10+20+30)/3 = 20
        assert quarterly["val"].iloc[0] == pytest.approx(20.0)
        # Q2 mean = (40+50+60)/3 = 50
        assert quarterly["val"].iloc[1] == pytest.approx(50.0)

    def test_sum_aggregation(self):
        dates = pd.date_range("2023-01-01", periods=3, freq="MS")
        df = pd.DataFrame({"val": [10.0, 20.0, 30.0]}, index=dates)
        quarterly = to_quarterly(df, "val", agg="sum")
        assert quarterly["val"].iloc[0] == pytest.approx(60.0)
