"""Tests for NOAA damage parsing and quarterly aggregation."""

import pandas as pd
import pytest

from src.data.noaa import parse_damage, _process_raw, _aggregate_quarterly


class TestParseDamage:
    def test_thousands(self):
        assert parse_damage("25K") == 0.025

    def test_millions(self):
        assert parse_damage("1.5M") == 1.5

    def test_billions(self):
        assert parse_damage("2B") == 2000.0

    def test_zero(self):
        assert parse_damage("0") == 0.0

    def test_empty(self):
        assert parse_damage("") == 0.0

    def test_none(self):
        assert parse_damage(None) == 0.0

    def test_plain_number(self):
        # Plain dollar amount → convert to millions
        assert parse_damage("5000000") == 5.0

    def test_lowercase(self):
        assert parse_damage("100k") == 0.1

    def test_decimal_k(self):
        assert parse_damage("2.5K") == 0.0025


class TestProcessRaw:
    def _make_raw(self, event_type="Tornado", damage="10M", yearmonth="202301"):
        return pd.DataFrame({
            "BEGIN_YEARMONTH": [yearmonth],
            "BEGIN_DAY": ["15"],
            "EVENT_TYPE": [event_type],
            "STATE": ["TEXAS"],
            "DAMAGE_PROPERTY": [damage],
            "DAMAGE_CROPS": ["0"],
        })

    def test_filters_irrelevant_events(self):
        df = self._make_raw(event_type="Dense Fog")
        result = _process_raw(df)
        assert len(result) == 0

    def test_keeps_relevant_events(self):
        df = self._make_raw(event_type="Tornado")
        result = _process_raw(df)
        assert len(result) == 1

    def test_insured_ratio_applied(self):
        df = self._make_raw(event_type="Tornado", damage="100M")
        result = _process_raw(df)
        # Tornado insured ratio = 0.55
        assert result["insured_loss_m"].iloc[0] == pytest.approx(55.0, rel=0.01)

    def test_date_parsing(self):
        df = self._make_raw(yearmonth="202307")
        result = _process_raw(df)
        assert result["date"].iloc[0].month == 7
        assert result["date"].iloc[0].year == 2023


class TestQuarterlyAggregation:
    def test_sums_within_quarter(self):
        events = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-04-01"]),
            "year": [2023, 2023, 2023],
            "month": [1, 2, 4],
            "EVENT_TYPE": ["Tornado", "Hail", "Tornado"],
            "STATE": ["TX", "TX", "TX"],
            "property_damage_m": [10.0, 5.0, 20.0],
            "insured_loss_m": [5.5, 3.25, 11.0],
        })
        quarterly = _aggregate_quarterly(events)
        # Q1 should sum Jan + Feb
        q1 = quarterly.loc[quarterly.index.month == 3]
        assert len(q1) == 1
        assert q1["cat_losses_quarterly"].iloc[0] == pytest.approx(8.75, rel=0.01)
