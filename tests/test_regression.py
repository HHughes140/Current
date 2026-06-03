"""Tests for regression model construction and carrier signal weighting."""

import pytest

from src.carriers import Carrier, get_carrier, CARRIER_REGISTRY


class TestCarrierSignalWeights:
    def test_progressive_auto_heavy(self):
        """Progressive is ~95% auto — VMT and used car CPI should dominate."""
        pgr = get_carrier("PGR")
        weights = pgr.signal_weights()
        # VMT should be the largest single signal
        assert weights["vmt_yoy"] > weights.get("payrolls_yoy", 0)
        assert weights["vmt_yoy"] > weights.get("cat_losses_quarterly", 0)

    def test_rnr_cat_heavy(self):
        """RenaissanceRe is reinsurance — cat losses should dominate."""
        rnr = get_carrier("RNR")
        weights = rnr.signal_weights()
        assert weights["cat_losses_quarterly"] > 0.5

    def test_hartford_workers_comp(self):
        """Hartford has large workers comp book — payrolls should be significant."""
        hig = get_carrier("HIG")
        weights = hig.signal_weights()
        assert weights["payrolls_yoy"] > 0.2

    def test_weights_sum_to_one(self):
        """All carrier signal weights should sum to 1.0."""
        for ticker, carrier in CARRIER_REGISTRY.items():
            weights = carrier.signal_weights()
            total = sum(weights.values())
            assert total == pytest.approx(1.0, abs=0.01), f"{ticker} weights sum to {total}"

    def test_all_carriers_have_weights(self):
        """Every carrier should produce non-empty signal weights."""
        for ticker, carrier in CARRIER_REGISTRY.items():
            weights = carrier.signal_weights()
            assert len(weights) > 0, f"{ticker} has no signal weights"

    def test_no_negative_weights(self):
        """Signal weights should all be non-negative."""
        for ticker, carrier in CARRIER_REGISTRY.items():
            weights = carrier.signal_weights()
            for sig, w in weights.items():
                assert w >= 0, f"{ticker} has negative weight for {sig}: {w}"
