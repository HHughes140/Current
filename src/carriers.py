"""Carrier registry, LOB weighting matrix, and signal-to-LOB mapping.

LOB weights are approximate and derived from each carrier's most recent
10-K segment disclosures. They should be updated annually.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Lines of business
# ---------------------------------------------------------------------------

LINES_OF_BUSINESS = [
    "personal_auto",
    "commercial_auto",
    "homeowners",
    "commercial_property",
    "workers_comp",
    "general_liability",
    "reinsurance_property",
    "specialty",
]

# ---------------------------------------------------------------------------
# Signal-to-LOB mapping — which proxy signals are relevant to each LOB
# ---------------------------------------------------------------------------

# Each LOB maps to a dict of {signal_name: relative_importance} within that LOB.
# Importances are normalized to sum to 1.0 within each LOB during model construction.
LOB_SIGNAL_MAP: dict[str, dict[str, float]] = {
    "personal_auto": {
        "vmt_yoy": 0.45,          # Vehicle miles traveled → frequency
        "used_car_cpi_yoy": 0.30, # Used car prices → severity (total loss cost)
        "medical_cpi_yoy": 0.25,  # Medical inflation → BI severity
    },
    "commercial_auto": {
        "vmt_yoy": 0.35,
        "used_car_cpi_yoy": 0.30,
        "medical_cpi_yoy": 0.35,
    },
    "homeowners": {
        "cat_losses_quarterly": 0.70,  # NOAA catastrophe property damage
        "medical_cpi_yoy": 0.30,       # Liability component
    },
    "commercial_property": {
        "cat_losses_quarterly": 0.85,
        "payrolls_yoy": 0.15,          # Exposure growth proxy
    },
    "workers_comp": {
        "payrolls_yoy": 0.55,          # Payroll = premium base and exposure
        "medical_cpi_yoy": 0.45,       # Medical severity
    },
    "general_liability": {
        "payrolls_yoy": 0.50,
        "medical_cpi_yoy": 0.50,
    },
    "reinsurance_property": {
        "cat_losses_quarterly": 0.95,
        "payrolls_yoy": 0.05,
    },
    "specialty": {
        "payrolls_yoy": 0.40,
        "medical_cpi_yoy": 0.30,
        "cat_losses_quarterly": 0.30,
    },
}


# ---------------------------------------------------------------------------
# Carrier definitions
# ---------------------------------------------------------------------------

@dataclass
class Carrier:
    ticker: str
    name: str
    cik: str
    lob_weights: dict[str, float]  # LOB name → fraction of net premiums earned

    def signal_weights(self) -> dict[str, float]:
        """Compute the carrier-level weight for each proxy signal.

        This blends LOB weights with the signal-to-LOB mapping to produce
        a single dict of {signal_name: weight} for regression/nowcast.
        Weights are normalized to sum to 1.0.
        """
        raw: dict[str, float] = {}
        for lob, lob_weight in self.lob_weights.items():
            signals = LOB_SIGNAL_MAP.get(lob, {})
            for signal, importance in signals.items():
                raw[signal] = raw.get(signal, 0.0) + lob_weight * importance
        # Normalize
        total = sum(raw.values())
        if total > 0:
            return {k: v / total for k, v in raw.items()}
        return raw


# ---------------------------------------------------------------------------
# Default carrier universe
# ---------------------------------------------------------------------------

CARRIER_REGISTRY: dict[str, Carrier] = {
    "PGR": Carrier(
        ticker="PGR",
        name="Progressive",
        cik="0000080661",
        lob_weights={
            "personal_auto": 0.80,
            "commercial_auto": 0.15,
            "homeowners": 0.05,
        },
    ),
    "TRV": Carrier(
        ticker="TRV",
        name="Travelers",
        cik="0000086312",
        lob_weights={
            "commercial_property": 0.30,
            "commercial_auto": 0.15,
            "workers_comp": 0.15,
            "general_liability": 0.15,
            "personal_auto": 0.10,
            "homeowners": 0.15,
        },
    ),
    "ALL": Carrier(
        ticker="ALL",
        name="Allstate",
        cik="0000899629",
        lob_weights={
            "personal_auto": 0.65,
            "homeowners": 0.25,
            "commercial_auto": 0.05,
            "general_liability": 0.05,
        },
    ),
    "CB": Carrier(
        ticker="CB",
        name="Chubb",
        cik="0000896159",
        lob_weights={
            "commercial_property": 0.25,
            "general_liability": 0.20,
            "workers_comp": 0.10,
            "commercial_auto": 0.10,
            "personal_auto": 0.10,
            "homeowners": 0.10,
            "specialty": 0.15,
        },
    ),
    "RNR": Carrier(
        ticker="RNR",
        name="RenaissanceRe",
        cik="0000913144",
        lob_weights={
            "reinsurance_property": 0.75,
            "specialty": 0.25,
        },
    ),
    "AIG": Carrier(
        ticker="AIG",
        name="American International Group",
        cik="0000005272",
        lob_weights={
            "commercial_property": 0.25,
            "general_liability": 0.25,
            "workers_comp": 0.10,
            "commercial_auto": 0.10,
            "personal_auto": 0.10,
            "homeowners": 0.10,
            "specialty": 0.10,
        },
    ),
    "HIG": Carrier(
        ticker="HIG",
        name="Hartford Financial",
        cik="0000874766",
        lob_weights={
            "workers_comp": 0.30,
            "commercial_auto": 0.20,
            "general_liability": 0.20,
            "commercial_property": 0.20,
            "specialty": 0.10,
        },
    ),
    "CNA": Carrier(
        ticker="CNA",
        name="CNA Financial",
        cik="0000021175",
        lob_weights={
            "commercial_property": 0.25,
            "general_liability": 0.25,
            "workers_comp": 0.20,
            "commercial_auto": 0.15,
            "specialty": 0.15,
        },
    ),
}


def get_carrier(ticker: str) -> Carrier:
    """Look up a carrier by ticker symbol."""
    ticker = ticker.upper()
    if ticker not in CARRIER_REGISTRY:
        raise ValueError(f"Unknown carrier: {ticker}. Known: {list(CARRIER_REGISTRY)}")
    return CARRIER_REGISTRY[ticker]


def get_all_carriers() -> list[Carrier]:
    """Return all carriers in the registry."""
    return list(CARRIER_REGISTRY.values())


def get_all_signal_names() -> list[str]:
    """Return the unique set of proxy signal names used across all LOBs."""
    signals: set[str] = set()
    for lob_signals in LOB_SIGNAL_MAP.values():
        signals.update(lob_signals.keys())
    return sorted(signals)
