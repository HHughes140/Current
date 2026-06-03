"""Per-carrier historical regression of loss ratios on proxy signals.

For each carrier, we estimate:
    loss_ratio_yoy_change ~ β₁·signal₁ + β₂·signal₂ + ... + α

where signals are the YoY changes in proxy variables (VMT, payrolls, medical CPI,
used car CPI, cat losses), weighted by the carrier's LOB mix.

We use YoY changes rather than levels to avoid spurious regression on trending
variables. Cat losses are used in levels (quarterly total) since they're already
stationary and event-driven.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.carriers import Carrier, get_all_carriers, get_all_signal_names

logger = logging.getLogger(__name__)


@dataclass
class CarrierModel:
    """Fitted regression model for a single carrier."""
    ticker: str
    coefficients: dict[str, float]  # signal_name → coefficient
    intercept: float
    r_squared: float
    r_squared_adj: float
    residual_se: float
    n_obs: int
    signal_names: list[str]


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def build_regression_panel(
    carrier: Carrier,
    fred_quarterly: pd.DataFrame,
    noaa_quarterly: pd.DataFrame,
    edgar_ratios: pd.DataFrame,
) -> pd.DataFrame | None:
    """Build the regression panel for a single carrier.

    Merges proxy signals with historical loss ratios, aligns on quarter,
    and computes the dependent variable (loss ratio YoY change).
    """
    # Get carrier's ratio history
    carrier_ratios = edgar_ratios[edgar_ratios["ticker"] == carrier.ticker].copy()

    if carrier_ratios.empty:
        logger.warning("No ratio data for %s — skipping", carrier.ticker)
        return None

    # Parse period into a datetime for merging
    def period_to_date(period: str) -> pd.Timestamp | None:
        """Convert '2023-Q4' or '2023-FY' to end-of-quarter timestamp."""
        parts = period.split("-")
        if len(parts) != 2:
            return None
        year = int(parts[0])
        q = parts[1]
        if q == "FY":
            # Annual — use Q4
            return pd.Timestamp(year=year, month=12, day=31)
        elif q.startswith("Q"):
            quarter = int(q[1])
            month = quarter * 3
            return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        return None

    carrier_ratios["quarter"] = carrier_ratios["period"].apply(period_to_date)
    carrier_ratios = carrier_ratios.dropna(subset=["quarter"])
    carrier_ratios = carrier_ratios.set_index("quarter").sort_index()

    # Use loss_ratio if available, else try to derive from combined - expense
    if "loss_ratio" in carrier_ratios.columns:
        dep_col = "loss_ratio"
    elif "combined_ratio" in carrier_ratios.columns:
        dep_col = "combined_ratio"
    else:
        logger.warning("No usable ratio column for %s", carrier.ticker)
        return None

    carrier_ratios["dep_var"] = pd.to_numeric(carrier_ratios[dep_col], errors="coerce")
    carrier_ratios = carrier_ratios.dropna(subset=["dep_var"])

    # Compute YoY change in the ratio
    carrier_ratios["dep_var_yoy"] = carrier_ratios["dep_var"].diff(4)  # 4 quarters = 1 year

    # Merge proxy signals
    panel = carrier_ratios[["dep_var", "dep_var_yoy"]].copy()

    # FRED signals — use 1-quarter lag (signals predict next quarter's ratio)
    fred_signals = ["vmt_yoy", "payrolls_yoy", "medical_cpi_yoy", "used_car_cpi_yoy"]
    for sig in fred_signals:
        if sig in fred_quarterly.columns:
            lagged = fred_quarterly[[sig]].shift(1)  # 1-quarter lag
            lagged.columns = [sig]
            panel = panel.join(lagged, how="left")

    # NOAA cat losses — use current quarter (cat events are concurrent)
    if "cat_losses_quarterly" in noaa_quarterly.columns:
        panel = panel.join(
            noaa_quarterly[["cat_losses_quarterly"]], how="left"
        )

    panel = panel.dropna()
    return panel


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_carrier(
    carrier: Carrier,
    panel: pd.DataFrame,
) -> CarrierModel | None:
    """Fit an OLS regression for a single carrier.

    Uses the carrier's LOB-weighted signal selection to determine which
    proxy variables enter the regression and their relative importance.
    """
    signal_weights = carrier.signal_weights()
    available_signals = [s for s in signal_weights if s in panel.columns]

    if len(available_signals) < 2:
        logger.warning(
            "Only %d signals available for %s — need at least 2",
            len(available_signals), carrier.ticker,
        )
        return None

    y = panel["dep_var_yoy"]
    X = panel[available_signals]

    # Drop any remaining NaN rows
    mask = y.notna() & X.notna().all(axis=1)
    y = y[mask]
    X = X[mask]

    if len(y) < 10:
        logger.warning(
            "Only %d observations for %s — need at least 10",
            len(y), carrier.ticker,
        )
        return None

    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        logger.error("OLS fit failed for %s: %s", carrier.ticker, e)
        return None

    coefficients = {
        sig: model.params[sig] for sig in available_signals
    }
    intercept = model.params.get("const", 0.0)

    result = CarrierModel(
        ticker=carrier.ticker,
        coefficients=coefficients,
        intercept=intercept,
        r_squared=model.rsquared,
        r_squared_adj=model.rsquared_adj,
        residual_se=np.sqrt(model.mse_resid),
        n_obs=int(model.nobs),
        signal_names=available_signals,
    )

    logger.info(
        "Fitted %s: R²=%.3f, adj R²=%.3f, SE=%.2f, n=%d",
        carrier.ticker, result.r_squared, result.r_squared_adj,
        result.residual_se, result.n_obs,
    )
    for sig, coef in coefficients.items():
        logger.info("  %s: β=%.4f", sig, coef)

    return result


# ---------------------------------------------------------------------------
# Full regression pipeline
# ---------------------------------------------------------------------------

def fit_all(
    fred_quarterly: pd.DataFrame,
    noaa_quarterly: pd.DataFrame,
    edgar_ratios: pd.DataFrame,
    model_dir: str,
    carriers: list[Carrier] | None = None,
) -> dict[str, CarrierModel]:
    """Fit regression models for all carriers and persist them.

    Returns a dict of {ticker: CarrierModel}.
    """
    if carriers is None:
        carriers = get_all_carriers()

    models: dict[str, CarrierModel] = {}

    for carrier in carriers:
        panel = build_regression_panel(
            carrier, fred_quarterly, noaa_quarterly, edgar_ratios
        )
        if panel is None:
            continue

        model = fit_carrier(carrier, panel)
        if model is None:
            continue

        models[carrier.ticker] = model

    # Persist all models
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, model_path / "carrier_models.joblib")
    logger.info("Saved %d carrier models to %s", len(models), model_path)

    return models


def load_models(model_dir: str) -> dict[str, CarrierModel] | None:
    """Load previously fitted models."""
    path = Path(model_dir) / "carrier_models.joblib"
    if path.exists():
        return joblib.load(path)
    return None
