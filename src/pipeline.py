"""CLI entry point — orchestrate ingest, model, score, and report.

Usage:
    pressure run              # Full pipeline
    pressure ingest           # Pull 13F, factors, volume, options, ETF, rates
    pressure model            # Fit demand model
    pressure score            # Compute pressure scores
    pressure report           # Generate report
    pressure detail TICKER    # Deep dive on a single stock
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import pandas as pd
import yaml

from src.universe import INSTITUTION_REGISTRY, ALL_TICKERS, INSURANCE_UNIVERSE
from src.data import edgar_13f, factors, volume, options, etf_flows, fred_rates, cache
from src.model import demand_model, residual, volume_model, pressure_score, accumulation
from src import report

logger = logging.getLogger(__name__)


def _load_config(config_path: str = "config.yaml") -> dict:
    local = Path(config_path).with_name("config.local.yaml")
    if local.exists():
        config_path = str(local)
    path = Path(config_path)
    if not path.exists():
        click.echo(f"Config not found: {config_path}", err=True)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--config", default="config.yaml", help="Path to config file")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, config, verbose):
    """Institutional Pressure Score — Insurance Equities."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = _load_config(config)
    ctx.obj["data_dir"] = ctx.obj["config"].get("data_dir", "data")
    ctx.obj["model_dir"] = ctx.obj["config"].get("model_dir", "data/models")


@cli.command()
@click.option("--force", is_flag=True, help="Ignore cache, re-download everything")
@click.pass_context
def ingest(ctx, force):
    """Pull data from all sources (13F, factors, volume, options, ETF, rates)."""
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]

    # 13F holdings
    click.echo("Ingesting 13F holdings...")
    holdings = edgar_13f.refresh_all(data_dir, force=force)
    click.echo(f"  13F: {len(holdings)} holdings records")

    # Factor data
    click.echo("Pulling factor data...")
    fac = factors.refresh_all(data_dir, force=force)
    click.echo(f"  Factors: {len(fac)} stocks")

    # Volume signals
    click.echo("Computing volume signals...")
    vol = volume.refresh_all(data_dir, force=force)
    click.echo(f"  Volume: {len(vol)} stocks")

    # Options activity
    click.echo("Pulling options data...")
    opts = options.refresh_all(data_dir, force=force)
    click.echo(f"  Options: {len(opts)} stocks")

    # ETF flows
    click.echo("Estimating ETF flows...")
    etf = etf_flows.refresh(data_dir, force=force)
    click.echo(f"  ETFs: {len(etf)} tracked")

    # Interest rates
    api_key = config.get("fred", {}).get("api_key", "")
    if api_key and api_key != "YOUR_FRED_API_KEY_HERE":
        click.echo("Pulling rate data...")
        rates = fred_rates.refresh(api_key, data_dir, force=force)
        click.echo(f"  Rates: {len(rates)} days")
    else:
        click.echo("  Skipping rates (no FRED API key)")

    click.echo("Data ingestion complete.")


@cli.command()
@click.pass_context
def model(ctx):
    """Fit institutional demand models."""
    data_dir = ctx.obj["data_dir"]
    model_dir = ctx.obj["model_dir"]

    holdings = edgar_13f.load_holdings(data_dir)
    fac = factors.load_snapshot(data_dir)
    rates_df = fred_rates.load_panel(data_dir)

    if holdings is None or fac is None:
        click.echo("Error: Run 'pressure ingest' first.", err=True)
        sys.exit(1)

    click.echo("Fitting demand models...")
    results = demand_model.fit_and_save(holdings, fac, model_dir, rates_df)

    for style, res in results.items():
        click.echo(
            f"  {style}: AUC={res.auc_walkforward:.3f}, n={res.n_train}, "
            f"features={res.n_features}"
        )
        top3 = list(res.feature_importances.items())[:3]
        for feat, imp in top3:
            click.echo(f"    {feat}: {imp:.4f}")


@cli.command()
@click.pass_context
def score(ctx):
    """Compute current Institutional Pressure Scores."""
    data_dir = ctx.obj["data_dir"]

    # Load all data
    holdings = edgar_13f.load_holdings(data_dir)
    fac = factors.load_snapshot(data_dir) or pd.DataFrame()
    vol = volume.load_latest(data_dir) or pd.DataFrame()
    opts = options.load_latest(data_dir) or pd.DataFrame()
    etf = etf_flows.load_latest(data_dir) or pd.DataFrame()

    if holdings is None or holdings.empty:
        click.echo("Error: No holdings data. Run 'pressure ingest' first.", err=True)
        sys.exit(1)

    model_dir = ctx.obj["model_dir"]

    # Use trained demand model if available, otherwise neutral baseline
    click.echo("Computing institutional residuals...")
    trained = demand_model.load_models(model_dir)
    holdings_with_expected = holdings.copy()

    if trained:
        click.echo("  Using trained demand model for expected probabilities")
        # Apply trained model to compute expected buy probability per holding
        for style_key in ["passive", "active"]:
            if style_key not in trained:
                continue
            m = trained[style_key]
            style_mask = holdings_with_expected["style"] == style_key
            style_holdings = holdings_with_expected[style_mask]
            if style_holdings.empty:
                continue

            features = [f for f in demand_model.FACTOR_FEATURES if f in fac.columns]
            if features and not fac.empty:
                merged = style_holdings.merge(fac[["ticker"] + features], on="ticker", how="left")
                X, used_features = demand_model._prepare_features(merged)
                for f in m["scaler"].feature_names_in_:
                    if f not in X.columns:
                        X[f] = 0
                X = X[list(m["scaler"].feature_names_in_)]
                X_scaled = m["scaler"].transform(X)
                proba = m["model"].predict_proba(X_scaled)[:, 1]
                holdings_with_expected.loc[style_mask, "expected_buy_prob"] = proba
    else:
        click.echo("  No trained model found — using neutral baseline (run 'pressure model' first for better residuals)")
        holdings_with_expected["expected_buy_prob"] = 0.5

    resid = residual.compute_institution_residuals(
        holdings_with_expected,
        holdings_with_expected,
    )
    agg_resid = residual.aggregate_residuals(resid)

    if agg_resid.empty:
        click.echo("Error: Could not compute residuals.", err=True)
        sys.exit(1)

    # Ownership concentration
    ownership = residual.compute_ownership_concentration(holdings)

    # Accumulation detection — link 13F streaks to current volume
    click.echo("Detecting accumulation patterns...")
    accum_signals = accumulation.detect_accumulation(holdings, vol)
    accum_summary = accumulation.summarize_by_stock(accum_signals)

    # Volume predictions
    click.echo("Predicting volume spikes...")
    vol_features = volume_model.build_volume_features(agg_resid, fac, vol, opts)
    vol_preds = volume_model.predict_volume_spikes(vol_features)

    # Composite pressure score
    click.echo("Computing pressure scores...")
    results = pressure_score.compute_pressure_scores(
        residuals=agg_resid,
        volume_signals=vol,
        options_signals=opts,
        etf_signals=etf,
        factors=fac,
        ownership_changes=ownership,
        volume_predictions=vol_preds,
        holdings=holdings,
    )

    # Output
    report.print_terminal_report(results)

    # Print accumulation signals if any
    if accum_signals:
        report.print_accumulation_report(accum_signals, accum_summary)

    md_path = report.generate_markdown_report(
        results, f"{data_dir}/reports/pressure_latest.md"
    )
    click.echo(f"Report saved to {md_path}")


@cli.command("report")
@click.pass_context
def generate_report(ctx):
    """Generate report from latest scores."""
    ctx.invoke(score)


@cli.command()
@click.argument("ticker")
@click.pass_context
def detail(ctx, ticker):
    """Deep dive on a single stock's pressure score."""
    ticker = ticker.upper()
    if ticker not in INSURANCE_UNIVERSE:
        click.echo(f"Unknown ticker: {ticker}", err=True)
        click.echo(f"Available: {', '.join(ALL_TICKERS)}")
        sys.exit(1)

    data_dir = ctx.obj["data_dir"]

    # Recompute scores (or load cached — for now recompute)
    holdings = edgar_13f.load_holdings(data_dir)
    fac = factors.load_snapshot(data_dir) or pd.DataFrame()
    vol = volume.load_latest(data_dir) or pd.DataFrame()
    opts = options.load_latest(data_dir) or pd.DataFrame()
    etf = etf_flows.load_latest(data_dir) or pd.DataFrame()

    if holdings is None or holdings.empty:
        click.echo("Error: No data. Run 'pressure ingest' first.", err=True)
        sys.exit(1)

    holdings_with_expected = holdings.copy()
    holdings_with_expected["expected_buy_prob"] = 0.5

    resid = residual.compute_institution_residuals(
        holdings_with_expected, holdings_with_expected
    )
    agg_resid = residual.aggregate_residuals(resid)
    ownership = residual.compute_ownership_concentration(holdings)

    vol_features = volume_model.build_volume_features(agg_resid, fac, vol, opts)
    vol_preds = volume_model.predict_volume_spikes(vol_features)

    results = pressure_score.compute_pressure_scores(
        residuals=agg_resid,
        volume_signals=vol,
        options_signals=opts,
        etf_signals=etf,
        factors=fac,
        ownership_changes=ownership,
        volume_predictions=vol_preds,
        holdings=holdings,
    )

    # Find the specific ticker
    target = [r for r in results if r.ticker == ticker]
    if not target:
        click.echo(f"No data available for {ticker}")
        return

    report.print_detail_report(target[0])


@cli.command()
@click.option("--force", is_flag=True)
@click.pass_context
def run(ctx, force):
    """Full pipeline: ingest -> model -> score -> report."""
    ctx.invoke(ingest, force=force)
    ctx.invoke(model)
    ctx.invoke(score)


if __name__ == "__main__":
    cli()
