"""CLI entry point — orchestrate refresh, regress, nowcast, and report.

Usage:
    nowcast run          # Full pipeline: refresh → regress → nowcast → report
    nowcast refresh      # Pull latest data only
    nowcast regress      # Fit/update regression models
    nowcast nowcast      # Run current-quarter nowcast
    nowcast report       # Generate report from latest nowcast
    nowcast verify-edgar # Review and correct EDGAR extractions
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import yaml

from src.carriers import CARRIER_REGISTRY, get_all_carriers
from src.data import fred, noaa, edgar, cache
from src.model import regression, nowcast
from src import report

logger = logging.getLogger(__name__)


def _load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    # Try local override first
    local = Path(config_path).with_suffix("").with_name("config.local.yaml")
    if local.exists():
        config_path = str(local)

    path = Path(config_path)
    if not path.exists():
        click.echo(f"Config file not found: {config_path}", err=True)
        click.echo("Copy config.yaml to config.local.yaml and set your FRED API key.", err=True)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
@click.option("--config", default="config.yaml", help="Path to config file")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, config, verbose):
    """Insurance Loss Ratio Nowcasting System."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = _load_config(config)
    ctx.obj["data_dir"] = ctx.obj["config"].get("data_dir", "data")
    ctx.obj["model_dir"] = ctx.obj["config"].get("model_dir", "data/models")


@cli.command()
@click.option("--force", is_flag=True, help="Ignore cache and re-download everything")
@click.pass_context
def refresh(ctx, force):
    """Pull latest data from all sources (FRED, NOAA, EDGAR)."""
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]

    api_key = config.get("fred", {}).get("api_key", "")
    if not api_key or api_key == "YOUR_FRED_API_KEY_HERE":
        click.echo("Error: Set your FRED API key in config.local.yaml", err=True)
        sys.exit(1)

    start_year = config.get("history", {}).get("start_year", 2014)
    end_year = config.get("history", {}).get("end_year", 2025)

    click.echo("Refreshing FRED data...")
    fred_q = fred.refresh_all(api_key, data_dir, force=force)
    click.echo(f"  FRED: {len(fred_q)} quarters")

    click.echo("Refreshing NOAA Storm Events...")
    noaa_q = noaa.refresh(data_dir, start_year=start_year, end_year=end_year, force=force)
    click.echo(f"  NOAA: {len(noaa_q)} quarters")

    click.echo("Refreshing EDGAR filings...")
    carrier_ciks = {
        t: c.cik for t, c in CARRIER_REGISTRY.items()
        if t in config.get("carriers", list(CARRIER_REGISTRY.keys()))
    }
    edgar_df = edgar.refresh_all(carrier_ciks, data_dir, force=force)
    click.echo(f"  EDGAR: {len(edgar_df)} ratio extractions")

    click.echo("Data refresh complete.")


@cli.command()
@click.pass_context
def regress(ctx):
    """Fit carrier-specific regression models."""
    data_dir = ctx.obj["data_dir"]
    model_dir = ctx.obj["model_dir"]

    fred_q = fred.load_quarterly(data_dir)
    noaa_q = noaa.load_quarterly(data_dir)
    edgar_df = edgar.load_ratios(data_dir)

    if fred_q is None or noaa_q is None or edgar_df is None:
        click.echo("Error: Run 'nowcast refresh' first to pull data.", err=True)
        sys.exit(1)

    click.echo("Fitting regression models...")
    models = regression.fit_all(fred_q, noaa_q, edgar_df, model_dir)

    click.echo(f"\nFitted {len(models)} carrier models:")
    for ticker, model in models.items():
        click.echo(
            f"  {ticker}: R²={model.r_squared:.3f}, "
            f"SE={model.residual_se:.2f}, n={model.n_obs}"
        )


@cli.command("nowcast")
@click.pass_context
def run_nowcast(ctx):
    """Run current-quarter loss ratio nowcast."""
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    model_dir = ctx.obj["model_dir"]

    models = regression.load_models(model_dir)
    if models is None:
        click.echo("Error: Run 'nowcast regress' first to fit models.", err=True)
        sys.exit(1)

    fred_q = fred.load_quarterly(data_dir)
    fred_m = fred.load_monthly(data_dir)
    noaa_q = noaa.load_quarterly(data_dir)
    edgar_df = edgar.load_ratios(data_dir)

    if fred_q is None or noaa_q is None or edgar_df is None:
        click.echo("Error: Run 'nowcast refresh' first.", err=True)
        sys.exit(1)

    # Load consensus from config
    consensus = config.get("consensus", {})
    # Filter out None values
    consensus = {k: v for k, v in consensus.items() if v is not None}

    results = nowcast.nowcast_all(
        models, fred_q, noaa_q, edgar_df,
        consensus=consensus if consensus else None,
        fred_monthly=fred_m,
    )

    report.print_terminal_report(results)

    # Save markdown report
    md_path = report.generate_markdown_report(
        results, f"{data_dir}/reports/nowcast_latest.md"
    )
    click.echo(f"Report saved to {md_path}")


@cli.command("report")
@click.option("--output", "-o", default=None, help="Output path for markdown report")
@click.pass_context
def generate_report(ctx, output):
    """Generate report from latest nowcast results."""
    # This re-runs the nowcast with cached data
    ctx.invoke(run_nowcast)


@cli.command()
@click.pass_context
def run(ctx):
    """Full pipeline: refresh → regress → nowcast → report."""
    ctx.invoke(refresh)
    ctx.invoke(regress)
    ctx.invoke(run_nowcast)


@cli.command("verify-edgar")
@click.pass_context
def verify_edgar(ctx):
    """Review and correct EDGAR ratio extractions."""
    data_dir = ctx.obj["data_dir"]
    df = edgar.load_ratios(data_dir)

    if df is None or df.empty:
        click.echo("No EDGAR data found. Run 'nowcast refresh' first.")
        return

    # Show low-confidence extractions
    if "confidence" in df.columns:
        low_conf = df[df["confidence"] < 0.7].sort_values(["ticker", "period"])
    else:
        low_conf = pd.DataFrame()

    if low_conf.empty:
        click.echo("All extractions have high confidence. No review needed.")
        click.echo(f"\nTotal extractions: {len(df)}")
        return

    click.echo(f"Found {len(low_conf)} low-confidence extractions:\n")

    for _, row in low_conf.iterrows():
        ticker = row.get("ticker", "?")
        period = row.get("period", "?")
        lr = row.get("loss_ratio", "—")
        cr = row.get("combined_ratio", "—")
        conf = row.get("confidence", 0)

        click.echo(f"  {ticker} {period}: LR={lr}, CR={cr} (confidence={conf:.0%})")

    click.echo(f"\nTo correct a value, use:")
    click.echo(f'  python -c "from src.data.edgar import save_manual_override; '
               f'save_manual_override(\'data\', \'TICKER\', \'YYYY-QN\', loss_ratio=XX.X)"')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
