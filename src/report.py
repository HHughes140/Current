"""Report generator — PM-readable output for nowcast results.

Outputs:
1. Rich terminal table with color-coded surprise flags
2. Markdown file for sharing
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.model.nowcast import NowcastResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_terminal_report(results: list[NowcastResult]) -> None:
    """Print a rich terminal table summarizing nowcast results."""
    console = Console()

    console.print()
    console.print(
        f"[bold]Insurance Loss Ratio Nowcast[/bold] — {datetime.now():%Y-%m-%d %H:%M}",
        style="cyan",
    )
    console.print()

    # Summary table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Ticker", style="bold", width=6)
    table.add_column("Carrier", width=22)
    table.add_column("Est LR", justify="right", width=8)
    table.add_column("±1 SE", justify="right", width=14)
    table.add_column("Cons LR", justify="right", width=8)
    table.add_column("Delta", justify="right", width=8)
    table.add_column("Signal", justify="center", width=10)
    table.add_column("R²", justify="right", width=6)

    for r in results:
        # Color-code the surprise flag
        if r.surprise_flag == "ADVERSE":
            flag_style = "bold red"
            flag_text = "ADVERSE"
        elif r.surprise_flag == "FAVORABLE":
            flag_style = "bold green"
            flag_text = "FAVORABLE"
        elif r.surprise_flag == "IN_LINE":
            flag_style = "dim"
            flag_text = "IN LINE"
        else:
            flag_style = "yellow"
            flag_text = "NO CONS"

        delta_str = f"{r.delta_vs_consensus:+.1f}pp" if r.delta_vs_consensus is not None else "—"
        cons_str = f"{r.consensus_loss_ratio:.1f}" if r.consensus_loss_ratio is not None else "—"

        ci = f"{r.confidence_interval_1se[0]:.1f}–{r.confidence_interval_1se[1]:.1f}"

        table.add_row(
            r.ticker,
            r.carrier_name,
            f"{r.loss_ratio_est:.1f}",
            ci,
            cons_str,
            delta_str,
            Text(flag_text, style=flag_style),
            f"{r.r_squared:.2f}",
        )

    console.print(table)
    console.print()

    # Key drivers for surprise candidates
    surprises = [r for r in results if r.surprise_flag in ("ADVERSE", "FAVORABLE")]
    if surprises:
        console.print("[bold]Key Drivers[/bold]", style="cyan")
        console.print()
        for r in surprises:
            direction = "above" if r.surprise_flag == "ADVERSE" else "below"
            console.print(f"  [bold]{r.ticker}[/bold] — Est {direction} consensus by {abs(r.delta_vs_consensus):.1f}pp")
            # Sort contributions by absolute magnitude
            sorted_contribs = sorted(
                r.signal_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            for sig, contrib in sorted_contribs[:3]:
                direction_word = "adding" if contrib > 0 else "reducing"
                console.print(f"    {_signal_label(sig)}: {direction_word} {abs(contrib):.2f}pp")
            console.print()


def _signal_label(signal_name: str) -> str:
    """Human-readable label for a signal name."""
    labels = {
        "vmt_yoy": "VMT (vehicle miles)",
        "payrolls_yoy": "Nonfarm payrolls",
        "medical_cpi_yoy": "Medical CPI",
        "used_car_cpi_yoy": "Used car prices",
        "cat_losses_quarterly": "Cat losses",
    }
    return labels.get(signal_name, signal_name)


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def generate_markdown_report(results: list[NowcastResult], output_path: str) -> Path:
    """Generate a markdown report file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Insurance Loss Ratio Nowcast",
        f"",
        f"**Generated:** {datetime.now():%Y-%m-%d %H:%M}",
        f"",
        f"## Summary",
        f"",
        f"| Ticker | Carrier | Est LR | ±1 SE | Consensus | Delta | Signal |",
        f"|--------|---------|--------|-------|-----------|-------|--------|",
    ]

    for r in results:
        cons = f"{r.consensus_loss_ratio:.1f}" if r.consensus_loss_ratio is not None else "—"
        delta = f"{r.delta_vs_consensus:+.1f}pp" if r.delta_vs_consensus is not None else "—"
        ci = f"{r.confidence_interval_1se[0]:.1f}–{r.confidence_interval_1se[1]:.1f}"
        flag = r.surprise_flag.replace("_", " ")
        lines.append(
            f"| {r.ticker} | {r.carrier_name} | {r.loss_ratio_est:.1f} | {ci} | {cons} | {delta} | {flag} |"
        )

    lines.append("")

    # Key drivers section
    surprises = [r for r in results if r.surprise_flag in ("ADVERSE", "FAVORABLE")]
    if surprises:
        lines.append("## Earnings Watch")
        lines.append("")
        for r in surprises:
            direction = "above" if r.surprise_flag == "ADVERSE" else "below"
            lines.append(
                f"### {r.ticker} ({r.carrier_name}) — {r.surprise_flag}"
            )
            lines.append("")
            lines.append(
                f"Estimated loss ratio of **{r.loss_ratio_est:.1f}%** is "
                f"**{abs(r.delta_vs_consensus):.1f}pp {direction}** consensus "
                f"({r.consensus_loss_ratio:.1f}%). Model R² = {r.r_squared:.2f}."
            )
            lines.append("")
            lines.append("**Key drivers:**")
            sorted_contribs = sorted(
                r.signal_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            for sig, contrib in sorted_contribs[:3]:
                direction_word = "Adding" if contrib > 0 else "Reducing"
                lines.append(
                    f"- {_signal_label(sig)}: {direction_word} {abs(contrib):.2f}pp to loss ratio change"
                )
            lines.append("")

    # Model details
    lines.append("## Model Details")
    lines.append("")
    lines.append("| Ticker | R² | SE | Obs | Signals Used |")
    lines.append("|--------|-----|-----|-----|--------------|")
    for r in results:
        signals = ", ".join(r.signal_contributions.keys())
        lines.append(
            f"| {r.ticker} | {r.r_squared:.3f} | {r.residual_se:.2f} | — | {signals} |"
        )
    lines.append("")

    path.write_text("\n".join(lines))
    logger.info("Markdown report written to %s", path)
    return path
