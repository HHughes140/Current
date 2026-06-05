"""Report generator — terminal and markdown output for pressure scores."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

from src.model.pressure_score import PressureResult

logger = logging.getLogger(__name__)


def print_terminal_report(results: list[PressureResult]) -> None:
    """Print a rich terminal table of pressure scores."""
    console = Console()

    console.print()
    console.print(
        Panel(
            f"[bold]Institutional Pressure Score — Insurance Equities[/bold]\n"
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
            style="cyan",
        )
    )

    # Main score table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Ticker", style="bold", width=6)
    table.add_column("Name", width=22)
    table.add_column("IPS", justify="right", width=7)
    table.add_column("Direction", justify="center", width=12)
    table.add_column("Strength", justify="center", width=10)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Vol Spike P", justify="right", width=10)
    table.add_column("Res Z", justify="right", width=7)

    for r in results:
        # Color-code direction
        if r.direction == "ACCUMULATE":
            dir_style = "bold green"
        elif r.direction == "DISTRIBUTE":
            dir_style = "bold red"
        else:
            dir_style = "dim"

        # Color-code strength
        strength_styles = {
            "STRONG": "bold",
            "MODERATE": "",
            "WEAK": "dim",
            "NEGLIGIBLE": "dim italic",
        }

        # Color-code score
        if r.score > 0:
            score_str = f"+{r.score:.0f}"
            score_style = "green" if r.score > 20 else ""
        elif r.score < 0:
            score_str = f"{r.score:.0f}"
            score_style = "red" if r.score < -20 else ""
        else:
            score_str = "0"
            score_style = "dim"

        table.add_row(
            r.ticker,
            r.name,
            Text(score_str, style=score_style),
            Text(r.direction, style=dir_style),
            Text(r.strength, style=strength_styles.get(r.strength, "")),
            f"{r.confidence:.0%}",
            f"{r.volume_spike_prob:.0%}",
            f"{r.residual_z:+.2f}",
        )

    console.print(table)

    # Actionable signals — stocks with strong scores
    strong = [r for r in results if r.strength in ("STRONG", "MODERATE")]
    if strong:
        console.print()
        console.print("[bold cyan]Actionable Signals[/bold cyan]")
        console.print()

        for r in strong:
            dir_word = "accumulating" if r.direction == "ACCUMULATE" else "distributing"
            color = "green" if r.direction == "ACCUMULATE" else "red"

            console.print(f"  [{color} bold]{r.ticker}[/{color} bold] ({r.name})")
            console.print(f"    Score: {r.score:+.0f} — Institutions {dir_word}")

            if r.top_institutions:
                console.print(f"    Key movers: {', '.join(r.top_institutions)}")

            # Component breakdown
            top_components = sorted(
                r.components.items(), key=lambda x: abs(x[1]), reverse=True
            )[:3]
            drivers = ", ".join(
                f"{k}: {v:+.2f}" for k, v in top_components
            )
            console.print(f"    Drivers: {drivers}")
            console.print()

    # ETF sector flow summary
    console.print("[dim]Score range: -100 (distribution) to +100 (accumulation)[/dim]")
    console.print()


def print_detail_report(result: PressureResult) -> None:
    """Print a detailed deep-dive for a single stock."""
    console = Console()
    color = "green" if result.direction == "ACCUMULATE" else ("red" if result.direction == "DISTRIBUTE" else "white")

    console.print()
    console.print(Panel(
        f"[bold]{result.ticker}[/bold] — {result.name}\n"
        f"[{color}]IPS: {result.score:+.0f} | {result.direction} | {result.strength}[/{color}]",
    ))

    # Component table
    comp_table = Table(title="Score Components", show_header=True)
    comp_table.add_column("Component", width=25)
    comp_table.add_column("Value", justify="right", width=10)
    comp_table.add_column("Contribution", justify="right", width=12)

    from src.model.pressure_score import DEFAULT_WEIGHTS
    for comp_name, comp_val in sorted(result.components.items(), key=lambda x: abs(x[1]), reverse=True):
        weight = DEFAULT_WEIGHTS.get(comp_name, 0)
        contribution = comp_val * weight * 100
        comp_table.add_row(
            comp_name.replace("_", " ").title(),
            f"{comp_val:+.3f}",
            f"{contribution:+.1f}",
        )

    console.print(comp_table)

    # Institution activity
    if result.top_institutions:
        console.print()
        console.print("[bold]Top Institutions:[/bold]")
        for inst in result.top_institutions:
            console.print(f"  - {inst}")

    console.print()
    console.print(f"Volume spike probability: {result.volume_spike_prob:.0%}")
    console.print(f"Residual z-score: {result.residual_z:+.3f}")
    console.print(f"Signal confidence: {result.confidence:.0%}")
    console.print()


def generate_markdown_report(results: list[PressureResult], output_path: str) -> Path:
    """Generate a markdown report file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Institutional Pressure Score Report",
        "",
        f"**Generated:** {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Scores",
        "",
        "| Ticker | Name | IPS | Direction | Strength | Confidence | Vol Spike P |",
        "|--------|------|-----|-----------|----------|------------|-------------|",
    ]

    for r in results:
        score_str = f"{r.score:+.0f}"
        lines.append(
            f"| {r.ticker} | {r.name} | {score_str} | {r.direction} | "
            f"{r.strength} | {r.confidence:.0%} | {r.volume_spike_prob:.0%} |"
        )

    lines.append("")

    # Actionable signals
    strong = [r for r in results if r.strength in ("STRONG", "MODERATE")]
    if strong:
        lines.append("## Actionable Signals")
        lines.append("")
        for r in strong:
            dir_word = "accumulating" if r.direction == "ACCUMULATE" else "distributing"
            lines.append(f"### {r.ticker} ({r.name}) — IPS {r.score:+.0f}")
            lines.append("")
            lines.append(f"Institutions are **{dir_word}**. Residual z-score: {r.residual_z:+.3f}")
            if r.top_institutions:
                lines.append(f"\nKey movers: {', '.join(r.top_institutions)}")
            lines.append("")
            top_comp = sorted(r.components.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            lines.append("**Drivers:**")
            for k, v in top_comp:
                lines.append(f"- {k.replace('_', ' ').title()}: {v:+.3f}")
            lines.append("")

    path.write_text("\n".join(lines))
    logger.info("Markdown report written to %s", path)
    return path
