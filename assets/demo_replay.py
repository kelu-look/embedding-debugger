"""
Scripted replay for GIF recording — prints actual benchmark output
section by section with deliberate pauses. No model inference needed.

Usage (inside asciinema rec):
    python assets/demo_replay.py
"""

import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich import print as rprint

console = Console(width=100, highlight=False)
MODEL = "all-MiniLM-L6-v2"


def pause(s: float = 1.0):
    time.sleep(s)


def header():
    console.print()
    console.print(Panel(
        "[bold]Embedding Debugger — Killer Demo[/bold]\n"
        "Diagnosing retrieval failures hidden by high embedding similarity.",
        border_style="bold white",
        width=100,
    ))
    console.print(f"\nModel: [cyan]EmbeddingModel(name={MODEL!r}, dim=384, device='cpu')[/cyan]\n")
    pause(2.5)


def section1():
    console.rule("[bold red]Section 1 — Retrieval Risk: Near-Tie Between Safe and Dangerous Procedures")
    console.print()
    console.print(Panel(
        "[bold]Query:[/bold] What are the steps to upgrade the software safely?\n\n"
        "[green]CORRECT:[/green]  Step 1: Back up all user data. Step 2: Close all applications.\n"
        "          Step 3: Run the installer. Step 4: Restart. Step 5: Verify.\n\n"
        "[red]DANGEROUS:[/red] Step 1: Restart. Step 2: Run the installer.\n"
        "          Step 3: Back up all user data. Step 4: Close all applications. Step 5: Verify.",
        title="The Setup",
        border_style="yellow",
        width=100,
    ))
    pause(3.0)

    t = Table(title=f"Retrieval results [{MODEL}]", show_lines=True, width=100)
    t.add_column("Rank", width=5)
    t.add_column("Score", width=8)
    t.add_column("Document")
    t.add_row("1", "0.9897", "[green]Step 1: Back up all user data…  ◀ CORRECT[/green]")
    t.add_row("2", "0.9865", "[red]Step 1: Restart when prompted…  ◀ DANGEROUS[/red]")
    t.add_row("3", "0.8234", "Read the release notes before installing…")
    t.add_row("4", "0.7891", "Check compatibility with your OS before upgrading…")
    t.add_row("5", "0.7543", "Contact the vendor for upgrade support…")
    console.print(t)

    console.print(f"\n  Safe score:      [bold]0.9897[/bold]")
    console.print(f"  Dangerous score: [bold]0.9865[/bold]")
    console.print(f"  Score gap:       [bold]+0.0032[/bold]")
    console.print("[bold red]  ⚠  Gap < 0.05 — model weakly separates safe from dangerous procedure.[/bold red]")
    console.print()
    console.print(f"  Cosine similarity (safe ↔ dangerous): [bold]0.9866[/bold]")
    console.print("[red]  ⚠  These documents are embedded at cosine > 0.90. "
                  "The model treats them as nearly identical.[/red]")
    console.print()
    pause(3.5)


def section2():
    console.rule("[bold yellow]Section 2 — Measuring Order Sensitivity")
    console.print()

    t = Table(title="Adversarial pairs — cosine similarity", show_lines=True, width=100)
    t.add_column("Original")
    t.add_column("Adversarial")
    t.add_column("Cosine", justify="right")
    t.add_column("Risk Type")
    t.add_column("Verdict")
    data = [
        ("The company acquired the startup.", "The startup acquired the company.", "0.9925", "entity reversal",   "[red]FAIL[/red]"),
        ("The patient sued the doctor.",      "The doctor sued the patient.",      "0.9941", "entity reversal",   "[red]FAIL[/red]"),
        ("Revenue exceeded expenses.",        "Expenses exceeded revenue.",        "0.9959", "entity reversal",   "[red]FAIL[/red]"),
        ("Smoking causes lung cancer.",       "Lung cancer causes smoking.",       "0.9779", "causal reversal",   "[red]FAIL[/red]"),
        ("Power outage caused server crash.", "Server crash caused power outage.", "0.9756", "causal reversal",   "[red]FAIL[/red]"),
        ("Step order: backup→install→verify", "Step order: verify→install→backup", "0.9969", "procedure reorder", "[red]FAIL[/red]"),
    ]
    for row in data:
        t.add_row(*row)
    console.print(t)
    console.print("\n  [bold]All 15 curated adversarial pairs score > 0.90 across all 4 tested models.[/bold]")
    console.print("  [dim]This pattern appears across the tested models and warrants targeted "
                  "debugging in retrieval pipelines.[/dim]")
    console.print()
    pause(3.5)


def section3():
    console.rule("[bold blue]Section 3 — Perturbation Robustness Sweep (all categories)")
    console.print()
    console.print("  [dim]lexical = surface noise (high sim expected ✅) | "
                  "semantic/retrieval_critical = meaning change (high sim = 🚨)[/dim]\n")

    t = Table(title=f"Perturbation sweep [{MODEL}]", show_lines=True, width=100)
    t.add_column("Category", style="dim")
    t.add_column("Perturbation")
    t.add_column("Mean Sim", justify="right")
    t.add_column("Verdict")
    data = [
        # Lexical — should be high sim (no meaning change)
        ("lexical",            "lowercase",             "0.9997", "[green]✅ OK[/green]"),
        ("lexical",            "strip_punctuation",     "0.9991", "[green]✅ OK[/green]"),
        ("lexical",            "add_typos",             "0.9954", "[green]✅ OK[/green]"),
        # Structural — high sim reveals order-insensitive behavior (expected but noteworthy)
        ("structural",         "shuffle_words",         "0.9821", "[yellow]⚠ order-insensitive[/yellow]"),
        ("structural",         "reverse_sentences",     "0.9774", "[yellow]⚠ order-insensitive[/yellow]"),
        # Retrieval-critical — meaning changes, high sim = failure
        ("retrieval_critical", "step_reorder",          "0.9969", "[red]🚨 FAILURE[/red]"),
        ("retrieval_critical", "causal_reversal",       "0.9847", "[red]🚨 FAILURE[/red]"),
        ("retrieval_critical", "subject_object_swap",   "0.9093", "[red]🚨 FAILURE[/red]"),
        # Semantic — meaning changes, high sim = failure
        ("semantic",           "antonym_swap",          "0.9967", "[red]🚨 FAILURE[/red]"),
        ("semantic",           "inject_negation",       "0.9124", "[red]🚨 FAILURE[/red]"),
        ("semantic",           "inject_contradiction",  "0.8543", "[red]🚨 FAILURE[/red]"),
    ]
    for row in data:
        t.add_row(*row)
    console.print(t)
    console.print(
        "\n  ✅ Correctly robust to surface noise (lexical)  |  "
        "[red]🚨 6/6 semantic or retrieval-critical perturbations remain highly similar "
        "under the current threshold (cosine ≥ 0.85)[/red]\n"
    )
    pause(3.0)


def section4():
    console.rule("[bold green]Section 4 — Summary")
    console.print()
    console.print(Panel(
        f"[bold]Model:[/bold] {MODEL}\n\n"
        "• Safe vs dangerous procedure score gap: [bold]+0.0032[/bold] ([red]very small margin[/red])\n"
        "• Safe ↔ dangerous cosine similarity:    [bold]0.9866[/bold]  ([red]nearly identical[/red])\n"
        "• Step reordering remains highly similar under embedding-only retrieval.\n\n"
        "[bold]Observed behavior:[/bold] The pooled representation provides weak\n"
        "separation under meaning-changing order edits.\n\n"
        "[bold]Implications:[/bold]\n"
        "  • RAG systems may surface dangerous or incorrect procedural documents\n"
        "  • Semantic search cannot distinguish cause from effect\n"
        "  • Priority-ordered lists may be difficult to distinguish under embedding-only retrieval\n\n"
        "[bold]Mitigation directions:[/bold]\n"
        "  • Use chunked retrieval (sentence-level, not document-level)\n"
        "  • Re-rank with a cross-encoder that attends to order\n"
        "  • Augment with structural metadata (step numbers, dependency labels)\n"
        "  • Evaluate with adversarial step-reorder test sets\n\n"
        "[dim]Report written → killer_demo_report.md[/dim]",
        title="[bold]Embedding Debugger — Failure Report[/bold]",
        border_style="green",
        width=100,
    ))
    pause(4.0)


if __name__ == "__main__":
    header()
    section1()
    section2()
    section3()
    section4()
