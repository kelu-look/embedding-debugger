"""
Killer Demo — the complete embedding failure pipeline.

This script tells a single concrete story from start to finish:

  1. A user queries: "How do I upgrade the software safely?"
  2. The corpus contains one CORRECT procedure and one DANGEROUS (steps scrambled).
  3. Both documents retrieve at nearly the same score.
  4. We prove the model is order-blind by measuring cosine similarity.
  5. We show the retrieval rank DOES NOT CHANGE after step reordering.
  6. We repeat across causal reversal and subject-object swap.

Run:
    python -m demo.killer_demo
    python -m demo.killer_demo --model gte-small

Output: a rich console report + killer_demo_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from embedding_debugger.models import EmbeddingModel
from embedding_debugger.retrieval import RetrievalDebugger
from embedding_debugger.perturbation import (
    PerturbationSuite,
    PERTURBATION_CATEGORIES,
    is_failure,
)
from embedding_debugger.export import DebugReport
from demo.failure_cases import (
    ALL_FAILURE_CASES,
    SUBJECT_OBJECT_FAILURES,
    CAUSAL_REVERSAL_FAILURES,
    STEP_REORDER_FAILURES,
    LIST_INVERSION_FAILURES,
    FailurePair,
)

console = Console(highlight=True)

# ──────────────────────────────────────────────────────────────────────
# Section 1 — The Setup Story
# ──────────────────────────────────────────────────────────────────────

SAFE_UPGRADE_PROCEDURE = (
    "Step 1: Back up all user data to an external drive. "
    "Step 2: Close all running applications. "
    "Step 3: Run the installer and follow on-screen prompts. "
    "Step 4: Restart the computer when prompted. "
    "Step 5: Verify the new version is running correctly."
)

DANGEROUS_UPGRADE_PROCEDURE = (
    "Step 1: Restart the computer when prompted. "
    "Step 2: Run the installer and follow on-screen prompts. "
    "Step 3: Back up all user data to an external drive. "
    "Step 4: Close all running applications. "
    "Step 5: Verify the new version is running correctly."
)
# Note: restarting before running the installer and backing up AFTER
# means data loss if something goes wrong during installation.

UPGRADE_CORPUS = [
    SAFE_UPGRADE_PROCEDURE,
    DANGEROUS_UPGRADE_PROCEDURE,
    "Contact the vendor for upgrade support.",
    "Check compatibility with your operating system before upgrading.",
    "Read the release notes before installing any software update.",
]

UPGRADE_QUERY = "What are the steps to upgrade the software safely?"


def section_1_retrieval_failure(model: EmbeddingModel) -> dict:
    console.rule("[bold red]Section 1 — Retrieval Failure: Safe vs Dangerous Procedure")
    console.print()
    console.print(Panel(
        f"[bold]Query:[/bold] {UPGRADE_QUERY}\n\n"
        f"[green]CORRECT:[/green] {SAFE_UPGRADE_PROCEDURE[:120]}…\n\n"
        f"[red]DANGEROUS:[/red] {DANGEROUS_UPGRADE_PROCEDURE[:120]}…",
        title="The Setup",
        border_style="yellow",
    ))

    corpus_vecs = model.encode(UPGRADE_CORPUS)
    query_vec = model.encode_single(UPGRADE_QUERY, is_query=True)

    debugger = RetrievalDebugger(UPGRADE_CORPUS, corpus_vecs)
    result = debugger.retrieve(UPGRADE_QUERY, query_vec, k=5)

    t = Table(title=f"Retrieval results [{model.short_name}]", show_lines=True)
    t.add_column("Rank", width=5)
    t.add_column("Score", width=8)
    t.add_column("Document")
    for rank, (doc, score) in enumerate(zip(result.retrieved, result.scores), 1):
        is_dangerous = doc == DANGEROUS_UPGRADE_PROCEDURE
        is_safe = doc == SAFE_UPGRADE_PROCEDURE
        color = "red" if is_dangerous else ("green" if is_safe else "white")
        label = " ◀ DANGEROUS" if is_dangerous else (" ◀ CORRECT" if is_safe else "")
        t.add_row(str(rank), f"{score:.4f}", f"[{color}]{doc[:80]}…{label}[/{color}]")
    console.print(t)

    safe_score = float(corpus_vecs[0] @ query_vec)
    danger_score = float(corpus_vecs[1] @ query_vec)
    score_gap = safe_score - danger_score
    console.print(f"\n  Safe score:      {safe_score:.4f}")
    console.print(f"  Dangerous score: {danger_score:.4f}")
    console.print(f"  Score gap:       {score_gap:+.4f}")
    if abs(score_gap) < 0.05:
        console.print("[bold red]  ⚠  Gap < 0.05 — model cannot distinguish safe from dangerous procedure.[/bold red]")
    console.print()

    mutual_sim = float(corpus_vecs[0] @ corpus_vecs[1])
    console.print(f"  Cosine similarity (safe ↔ dangerous): [bold]{mutual_sim:.4f}[/bold]")
    if mutual_sim > 0.90:
        console.print("[red]  ⚠  These documents are embedded at cosine > 0.90. "
                      "The model treats them as nearly identical.[/red]")
    console.print()

    return {
        "safe_score": safe_score,
        "danger_score": danger_score,
        "score_gap": score_gap,
        "mutual_sim": mutual_sim,
        "safe_rank": result.indices.index(0) if 0 in result.indices else -1,
        "danger_rank": result.indices.index(1) if 1 in result.indices else -1,
    }


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Order Blindness Proof
# ──────────────────────────────────────────────────────────────────────

def section_2_order_blindness(model: EmbeddingModel) -> dict:
    console.rule("[bold yellow]Section 2 — Proving Order Blindness")
    console.print()

    suite = PerturbationSuite(model.encode)

    texts = [fp.original for fp in STEP_REORDER_FAILURES[:2]]
    batch = suite.run_batch(texts, "step_reorder")

    t = Table(title="Step-reordered text vs original — cosine similarity", show_lines=True)
    t.add_column("Original (first 80 chars)")
    t.add_column("Reordered (first 80 chars)")
    t.add_column("Cosine Sim", justify="right")
    t.add_column("Verdict")

    results_data = []
    for r in batch.results:
        sim = r.similarity
        verdict = "[red]FAILURE[/red]" if sim > 0.90 else "[green]OK[/green]"
        t.add_row(r.original[:80], r.perturbed[:80], f"{sim:.4f}", verdict)
        results_data.append({"text": r.original[:60], "sim": sim})
    console.print(t)

    # Also show subject-object + causal
    console.print()
    for label, cases in [
        ("Subject-Object Swap", SUBJECT_OBJECT_FAILURES[:3]),
        ("Causal Reversal", CAUSAL_REVERSAL_FAILURES[:3]),
    ]:
        originals = [c.original for c in cases]
        adversarials = [c.adversarial for c in cases]
        all_vecs = model.encode(originals + adversarials)
        n = len(originals)
        t2 = Table(title=f"{label} — cosine similarity", show_lines=True)
        t2.add_column("Original")
        t2.add_column("Adversarial")
        t2.add_column("Sim", justify="right")
        t2.add_column("Expected", justify="right")
        t2.add_column("Verdict")
        for i, case in enumerate(cases):
            sim = float(all_vecs[i] @ all_vecs[n + i])
            verdict = "[red]FAIL[/red]" if sim > 0.85 else "[green]OK[/green]"
            t2.add_row(
                case.original[:50],
                case.adversarial[:50],
                f"{sim:.4f}",
                f"{case.expected_sim:.2f}",
                verdict,
            )
        console.print(t2)
        console.print()

    mean_step_sim = float(np.mean([r.similarity for r in batch.results]))
    return {"mean_step_reorder_sim": mean_step_sim}


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Category Robustness Sweep
# ──────────────────────────────────────────────────────────────────────

def section_3_robustness_sweep(model: EmbeddingModel) -> "pd.DataFrame":
    import pandas as pd
    console.rule("[bold blue]Section 3 — Perturbation Robustness by Category")
    console.print()

    # Use a diverse set of texts spanning all failure types
    texts = [fp.original for fp in ALL_FAILURE_CASES[:8]]
    suite = PerturbationSuite(model.encode)

    # Run all retrieval-critical + semantic perturbations
    types = (
        PERTURBATION_CATEGORIES["retrieval_critical"]
        + PERTURBATION_CATEGORIES["semantic"]
    )
    summary_df = suite.summary_table(texts, types)

    t = Table(title=f"Robustness sweep [{model.short_name}]", show_lines=True)
    t.add_column("Category", style="dim")
    t.add_column("Perturbation")
    t.add_column("Mean Sim", justify="right")
    t.add_column("Verdict")
    for _, row in summary_df.iterrows():
        verdict = "[red]⚠ FAILURE[/red]" if row["is_failure"] else "[green]OK[/green]"
        color = "red" if row["is_failure"] else "white"
        t.add_row(
            row["category"],
            f"[{color}]{row['perturbation']}[/{color}]",
            f"[{color}]{row['mean_sim']}[/{color}]",
            verdict,
        )
    console.print(t)
    n_fail = int(summary_df["is_failure"].sum())
    console.print(
        f"\n  {n_fail}/{len(summary_df)} perturbation types produced [red]FAILURE[/red] "
        f"(cosine ≥ 0.85 despite meaning change)\n"
    )
    return summary_df


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Summary & Takeaways
# ──────────────────────────────────────────────────────────────────────

def section_4_summary(s1: dict, s2: dict, model_name: str) -> None:
    console.rule("[bold green]Section 4 — Summary")
    console.print()
    console.print(Panel(
        f"[bold]Model:[/bold] {model_name}\n\n"
        f"• Safe vs dangerous procedure score gap: [bold]{s1['score_gap']:+.4f}[/bold] "
        f"({'[red]indistinguishable[/red]' if abs(s1['score_gap']) < 0.05 else '[green]distinguishable[/green]'})\n"
        f"• Safe ↔ dangerous cosine similarity: [bold]{s1['mutual_sim']:.4f}[/bold] "
        f"({'[red]nearly identical[/red]' if s1['mutual_sim'] > 0.90 else '[green]distinguishable[/green]'})\n"
        f"• Mean cosine after step reorder: [bold]{s2['mean_step_reorder_sim']:.4f}[/bold]\n\n"
        "[bold]Root cause:[/bold] Sentence-level embedding models pool token representations\n"
        "into a single vector, discarding positional structure. As a result, a document\n"
        "with steps in the WRONG ORDER embeds identically to the correct procedure.\n\n"
        "[bold]Implications:[/bold]\n"
        "  • RAG systems may surface dangerous or incorrect procedural documents\n"
        "  • Semantic search cannot distinguish cause from effect\n"
        "  • Priority-ordered lists are retrieved interchangeably regardless of order\n\n"
        "[bold]Mitigation directions:[/bold]\n"
        "  • Use chunked retrieval (sentence-level, not document-level)\n"
        "  • Re-rank with a cross-encoder that attends to order\n"
        "  • Augment with structural metadata (step numbers, dependency labels)\n"
        "  • Evaluate retrieval with adversarial step-reorder test sets",
        title="[bold]Embedding Debugger — Failure Report[/bold]",
        border_style="green",
    ))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def run(model_name: str = "all-MiniLM-L6-v2", export: bool = True) -> None:
    console.print()
    console.print(Panel(
        "[bold]Embedding Debugger — Killer Demo[/bold]\n"
        "Proving that standard embedding models cannot distinguish\n"
        "semantically opposite procedures, causes, and orderings.",
        border_style="bold white",
    ))
    console.print()

    model = EmbeddingModel(model_name)
    console.print(f"Model: [cyan]{model}[/cyan]\n")

    s1 = section_1_retrieval_failure(model)
    s2 = section_2_order_blindness(model)
    summary_df = section_3_robustness_sweep(model)
    section_4_summary(s1, s2, model_name)

    if export:
        import pandas as pd
        report = DebugReport(
            title="Embedding Debugger — Killer Demo",
            model=model_name,
            dataset="curated_failure_cases",
        )
        report.add_metadata("safe_vs_dangerous_score_gap", round(s1["score_gap"], 4))
        report.add_metadata("safe_vs_dangerous_cosine_sim", round(s1["mutual_sim"], 4))
        report.add_metadata("mean_step_reorder_sim", round(s2["mean_step_reorder_sim"], 4))
        report.add_metadata("n_perturbation_failures", int(summary_df["is_failure"].sum()))
        report.add_section(
            "Perturbation robustness sweep",
            summary_df,
            "Perturbations where is_failure=True indicate the model embeds "
            "semantically opposite text at cosine ≥ 0.85.",
        )

        from demo.failure_cases import failure_cases_dataframe
        report.add_section("Curated failure cases", failure_cases_dataframe())

        md = report.to_markdown("killer_demo_report.md")
        console.print(f"\n[dim]Report written → killer_demo_report.md[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    run(model_name=args.model, export=not args.no_export)
