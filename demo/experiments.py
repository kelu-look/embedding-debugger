"""
Pre-built experiments demonstrating key embedding failure modes.

Run from the command line:
    python -m demo.experiments

Or import individual experiments for use in notebooks.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from embedding_debugger.models import EmbeddingModel
from embedding_debugger.perturbation import PerturbationSuite, ORDER_PERTURBATIONS
from embedding_debugger.similarity import SimilarityAnalyzer, top_k_neighbors
from embedding_debugger.retrieval import RetrievalDebugger
from embedding_debugger.clustering import ClusteringAnalyzer
from embedding_debugger.drift import DriftAnalyzer
from .datasets import load_order_blind_pairs, load_faq, load_news, load_nli, load_products

console = Console()


# ==================================================================
# Experiment 1: Order-Blind Retrieval
# ==================================================================

def experiment_order_blindness(model_name: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    """
    Show that models treat meaning-opposite sentences as highly similar.
    e.g., "The dog bit the man" ≈ "The man bit the dog"
    """
    console.rule("[bold red]Experiment 1: Order Blindness")
    texts, meta = load_order_blind_pairs()
    model = EmbeddingModel(model_name)
    originals = meta["originals"]
    perturbed = meta["perturbed"]

    all_texts = originals + perturbed
    vecs = model.encode(all_texts)
    n = len(originals)

    rows = []
    for i in range(n):
        sim = float(vecs[i] @ vecs[n + i])
        rows.append(
            {
                "original": originals[i],
                "order_reversed": perturbed[i],
                "cosine_similarity": round(sim, 4),
                "semantically_same": False,
            }
        )

    df = pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False)

    t = Table(title=f"Order-Blind Pairs [{model_name}]", show_lines=True)
    t.add_column("Original", style="green", max_width=40)
    t.add_column("Reversed", style="red", max_width=40)
    t.add_column("Cosine Sim", justify="right")
    for _, row in df.iterrows():
        t.add_row(row["original"], row["order_reversed"], str(row["cosine_similarity"]))
    console.print(t)
    console.print(f"\n[yellow]Mean similarity for meaning-opposite pairs:[/yellow] {df['cosine_similarity'].mean():.4f}")
    console.print("[dim]A perfect model would assign ~0 similarity to these pairs.[/dim]\n")
    return df


# ==================================================================
# Experiment 2: Perturbation Robustness
# ==================================================================

def experiment_perturbation_robustness(
    model_name: str = "all-MiniLM-L6-v2",
    n_samples: int = 10,
) -> pd.DataFrame:
    """
    Compare similarity preservation across perturbation types.
    Order perturbations should ideally show LOW similarity (meaning changes),
    but most models show HIGH similarity (order-blind).
    """
    console.rule("[bold yellow]Experiment 2: Perturbation Robustness")
    texts, _ = load_faq()
    texts = texts[:n_samples]
    model = EmbeddingModel(model_name)
    suite = PerturbationSuite(model.encode)
    df = suite.summary_table(texts)

    t = Table(title=f"Perturbation Robustness [{model_name}]", show_lines=True)
    t.add_column("Perturbation", style="cyan")
    t.add_column("Mean Sim", justify="right")
    t.add_column("Min Sim", justify="right")
    t.add_column("Std Sim", justify="right")
    for _, row in df.iterrows():
        color = "red" if row["mean_sim"] > 0.7 and "shuffle" in row["perturbation"] else "white"
        t.add_row(
            row["perturbation"],
            f"[{color}]{row['mean_sim']}[/{color}]",
            str(row["min_sim"]),
            str(row["std_sim"]),
        )
    console.print(t)
    console.print("[dim]Red = high similarity despite meaning-altering perturbation (bad)[/dim]\n")
    return df


# ==================================================================
# Experiment 3: FAQ Retrieval Failure Analysis
# ==================================================================

def experiment_retrieval_failures(model_name: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    """
    Build a FAISS index over FAQ answers.
    Query with original questions and check if the correct answer is rank-1.
    Then perturb the questions and track rank shift.
    """
    console.rule("[bold blue]Experiment 3: Retrieval Failure Analysis")
    texts, meta = load_faq()
    questions = meta["questions"]
    answers = meta["answers"]
    n = len(questions)

    model = EmbeddingModel(model_name)
    q_vecs = model.encode(questions, is_query=True)
    a_vecs = model.encode(answers)

    debugger = RetrievalDebugger(answers, a_vecs)
    expected_indices = list(range(n))
    failures, df = debugger.analyze_failures(questions, q_vecs, expected_indices, k=n)

    mrr = debugger.mrr_at_k(df, k=5)
    r1 = debugger.recall_at_k(df, k=1)
    r5 = debugger.recall_at_k(df, k=5)

    console.print(f"[bold]Results for {model_name}[/bold]")
    console.print(f"  Recall@1  = {r1:.3f}")
    console.print(f"  Recall@5  = {r5:.3f}")
    console.print(f"  MRR@5     = {mrr:.3f}")

    failures_df = df[df["is_failure"]]
    if len(failures_df) > 0:
        console.print(f"\n[red]Retrieval failures (expected rank != 0):[/red]")
        for _, row in failures_df.head(5).iterrows():
            console.print(f"  Q: {row['query']}")
            console.print(f"  Expected: {row['expected_doc']}")
            console.print(f"  Top-1 score: {row['top1_score']}")
            console.print()
    else:
        console.print("[green]All questions retrieved correct answer at rank 1![/green]")

    return df


# ==================================================================
# Experiment 4: Model Comparison
# ==================================================================

def experiment_model_comparison(
    models: List[str] = None,
    n_texts: int = 15,
) -> pd.DataFrame:
    """
    Compare SBERT, GTE, E5 on order-blind pairs.
    Shows which model is most order-sensitive.
    """
    console.rule("[bold magenta]Experiment 4: Model Comparison on Order-Blind Pairs")
    if models is None:
        models = ["all-MiniLM-L6-v2", "gte-small", "e5-small-v2"]

    texts, meta = load_order_blind_pairs()
    originals = meta["originals"][:n_texts]
    perturbed = meta["perturbed"][:n_texts]

    rows = []
    all_vecs: dict[str, np.ndarray] = {}

    for mname in models:
        model = EmbeddingModel(mname)
        vecs_o = model.encode(originals)
        vecs_p = model.encode(perturbed)
        all_vecs[mname] = model.encode(originals + perturbed)
        sims = (vecs_o * vecs_p).sum(axis=1)
        rows.append(
            {
                "model": mname,
                "mean_sim_order_pairs": round(float(sims.mean()), 4),
                "min_sim": round(float(sims.min()), 4),
                "max_sim": round(float(sims.max()), 4),
            }
        )

    df = pd.DataFrame(rows).sort_values("mean_sim_order_pairs")

    t = Table(title="Order-Blindness by Model (lower = better)", show_lines=True)
    t.add_column("Model", style="cyan")
    t.add_column("Mean Sim", justify="right")
    t.add_column("Min Sim", justify="right")
    t.add_column("Max Sim", justify="right")
    for _, row in df.iterrows():
        color = "green" if row["mean_sim_order_pairs"] < 0.6 else "red"
        t.add_row(
            row["model"],
            f"[{color}]{row['mean_sim_order_pairs']}[/{color}]",
            str(row["min_sim"]),
            str(row["max_sim"]),
        )
    console.print(t)
    console.print("[dim]Lower mean_sim on order-reversed pairs = more sensitive to word order[/dim]\n")

    # Drift comparison
    if len(models) >= 2:
        console.print("[bold]Pairwise model drift:[/bold]")
        drift_df = DriftAnalyzer.compare_models(
            originals,
            {m: EmbeddingModel(m).encode(originals) for m in models},
        )
        console.print(drift_df.to_string(index=False))

    return df


# ==================================================================
# Experiment 5: Cluster Geometry
# ==================================================================

def experiment_cluster_geometry(model_name: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    """
    Visualize news headline clusters and check if category structure emerges.
    """
    console.rule("[bold green]Experiment 5: Cluster Geometry (News Headlines)")
    texts, meta = load_news()
    categories = meta["categories"]
    model = EmbeddingModel(model_name)
    vecs = model.encode(texts)
    ca = ClusteringAnalyzer(texts, vecs)
    elbow_df = ca.elbow_dataframe()
    console.print(elbow_df.to_string(index=False))

    best_k_val = elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"]
    console.print(f"\n[bold]Best k by silhouette:[/bold] {best_k_val}")
    console.print(f"[dim]True number of categories: 5[/dim]\n")

    df = ca.build_dataframe(method="pca", k=5, labels=categories)
    return df


# ==================================================================
# Run all experiments
# ==================================================================

def run_all(model_name: str = "all-MiniLM-L6-v2") -> None:
    console.print("\n[bold underline]Embedding Debugger — Diagnostic Experiments[/bold underline]\n")
    experiment_order_blindness(model_name)
    experiment_perturbation_robustness(model_name)
    experiment_retrieval_failures(model_name)
    experiment_model_comparison()
    experiment_cluster_geometry(model_name)
    console.print("[bold green]All experiments complete.[/bold green]")


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "all-MiniLM-L6-v2"
    run_all(model)
