"""
Killer Demo page — the complete embedding failure pipeline in the UI.

Tells the story:
  query → wrong retrieval → high cosine despite semantic error → perturbation explains failure
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from embedding_debugger.perturbation import PerturbationSuite, PERTURBATION_CATEGORIES
from embedding_debugger.retrieval import RetrievalDebugger
from embedding_debugger.export import DebugReport, perturbation_report
from app.pages._shared import encode_texts, get_model, model_selector
from demo.failure_cases import (
    ALL_FAILURE_CASES,
    SUBJECT_OBJECT_FAILURES,
    CAUSAL_REVERSAL_FAILURES,
    STEP_REORDER_FAILURES,
    LIST_INVERSION_FAILURES,
    failure_cases_dataframe,
)

# ──────────────────────────────────────────────────────────────────────
# Corpus for the main story
# ──────────────────────────────────────────────────────────────────────

SAFE_PROCEDURE = (
    "Step 1: Back up all user data to an external drive. "
    "Step 2: Close all running applications. "
    "Step 3: Run the installer and follow on-screen prompts. "
    "Step 4: Restart the computer when prompted. "
    "Step 5: Verify the new version is running correctly."
)

DANGEROUS_PROCEDURE = (
    "Step 1: Restart the computer when prompted. "
    "Step 2: Run the installer and follow on-screen prompts. "
    "Step 3: Back up all user data to an external drive. "
    "Step 4: Close all running applications. "
    "Step 5: Verify the new version is running correctly."
)

UPGRADE_CORPUS = [
    SAFE_PROCEDURE,
    DANGEROUS_PROCEDURE,
    "Contact the vendor for upgrade support.",
    "Check compatibility with your operating system before upgrading.",
    "Read the release notes before installing any software update.",
]

QUERY = "What are the steps to upgrade the software safely?"


def render() -> None:
    st.title("🎯 Killer Demo")
    st.markdown(
        "**The complete embedding failure pipeline** — from a plausible query "
        "to a dangerous retrieval result, explained step by step."
    )

    with st.sidebar:
        st.header("Settings")
        model_name = model_selector(key="kd_model")
        failure_threshold = st.slider("Failure threshold (cosine)", 0.7, 0.99, 0.85, 0.01, key="kd_thresh")

    # ── Step 1: The Setup ──────────────────────────────────────────────
    st.divider()
    st.markdown("## Step 1 — The Setup")
    st.info(f"**Query:** {QUERY}")

    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ CORRECT procedure (safe order)")
        st.code(SAFE_PROCEDURE.replace(". Step", ".\nStep"), language="text")
    with col2:
        st.error("⚠️ DANGEROUS procedure (scrambled order)")
        st.code(DANGEROUS_PROCEDURE.replace(". Step", ".\nStep"), language="text")

    st.caption("Both documents contain identical vocabulary. Only the step order differs.")

    # ── Step 2: Retrieval Results ─────────────────────────────────────
    st.divider()
    st.markdown("## Step 2 — Retrieval Results")

    corpus_vecs = encode_texts(model_name, tuple(UPGRADE_CORPUS))
    query_vec = get_model(model_name).encode_single(QUERY, is_query=True)

    debugger = RetrievalDebugger(UPGRADE_CORPUS, corpus_vecs)
    result = debugger.retrieve(QUERY, query_vec, k=5)

    safe_score = float(corpus_vecs[0] @ query_vec)
    danger_score = float(corpus_vecs[1] @ query_vec)
    mutual_sim = float(corpus_vecs[0] @ corpus_vecs[1])
    score_gap = abs(safe_score - danger_score)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Safe procedure score", f"{safe_score:.4f}")
    c2.metric("Dangerous procedure score", f"{danger_score:.4f}")
    c3.metric("Score gap", f"{score_gap:.4f}", delta_color="inverse",
              delta="-insufficient" if score_gap < 0.05 else "+sufficient")
    c4.metric("Safe ↔ Dangerous cosine", f"{mutual_sim:.4f}")

    if score_gap < 0.05:
        st.error(
            f"⚠️ **Retrieval failure detected** — score gap is only {score_gap:.4f}. "
            "The model cannot distinguish the safe procedure from the dangerous one."
        )
    if mutual_sim > 0.90:
        st.error(
            f"⚠️ **Embedding collapse** — cosine similarity between correct and dangerous "
            f"procedure is {mutual_sim:.4f}. They occupy nearly the same point in embedding space."
        )

    # Retrieval table
    retrieval_data = []
    for rank, (doc, score, idx) in enumerate(zip(result.retrieved, result.scores, result.indices), 1):
        tag = ""
        if idx == 0:
            tag = "✅ CORRECT"
        elif idx == 1:
            tag = "⚠️ DANGEROUS"
        retrieval_data.append({
            "Rank": rank,
            "Score": round(score, 4),
            "Tag": tag,
            "Document (first 90 chars)": doc[:90] + ("…" if len(doc) > 90 else ""),
        })
    ret_df = pd.DataFrame(retrieval_data)
    st.dataframe(ret_df, use_container_width=True, hide_index=True)

    # Score bar chart
    fig = go.Figure()
    colors = ["#4daf4a", "#e41a1c"] + ["#aaaaaa"] * (len(UPGRADE_CORPUS) - 2)
    labels = ["CORRECT", "DANGEROUS"] + [f"Doc {i}" for i in range(2, len(UPGRADE_CORPUS))]
    scores_all = [float(corpus_vecs[i] @ query_vec) for i in range(len(UPGRADE_CORPUS))]
    fig.add_trace(go.Bar(
        x=labels, y=scores_all,
        marker_color=colors,
        text=[f"{s:.4f}" for s in scores_all],
        textposition="outside",
    ))
    fig.update_layout(
        title="Retrieval scores per document",
        yaxis_title="Cosine similarity to query",
        yaxis_range=[0, 1.05],
        showlegend=False,
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Step 3: Why This Happens ──────────────────────────────────────
    st.divider()
    st.markdown("## Step 3 — Why This Happens: Order Blindness Proof")
    st.markdown(
        "We now test whether the model is sensitive to **step reordering** "
        "across a set of known failure cases."
    )

    fail_df = failure_cases_dataframe()
    originals = fail_df["original"].tolist()
    adversarials = fail_df["adversarial"].tolist()
    all_texts = originals + adversarials
    all_vecs = encode_texts(model_name, tuple(all_texts))
    n = len(originals)
    sims = [float(all_vecs[i] @ all_vecs[n + i]) for i in range(n)]

    fail_df["actual_sim"] = [round(s, 4) for s in sims]
    fail_df["failure"] = fail_df["actual_sim"] >= failure_threshold

    # Scatter: expected vs actual similarity
    fig2 = px.scatter(
        fail_df,
        x="expected_sim",
        y="actual_sim",
        color="failure",
        color_discrete_map={True: "#e41a1c", False: "#4daf4a"},
        hover_data=["original", "adversarial", "category"],
        facet_col="category",
        title=f"Expected vs actual cosine similarity — {model_name}",
        labels={"expected_sim": "Expected (ideal)", "actual_sim": "Actual (model)"},
    )
    fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                   line=dict(dash="dash", color="gray"))
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    n_failures = int(fail_df["failure"].sum())
    st.markdown(
        f"**{n_failures}/{len(fail_df)} cases** scored ≥ {failure_threshold} despite semantically "
        f"opposite meaning. Points above the dashed line are model failures."
    )

    # Failure table
    st.dataframe(
        fail_df[["category", "original", "adversarial", "expected_sim", "actual_sim", "failure"]]
        .sort_values("actual_sim", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # ── Step 4: Perturbation Robustness (Retrieval-Critical) ──────────
    st.divider()
    st.markdown("## Step 4 — Retrieval-Critical Perturbation Sweep")
    st.caption(
        "All four retrieval-critical perturbation types run on the curated failure cases. "
        "High cosine similarity = model failure."
    )

    suite = PerturbationSuite(lambda t: encode_texts(model_name, tuple(t)))
    rc_types = PERTURBATION_CATEGORIES["retrieval_critical"]
    with st.spinner("Running perturbations…"):
        rc_summary = suite.summary_table(originals[:8], rc_types, failure_threshold)

    colors_rc = ["#e41a1c" if f else "#377eb8" for f in rc_summary["is_failure"]]
    fig3 = go.Figure(go.Bar(
        x=rc_summary["mean_sim"],
        y=rc_summary["perturbation"],
        orientation="h",
        marker_color=colors_rc,
        text=[f"{s:.3f}" for s in rc_summary["mean_sim"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Mean sim: %{x:.4f}<extra></extra>",
    ))
    fig3.add_vline(
        x=failure_threshold, line_dash="dash", line_color="orange",
        annotation_text=f"failure threshold ({failure_threshold})",
    )
    fig3.update_layout(
        title="Retrieval-critical perturbations — mean cosine similarity",
        xaxis_range=[0, 1.05],
        xaxis_title="Mean Cosine Similarity (higher = model failure)",
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(rc_summary, use_container_width=True, hide_index=True)

    # ── Export ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("## Export Results")

    report = DebugReport(
        title="Embedding Debugger — Killer Demo Report",
        model=model_name,
        dataset="curated_failure_cases",
    )
    report.add_metadata("score_gap_safe_vs_dangerous", round(score_gap, 4))
    report.add_metadata("mutual_sim_safe_dangerous", round(mutual_sim, 4))
    report.add_metadata("n_failures_out_of", n_failures)
    report.add_metadata("failure_threshold", failure_threshold)
    report.add_section(
        "Failure case analysis",
        fail_df[["category", "original", "adversarial", "expected_sim", "actual_sim", "failure"]],
        "Curated semantically-opposite pairs and their cosine similarity under the tested model.",
    )
    report.add_section("Retrieval-critical perturbation sweep", rc_summary)

    report.streamlit_download_buttons()
