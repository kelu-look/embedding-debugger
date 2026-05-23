"""
Perturbation Lab — organized by category, with failure highlighting.

Categories:
  lexical            — surface noise (should NOT change meaning)
  structural         — reordering (should NOT change meaning; reveals order-insensitive behavior)
  semantic           — meaning-altering (should change meaning; failure = high sim)
  retrieval_critical — most dangerous (step reorder, causal reversal, etc.)
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from embedding_debugger.perturbation import (
    PerturbationSuite,
    PERTURBATION_REGISTRY,
    PERTURBATION_CATEGORIES,
    PERTURBATION_CATEGORY_MAP,
    CATEGORY_EXPECTED_LOW_SIM,
    ORDER_PERTURBATIONS,
    is_failure,
)
from embedding_debugger.export import perturbation_report
from app.pages._shared import encode_texts, get_dataset, get_model, model_selector, dataset_selector

CATEGORY_COLORS = {
    "lexical":            "#aaaaaa",
    "structural":         "#377eb8",
    "semantic":           "#ff7f00",
    "retrieval_critical": "#e41a1c",
}

CATEGORY_DESCRIPTIONS = {
    "lexical":            "Surface-form noise — casing, punctuation, typos. Models should be robust here.",
    "structural":         "Word/sentence reordering. High similarity reveals order-insensitive behavior.",
    "semantic":           "Meaning-altering edits — negation, antonyms. High similarity = model failure.",
    "retrieval_critical": "🚨 Most dangerous. Step reorder, causal reversal, subject-object swap. High similarity = model cannot distinguish semantically opposite documents.",
}


def render() -> None:
    st.title("⚡ Perturbation Lab")
    st.markdown(
        "Measure how much each perturbation changes cosine similarity. "
        "Perturbations are grouped by **expected model behavior**: "
        "🔴 red = high similarity is a **failure**, ⚪ gray = high similarity is **expected**."
    )

    with st.sidebar:
        st.header("Settings")
        model_name = model_selector(key="pert_model")
        dataset_name = dataset_selector(key="pert_dataset")
        n_samples = st.slider("Number of texts", 5, 30, 10, key="pert_n")
        failure_threshold = st.slider("Failure threshold (cosine)", 0.7, 0.99, 0.85, 0.01, key="pert_thresh")

        st.markdown("**Perturbation categories**")
        show_lexical = st.checkbox("Lexical", value=True, key="pert_lex")
        show_structural = st.checkbox("Structural", value=True, key="pert_struct")
        show_semantic = st.checkbox("Semantic", value=True, key="pert_sem")
        show_rc = st.checkbox("Retrieval-critical 🚨", value=True, key="pert_rc")

    texts, _ = get_dataset(dataset_name)
    texts = list(texts)[:n_samples]

    selected_types: list[str] = []
    if show_lexical:
        selected_types += PERTURBATION_CATEGORIES["lexical"]
    if show_structural:
        selected_types += PERTURBATION_CATEGORIES["structural"]
    if show_semantic:
        selected_types += PERTURBATION_CATEGORIES["semantic"]
    if show_rc:
        selected_types += PERTURBATION_CATEGORIES["retrieval_critical"]

    if not selected_types:
        st.warning("Select at least one category.")
        return

    suite = PerturbationSuite(lambda t: encode_texts(model_name, tuple(t)))

    with st.spinner("Running perturbations…"):
        summary_df = suite.summary_table(texts, selected_types, failure_threshold)

    # ── Category legend ────────────────────────────────────────────────
    st.markdown(
        " &nbsp; ".join(
            f"<span style='color:{c};'>■</span> **{cat}** — {CATEGORY_DESCRIPTIONS[cat][:60]}…"
            for cat, c in CATEGORY_COLORS.items()
        ),
        unsafe_allow_html=True,
    )
    st.caption("")

    # ── Bar chart ─────────────────────────────────────────────────────
    bar_colors = []
    for _, row in summary_df.iterrows():
        cat = PERTURBATION_CATEGORY_MAP.get(row["perturbation"], "unknown")
        if row["is_failure"]:
            bar_colors.append("#e41a1c")
        else:
            bar_colors.append(CATEGORY_COLORS.get(cat, "#888888"))

    fig = go.Figure(go.Bar(
        x=summary_df["mean_sim"],
        y=summary_df["perturbation"],
        orientation="h",
        marker_color=bar_colors,
        error_x=dict(type="data", array=summary_df["std_sim"].tolist()),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Mean sim: %{x:.4f}<extra></extra>"
        ),
    ))
    fig.add_vline(
        x=failure_threshold, line_dash="dash", line_color="orange",
        annotation_text=f"failure ≥ {failure_threshold}",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="Mean Cosine Similarity",
        xaxis_range=[0, 1.08],
        height=max(350, 32 * len(summary_df) + 80),
        showlegend=False,
        margin=dict(l=20, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary table with color coding ───────────────────────────────
    st.subheader("Summary table")
    display_df = summary_df.copy()

    def row_style(row):
        if row["is_failure"]:
            return ["background-color: #ffeeee"] * len(row)
        cat = PERTURBATION_CATEGORY_MAP.get(row["perturbation"], "")
        if cat in ("lexical", "structural"):
            return ["background-color: #f0f0f0"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.apply(row_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    n_failures = int(summary_df["is_failure"].sum())
    if n_failures:
        st.error(
            f"⚠️ **{n_failures} failure(s)** — perturbations that should change meaning "
            f"scored ≥ {failure_threshold} on this model."
        )
    else:
        st.success("✅ No failures detected at this threshold.")

    # ── Per-text drill-down ────────────────────────────────────────────
    st.subheader("Per-text drill-down")
    drill_type = st.selectbox("Select perturbation", selected_types, key="pert_drill")
    batch = suite.run_batch(texts, drill_type)
    detail_df = pd.DataFrame({
        "original": [r.original for r in batch.results],
        "perturbed": [r.perturbed for r in batch.results],
        "similarity": [round(r.similarity, 4) for r in batch.results],
        "failure": [r.similarity >= failure_threshold and is_failure(drill_type, r.similarity, failure_threshold) for r in batch.results],
    }).sort_values("similarity", ascending=False)

    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    cat = PERTURBATION_CATEGORY_MAP.get(drill_type, "")
    bar_color = "#e41a1c" if cat in CATEGORY_EXPECTED_LOW_SIM else "#377eb8"
    fig2 = px.histogram(
        detail_df, x="similarity", nbins=15,
        title=f"Similarity distribution — {drill_type}  [{cat}]",
        color_discrete_sequence=[bar_color],
    )
    fig2.add_vline(x=failure_threshold, line_dash="dash", annotation_text=f"threshold {failure_threshold}")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Failure report ─────────────────────────────────────────────────
    with st.expander("📋 Full failure report (semantic + retrieval-critical only)", expanded=False):
        with st.spinner("Computing failure report…"):
            fail_df = suite.failure_report(texts, failure_threshold=failure_threshold)
        if len(fail_df):
            st.dataframe(fail_df, use_container_width=True, hide_index=True)
        else:
            st.success("No failures detected at this threshold.")

    # ── Export ─────────────────────────────────────────────────────────
    st.subheader("Export")
    report = perturbation_report(
        model=model_name,
        dataset=dataset_name,
        summary_df=summary_df,
        failures_df=fail_df if "fail_df" in dir() else None,
    )
    report.streamlit_download_buttons()
