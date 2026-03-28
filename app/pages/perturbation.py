"""
Perturbation Lab page.

Shows how much cosine similarity changes across perturbation types,
with special focus on order-swap failures.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from embedding_debugger.perturbation import (
    PerturbationSuite,
    PERTURBATION_REGISTRY,
    ORDER_PERTURBATIONS,
)
from app.pages._shared import encode_texts, get_dataset, get_model, model_selector, dataset_selector


def render() -> None:
    st.title("⚡ Perturbation Lab")
    st.caption(
        "Test how much perturbations change cosine similarity. "
        "Order-altering perturbations should change meaning — "
        "high similarity after order swap = **order blindness**."
    )

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Settings")
        model_name = model_selector(key="pert_model")
        dataset_name = dataset_selector(key="pert_dataset")
        n_samples = st.slider("Number of texts to test", 5, 40, 10, key="pert_n")
        selected_types = st.multiselect(
            "Perturbation types",
            list(PERTURBATION_REGISTRY.keys()),
            default=ORDER_PERTURBATIONS + ["drop_words_20pct", "lowercase", "strip_punctuation"],
            key="pert_types",
        )
        highlight_order = st.checkbox("Highlight order perturbations", value=True)

    texts, _ = get_dataset(dataset_name)
    texts = list(texts)[:n_samples]

    if not selected_types:
        st.warning("Select at least one perturbation type.")
        return

    model = get_model(model_name)
    suite = PerturbationSuite(lambda t: encode_texts(model_name, tuple(t)))

    with st.spinner("Running perturbations…"):
        summary_df = suite.summary_table(texts, selected_types)

    # ------------------------------------------------------------------
    # Summary bar chart
    # ------------------------------------------------------------------
    st.subheader("Mean cosine similarity after perturbation")
    summary_df["is_order"] = summary_df["perturbation"].isin(ORDER_PERTURBATIONS)

    color_map = {}
    if highlight_order:
        for p in summary_df["perturbation"]:
            color_map[p] = "#e41a1c" if p in ORDER_PERTURBATIONS else "#377eb8"
        colors = summary_df["perturbation"].map(color_map).tolist()
    else:
        colors = ["#377eb8"] * len(summary_df)

    fig = go.Figure(
        go.Bar(
            x=summary_df["mean_sim"],
            y=summary_df["perturbation"],
            orientation="h",
            marker_color=colors,
            error_x=dict(type="data", array=summary_df["std_sim"].tolist()),
            hovertemplate="<b>%{y}</b><br>Mean sim: %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0.9, line_dash="dash", line_color="orange", annotation_text="0.9 threshold")
    fig.update_layout(
        xaxis_title="Mean Cosine Similarity",
        xaxis_range=[0, 1.05],
        height=max(300, 35 * len(summary_df) + 100),
        showlegend=False,
    )
    if highlight_order:
        fig.add_annotation(
            text="🔴 = order perturbation (should be low for order-sensitive models)",
            xref="paper", yref="paper", x=0.5, y=-0.1,
            showarrow=False, font_size=11,
        )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.dataframe(summary_df[["perturbation", "mean_sim", "min_sim", "std_sim", "n"]], use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Per-text drill-down
    # ------------------------------------------------------------------
    st.subheader("Per-text drill-down")
    selected_ptype = st.selectbox(
        "Select perturbation type to inspect",
        selected_types,
        key="pert_drill",
    )

    batch = suite.run_batch(texts, selected_ptype)
    detail_df = pd.DataFrame(
        {
            "original": [r.original for r in batch.results],
            "perturbed": [r.perturbed for r in batch.results],
            "similarity": [round(r.similarity, 4) for r in batch.results],
        }
    ).sort_values("similarity")

    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # Distribution
    fig2 = px.histogram(
        detail_df,
        x="similarity",
        nbins=20,
        title=f"Similarity distribution — {selected_ptype}",
        color_discrete_sequence=["#e41a1c" if selected_ptype in ORDER_PERTURBATIONS else "#377eb8"],
    )
    fig2.update_layout(bargap=0.1)
    st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------------------------------
    # Order sensitivity report
    # ------------------------------------------------------------------
    with st.expander("📋 Order sensitivity report (all order perturbations)"):
        order_df = suite.order_sensitivity_report(texts)
        st.dataframe(order_df, use_container_width=True, hide_index=True)
        mean_order_sim = order_df["similarity"].mean()
        if mean_order_sim > 0.75:
            st.error(
                f"⚠️ **Order blindness detected**: mean cosine sim after order perturbation = {mean_order_sim:.3f}. "
                "This model treats order-reversed text as nearly identical."
            )
        else:
            st.success(f"✅ Order sensitivity: mean sim after order perturbation = {mean_order_sim:.3f}")
