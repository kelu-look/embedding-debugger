"""
Drift Tracker page.

Visualizes how the embedding space changes between two models:
- Per-text cosine agreement
- Neighborhood stability heatmap
- Procrustes disparity
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from embedding_debugger.drift import DriftAnalyzer
from embedding_debugger.models import PRESET_MODELS
from app.pages._shared import encode_texts, get_dataset


def render() -> None:
    st.title("📈 Drift Tracker")
    st.caption("Compare two embedding models on the same corpus.")

    with st.sidebar:
        st.header("Settings")
        all_models = list(PRESET_MODELS.keys())
        model_a = st.selectbox("Model A", all_models, index=0, key="drift_ma")
        model_b = st.selectbox("Model B", all_models, index=2, key="drift_mb")
        dataset_name = st.selectbox("Dataset", ["news", "faq", "products", "order_blind"], key="drift_ds")
        k_nn = st.slider("k for neighborhood stability", 5, 20, 10, key="drift_k")

    texts, _ = get_dataset(dataset_name)
    texts = list(texts)

    if model_a == model_b:
        st.info("Select two different models to compare.")
        return

    with st.spinner("Encoding with both models…"):
        vecs_a = encode_texts(model_a, tuple(texts))
        vecs_b = encode_texts(model_b, tuple(texts))

    da = DriftAnalyzer(texts, vecs_a, vecs_b, name_a=model_a, name_b=model_b)

    with st.spinner("Computing drift report…"):
        report = da.report(k=k_nn)

    # ------------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean cosine diff", f"{report.mean_cosine_diff:.4f}", help="1 - mean cosine similarity per text")
    c2.metric("Procrustes disparity", f"{report.procrustes_disparity:.4f}")
    c3.metric(f"Nbhd stability @{k_nn}", f"{report.neighborhood_stability:.4f}", help="Mean RBO of k-NN lists")
    c4.metric("Centroid drift", f"{report.centroid_drift:.4f}")

    # ------------------------------------------------------------------
    # Per-text cosine agreement
    # ------------------------------------------------------------------
    st.subheader("Per-text cosine agreement")
    cos_df = da.cosine_agreement_df()
    col_name = cos_df.columns[1]

    fig = px.histogram(
        cos_df,
        x=col_name,
        nbins=30,
        title="Distribution of per-text cosine similarity between models",
        color_discrete_sequence=["#377eb8"],
    )
    fig.add_vline(x=cos_df[col_name].mean(), line_dash="dash", annotation_text="mean")
    st.plotly_chart(fig, use_container_width=True)

    # Lowest agreement texts
    st.markdown("**Texts where models disagree most (lowest cosine sim):**")
    st.dataframe(cos_df.head(10), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Neighborhood stability
    # ------------------------------------------------------------------
    st.subheader(f"Neighborhood stability @{k_nn}")
    stab_df = da.stability_df(k=k_nn)

    fig2 = px.histogram(
        stab_df,
        x="neighborhood_stability",
        nbins=30,
        title="Distribution of neighborhood stability (RBO)",
        color_discrete_sequence=["#4daf4a"],
    )
    fig2.add_vline(x=stab_df["neighborhood_stability"].mean(), line_dash="dash")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Most unstable neighborhoods:**")
    st.dataframe(stab_df.head(10), use_container_width=True, hide_index=True)
