"""
Cluster Geometry page.

- PCA or UMAP projection to 2-D scatter
- KMeans clustering with automatic k selection
- Elbow curve
- Cluster summary table
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from embedding_debugger.clustering import ClusteringAnalyzer
from app.pages._shared import encode_texts, get_dataset, model_selector, dataset_selector


def render() -> None:
    st.title("🗺️ Cluster Geometry")
    st.caption("Project embeddings to 2-D and inspect cluster structure.")

    with st.sidebar:
        st.header("Settings")
        model_name = model_selector(key="clust_model")
        dataset_name = dataset_selector(key="clust_dataset")
        proj_method = st.radio("Projection method", ["pca", "umap"], key="clust_proj")
        k_mode = st.radio("Cluster k", ["Auto (best silhouette)", "Manual"], key="clust_kmode")
        if k_mode == "Manual":
            k_val = st.slider("k", 2, 15, 5, key="clust_k")
        else:
            k_val = None
        label_col = st.checkbox("Show text labels on hover", value=True, key="clust_labels")
        show_elbow = st.checkbox("Show elbow curve", value=True, key="clust_elbow")

    texts, meta = get_dataset(dataset_name)
    texts = list(texts)
    vecs = encode_texts(model_name, tuple(texts))

    ca = ClusteringAnalyzer(texts, vecs)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    with st.spinner(f"Computing {proj_method.upper()} projection…"):
        try:
            df = ca.build_dataframe(method=proj_method, k=k_val)
        except ImportError:
            st.warning("umap-learn not installed — falling back to PCA")
            df = ca.build_dataframe(method="pca", k=k_val)

    # Add ground-truth label if available
    if "categories" in meta:
        df["category"] = meta["categories"][: len(df)]
        color_col = "category"
        symbol_col = "cluster_id"
    else:
        color_col = "cluster_id"
        symbol_col = None

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------
    hover_data = {"text": True, "x": False, "y": False}
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_col,
        symbol=symbol_col,
        hover_name="text" if label_col else None,
        hover_data=hover_data,
        title=f"Embedding projection — {proj_method.upper()} · {model_name}",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig.update_traces(marker_size=8, opacity=0.85)
    fig.update_layout(height=600, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Elbow curve
    # ------------------------------------------------------------------
    if show_elbow:
        with st.expander("📈 Elbow curve (KMeans silhouette)", expanded=True):
            elbow_df = ca.elbow_dataframe()
            col1, col2 = st.columns(2)
            with col1:
                fig_sil = px.line(
                    elbow_df, x="k", y="silhouette", markers=True,
                    title="Silhouette score by k",
                )
                best_k = int(elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"])
                fig_sil.add_vline(x=best_k, line_dash="dash", annotation_text=f"best k={best_k}")
                st.plotly_chart(fig_sil, use_container_width=True)
            with col2:
                fig_in = px.line(
                    elbow_df, x="k", y="inertia", markers=True,
                    title="Inertia (elbow) by k",
                )
                st.plotly_chart(fig_in, use_container_width=True)

    # ------------------------------------------------------------------
    # Cluster summary
    # ------------------------------------------------------------------
    with st.expander("📋 Cluster summary"):
        k_for_summary = k_val or int(elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"]) if show_elbow else 5
        summary = ca.cluster_summary(k=k_for_summary)
        for _, row in summary.iterrows():
            with st.container():
                st.markdown(f"**Cluster {row['cluster_id']}** — {row['size']} texts")
                for t in row["sample_texts"]:
                    st.markdown(f"  - {t[:80]}")
