"""
Outlier Detector page.

Surface anomalous embeddings using LOF, Isolation Forest, and centroid distance.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from embedding_debugger.outliers import OutlierDetector
from embedding_debugger.clustering import ClusteringAnalyzer
from app.pages._shared import encode_texts, get_dataset, model_selector, dataset_selector


def render() -> None:
    st.title("🚨 Outlier Detector")
    st.caption("Find anomalous or lonely embeddings in a corpus.")

    with st.sidebar:
        st.header("Settings")
        model_name = model_selector(key="out_model")
        dataset_name = dataset_selector(key="out_dataset")
        method = st.selectbox(
            "Detection method",
            ["lof", "isolation_forest", "centroid_distance", "low_max_similarity", "consensus"],
            key="out_method",
        )
        top_n = st.slider("Top-N outliers", 3, 20, 8, key="out_topn")

    texts, meta = get_dataset(dataset_name)
    texts = list(texts)
    vecs = encode_texts(model_name, tuple(texts))

    detector = OutlierDetector(texts, vecs)

    with st.spinner("Detecting outliers…"):
        if method == "consensus":
            result_df = detector.consensus_outliers(top_n=top_n)
        else:
            result_df = detector.summary(method=method, top_n=top_n)

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    st.subheader(f"Top-{top_n} outliers — {method}")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Visualize outliers in 2-D projection
    # ------------------------------------------------------------------
    st.subheader("Outlier positions in 2-D projection")
    ca = ClusteringAnalyzer(texts, vecs)
    with st.spinner("Computing PCA projection…"):
        proj_df = ca.build_dataframe(method="pca", k=5)

    outlier_indices = set(result_df["index"].tolist())
    proj_df["is_outlier"] = proj_df.index.isin(outlier_indices)
    proj_df["marker_size"] = proj_df["is_outlier"].map({True: 14, False: 7})
    proj_df["point_type"] = proj_df["is_outlier"].map({True: "Outlier", False: "Normal"})

    fig = px.scatter(
        proj_df,
        x="x",
        y="y",
        color="point_type",
        size="marker_size",
        hover_name="text",
        color_discrete_map={"Outlier": "#e41a1c", "Normal": "#aaaaaa"},
        title=f"Outliers highlighted ({method})",
        opacity=0.8,
    )
    fig.update_layout(height=550, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Score distribution
    # ------------------------------------------------------------------
    if method != "consensus":
        with st.expander("Score distribution across all texts"):
            full_df = detector.summary(method=method, top_n=len(texts))
            fig2 = px.histogram(full_df, x="score", nbins=30, title=f"{method} scores")
            st.plotly_chart(fig2, use_container_width=True)
