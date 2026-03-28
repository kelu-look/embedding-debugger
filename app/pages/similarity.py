"""
Similarity Inspector page.

Features:
- Free-text query against a corpus (built-in or custom)
- Sorted nearest-neighbor table with scores
- Pairwise similarity heatmap for a selected subset
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from embedding_debugger.similarity import SimilarityAnalyzer
from app.pages._shared import (
    encode_texts,
    get_dataset,
    get_model,
    k_slider,
    model_selector,
    dataset_selector,
)


def render() -> None:
    st.title("🔍 Similarity Inspector")
    st.caption("Query a corpus and inspect nearest-neighbor rankings.")

    # ------------------------------------------------------------------
    # Sidebar controls
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Settings")
        model_name = model_selector(key="sim_model")
        dataset_name = dataset_selector(key="sim_dataset")
        k = k_slider(key="sim_k")
        show_heatmap = st.checkbox("Show pairwise heatmap", value=True, key="sim_heatmap")
        heatmap_n = st.slider("Heatmap subset size", 5, 40, 15, key="sim_heatmap_n")

    # ------------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------------
    texts, meta = get_dataset(dataset_name)
    corpus_vecs = encode_texts(model_name, tuple(texts))

    # ------------------------------------------------------------------
    # Custom corpus input
    # ------------------------------------------------------------------
    with st.expander("✏️ Use custom corpus (one text per line)", expanded=False):
        custom_raw = st.text_area("Custom corpus", height=150, key="sim_custom")
        if custom_raw.strip():
            texts = [t.strip() for t in custom_raw.strip().splitlines() if t.strip()]
            corpus_vecs = encode_texts(model_name, tuple(texts))

    st.info(f"Corpus: **{len(texts)} texts** · Model: **{model_name}**")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    st.subheader("Query")
    query_text = st.text_input(
        "Enter query text",
        value="How do I reset my password?",
        key="sim_query",
    )

    if query_text.strip():
        model = get_model(model_name)
        query_vec = model.encode_single(query_text, is_query=True)

        analyzer = SimilarityAnalyzer(texts, corpus_vecs)
        result = analyzer.query(query_text, query_vec, k=k)

        st.subheader(f"Top-{k} neighbors")
        df = pd.DataFrame(
            {
                "rank": range(1, len(result.neighbors) + 1),
                "text": result.neighbors,
                "score": [round(s, 4) for s in result.scores],
            }
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Similarity bar chart
        fig = px.bar(
            df,
            x="score",
            y="text",
            orientation="h",
            color="score",
            color_continuous_scale="RdYlGn",
            range_color=[0, 1],
            title="Similarity scores",
            labels={"text": "", "score": "Cosine Similarity"},
        )
        fig.update_layout(yaxis={"autorange": "reversed"}, height=40 * k + 100)
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Pairwise heatmap
    # ------------------------------------------------------------------
    if show_heatmap:
        st.subheader("Pairwise Similarity Heatmap")
        subset_texts = texts[:heatmap_n]
        subset_vecs = corpus_vecs[:heatmap_n]

        analyzer = SimilarityAnalyzer(subset_texts, subset_vecs)
        mat = analyzer.similarity_matrix(subset_vecs)

        labels = [t[:40] + "…" if len(t) > 40 else t for t in subset_texts]
        fig = px.imshow(
            mat,
            x=labels,
            y=labels,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Cosine Similarity Matrix",
            aspect="auto",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Most / least similar pairs
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Most similar pairs**")
            st.dataframe(
                analyzer.most_similar_pairs(subset_texts, subset_vecs, top_n=5)[
                    ["text_i", "text_j", "score"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        with col2:
            st.markdown("**Least similar pairs**")
            st.dataframe(
                analyzer.least_similar_pairs(subset_texts, subset_vecs, top_n=5)[
                    ["text_i", "text_j", "score"]
                ],
                use_container_width=True,
                hide_index=True,
            )
