"""
Model Comparison page.

Compare SBERT, E5, GTE embeddings side by side:
- Order-blind pair similarity scores
- Top-k neighbor agreement
- Pairwise overlap on a shared corpus
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from embedding_debugger.models import PRESET_MODELS
from embedding_debugger.similarity import top_k_neighbors, rank_biased_overlap
from app.pages._shared import encode_texts, get_dataset, get_model


def render() -> None:
    st.title("🤝 Model Comparison")
    st.caption("Compare how different models embed the same texts.")

    with st.sidebar:
        st.header("Settings")
        all_models = list(PRESET_MODELS.keys())
        selected = st.multiselect(
            "Models to compare",
            all_models,
            default=["all-MiniLM-L6-v2", "gte-small", "e5-small-v2"],
            key="cmp_models",
        )
        dataset_name = st.selectbox("Dataset", ["order_blind", "faq", "news", "products"], key="cmp_dataset")
        k_nn = st.slider("k for neighbor overlap", 5, 20, 10, key="cmp_k")

    if len(selected) < 2:
        st.warning("Select at least 2 models to compare.")
        return

    texts, meta = get_dataset(dataset_name)
    texts = list(texts)

    # ------------------------------------------------------------------
    # Encode with all models
    # ------------------------------------------------------------------
    with st.spinner("Encoding with all models…"):
        model_vecs: Dict[str, np.ndarray] = {}
        for mname in selected:
            model_vecs[mname] = encode_texts(mname, tuple(texts))

    # ------------------------------------------------------------------
    # Tab 1: Order-blind pair similarity (order_blind dataset only)
    # ------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["Order sensitivity", "Neighbor overlap", "Pairwise drift"])

    with tab1:
        if dataset_name == "order_blind":
            originals = meta["originals"]
            perturbed = meta["perturbed"]
            n = len(originals)
            rows = []
            for mname, vecs in model_vecs.items():
                o_vecs = vecs[:n]
                p_vecs = vecs[n:]
                sims = (o_vecs * p_vecs).sum(axis=1)
                rows.append(
                    {
                        "model": mname,
                        "mean_sim": round(float(sims.mean()), 4),
                        "min_sim": round(float(sims.min()), 4),
                        "max_sim": round(float(sims.max()), 4),
                    }
                )
            df_ord = pd.DataFrame(rows).sort_values("mean_sim")
            st.subheader("Similarity on order-reversed pairs (lower = more order-sensitive)")
            fig = px.bar(
                df_ord,
                x="model",
                y="mean_sim",
                error_y=None,
                color="mean_sim",
                color_continuous_scale="RdYlGn_r",
                range_color=[0.3, 1.0],
                title="Mean cosine sim: original vs order-reversed",
            )
            fig.add_hline(y=0.9, line_dash="dash", annotation_text="0.9 — severe order blindness")
            st.plotly_chart(fig, use_container_width=True)

            # Per-pair breakdown
            detail_rows = []
            for mname, vecs in model_vecs.items():
                o_vecs = vecs[:n]
                p_vecs = vecs[n:]
                for i, (o, p) in enumerate(meta["pairs"]):
                    sim = float(o_vecs[i] @ p_vecs[i])
                    detail_rows.append({"model": mname, "original": o, "reversed": p, "sim": round(sim, 4)})
            detail_df = pd.DataFrame(detail_rows)
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
        else:
            st.info("Switch to the **order_blind** dataset to see order sensitivity comparison.")

    with tab2:
        st.subheader(f"Top-{k_nn} neighbor overlap (Jaccard & RBO)")

        # For each pair of models, compute mean neighbor overlap
        n_texts = len(texts)
        # Compute NN for each model
        nn_indices: Dict[str, np.ndarray] = {}
        for mname, vecs in model_vecs.items():
            idx, _ = top_k_neighbors(vecs, vecs, k=k_nn, exclude_self=True)
            nn_indices[mname] = idx

        overlap_rows = []
        model_list = list(selected)
        for i in range(len(model_list)):
            for j in range(i + 1, len(model_list)):
                ma, mb = model_list[i], model_list[j]
                rbo_scores = []
                jaccard_scores = []
                for t_idx in range(n_texts):
                    na_set = set(nn_indices[ma][t_idx].tolist())
                    nb_set = set(nn_indices[mb][t_idx].tolist())
                    jacc = len(na_set & nb_set) / len(na_set | nb_set) if na_set | nb_set else 0.0
                    rbo = rank_biased_overlap(
                        nn_indices[ma][t_idx].tolist(),
                        nn_indices[mb][t_idx].tolist(),
                    )
                    rbo_scores.append(rbo)
                    jaccard_scores.append(jacc)
                overlap_rows.append(
                    {
                        "model_a": ma,
                        "model_b": mb,
                        "mean_jaccard": round(float(np.mean(jaccard_scores)), 4),
                        "mean_rbo": round(float(np.mean(rbo_scores)), 4),
                    }
                )

        overlap_df = pd.DataFrame(overlap_rows)
        st.dataframe(overlap_df, use_container_width=True, hide_index=True)

        fig2 = px.bar(
            overlap_df,
            x=overlap_df.apply(lambda r: f"{r.model_a} vs {r.model_b}", axis=1),
            y=["mean_jaccard", "mean_rbo"],
            barmode="group",
            title=f"Neighbor overlap at k={k_nn}",
            labels={"value": "Score", "variable": "Metric"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Pairwise embedding space drift")
        from embedding_debugger.drift import DriftAnalyzer
        with st.spinner("Computing drift…"):
            drift_df = DriftAnalyzer.compare_models(texts, model_vecs, k=k_nn)
        st.dataframe(drift_df, use_container_width=True, hide_index=True)

        fig3 = px.bar(
            drift_df,
            x=drift_df.apply(lambda r: f"{r.model_a} vs {r.model_b}", axis=1),
            y=["neighborhood_stability", "mean_cosine_diff"],
            barmode="group",
            title="Drift metrics across model pairs",
        )
        st.plotly_chart(fig3, use_container_width=True)
