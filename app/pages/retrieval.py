"""
Retrieval Debugger page.

- Build FAISS index over corpus
- Query with questions; check if correct answer is rank-1
- Show failure cases
- Optionally perturb queries and track rank drift
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from embedding_debugger.retrieval import RetrievalDebugger
from embedding_debugger.perturbation import PERTURBATION_REGISTRY
from app.pages._shared import encode_texts, get_dataset, get_model, model_selector, k_slider


def render() -> None:
    st.title("📡 Retrieval Debugger")
    st.caption(
        "Analyze FAISS-based retrieval: rank failures, score gaps, "
        "and what happens when queries are perturbed."
    )

    with st.sidebar:
        st.header("Settings")
        model_name = model_selector(key="ret_model")
        k = k_slider("Retrieve top-k", default=10, key="ret_k")
        perturb_type = st.selectbox(
            "Perturbation for rank drift",
            ["(none)"] + list(PERTURBATION_REGISTRY.keys()),
            key="ret_pert",
        )

    # ------------------------------------------------------------------
    # Dataset: FAQ (question → answer matching)
    # ------------------------------------------------------------------
    st.subheader("Dataset: FAQ Q&A")
    texts, meta = get_dataset("faq")
    questions = meta["questions"]
    answers = meta["answers"]
    n = len(questions)

    model = get_model(model_name)

    q_vecs = encode_texts(model_name, tuple(questions), is_query=True)
    a_vecs = encode_texts(model_name, tuple(answers))

    debugger = RetrievalDebugger(answers, a_vecs)
    expected_indices = list(range(n))

    with st.spinner("Running retrieval analysis…"):
        failures, df = debugger.analyze_failures(questions, q_vecs, expected_indices, k=k)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    r1 = debugger.recall_at_k(df, k=1)
    r5 = debugger.recall_at_k(df, k=5)
    r10 = debugger.recall_at_k(df, k=min(10, k))
    mrr = debugger.mrr_at_k(df, k=k)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall@1", f"{r1:.1%}")
    c2.metric("Recall@5", f"{r5:.1%}")
    c3.metric(f"Recall@{min(10,k)}", f"{r10:.1%}")
    c4.metric(f"MRR@{k}", f"{mrr:.4f}")

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    st.subheader("Query-level results")
    display_df = df[["query", "expected_doc", "expected_rank", "top1_score", "expected_score", "is_failure"]].copy()
    display_df.columns = ["Query", "Expected Answer", "Rank", "Top-1 Score", "Expected Score", "Failure"]
    st.dataframe(
        display_df.style.apply(
            lambda row: ["background-color: #ffcccc" if row["Failure"] else "" for _ in row],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Score distribution
    fig = px.histogram(
        df,
        x="expected_score",
        nbins=20,
        title="Distribution of expected-document scores",
        color_discrete_sequence=["#377eb8"],
    )
    fig.add_vline(x=df["expected_score"].mean(), line_dash="dash", annotation_text="mean")
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Single query explorer
    # ------------------------------------------------------------------
    st.subheader("Single query explorer")
    query_idx = st.selectbox(
        "Select a query",
        range(len(questions)),
        format_func=lambda i: f"[{i}] {questions[i][:60]}",
        key="ret_query_select",
    )
    result = debugger.retrieve(questions[query_idx], q_vecs[query_idx], k=k)
    res_df = pd.DataFrame(
        {
            "rank": range(1, len(result.retrieved) + 1),
            "answer": result.retrieved,
            "score": [round(s, 4) for s in result.scores],
            "is_expected": [idx == query_idx for idx in result.indices],
        }
    )
    st.dataframe(
        res_df.style.apply(
            lambda row: ["background-color: #ccffcc" if row["is_expected"] else "" for _ in row],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------------------
    # Perturbation rank drift
    # ------------------------------------------------------------------
    if perturb_type != "(none)":
        st.subheader(f"Rank drift after perturbation: `{perturb_type}`")
        from embedding_debugger.perturbation import PERTURBATION_REGISTRY as REG
        fn = REG[perturb_type]
        perturbed_questions = [fn(q) for q in questions]
        p_vecs = encode_texts(model_name, tuple(perturbed_questions), is_query=True)

        drift_df = debugger.perturbation_rank_drift(
            questions, q_vecs, perturbed_questions, p_vecs, k=k
        )
        changed = drift_df["top1_changed"].mean()
        st.metric("Top-1 changed", f"{changed:.1%}", help="Fraction of queries where top-1 result changed after perturbation")

        st.dataframe(
            drift_df[["original_query", "perturbed_query", "original_top1", "perturbed_top1", "rank_shift", "top1_changed"]],
            use_container_width=True,
            hide_index=True,
        )

        fig2 = px.histogram(
            drift_df,
            x="rank_shift",
            title=f"Rank shift distribution — {perturb_type}",
            color_discrete_sequence=["#e41a1c"],
        )
        st.plotly_chart(fig2, use_container_width=True)
