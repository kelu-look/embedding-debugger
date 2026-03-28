"""
Retrieval Debugger — Pillar 1.

Shows rank failures, score gaps, and rank drift after perturbation.
Exports full debug report as JSON or Markdown.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from embedding_debugger.retrieval import RetrievalDebugger
from embedding_debugger.perturbation import PERTURBATION_REGISTRY, PERTURBATION_CATEGORIES
from embedding_debugger.export import retrieval_report
from app.pages._shared import encode_texts, get_dataset, get_model, model_selector, k_slider


def render() -> None:
    st.title("📡 Retrieval Debugger")
    st.caption(
        "**Pillar 1 — Retrieval Debugging.** "
        "Find rank failures, measure score gaps, and track how perturbations destroy retrieval quality."
    )

    with st.sidebar:
        st.header("Settings")
        model_name = model_selector(key="ret_model")
        k = k_slider("Retrieve top-k", default=10, key="ret_k")

        st.markdown("**Perturbation for rank drift**")
        cat = st.selectbox("Category", ["(none)", "retrieval_critical", "structural", "semantic", "lexical"], key="ret_cat")
        if cat != "(none)":
            pert_options = PERTURBATION_CATEGORIES[cat]
        else:
            pert_options = ["(none)"]
        perturb_type = st.selectbox("Perturbation type", pert_options, key="ret_pert")

    # ── Dataset ─────────────────────────────────────────────────────
    st.subheader("Dataset — FAQ Q&A retrieval")
    st.caption("20 questions paired with 20 answers. Correct retrieval = question matches its own answer at rank 1.")
    texts, meta = get_dataset("faq")
    questions = meta["questions"]
    answers = meta["answers"]
    n = len(questions)

    q_vecs = encode_texts(model_name, tuple(questions), is_query=True)
    a_vecs = encode_texts(model_name, tuple(answers))

    debugger = RetrievalDebugger(answers, a_vecs)
    with st.spinner("Running retrieval…"):
        _, df = debugger.analyze_failures(questions, q_vecs, list(range(n)), k=k)

    # ── Metrics ──────────────────────────────────────────────────────
    r1 = debugger.recall_at_k(df, k=1)
    r5 = debugger.recall_at_k(df, k=5)
    r10 = debugger.recall_at_k(df, k=min(10, k))
    mrr = debugger.mrr_at_k(df, k=k)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall@1", f"{r1:.1%}")
    c2.metric("Recall@5", f"{r5:.1%}")
    c3.metric(f"Recall@{min(10,k)}", f"{r10:.1%}")
    c4.metric(f"MRR@{k}", f"{mrr:.4f}")

    # ── Results table ─────────────────────────────────────────────────
    st.subheader("Query-level results")
    display_df = df[["query", "expected_doc", "expected_rank", "top1_score", "expected_score", "is_failure"]].copy()
    display_df.columns = ["Query", "Expected Answer", "Rank", "Top-1 Score", "Expected Score", "Failure"]
    st.dataframe(
        display_df.style.apply(
            lambda row: ["background-color: #ffdddd" if row["Failure"] else "" for _ in row],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ── Score gap analysis ────────────────────────────────────────────
    df["score_gap"] = df["top1_score"] - df["expected_score"]
    fig = px.histogram(
        df, x="expected_score", nbins=20, color="is_failure",
        color_discrete_map={True: "#e41a1c", False: "#4daf4a"},
        title="Expected-document score distribution (red = retrieval failure)",
        labels={"expected_score": "Cosine similarity to correct answer"},
    )
    fig.add_vline(x=df["expected_score"].mean(), line_dash="dash",
                  annotation_text=f"mean={df['expected_score'].mean():.3f}")
    st.plotly_chart(fig, use_container_width=True)

    # ── Single query explorer ─────────────────────────────────────────
    st.subheader("Single query explorer")
    query_idx = st.selectbox(
        "Select a question",
        range(n),
        format_func=lambda i: f"[{i}] {questions[i][:60]}",
        key="ret_query_select",
    )
    result = debugger.retrieve(questions[query_idx], q_vecs[query_idx], k=k)
    res_df = pd.DataFrame({
        "rank": range(1, len(result.retrieved) + 1),
        "answer": result.retrieved,
        "score": [round(s, 4) for s in result.scores],
        "is_expected": [idx == query_idx for idx in result.indices],
    })
    st.dataframe(
        res_df.style.apply(
            lambda row: ["background-color: #ddffdd" if row["is_expected"] else "" for _ in row],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ── Perturbation rank drift ────────────────────────────────────────
    if perturb_type not in ("(none)",):
        st.subheader(f"Rank drift — `{perturb_type}`")
        st.caption(
            "For each query: does the correct answer stay at rank 1 after perturbation? "
            "Rank shift > 0 = retrieval failure caused by this perturbation."
        )
        fn = PERTURBATION_REGISTRY[perturb_type]
        perturbed_questions = [fn(q) for q in questions]
        p_vecs = encode_texts(model_name, tuple(perturbed_questions), is_query=True)

        drift_df = debugger.perturbation_rank_drift(questions, q_vecs, perturbed_questions, p_vecs, k=k)
        changed = drift_df["top1_changed"].mean()

        st.metric(
            "Queries where top-1 changed",
            f"{changed:.1%}",
            help="How often the #1 retrieval result changes after applying this perturbation",
        )

        drift_display = drift_df[[
            "original_query", "perturbed_query",
            "original_top1", "perturbed_top1",
            "rank_shift", "top1_changed",
        ]].copy()
        st.dataframe(
            drift_display.style.apply(
                lambda row: ["background-color: #ffdddd" if row["top1_changed"] else "" for _ in row],
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

        fig2 = px.histogram(
            drift_df, x="rank_shift",
            title=f"Rank shift distribution after `{perturb_type}`",
            color_discrete_sequence=["#e41a1c"],
            labels={"rank_shift": "Rank shift (0 = no change, positive = correct answer fell)"},
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        drift_df = None

    # ── Export ────────────────────────────────────────────────────────
    st.subheader("Export")
    metrics = {"recall_at_1": r1, "recall_at_5": r5, "mrr": mrr}
    report = retrieval_report(
        model=model_name,
        dataset="faq",
        failures_df=df,
        metrics=metrics,
        perturbation_df=drift_df,
    )
    report.streamlit_download_buttons()
