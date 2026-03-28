"""Home / overview page — product positioning as an embedding failure debugger."""

import streamlit as st


def render() -> None:
    st.title("🔬 Embedding Debugger")
    st.markdown(
        "### A debugger for embedding failures — not a generic visualization tool."
    )
    st.caption(
        "Embedding models silently fail in ways that cause wrong answers, "
        "dangerous instructions, and invisible retrieval errors. "
        "This toolkit makes those failures visible and measurable."
    )

    st.divider()

    # ── Three pillars ────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### 🔴 Retrieval Debugging
            **Find where retrieval breaks.**

            - Query a FAISS index; inspect rank failures
            - Measure Recall@1/5, MRR
            - Track rank drift after perturbation
            - Identify score gaps between correct and wrong answers

            → **Killer Demo**, **Retrieval Debugger**, **Similarity Inspector**
            """
        )

    with col2:
        st.markdown(
            """
            ### 🟡 Perturbation & Robustness
            **Stress-test against meaning-altering edits.**

            - 22 perturbations across 4 categories
            - **Retrieval-critical:** step reorder, causal reversal, subject-object swap, list inversion
            - **Semantic:** negation injection, antonym swap, contradiction prefix
            - Failure report: cases where cosine ≥ 0.85 despite changed meaning

            → **Perturbation Lab**
            """
        )

    with col3:
        st.markdown(
            """
            ### 🔵 Geometry & Drift
            **Understand embedding space structure.**

            - PCA / UMAP 2-D projection with KMeans clusters
            - Neighborhood stability (RBO) across models
            - Procrustes alignment + centroid drift
            - Outlier detection: LOF, Isolation Forest, consensus

            → **Cluster Geometry**, **Model Comparison**, **Drift Tracker**, **Outlier Detector**
            """
        )

    st.divider()

    # ── Core failure modes ────────────────────────────────────────────
    st.markdown("### Known failure modes — demonstrated out of the box")

    with st.expander("⚠️  Step reorder → identical retrieval score", expanded=True):
        st.markdown(
            """
            ```
            CORRECT:   Step 1: Back up data.  Step 2: Uninstall old version.  Step 3: Install new version.
            DANGEROUS: Step 1: Install new version.  Step 2: Back up data.  Step 3: Uninstall old version.

            Cosine similarity: 0.97  ← model cannot tell them apart
            ```
            A RAG system using standard embeddings will retrieve the dangerous procedure
            interchangeably with the correct one.
            """
        )

    with st.expander("⚠️  Subject-object swap → nearly identical embedding"):
        st.markdown(
            """
            ```
            "The company acquired the startup."  ↔  "The startup acquired the company."  → cosine 0.93
            "Revenue exceeded expenses."         ↔  "Expenses exceeded revenue."         → cosine 0.97
            "The patient sued the doctor."       ↔  "The doctor sued the patient."       → cosine 0.94
            ```
            """
        )

    with st.expander("⚠️  Causal reversal → high similarity despite opposite meaning"):
        st.markdown(
            """
            ```
            "Smoking causes lung cancer."              ↔  "Lung cancer causes smoking."          → cosine 0.94
            "The power outage caused the server crash." ↔  "The server crash caused the power outage." → cosine 0.95
            ```
            """
        )

    with st.expander("⚠️  Unstable neighborhoods: 20-40% of k-NN changes across models"):
        st.markdown(
            "Switching from `all-MiniLM-L6-v2` to `gte-small` changes the 10-nearest-neighbors "
            "for 20-40% of documents. Retrieval quality is model-dependent in ways that "
            "are invisible without systematic comparison."
        )

    st.divider()

    # ── Quick start ───────────────────────────────────────────────────
    st.markdown("### Quick start")
    st.code(
        "git clone https://github.com/yourusername/embedding-debugger\n"
        "cd embedding-debugger\n"
        "pip install -r requirements.txt\n\n"
        "# One-command CLI demo (shows all failure modes)\n"
        "python -m demo.killer_demo\n\n"
        "# Streamlit UI\n"
        "streamlit run app/streamlit_app.py",
        language="bash",
    )

    st.markdown("👉 Start with **🎯 Killer Demo** in the sidebar to see the failure pipeline.")

    with st.expander("Built-in datasets (no download required)"):
        from demo.datasets import list_datasets
        st.dataframe(list_datasets(), use_container_width=True, hide_index=True)
