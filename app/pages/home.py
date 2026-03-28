"""Home / landing page."""

import streamlit as st


def render() -> None:
    st.title("🔬 Embedding Debugger")
    st.subheader("Local-first toolkit for analyzing text embedding behavior")

    st.markdown(
        """
        Embedding models power search, RAG, and recommendations — but they fail
        in subtle ways that are hard to detect without the right tools.
        This toolkit helps you **understand, stress-test, and compare** embedding models.

        ---

        ### What you can do here

        | Page | What it shows |
        |------|--------------|
        | 🔍 **Similarity Inspector** | Query a corpus and inspect ranked neighbors |
        | ⚡ **Perturbation Lab** | Test robustness to word order, drops, casing, etc. |
        | 📡 **Retrieval Debugger** | Find cases where FAISS retrieval fails |
        | 🗺️ **Cluster Geometry** | 2-D PCA / UMAP projection with KMeans clusters |
        | 🤝 **Model Comparison** | Compare SBERT, E5, GTE side by side |
        | 📈 **Drift Tracker** | Measure how much embedding spaces differ across models |
        | 🚨 **Outlier Detector** | Surface anomalous embeddings in a corpus |

        ---

        ### Known failure modes demonstrated

        - **Order blindness** — *"The dog bit the man"* ≈ *"The man bit the dog"* for most models
        - **Unstable neighborhoods** — nearest neighbors change dramatically with minor perturbations
        - **Retrieval rank collapse** — shuffling word order causes rank shifts from 1 to >10
        - **Model divergence** — SBERT and E5 agree on 60-80% of neighborhoods

        ---

        ### Quick start

        1. Pick a page from the left sidebar
        2. Select a model and demo dataset (no downloads needed — datasets are built-in)
        3. Explore the diagnostics

        **Models are downloaded on first use** (~80-400 MB depending on model).
        They are cached locally by `sentence-transformers`.
        """
    )

    with st.expander("Built-in demo datasets"):
        from demo.datasets import list_datasets
        st.dataframe(list_datasets(), use_container_width=True)
