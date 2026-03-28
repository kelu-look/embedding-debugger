"""
Shared helpers for Streamlit pages: model loading, dataset loading,
and common UI widgets.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import streamlit as st

from embedding_debugger.models import EmbeddingModel, PRESET_MODELS
from demo.datasets import load_dataset, DATASETS


# ------------------------------------------------------------------
# Model cache (avoid reloading on every rerun)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def get_model(model_name: str) -> EmbeddingModel:
    return EmbeddingModel(model_name)


@st.cache_data(show_spinner="Encoding texts…")
def encode_texts(
    model_name: str,
    texts: Tuple[str, ...],
    is_query: bool = False,
) -> np.ndarray:
    model = get_model(model_name)
    return model.encode(list(texts), is_query=is_query)


# ------------------------------------------------------------------
# Dataset cache
# ------------------------------------------------------------------

@st.cache_data
def get_dataset(name: str):
    return load_dataset(name)


# ------------------------------------------------------------------
# Sidebar widgets
# ------------------------------------------------------------------

def model_selector(key: str = "model", default: str = "all-MiniLM-L6-v2") -> str:
    options = list(PRESET_MODELS.keys())
    idx = options.index(default) if default in options else 0
    return st.selectbox(
        "Embedding model",
        options,
        index=idx,
        key=key,
        help="Models are downloaded from HuggingFace on first use",
    )


def dataset_selector(key: str = "dataset") -> str:
    return st.selectbox(
        "Demo dataset",
        list(DATASETS.keys()),
        key=key,
    )


def k_slider(label: str = "Top-k neighbors", default: int = 5, key: str = "k") -> int:
    return st.slider(label, 1, 20, default, key=key)


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------

def sim_color(score: float) -> str:
    if score >= 0.85:
        return "🟢"
    elif score >= 0.60:
        return "🟡"
    else:
        return "🔴"


def show_neighbor_table(texts: List[str], scores: List[float]) -> None:
    import pandas as pd
    df = pd.DataFrame({"text": texts, "score": [round(s, 4) for s in scores]})
    df.index += 1
    st.dataframe(df, use_container_width=True)
