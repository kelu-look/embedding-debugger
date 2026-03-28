"""
Shared utilities: caching, display helpers, color palettes.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np


# ------------------------------------------------------------------
# Simple disk cache for embeddings
# ------------------------------------------------------------------

CACHE_DIR = Path(os.environ.get("EMBDBG_CACHE_DIR", "./.emb_cache"))


def cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.pkl"


def cache_save(key: str, obj: Any) -> None:
    with open(cache_path(key), "wb") as f:
        pickle.dump(obj, f, protocol=4)


def cache_load(key: str) -> Optional[Any]:
    p = cache_path(key)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def embed_with_cache(
    texts: List[str],
    encode_fn: Callable[[List[str]], np.ndarray],
    model_name: str = "unknown",
) -> np.ndarray:
    """Encode texts, using disk cache keyed by (model_name, texts_hash)."""
    h = hashlib.sha256((model_name + "\n" + "\n".join(texts)).encode()).hexdigest()[:20]
    key = f"emb_{model_name.replace('/', '_')}_{h}"
    cached = cache_load(key)
    if cached is not None:
        return cached
    vecs = encode_fn(texts)
    cache_save(key, vecs)
    return vecs


# ------------------------------------------------------------------
# Text truncation for display
# ------------------------------------------------------------------

def truncate_text(text: str, max_len: int = 80) -> str:
    return text if len(text) <= max_len else text[:max_len] + "…"


def truncate_list(texts: List[str], max_len: int = 80) -> List[str]:
    return [truncate_text(t, max_len) for t in texts]


# ------------------------------------------------------------------
# Color palette for cluster visualization
# ------------------------------------------------------------------

CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
]


def cluster_color(cluster_id: int) -> str:
    return CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]


# ------------------------------------------------------------------
# Cosine helpers
# ------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ------------------------------------------------------------------
# Pretty-print a results table
# ------------------------------------------------------------------

def print_neighbors(result: Any, max_chars: int = 70) -> None:
    """Print a NeighborResult or RetrievalResult in a readable format."""
    print(f"\nQuery: {truncate_text(result.query, max_chars)}")
    print("-" * 72)
    for i, (doc, score) in enumerate(zip(result.retrieved if hasattr(result, 'retrieved') else result.neighbors, result.scores)):
        print(f"  {i+1:2d}. [{score:.4f}]  {truncate_text(doc, max_chars)}")
    print()
