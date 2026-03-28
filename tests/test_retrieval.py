"""Tests for retrieval module (numpy fallback, no FAISS required)."""

import numpy as np
import pytest

from embedding_debugger.retrieval import FAISSIndex, RetrievalDebugger


def make_vecs(n: int, d: int = 16, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_faiss_index_search_shape():
    vecs = make_vecs(20)
    idx = FAISSIndex(vecs)
    scores, indices = idx.search(vecs[:3], k=5)
    assert scores.shape == (3, 5)
    assert indices.shape == (3, 5)


def test_faiss_index_self_retrieval():
    """Each vector should retrieve itself as rank 1."""
    vecs = make_vecs(20)
    idx = FAISSIndex(vecs)
    scores, indices = idx.search(vecs, k=1)
    for i in range(20):
        assert indices[i, 0] == i


def test_retrieval_debugger_recall():
    n = 15
    vecs = make_vecs(n)
    texts = [f"doc_{i}" for i in range(n)]
    debugger = RetrievalDebugger(texts, vecs)
    # query with the same vectors — should get rank 1 for each
    _, df = debugger.analyze_failures(
        texts, vecs, expected_indices=list(range(n)), k=5
    )
    r1 = debugger.recall_at_k(df, k=1)
    assert r1 == pytest.approx(1.0)


def test_retrieval_rank_drift():
    n = 10
    vecs = make_vecs(n)
    texts = [f"doc_{i}" for i in range(n)]
    debugger = RetrievalDebugger(texts, vecs)
    # Use perturbed vecs (random) — just check shape
    pert_vecs = make_vecs(n, seed=99)
    drift_df = debugger.perturbation_rank_drift(
        texts, vecs, texts, pert_vecs, k=5
    )
    assert len(drift_df) == n
    assert "rank_shift" in drift_df.columns
