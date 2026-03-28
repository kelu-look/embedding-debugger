"""Tests for similarity module (numpy-only, no model download needed)."""

import numpy as np
import pytest

from embedding_debugger.similarity import (
    cosine_matrix,
    top_k_neighbors,
    rank_biased_overlap,
    neighborhood_stability,
    SimilarityAnalyzer,
)


def make_vecs(n: int = 10, d: int = 16, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_cosine_matrix_self():
    v = make_vecs()
    mat = cosine_matrix(v)
    assert mat.shape == (10, 10)
    # diagonal should be ~1.0
    np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-5)


def test_cosine_matrix_bounds():
    v = make_vecs()
    mat = cosine_matrix(v)
    assert mat.min() >= -1.0
    assert mat.max() <= 1.0


def test_top_k_shape():
    v = make_vecs(20, 16)
    idx, scores = top_k_neighbors(v[:5], v, k=3)
    assert idx.shape == (5, 3)
    assert scores.shape == (5, 3)


def test_top_k_sorted():
    v = make_vecs(20, 16)
    _, scores = top_k_neighbors(v[:5], v, k=5)
    # scores should be descending along axis 1
    for row in scores:
        assert all(row[i] >= row[i + 1] for i in range(len(row) - 1))


def test_rank_biased_overlap_identical():
    lst = [1, 2, 3, 4, 5]
    assert abs(rank_biased_overlap(lst, lst) - 1.0) < 1e-6


def test_rank_biased_overlap_disjoint():
    assert rank_biased_overlap([1, 2, 3], [4, 5, 6]) == pytest.approx(0.0, abs=1e-6)


def test_neighborhood_stability_same():
    v = make_vecs(20, 16)
    stab = neighborhood_stability(v, v, k=5)
    np.testing.assert_allclose(stab, 1.0, atol=1e-5)


def test_similarity_analyzer_query():
    v = make_vecs(20, 16)
    texts = [f"text_{i}" for i in range(20)]
    analyzer = SimilarityAnalyzer(texts, v)
    result = analyzer.query("text_0", v[0], k=5)
    assert len(result.neighbors) == 5
    assert len(result.scores) == 5
    assert result.scores[0] >= result.scores[-1]


def test_pairwise_dataframe():
    v = make_vecs(5, 16)
    texts = [f"t{i}" for i in range(5)]
    analyzer = SimilarityAnalyzer(texts, v)
    df = analyzer.pairwise_dataframe(texts, v)
    assert len(df) == 10  # C(5, 2)
    assert "score" in df.columns
