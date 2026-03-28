"""Tests for clustering module."""

import numpy as np
import pytest

from embedding_debugger.clustering import (
    project_pca,
    kmeans_cluster,
    elbow_search,
    best_k,
    ClusteringAnalyzer,
)


def make_vecs(n: int = 30, d: int = 16, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_pca_shape():
    v = make_vecs(20)
    res = project_pca(v, n_components=2)
    assert res.coords.shape == (20, 2)
    assert res.method == "pca"


def test_pca_3d():
    v = make_vecs(20)
    res = project_pca(v, n_components=3)
    assert res.coords.shape == (20, 3)


def test_kmeans_labels_range():
    v = make_vecs(30)
    res = kmeans_cluster(v, k=4)
    assert res.k == 4
    assert set(res.labels).issubset(set(range(4)))


def test_elbow_search():
    v = make_vecs(30)
    df = elbow_search(v, k_range=range(2, 7))
    assert len(df) == 5
    assert "silhouette" in df.columns
    assert "inertia" in df.columns


def test_best_k():
    v = make_vecs(30)
    df = elbow_search(v, k_range=range(2, 7))
    k = best_k(df)
    assert 2 <= k <= 6


def test_clustering_analyzer_build_df():
    v = make_vecs(20)
    texts = [f"text_{i}" for i in range(20)]
    ca = ClusteringAnalyzer(texts, v)
    df = ca.build_dataframe(method="pca", k=3)
    assert len(df) == 20
    assert "x" in df.columns
    assert "y" in df.columns
    assert "cluster_id" in df.columns
