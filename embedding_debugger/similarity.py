"""
Similarity analysis utilities.

Provides:
  - cosine_matrix   : pairwise cosine similarity matrix
  - top_k_neighbors : nearest neighbors for each query
  - rank_biased_overlap: soft rank-correlation metric for comparing NN lists
  - SimilarityAnalyzer : high-level class wrapping all of the above
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Low-level helpers
# ------------------------------------------------------------------

def cosine_matrix(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return cosine similarity matrix.
    If b is None, compute self-similarity of a.
    Assumes rows are L2-normalised (dot product == cosine).
    """
    if b is None:
        b = a
    return (a @ b.T).clip(-1.0, 1.0)


def top_k_neighbors(
    query_vecs: np.ndarray,
    corpus_vecs: np.ndarray,
    k: int = 10,
    exclude_self: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (indices, scores) of top-k neighbors for each query.

    Returns
    -------
    indices : (n_queries, k) int array
    scores  : (n_queries, k) float array
    """
    sim = cosine_matrix(query_vecs, corpus_vecs)  # (n_queries, n_corpus)
    if exclude_self:
        np.fill_diagonal(sim, -2.0)  # push self out of top-k
    k = min(k, sim.shape[1])
    # argsort descending
    idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    # sort within partition
    sorted_order = np.argsort(-sim[np.arange(len(sim))[:, None], idx], axis=1)
    idx = idx[np.arange(len(sim))[:, None], sorted_order]
    scores = sim[np.arange(len(sim))[:, None], idx]
    return idx, scores


def rank_biased_overlap(list1: List[int], list2: List[int], p: float = 0.9) -> float:
    """
    Rank-Biased Overlap (RBO-ext) — top-heavy rank-correlation metric.
    Returns a value in [0, 1]; 1 = identical ranking.

    Uses the "extrapolated" variant from Webber et al. (2010) which adds the
    residual term p^D * overlap@D so that identical finite lists score 1.0.
    """
    if not list1 or not list2:
        return 0.0
    depth = min(len(list1), len(list2))
    rbo_min = 0.0
    weight = 1.0
    last_overlap = 0.0
    for d in range(1, depth + 1):
        s1 = set(list1[:d])
        s2 = set(list2[:d])
        last_overlap = len(s1 & s2) / d
        rbo_min += last_overlap * weight
        weight *= p
    rbo_min *= (1 - p)
    # residual / extrapolation term
    residual = (p ** depth) * last_overlap
    return min(1.0, rbo_min + residual)


def neighborhood_stability(
    vecs_a: np.ndarray,
    vecs_b: np.ndarray,
    k: int = 10,
    p: float = 0.9,
) -> np.ndarray:
    """
    For each point, compute RBO between its k-NN under vecs_a vs vecs_b.
    Returns array of shape (n,) with stability scores in [0, 1].
    """
    idx_a, _ = top_k_neighbors(vecs_a, vecs_a, k=k, exclude_self=True)
    idx_b, _ = top_k_neighbors(vecs_b, vecs_b, k=k, exclude_self=True)
    stabilities = []
    for i in range(len(vecs_a)):
        rbo = rank_biased_overlap(idx_a[i].tolist(), idx_b[i].tolist(), p=p)
        stabilities.append(rbo)
    return np.array(stabilities)


# ------------------------------------------------------------------
# High-level class
# ------------------------------------------------------------------

@dataclass
class NeighborResult:
    query: str
    neighbors: List[str]
    scores: List[float]
    indices: List[int]


class SimilarityAnalyzer:
    """
    High-level similarity interface.

    Usage
    -----
    analyzer = SimilarityAnalyzer(corpus_texts, corpus_vecs)
    results  = analyzer.query("what is machine learning?", query_vec, k=5)
    df       = analyzer.pairwise_dataframe(texts, vecs)
    """

    def __init__(
        self,
        corpus_texts: List[str],
        corpus_vecs: np.ndarray,
    ) -> None:
        self.corpus_texts = corpus_texts
        self.corpus_vecs = corpus_vecs.astype(np.float32)

    def query(
        self,
        query_text: str,
        query_vec: np.ndarray,
        k: int = 10,
    ) -> NeighborResult:
        idx, scores = top_k_neighbors(
            query_vec[None, :], self.corpus_vecs, k=k
        )
        idx = idx[0].tolist()
        scores = scores[0].tolist()
        return NeighborResult(
            query=query_text,
            neighbors=[self.corpus_texts[i] for i in idx],
            scores=scores,
            indices=idx,
        )

    def pairwise_dataframe(
        self,
        texts: List[str],
        vecs: np.ndarray,
    ) -> pd.DataFrame:
        """Return a tidy (i, j, text_i, text_j, score) DataFrame."""
        mat = cosine_matrix(vecs)
        rows = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                rows.append(
                    {
                        "i": i,
                        "j": j,
                        "text_i": texts[i],
                        "text_j": texts[j],
                        "score": float(mat[i, j]),
                    }
                )
        return pd.DataFrame(rows)

    def similarity_matrix(self, vecs: np.ndarray) -> np.ndarray:
        return cosine_matrix(vecs)

    def most_similar_pairs(
        self, texts: List[str], vecs: np.ndarray, top_n: int = 20
    ) -> pd.DataFrame:
        df = self.pairwise_dataframe(texts, vecs)
        return df.nlargest(top_n, "score").reset_index(drop=True)

    def least_similar_pairs(
        self, texts: List[str], vecs: np.ndarray, top_n: int = 20
    ) -> pd.DataFrame:
        df = self.pairwise_dataframe(texts, vecs)
        return df.nsmallest(top_n, "score").reset_index(drop=True)
