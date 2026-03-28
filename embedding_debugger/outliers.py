"""
Outlier detection for embedding corpora.

Methods:
  - LOF (Local Outlier Factor) — density-based
  - Isolation Forest
  - Distance-to-centroid
  - Low-similarity score: texts whose max cosine to any neighbor is low
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import normalize


@dataclass
class OutlierResult:
    indices: List[int]
    scores: np.ndarray       # higher = more anomalous
    texts: List[str]
    method: str


class OutlierDetector:
    """
    Detect anomalous embeddings in a corpus.

    Parameters
    ----------
    texts : list of str
    vecs  : float32 array, shape (N, dim)
    """

    def __init__(self, texts: List[str], vecs: np.ndarray) -> None:
        self.texts = texts
        self.vecs = normalize(vecs.astype(np.float32))

    def _rank_indices(self, scores: np.ndarray, top_n: int) -> List[int]:
        return np.argsort(-scores)[:top_n].tolist()

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def lof(
        self,
        n_neighbors: int = 20,
        top_n: int = 10,
    ) -> OutlierResult:
        n_neighbors = min(n_neighbors, len(self.vecs) - 1)
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto")
        clf.fit(self.vecs)
        # LOF gives negative scores; negate for "more outlier = higher"
        scores = -clf.negative_outlier_factor_
        idx = self._rank_indices(scores, top_n)
        return OutlierResult(
            indices=idx,
            scores=scores,
            texts=[self.texts[i] for i in idx],
            method="lof",
        )

    def isolation_forest(
        self,
        contamination: float = 0.05,
        top_n: int = 10,
        seed: int = 42,
    ) -> OutlierResult:
        clf = IsolationForest(contamination=contamination, random_state=seed)
        clf.fit(self.vecs)
        # decision_function: lower = more anomalous; negate
        scores = -clf.decision_function(self.vecs)
        idx = self._rank_indices(scores, top_n)
        return OutlierResult(
            indices=idx,
            scores=scores,
            texts=[self.texts[i] for i in idx],
            method="isolation_forest",
        )

    def centroid_distance(self, top_n: int = 10) -> OutlierResult:
        centroid = self.vecs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        scores = 1.0 - (self.vecs @ centroid)  # 1 - cosine_to_centroid
        idx = self._rank_indices(scores, top_n)
        return OutlierResult(
            indices=idx,
            scores=scores,
            texts=[self.texts[i] for i in idx],
            method="centroid_distance",
        )

    def low_max_similarity(self, k: int = 10, top_n: int = 10) -> OutlierResult:
        """
        For each point, compute its max cosine similarity to any other point.
        Points with low max-similarity are 'lonely' outliers.
        """
        sim = self.vecs @ self.vecs.T
        np.fill_diagonal(sim, -2.0)
        max_sim = sim.max(axis=1)
        scores = 1.0 - max_sim  # higher = more isolated
        idx = self._rank_indices(scores, top_n)
        return OutlierResult(
            indices=idx,
            scores=scores,
            texts=[self.texts[i] for i in idx],
            method="low_max_similarity",
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(
        self,
        method: Literal["lof", "isolation_forest", "centroid_distance", "low_max_similarity"] = "lof",
        top_n: int = 10,
    ) -> pd.DataFrame:
        dispatch = {
            "lof": self.lof,
            "isolation_forest": self.isolation_forest,
            "centroid_distance": self.centroid_distance,
            "low_max_similarity": self.low_max_similarity,
        }
        result = dispatch[method](top_n=top_n)
        return pd.DataFrame(
            {
                "rank": range(1, len(result.indices) + 1),
                "index": result.indices,
                "text": result.texts,
                "score": [round(float(result.scores[i]), 4) for i in result.indices],
                "method": method,
            }
        )

    def consensus_outliers(self, top_n: int = 10) -> pd.DataFrame:
        """
        Return texts flagged as outliers by multiple methods.
        """
        methods = ["lof", "isolation_forest", "centroid_distance", "low_max_similarity"]
        vote_counts = np.zeros(len(self.texts))
        for m in methods:
            res = self.summary(method=m, top_n=top_n)
            for idx in res["index"]:
                vote_counts[idx] += 1
        top_idx = np.argsort(-vote_counts)[:top_n]
        return pd.DataFrame(
            {
                "index": top_idx,
                "text": [self.texts[i] for i in top_idx],
                "vote_count": vote_counts[top_idx].astype(int),
            }
        )
