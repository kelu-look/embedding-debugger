"""
Embedding drift analysis.

Measures how much the embedding space changes when:
  - Switching between models
  - Encoding different text versions (paraphrases, translations)
  - Comparing corpus subsets

Methods:
  - Pairwise cosine distance distributions
  - Neighborhood stability (RBO)
  - Procrustes alignment (structural drift)
  - Centroid drift
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize


# ------------------------------------------------------------------
# Procrustes alignment
# ------------------------------------------------------------------

def procrustes_align(
    A: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Find orthogonal matrix R that minimises ||A - B @ R||_F.
    Returns (B_aligned, disparity).
    Assumes A and B are already centered and scaled.
    """
    M = B.T @ A
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    B_aligned = B @ R
    disparity = float(np.linalg.norm(A - B_aligned, "fro"))
    return B_aligned, disparity


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------

@dataclass
class DriftReport:
    model_a: str
    model_b: str
    n_texts: int
    mean_cosine_diff: float       # mean |cos(a_i, b_i) - 1|
    procrustes_disparity: float
    neighborhood_stability: float  # mean RBO@10
    centroid_drift: float


# ------------------------------------------------------------------
# DriftAnalyzer
# ------------------------------------------------------------------

class DriftAnalyzer:
    """
    Compare two sets of embeddings for the same texts.

    Parameters
    ----------
    texts          : the shared text corpus
    vecs_a, vecs_b : embeddings from model A and B (same order)
    name_a, name_b : display names for the two models
    """

    def __init__(
        self,
        texts: List[str],
        vecs_a: np.ndarray,
        vecs_b: np.ndarray,
        name_a: str = "model_a",
        name_b: str = "model_b",
    ) -> None:
        self.texts = texts
        self.vecs_a = normalize(vecs_a.astype(np.float32))
        self.vecs_b = normalize(vecs_b.astype(np.float32))
        self.name_a = name_a
        self.name_b = name_b

    # ------------------------------------------------------------------
    # Per-text cosine agreement
    # ------------------------------------------------------------------

    def pointwise_cosine(self) -> np.ndarray:
        """Cosine similarity between corresponding embeddings."""
        return (self.vecs_a * self.vecs_b).sum(axis=1).clip(-1.0, 1.0)

    def cosine_agreement_df(self) -> pd.DataFrame:
        scores = self.pointwise_cosine()
        return pd.DataFrame(
            {
                "text": self.texts,
                f"sim_{self.name_a}_vs_{self.name_b}": scores,
            }
        ).sort_values(f"sim_{self.name_a}_vs_{self.name_b}")

    # ------------------------------------------------------------------
    # Neighborhood stability
    # ------------------------------------------------------------------

    def neighborhood_stability(self, k: int = 10) -> np.ndarray:
        from .similarity import neighborhood_stability as _ns
        return _ns(self.vecs_a, self.vecs_b, k=k)

    def stability_df(self, k: int = 10) -> pd.DataFrame:
        stab = self.neighborhood_stability(k=k)
        return pd.DataFrame(
            {"text": self.texts, "neighborhood_stability": stab}
        ).sort_values("neighborhood_stability")

    # ------------------------------------------------------------------
    # Structural (Procrustes) drift
    # ------------------------------------------------------------------

    def procrustes(self) -> Tuple[np.ndarray, float]:
        """Align vecs_b to vecs_a; return (aligned_b, disparity)."""
        return procrustes_align(self.vecs_a, self.vecs_b)

    # ------------------------------------------------------------------
    # Centroid drift
    # ------------------------------------------------------------------

    def centroid_drift(self) -> float:
        ca = self.vecs_a.mean(axis=0)
        cb = self.vecs_b.mean(axis=0)
        return float(np.linalg.norm(ca - cb))

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def report(self, k: int = 10) -> DriftReport:
        cos = self.pointwise_cosine()
        _, disp = self.procrustes()
        stab = self.neighborhood_stability(k=k)
        return DriftReport(
            model_a=self.name_a,
            model_b=self.name_b,
            n_texts=len(self.texts),
            mean_cosine_diff=float(1.0 - cos.mean()),
            procrustes_disparity=round(disp, 4),
            neighborhood_stability=round(float(stab.mean()), 4),
            centroid_drift=round(self.centroid_drift(), 4),
        )

    # ------------------------------------------------------------------
    # Multi-model comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_models(
        texts: List[str],
        model_vecs: Dict[str, np.ndarray],
        k: int = 10,
    ) -> pd.DataFrame:
        """
        Pairwise drift between all model pairs.
        Returns a DataFrame with columns: model_a, model_b, + DriftReport fields.
        """
        names = list(model_vecs.keys())
        rows = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                na, nb = names[i], names[j]
                da = DriftAnalyzer(
                    texts, model_vecs[na], model_vecs[nb], name_a=na, name_b=nb
                )
                r = da.report(k=k)
                rows.append(
                    {
                        "model_a": r.model_a,
                        "model_b": r.model_b,
                        "mean_cosine_diff": r.mean_cosine_diff,
                        "procrustes_disparity": r.procrustes_disparity,
                        "neighborhood_stability": r.neighborhood_stability,
                        "centroid_drift": r.centroid_drift,
                    }
                )
        return pd.DataFrame(rows)
