"""
FAISS-based retrieval engine with failure diagnostics.

Key features:
  - Build a flat (exact) FAISS index over a corpus
  - Query with top-k retrieval
  - Failure analysis: identify cases where expected docs are not retrieved
  - Rank tracking across perturbations
  - RetrievalDebugger: high-level class
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------

@dataclass
class RetrievalResult:
    query: str
    retrieved: List[str]
    scores: List[float]
    indices: List[int]


@dataclass
class FailureCase:
    query: str
    expected_doc: str
    expected_idx: int
    actual_rank: int          # rank of expected doc in retrieved list (0-indexed), -1 if not found
    top1_doc: str
    top1_score: float
    expected_score: Optional[float] = None


# ------------------------------------------------------------------
# FAISS index wrapper
# ------------------------------------------------------------------

class FAISSIndex:
    """
    Simple wrapper around faiss.IndexFlatIP (inner product = cosine for L2-norm vecs).
    Falls back to numpy brute-force if faiss is not installed.
    """

    def __init__(self, vecs: np.ndarray) -> None:
        self.vecs = vecs.astype(np.float32)
        self.dim = vecs.shape[1]
        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(self.vecs)
        else:
            self._index = None  # numpy fallback

    def search(self, query_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (scores, indices) each of shape (n_queries, k).
        """
        q = query_vecs.astype(np.float32)
        k = min(k, len(self.vecs))
        if self._index is not None:
            scores, indices = self._index.search(q, k)
        else:
            # numpy fallback: O(n*d) brute-force
            sim = q @ self.vecs.T
            indices = np.argsort(-sim, axis=1)[:, :k]
            scores = sim[np.arange(len(q))[:, None], indices]
        return scores, indices

    def __len__(self) -> int:
        return len(self.vecs)


# ------------------------------------------------------------------
# RetrievalDebugger
# ------------------------------------------------------------------

class RetrievalDebugger:
    """
    Debug retrieval quality for a corpus.

    Parameters
    ----------
    corpus_texts : list of str
    corpus_vecs  : float32 numpy array, shape (N, dim)
    """

    def __init__(
        self,
        corpus_texts: List[str],
        corpus_vecs: np.ndarray,
    ) -> None:
        self.corpus_texts = corpus_texts
        self.corpus_vecs = corpus_vecs.astype(np.float32)
        self._index = FAISSIndex(self.corpus_vecs)

    def retrieve(
        self,
        query_text: str,
        query_vec: np.ndarray,
        k: int = 10,
    ) -> RetrievalResult:
        scores, indices = self._index.search(query_vec[None, :], k)
        scores = scores[0].tolist()
        indices = indices[0].tolist()
        return RetrievalResult(
            query=query_text,
            retrieved=[self.corpus_texts[i] for i in indices],
            scores=scores,
            indices=indices,
        )

    def retrieve_batch(
        self,
        query_texts: List[str],
        query_vecs: np.ndarray,
        k: int = 10,
    ) -> List[RetrievalResult]:
        scores_all, indices_all = self._index.search(query_vecs, k)
        results = []
        for i, (q, s, idx) in enumerate(zip(query_texts, scores_all, indices_all)):
            results.append(
                RetrievalResult(
                    query=q,
                    retrieved=[self.corpus_texts[j] for j in idx],
                    scores=s.tolist(),
                    indices=idx.tolist(),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Failure analysis
    # ------------------------------------------------------------------

    def find_rank(self, result: RetrievalResult, expected_idx: int) -> int:
        """Return 0-based rank of expected_idx in result, -1 if absent."""
        try:
            return result.indices.index(expected_idx)
        except ValueError:
            return -1

    def analyze_failures(
        self,
        query_texts: List[str],
        query_vecs: np.ndarray,
        expected_indices: List[int],
        k: int = 20,
    ) -> Tuple[List[FailureCase], pd.DataFrame]:
        """
        Given queries and their expected documents, find retrieval failures.

        Returns
        -------
        failures : list of FailureCase
        df       : summary DataFrame
        """
        results = self.retrieve_batch(query_texts, query_vecs, k=k)
        failures: List[FailureCase] = []
        rows = []
        for i, (q, res, exp_idx) in enumerate(zip(query_texts, results, expected_indices)):
            rank = self.find_rank(res, exp_idx)
            exp_score = self.corpus_vecs[exp_idx] @ query_vecs[i]
            fc = FailureCase(
                query=q,
                expected_doc=self.corpus_texts[exp_idx],
                expected_idx=exp_idx,
                actual_rank=rank,
                top1_doc=res.retrieved[0] if res.retrieved else "",
                top1_score=res.scores[0] if res.scores else 0.0,
                expected_score=float(exp_score),
            )
            failures.append(fc)
            rows.append(
                {
                    "query": q[:60],
                    "expected_doc": self.corpus_texts[exp_idx][:60],
                    "expected_rank": rank if rank >= 0 else "> k",
                    "top1_score": round(fc.top1_score, 4),
                    "expected_score": round(float(exp_score), 4),
                    "is_failure": rank != 0,
                    "not_retrieved": rank == -1,
                }
            )
        return failures, pd.DataFrame(rows)

    def perturbation_rank_drift(
        self,
        original_texts: List[str],
        original_vecs: np.ndarray,
        perturbed_texts: List[str],
        perturbed_vecs: np.ndarray,
        k: int = 10,
    ) -> pd.DataFrame:
        """
        For each query, compare retrieval rank of the same document
        before and after perturbation.
        """
        orig_results = self.retrieve_batch(original_texts, original_vecs, k=k)
        pert_results = self.retrieve_batch(perturbed_texts, perturbed_vecs, k=k)
        rows = []
        for i, (orig, pert) in enumerate(zip(orig_results, pert_results)):
            # track rank of top-1 original doc in perturbed result
            expected_idx = orig.indices[0] if orig.indices else -1
            pert_rank = self.find_rank(pert, expected_idx)
            rows.append(
                {
                    "original_query": original_texts[i][:60],
                    "perturbed_query": perturbed_texts[i][:60],
                    "original_top1": orig.retrieved[0][:60] if orig.retrieved else "",
                    "perturbed_top1": pert.retrieved[0][:60] if pert.retrieved else "",
                    "original_rank": 0,
                    "perturbed_rank": pert_rank if pert_rank >= 0 else k + 1,
                    "rank_shift": pert_rank if pert_rank >= 0 else k + 1,
                    "top1_changed": orig.indices[0] != pert.indices[0] if orig.indices and pert.indices else True,
                }
            )
        return pd.DataFrame(rows)

    def mrr_at_k(self, failures_df: pd.DataFrame, k: int = 10) -> float:
        """Mean Reciprocal Rank @k."""
        ranks = failures_df["expected_rank"].apply(
            lambda r: r if isinstance(r, int) and r >= 0 else k
        )
        return float(np.mean([1.0 / (r + 1) for r in ranks]))

    def recall_at_k(self, failures_df: pd.DataFrame, k: int = 10) -> float:
        hits = failures_df["expected_rank"].apply(
            lambda r: 1 if isinstance(r, int) and 0 <= r < k else 0
        )
        return float(hits.mean())
