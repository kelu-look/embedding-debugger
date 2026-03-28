"""
Embedding Debugger — local-first toolkit for analyzing text embeddings.

Modules:
  models       — load and cache embedding models
  similarity   — cosine similarity, pairwise, ranked neighbors
  perturbation — order swaps, word drops, structured rewriting
  clustering   — KMeans + UMAP/PCA 2-D projections
  retrieval    — FAISS-based retrieval + failure diagnostics
  drift        — compare embedding spaces across models or time
  outliers     — detect anomalous embeddings
  utils        — shared helpers
"""

from .models import EmbeddingModel
from .similarity import SimilarityAnalyzer
from .perturbation import PerturbationSuite
from .clustering import ClusteringAnalyzer
from .retrieval import RetrievalDebugger
from .drift import DriftAnalyzer
from .outliers import OutlierDetector

__version__ = "0.1.0"
__all__ = [
    "EmbeddingModel",
    "SimilarityAnalyzer",
    "PerturbationSuite",
    "ClusteringAnalyzer",
    "RetrievalDebugger",
    "DriftAnalyzer",
    "OutlierDetector",
]
