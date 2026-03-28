"""
Clustering and dimensionality-reduction for embedding visualization.

Features:
  - KMeans clustering with automatic elbow detection
  - PCA projection to 2-D / 3-D
  - UMAP projection (optional; falls back to PCA if unavailable)
  - ClusteringAnalyzer: high-level wrapper returning tidy DataFrames
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------
# UMAP (optional)
# ------------------------------------------------------------------
try:
    import umap  # type: ignore
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------

@dataclass
class ProjectionResult:
    coords: np.ndarray          # shape (n, 2) or (n, 3)
    method: str                 # "pca" | "umap"
    variance_explained: Optional[np.ndarray] = None  # for PCA only


@dataclass
class ClusterResult:
    labels: np.ndarray          # shape (n,) cluster ids
    centroids: np.ndarray       # shape (k, dim)
    k: int
    inertia: float
    silhouette: float


# ------------------------------------------------------------------
# Projection helpers
# ------------------------------------------------------------------

def project_pca(vecs: np.ndarray, n_components: int = 2) -> ProjectionResult:
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(vecs)
    return ProjectionResult(
        coords=coords,
        method="pca",
        variance_explained=pca.explained_variance_ratio_,
    )


def project_umap(
    vecs: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> ProjectionResult:
    if not _UMAP_AVAILABLE:
        raise ImportError("umap-learn is not installed. Install with: pip install umap-learn")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(vecs)
    return ProjectionResult(coords=coords, method="umap")


def project(
    vecs: np.ndarray,
    method: Literal["pca", "umap"] = "pca",
    n_components: int = 2,
    **umap_kwargs,
) -> ProjectionResult:
    if method == "umap":
        return project_umap(vecs, n_components=n_components, **umap_kwargs)
    return project_pca(vecs, n_components=n_components)


# ------------------------------------------------------------------
# Clustering helpers
# ------------------------------------------------------------------

def kmeans_cluster(vecs: np.ndarray, k: int, seed: int = 42) -> ClusterResult:
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(vecs)
    sil = silhouette_score(vecs, labels) if k > 1 else 0.0
    return ClusterResult(
        labels=labels,
        centroids=km.cluster_centers_,
        k=k,
        inertia=float(km.inertia_),
        silhouette=float(sil),
    )


def elbow_search(
    vecs: np.ndarray,
    k_range: range = range(2, 12),
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame with k, inertia, silhouette for elbow detection."""
    rows = []
    for k in k_range:
        if k >= len(vecs):
            break
        res = kmeans_cluster(vecs, k, seed=seed)
        rows.append({"k": k, "inertia": res.inertia, "silhouette": res.silhouette})
    return pd.DataFrame(rows)


def best_k(elbow_df: pd.DataFrame) -> int:
    """Pick k with highest silhouette score."""
    return int(elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"])


# ------------------------------------------------------------------
# High-level class
# ------------------------------------------------------------------

class ClusteringAnalyzer:
    """
    Cluster and project embeddings, returning tidy DataFrames.

    Usage
    -----
    ca = ClusteringAnalyzer(texts, vecs)
    df = ca.build_dataframe(method="umap", k=5)
    """

    def __init__(self, texts: List[str], vecs: np.ndarray) -> None:
        self.texts = texts
        self.vecs = vecs.astype(np.float32)

    def project(
        self,
        method: Literal["pca", "umap"] = "pca",
        n_components: int = 2,
        **kwargs,
    ) -> ProjectionResult:
        return project(self.vecs, method=method, n_components=n_components, **kwargs)

    def cluster(self, k: int = 5) -> ClusterResult:
        return kmeans_cluster(self.vecs, k)

    def auto_cluster(self, k_range: range = range(2, 12)) -> ClusterResult:
        df = elbow_search(self.vecs, k_range)
        k = best_k(df)
        return kmeans_cluster(self.vecs, k)

    def elbow_dataframe(self, k_range: range = range(2, 12)) -> pd.DataFrame:
        return elbow_search(self.vecs, k_range)

    def build_dataframe(
        self,
        method: Literal["pca", "umap"] = "pca",
        k: Optional[int] = None,
        labels: Optional[List[str]] = None,
        extra_cols: Optional[dict] = None,
        **proj_kwargs,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
          text, x, y[, z], cluster_id, label[, extra cols...]
        """
        proj = self.project(method=method, **proj_kwargs)
        n_components = proj.coords.shape[1]

        if k is None:
            cluster_res = self.auto_cluster()
        else:
            cluster_res = self.cluster(k=k)

        data: dict = {
            "text": self.texts,
            "x": proj.coords[:, 0],
            "y": proj.coords[:, 1],
            "cluster_id": cluster_res.labels.astype(str),
        }
        if n_components == 3:
            data["z"] = proj.coords[:, 2]

        if labels is not None:
            data["label"] = labels

        if extra_cols:
            data.update(extra_cols)

        return pd.DataFrame(data)

    def cluster_summary(self, k: Optional[int] = None) -> pd.DataFrame:
        """Return per-cluster statistics."""
        if k is None:
            cluster_res = self.auto_cluster()
        else:
            cluster_res = self.cluster(k=k)
        rows = []
        for cid in range(cluster_res.k):
            mask = cluster_res.labels == cid
            members = [self.texts[i] for i, m in enumerate(mask) if m]
            rows.append(
                {
                    "cluster_id": cid,
                    "size": int(mask.sum()),
                    "sample_texts": members[:3],
                }
            )
        return pd.DataFrame(rows)
