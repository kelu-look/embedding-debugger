"""
Model loading and embedding generation.

Wraps sentence-transformers with an in-process cache so that repeated calls
within a session don't re-load weights.  Supports multiple named presets
(SBERT, E5, GTE) and any arbitrary HuggingFace model name.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# Preset catalogue
# ------------------------------------------------------------------
PRESET_MODELS: dict[str, str] = {
    "all-MiniLM-L6-v2":       "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2":       "sentence-transformers/all-mpnet-base-v2",
    "e5-small-v2":             "intfloat/e5-small-v2",
    "e5-base-v2":              "intfloat/e5-base-v2",
    "gte-small":               "thenlper/gte-small",
    "gte-base":                "thenlper/gte-base",
    "bge-small-en-v1.5":       "BAAI/bge-small-en-v1.5",
    "paraphrase-MiniLM-L6-v2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
}

# Models that expect a prefix on the query/passage side (E5 convention)
E5_PREFIX_MODELS = {"e5-small-v2", "e5-base-v2"}
BGE_PREFIX_MODELS = {"bge-small-en-v1.5"}


# ------------------------------------------------------------------
# EmbeddingModel
# ------------------------------------------------------------------
@dataclass
class EmbeddingStats:
    model_name: str
    dim: int
    encode_time_s: float
    n_texts: int


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer.

    Parameters
    ----------
    model_name : str
        Either a key from PRESET_MODELS or a raw HuggingFace model ID.
    device : str
        "cpu", "cuda", or "mps".  Defaults to "cpu".
    normalize : bool
        L2-normalise embeddings (recommended for cosine similarity).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self.short_name = model_name
        self.model_path = PRESET_MODELS.get(model_name, model_name)
        self.device = device
        self.normalize = normalize
        self._model: Optional[SentenceTransformer] = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_path, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _apply_prefix(self, texts: List[str], is_query: bool = False) -> List[str]:
        """Add model-specific prefixes (E5, BGE)."""
        if self.short_name in E5_PREFIX_MODELS:
            prefix = "query: " if is_query else "passage: "
            return [prefix + t for t in texts]
        if self.short_name in BGE_PREFIX_MODELS and is_query:
            return ["Represent this sentence for searching relevant passages: " + t for t in texts]
        return texts

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        is_query: bool = False,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts → float32 numpy array of shape (N, dim).
        """
        prefixed = self._apply_prefix(texts, is_query=is_query)
        t0 = time.perf_counter()
        vecs = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        elapsed = time.perf_counter() - t0
        self._last_stats = EmbeddingStats(
            model_name=self.short_name,
            dim=int(vecs.shape[1]),
            encode_time_s=round(elapsed, 4),
            n_texts=len(texts),
        )
        return vecs.astype(np.float32)

    def encode_single(self, text: str, is_query: bool = True) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @property
    def last_stats(self) -> Optional[EmbeddingStats]:
        return getattr(self, "_last_stats", None)

    def fingerprint(self, texts: List[str]) -> str:
        """SHA-256 of joined texts — useful for cache keys."""
        joined = "\n".join(texts)
        return hashlib.sha256(joined.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        return f"EmbeddingModel(name={self.short_name!r}, dim={self.dim}, device={self.device!r})"


# ------------------------------------------------------------------
# Convenience: load multiple models at once
# ------------------------------------------------------------------
def load_models(names: List[str], device: str = "cpu") -> dict[str, EmbeddingModel]:
    return {name: EmbeddingModel(name, device=device) for name in names}
