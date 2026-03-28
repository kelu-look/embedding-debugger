"""
Perturbation suite for testing embedding robustness.

Key failure modes probed:
  - Order blindness  : shuffling word or sentence order
  - Lexical swap     : synonym replacement, antonym injection
  - Structural noise : punctuation stripping, casing, number corruption
  - Prefix injection : prepending irrelevant context
  - Truncation       : cutting text at various lengths

Each perturbation returns a PerturbationResult with original/perturbed text,
the original & perturbed embeddings, and a similarity score.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------

@dataclass
class PerturbationResult:
    original: str
    perturbed: str
    perturbation_type: str
    original_vec: np.ndarray
    perturbed_vec: np.ndarray
    similarity: float        # cosine similarity between original & perturbed
    rank_shift: Optional[int] = None  # how much did retrieval rank change?


@dataclass
class PerturbationBatchResult:
    results: List[PerturbationResult]
    perturbation_type: str

    @property
    def mean_similarity(self) -> float:
        return float(np.mean([r.similarity for r in self.results]))

    @property
    def min_similarity(self) -> float:
        return float(np.min([r.similarity for r in self.results]))

    @property
    def std_similarity(self) -> float:
        return float(np.std([r.similarity for r in self.results]))


# ------------------------------------------------------------------
# Perturbation functions  (str → str)
# ------------------------------------------------------------------

def shuffle_words(text: str, seed: Optional[int] = None) -> str:
    """Randomly shuffle all words in the text."""
    rng = random.Random(seed)
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def reverse_words(text: str) -> str:
    return " ".join(text.split()[::-1])


def shuffle_sentences(text: str, seed: Optional[int] = None) -> str:
    """Shuffle sentence order (split on . ? !)."""
    rng = random.Random(seed)
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    rng.shuffle(sentences)
    return " ".join(sentences)


def reverse_sentences(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    return " ".join(reversed(sentences))


def drop_random_words(text: str, drop_fraction: float = 0.2, seed: Optional[int] = None) -> str:
    rng = random.Random(seed)
    words = text.split()
    n_drop = max(1, int(len(words) * drop_fraction))
    drop_idx = set(rng.sample(range(len(words)), min(n_drop, len(words))))
    return " ".join(w for i, w in enumerate(words) if i not in drop_idx)


def drop_first_half(text: str) -> str:
    words = text.split()
    return " ".join(words[len(words) // 2 :])


def drop_second_half(text: str) -> str:
    words = text.split()
    return " ".join(words[: len(words) // 2])


def lowercase(text: str) -> str:
    return text.lower()


def uppercase(text: str) -> str:
    return text.upper()


def strip_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text)


def inject_prefix(text: str, prefix: str = "Irrelevant context. ") -> str:
    return prefix + text


def inject_negation(text: str) -> str:
    """Prepend 'NOT' to a few random words — a naive negation probe."""
    words = text.split()
    if len(words) < 4:
        return "NOT " + text
    idx = len(words) // 3
    words.insert(idx, "NOT")
    return " ".join(words)


def swap_numbers(text: str) -> str:
    """Replace digits with different random digits."""
    def replace_digit(m: re.Match) -> str:
        d = int(m.group())
        return str((d + random.randint(1, 8)) % 10)
    return re.sub(r"\d", replace_digit, text)


def truncate(text: str, max_words: int = 10) -> str:
    return " ".join(text.split()[:max_words])


# ------------------------------------------------------------------
# Named registry
# ------------------------------------------------------------------

PERTURBATION_REGISTRY: Dict[str, Callable[[str], str]] = {
    "shuffle_words":       shuffle_words,
    "reverse_words":       reverse_words,
    "shuffle_sentences":   shuffle_sentences,
    "reverse_sentences":   reverse_sentences,
    "drop_words_20pct":    lambda t: drop_random_words(t, 0.2),
    "drop_words_50pct":    lambda t: drop_random_words(t, 0.5),
    "drop_first_half":     drop_first_half,
    "drop_second_half":    drop_second_half,
    "lowercase":           lowercase,
    "uppercase":           uppercase,
    "strip_punctuation":   strip_punctuation,
    "inject_prefix":       inject_prefix,
    "inject_negation":     inject_negation,
    "swap_numbers":        swap_numbers,
    "truncate_10w":        lambda t: truncate(t, 10),
    "truncate_5w":         lambda t: truncate(t, 5),
}

ORDER_PERTURBATIONS = [
    "shuffle_words",
    "reverse_words",
    "shuffle_sentences",
    "reverse_sentences",
    "drop_first_half",
    "drop_second_half",
]


# ------------------------------------------------------------------
# PerturbationSuite
# ------------------------------------------------------------------

class PerturbationSuite:
    """
    Apply perturbations to a set of texts and compare embeddings.

    Parameters
    ----------
    encode_fn : callable
        A function (List[str]) -> np.ndarray that encodes texts.
    """

    def __init__(self, encode_fn: Callable[[List[str]], np.ndarray]) -> None:
        self.encode_fn = encode_fn

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-9:
            return 0.0
        return float(np.dot(a, b) / denom)

    def run_single(
        self,
        text: str,
        perturbation_type: str,
    ) -> PerturbationResult:
        fn = PERTURBATION_REGISTRY[perturbation_type]
        perturbed = fn(text)
        vecs = self.encode_fn([text, perturbed])
        return PerturbationResult(
            original=text,
            perturbed=perturbed,
            perturbation_type=perturbation_type,
            original_vec=vecs[0],
            perturbed_vec=vecs[1],
            similarity=self._cosine(vecs[0], vecs[1]),
        )

    def run_batch(
        self,
        texts: List[str],
        perturbation_type: str,
    ) -> PerturbationBatchResult:
        fn = PERTURBATION_REGISTRY[perturbation_type]
        perturbed_texts = [fn(t) for t in texts]
        # encode all at once for efficiency
        all_texts = texts + perturbed_texts
        all_vecs = self.encode_fn(all_texts)
        n = len(texts)
        orig_vecs = all_vecs[:n]
        pert_vecs = all_vecs[n:]
        results = []
        for i in range(n):
            results.append(
                PerturbationResult(
                    original=texts[i],
                    perturbed=perturbed_texts[i],
                    perturbation_type=perturbation_type,
                    original_vec=orig_vecs[i],
                    perturbed_vec=pert_vecs[i],
                    similarity=self._cosine(orig_vecs[i], pert_vecs[i]),
                )
            )
        return PerturbationBatchResult(results=results, perturbation_type=perturbation_type)

    def run_all(
        self,
        texts: List[str],
        perturbation_types: Optional[List[str]] = None,
    ) -> Dict[str, PerturbationBatchResult]:
        types = perturbation_types or list(PERTURBATION_REGISTRY.keys())
        return {t: self.run_batch(texts, t) for t in types}

    def order_sensitivity_report(
        self, texts: List[str]
    ) -> "pd.DataFrame":
        import pandas as pd
        rows = []
        for ptype in ORDER_PERTURBATIONS:
            batch = self.run_batch(texts, ptype)
            for r in batch.results:
                rows.append(
                    {
                        "perturbation": ptype,
                        "original": r.original[:80],
                        "perturbed": r.perturbed[:80],
                        "similarity": round(r.similarity, 4),
                    }
                )
        return pd.DataFrame(rows)

    def summary_table(
        self,
        texts: List[str],
        perturbation_types: Optional[List[str]] = None,
    ) -> "pd.DataFrame":
        import pandas as pd
        results = self.run_all(texts, perturbation_types)
        rows = []
        for ptype, batch in results.items():
            rows.append(
                {
                    "perturbation": ptype,
                    "mean_sim": round(batch.mean_similarity, 4),
                    "min_sim": round(batch.min_similarity, 4),
                    "std_sim": round(batch.std_similarity, 4),
                    "n": len(batch.results),
                }
            )
        return pd.DataFrame(rows).sort_values("mean_sim")
