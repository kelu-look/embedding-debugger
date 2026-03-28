"""
Perturbation suite — structured stress-tests for embedding robustness.

Perturbations are organized into four categories that map directly to failure modes:

  LEXICAL     — surface-form noise (casing, punctuation, numbers)
                Should NOT change cosine sim much; high sim here is correct.

  STRUCTURAL  — reordering words or sentences
                Should NOT change cosine sim; high sim here reveals order blindness.

  SEMANTIC    — meaning-altering edits (negation, content injection)
                SHOULD change cosine sim; high sim here is a model failure.

  RETRIEVAL_CRITICAL — edits that destroy ranking semantics
                (step reorder, causal reversal, subject-object swap, list inversion)
                The most dangerous category: model treats opposite-meaning text as
                near-identical, silently surfacing wrong or dangerous answers.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PerturbationResult:
    original: str
    perturbed: str
    perturbation_type: str
    category: str
    original_vec: np.ndarray
    perturbed_vec: np.ndarray
    similarity: float
    rank_shift: Optional[int] = None


@dataclass
class PerturbationBatchResult:
    results: List[PerturbationResult]
    perturbation_type: str
    category: str

    @property
    def mean_similarity(self) -> float:
        return float(np.mean([r.similarity for r in self.results]))

    @property
    def min_similarity(self) -> float:
        return float(np.min([r.similarity for r in self.results]))

    @property
    def std_similarity(self) -> float:
        return float(np.std([r.similarity for r in self.results]))


# ──────────────────────────────────────────────────────────────────────
# LEXICAL perturbations
# ──────────────────────────────────────────────────────────────────────

def lowercase(text: str) -> str:
    return text.lower()

def uppercase(text: str) -> str:
    return text.upper()

def strip_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text)

def swap_numbers(text: str) -> str:
    """Replace each digit with a different random digit."""
    def _replace(m: re.Match) -> str:
        d = int(m.group())
        return str((d + random.randint(1, 8)) % 10)
    return re.sub(r"\d", _replace, text)

def add_typos(text: str, rate: float = 0.1, seed: Optional[int] = None) -> str:
    """Randomly swap adjacent characters to simulate typos."""
    rng = random.Random(seed)
    chars = list(text)
    for i in range(len(chars) - 1):
        if chars[i] != " " and chars[i + 1] != " " and rng.random() < rate:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


# ──────────────────────────────────────────────────────────────────────
# STRUCTURAL perturbations
# ──────────────────────────────────────────────────────────────────────

def shuffle_words(text: str, seed: Optional[int] = None) -> str:
    rng = random.Random(seed)
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)

def reverse_words(text: str) -> str:
    return " ".join(text.split()[::-1])

def shuffle_sentences(text: str, seed: Optional[int] = None) -> str:
    rng = random.Random(seed)
    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    rng.shuffle(sents)
    return " ".join(sents)

def reverse_sentences(text: str) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    return " ".join(reversed(sents))

def drop_random_words(text: str, drop_fraction: float = 0.2, seed: Optional[int] = None) -> str:
    rng = random.Random(seed)
    words = text.split()
    n_drop = max(1, int(len(words) * drop_fraction))
    drop_idx = set(rng.sample(range(len(words)), min(n_drop, len(words))))
    return " ".join(w for i, w in enumerate(words) if i not in drop_idx)

def drop_first_half(text: str) -> str:
    words = text.split()
    return " ".join(words[len(words) // 2:])

def drop_second_half(text: str) -> str:
    words = text.split()
    return " ".join(words[:len(words) // 2])

def truncate(text: str, max_words: int = 10) -> str:
    return " ".join(text.split()[:max_words])


# ──────────────────────────────────────────────────────────────────────
# SEMANTIC perturbations
# ──────────────────────────────────────────────────────────────────────

def inject_negation(text: str) -> str:
    """Insert 'NOT' before the first verb-like word."""
    words = text.split()
    if len(words) < 3:
        return "NOT " + text
    # inject after first 2-3 words (typically subject)
    idx = min(2, len(words) // 3)
    words.insert(idx, "NOT")
    return " ".join(words)

def inject_irrelevant_prefix(text: str, prefix: str = "Unrelated context: ") -> str:
    return prefix + text

def inject_contradiction_prefix(text: str) -> str:
    """Prepend a flat denial — strongest semantic perturbation."""
    return "The following is false. " + text

def antonym_swap(text: str) -> str:
    """Swap a small set of common antonym pairs."""
    ANTONYMS = {
        "increase": "decrease", "decrease": "increase",
        "start": "stop", "stop": "start",
        "open": "close", "close": "open",
        "add": "remove", "remove": "add",
        "enable": "disable", "disable": "enable",
        "connect": "disconnect", "disconnect": "connect",
        "positive": "negative", "negative": "positive",
        "success": "failure", "failure": "success",
        "before": "after", "after": "before",
        "first": "last", "last": "first",
        "more": "less", "less": "more",
        "fast": "slow", "slow": "fast",
    }
    tokens = text.split()
    return " ".join(ANTONYMS.get(w.lower(), w) for w in tokens)


# ──────────────────────────────────────────────────────────────────────
# RETRIEVAL-CRITICAL perturbations
# These are the most dangerous: they produce semantically opposite text
# that most models embed at cosine ≥ 0.90.
# ──────────────────────────────────────────────────────────────────────

def subject_object_swap(text: str) -> str:
    """
    Swap subject and object using simple heuristics.
    Targets: "X [verb] Y" → "Y [verb] X"
    Works well on short declarative sentences.
    """
    # Pattern: "A <verb> B" where A and B are simple noun phrases
    match = re.match(
        r"^((?:The |A |An )?\w+(?:\s+\w+)?)\s+(acquired|defeated|sued|replaced|"
        r"caused|exceeds?|exceeded|loves?|loved|hates?|hated|bites?|bit|"
        r"supports?|supported|blocks?|blocked|owns?|owned|controls?|controlled|"
        r"prevents?|prevented)\s+((?:the |a |an )?\w+(?:\s+\w+)?)(.*)$",
        text,
        re.IGNORECASE,
    )
    if match:
        subj, verb, obj, rest = match.groups()
        # Normalize capitalization
        new_subj = obj[0].upper() + obj[1:] if obj else obj
        new_obj = subj[0].lower() + subj[1:] if subj else subj
        return f"{new_subj} {verb} {new_obj}{rest}"
    # Fallback: just reverse the words
    return reverse_words(text)


def causal_reversal(text: str) -> str:
    """
    Swap cause and effect.
    Targets connectives: "X causes Y" → "Y causes X",
                         "X leads to Y" → "Y leads to X", etc.
    """
    CAUSAL_PATTERNS = [
        (r"(.+?)\s+causes?\s+(.+)", r"\2 causes \1"),
        (r"(.+?)\s+leads?\s+to\s+(.+)", r"\2 leads to \1"),
        (r"(.+?)\s+results?\s+in\s+(.+)", r"\2 results in \1"),
        (r"(.+?)\s+triggers?\s+(.+)", r"\2 triggers \1"),
        (r"(.+?)\s+produces?\s+(.+)", r"\2 produces \1"),
        (r"If\s+(.+?),\s+then\s+(.+)", r"If \2, then \1"),
    ]
    for pattern, replacement in CAUSAL_PATTERNS:
        m = re.match(pattern, text, re.IGNORECASE)
        if m:
            return re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Fallback: reverse sentences
    return reverse_sentences(text)


def step_reorder(text: str, seed: Optional[int] = None) -> str:
    """
    Detect numbered steps (Step 1:, 1., 1), etc.) and shuffle their order
    while keeping their original labels.
    This simulates a document where procedural order is scrambled.
    """
    rng = random.Random(seed)

    # Match "Step N:" or "N." or "N)" at the start of segments
    step_pattern = re.compile(
        r"(?:^|\n|(?<=\.\s))((?:Step\s+)?\d+[.:)]\s*)",
        re.IGNORECASE,
    )
    splits = step_pattern.split(text)

    # splits alternates: [preamble, label1, content1, label2, content2, ...]
    if len(splits) < 5:
        # Not enough steps found — fall back to sentence shuffle
        return shuffle_sentences(text, seed=seed)

    preamble = splits[0]
    steps = []
    for i in range(1, len(splits) - 1, 2):
        label = splits[i]
        content = splits[i + 1] if i + 1 < len(splits) else ""
        steps.append((label, content))

    if len(steps) < 2:
        return shuffle_sentences(text, seed=seed)

    rng.shuffle(steps)
    body = "".join(label + content for label, content in steps)
    return preamble + body


def list_item_reversal(text: str) -> str:
    """
    Reverse the order of bullet/numbered list items.
    E.g., "priorities: safety, speed, cost" → "priorities: cost, speed, safety"
    """
    # Try comma-separated list after a colon
    m = re.match(r"^(.*?:\s*)(.+)$", text, re.DOTALL)
    if m:
        prefix, rest = m.groups()
        items = [i.strip() for i in rest.split(",")]
        if len(items) >= 2:
            return prefix + ", ".join(reversed(items))

    # Try numbered list lines
    lines = text.split("\n")
    numbered = [l for l in lines if re.match(r"^\s*\d+[.:)]", l)]
    if len(numbered) >= 2:
        non_numbered = [l for l in lines if not re.match(r"^\s*\d+[.:)]", l)]
        return "\n".join(non_numbered + list(reversed(numbered)))

    return reverse_sentences(text)


# ──────────────────────────────────────────────────────────────────────
# Registry + category map
# ──────────────────────────────────────────────────────────────────────

PERTURBATION_REGISTRY: Dict[str, Callable[[str], str]] = {
    # Lexical
    "lowercase":                  lowercase,
    "uppercase":                  uppercase,
    "strip_punctuation":          strip_punctuation,
    "swap_numbers":               swap_numbers,
    "add_typos":                  add_typos,
    # Structural
    "shuffle_words":              shuffle_words,
    "reverse_words":              reverse_words,
    "shuffle_sentences":          shuffle_sentences,
    "reverse_sentences":          reverse_sentences,
    "drop_words_20pct":           lambda t: drop_random_words(t, 0.2),
    "drop_words_50pct":           lambda t: drop_random_words(t, 0.5),
    "drop_first_half":            drop_first_half,
    "drop_second_half":           drop_second_half,
    "truncate_10w":               lambda t: truncate(t, 10),
    "truncate_5w":                lambda t: truncate(t, 5),
    # Semantic
    "inject_negation":            inject_negation,
    "inject_irrelevant_prefix":   inject_irrelevant_prefix,
    "inject_contradiction_prefix": inject_contradiction_prefix,
    "antonym_swap":               antonym_swap,
    # Retrieval-critical
    "subject_object_swap":        subject_object_swap,
    "causal_reversal":            causal_reversal,
    "step_reorder":               step_reorder,
    "list_item_reversal":         list_item_reversal,
}

# category → list of perturbation names
PERTURBATION_CATEGORIES: Dict[str, List[str]] = {
    "lexical": [
        "lowercase", "uppercase", "strip_punctuation", "swap_numbers", "add_typos",
    ],
    "structural": [
        "shuffle_words", "reverse_words", "shuffle_sentences", "reverse_sentences",
        "drop_words_20pct", "drop_words_50pct", "drop_first_half", "drop_second_half",
        "truncate_10w", "truncate_5w",
    ],
    "semantic": [
        "inject_negation", "inject_irrelevant_prefix",
        "inject_contradiction_prefix", "antonym_swap",
    ],
    "retrieval_critical": [
        "subject_object_swap", "causal_reversal", "step_reorder", "list_item_reversal",
    ],
}

# Reverse map: name → category
PERTURBATION_CATEGORY_MAP: Dict[str, str] = {
    name: cat
    for cat, names in PERTURBATION_CATEGORIES.items()
    for name in names
}

# Expected model behavior per category (for UI annotation)
CATEGORY_EXPECTED_HIGH_SIM = {"lexical", "structural"}    # high sim is FINE
CATEGORY_EXPECTED_LOW_SIM  = {"semantic", "retrieval_critical"}  # high sim = FAILURE

# Legacy aliases used elsewhere in the codebase
ORDER_PERTURBATIONS = PERTURBATION_CATEGORIES["structural"]


def get_category(name: str) -> str:
    return PERTURBATION_CATEGORY_MAP.get(name, "unknown")


def is_failure(name: str, similarity: float, threshold: float = 0.85) -> bool:
    """Return True if a high similarity on this perturbation is a model failure."""
    cat = get_category(name)
    if cat in CATEGORY_EXPECTED_LOW_SIM and similarity >= threshold:
        return True
    return False


# ──────────────────────────────────────────────────────────────────────
# PerturbationSuite
# ──────────────────────────────────────────────────────────────────────

class PerturbationSuite:
    """
    Apply perturbations to a corpus and return similarity statistics.

    Parameters
    ----------
    encode_fn : (List[str]) -> np.ndarray
    """

    def __init__(self, encode_fn: Callable[[List[str]], np.ndarray]) -> None:
        self.encode_fn = encode_fn

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-9:
            return 0.0
        return float(np.dot(a, b) / denom)

    def run_single(self, text: str, perturbation_type: str) -> PerturbationResult:
        fn = PERTURBATION_REGISTRY[perturbation_type]
        perturbed = fn(text)
        vecs = self.encode_fn([text, perturbed])
        return PerturbationResult(
            original=text,
            perturbed=perturbed,
            perturbation_type=perturbation_type,
            category=get_category(perturbation_type),
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
        all_vecs = self.encode_fn(texts + perturbed_texts)
        n = len(texts)
        orig_vecs, pert_vecs = all_vecs[:n], all_vecs[n:]
        results = [
            PerturbationResult(
                original=texts[i],
                perturbed=perturbed_texts[i],
                perturbation_type=perturbation_type,
                category=get_category(perturbation_type),
                original_vec=orig_vecs[i],
                perturbed_vec=pert_vecs[i],
                similarity=self._cosine(orig_vecs[i], pert_vecs[i]),
            )
            for i in range(n)
        ]
        return PerturbationBatchResult(
            results=results,
            perturbation_type=perturbation_type,
            category=get_category(perturbation_type),
        )

    def run_all(
        self,
        texts: List[str],
        perturbation_types: Optional[List[str]] = None,
    ) -> Dict[str, PerturbationBatchResult]:
        types = perturbation_types or list(PERTURBATION_REGISTRY.keys())
        return {t: self.run_batch(texts, t) for t in types}

    def run_category(
        self,
        texts: List[str],
        category: str,
    ) -> Dict[str, PerturbationBatchResult]:
        return self.run_all(texts, PERTURBATION_CATEGORIES[category])

    def summary_table(
        self,
        texts: List[str],
        perturbation_types: Optional[List[str]] = None,
        failure_threshold: float = 0.85,
    ) -> "pd.DataFrame":
        import pandas as pd
        results = self.run_all(texts, perturbation_types)
        rows = []
        for ptype, batch in results.items():
            cat = get_category(ptype)
            mean_sim = batch.mean_similarity
            rows.append({
                "category": cat,
                "perturbation": ptype,
                "mean_sim": round(mean_sim, 4),
                "min_sim": round(batch.min_similarity, 4),
                "std_sim": round(batch.std_similarity, 4),
                "n": len(batch.results),
                "is_failure": is_failure(ptype, mean_sim, failure_threshold),
            })
        return (
            pd.DataFrame(rows)
            .sort_values(["category", "mean_sim"])
            .reset_index(drop=True)
        )

    def order_sensitivity_report(self, texts: List[str]) -> "pd.DataFrame":
        import pandas as pd
        rows = []
        for ptype in ORDER_PERTURBATIONS:
            batch = self.run_batch(texts, ptype)
            for r in batch.results:
                rows.append({
                    "perturbation": ptype,
                    "original": r.original[:80],
                    "perturbed": r.perturbed[:80],
                    "similarity": round(r.similarity, 4),
                })
        return pd.DataFrame(rows)

    def failure_report(
        self,
        texts: List[str],
        perturbation_types: Optional[List[str]] = None,
        threshold: float = 0.85,
    ) -> "pd.DataFrame":
        """
        Return only the cases where a meaning-altering perturbation
        produced dangerously high similarity.
        """
        import pandas as pd
        types = perturbation_types or (
            PERTURBATION_CATEGORIES["semantic"]
            + PERTURBATION_CATEGORIES["retrieval_critical"]
        )
        rows = []
        for ptype in types:
            batch = self.run_batch(texts, ptype)
            for r in batch.results:
                if r.similarity >= threshold:
                    rows.append({
                        "perturbation": r.perturbation_type,
                        "category": r.category,
                        "original": r.original,
                        "perturbed": r.perturbed,
                        "similarity": round(r.similarity, 4),
                        "verdict": "FAILURE — semantically different, embedded as same",
                    })
        return pd.DataFrame(rows).sort_values("similarity", ascending=False)
