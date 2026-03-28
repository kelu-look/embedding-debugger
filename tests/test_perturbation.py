"""Tests for perturbation functions (no model download)."""

import numpy as np
import pytest

from embedding_debugger.perturbation import (
    shuffle_words,
    reverse_words,
    shuffle_sentences,
    drop_random_words,
    lowercase,
    strip_punctuation,
    inject_prefix,
    truncate,
    PERTURBATION_REGISTRY,
    PerturbationSuite,
)


TEXT = "The quick brown fox jumps over the lazy dog."
MULTI = "The cat sat on the mat. The dog ran outside. It was a sunny day."


def test_shuffle_words_same_words():
    result = shuffle_words(TEXT, seed=0)
    assert sorted(result.split()) == sorted(TEXT.split())


def test_reverse_words():
    words = TEXT.split()
    result = reverse_words(TEXT)
    assert result.split() == list(reversed(words))


def test_shuffle_sentences_contains_all():
    result = shuffle_sentences(MULTI, seed=0)
    # all original content should still be there (just reordered)
    assert len(result) > 0


def test_drop_words_fewer():
    result = drop_random_words(TEXT, 0.5, seed=0)
    assert len(result.split()) < len(TEXT.split())


def test_lowercase():
    assert lowercase("Hello World") == "hello world"


def test_strip_punctuation():
    result = strip_punctuation("Hello, World!")
    assert "," not in result
    assert "!" not in result


def test_inject_prefix():
    result = inject_prefix(TEXT, "PREFIX. ")
    assert result.startswith("PREFIX. ")


def test_truncate():
    result = truncate(TEXT, max_words=3)
    assert len(result.split()) <= 3


def test_registry_all_callable():
    for name, fn in PERTURBATION_REGISTRY.items():
        result = fn(TEXT)
        assert isinstance(result, str), f"{name} did not return str"


def fake_encode(texts):
    """Deterministic fake embedder for testing (no GPU/model needed)."""
    rng = np.random.default_rng(0)
    v = rng.random((len(texts), 8)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_perturbation_suite_run_single():
    suite = PerturbationSuite(fake_encode)
    result = suite.run_single(TEXT, "reverse_words")
    assert result.perturbation_type == "reverse_words"
    assert isinstance(result.similarity, float)
    assert -1.0 <= result.similarity <= 1.0


def test_perturbation_suite_run_batch():
    suite = PerturbationSuite(fake_encode)
    texts = [TEXT, "another sample text", "one more example here"]
    batch = suite.run_batch(texts, "shuffle_words")
    assert len(batch.results) == 3
    assert 0.0 <= batch.mean_similarity <= 1.0


def test_summary_table():
    suite = PerturbationSuite(fake_encode)
    texts = [TEXT, "another example text for testing"]
    df = suite.summary_table(texts, ["reverse_words", "lowercase"])
    assert len(df) == 2
    assert "mean_sim" in df.columns
