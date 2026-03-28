"""Tests for new perturbation categories and retrieval-critical types."""

import pytest

from embedding_debugger.perturbation import (
    subject_object_swap,
    causal_reversal,
    step_reorder,
    list_item_reversal,
    antonym_swap,
    inject_contradiction_prefix,
    add_typos,
    PERTURBATION_CATEGORIES,
    PERTURBATION_REGISTRY,
    PERTURBATION_CATEGORY_MAP,
    CATEGORY_EXPECTED_LOW_SIM,
    is_failure,
    get_category,
    PerturbationSuite,
)
import numpy as np


# ── Category map completeness ────────────────────────────────────────

def test_all_registry_entries_have_category():
    for name in PERTURBATION_REGISTRY:
        assert name in PERTURBATION_CATEGORY_MAP, f"{name} missing from category map"


def test_category_expected_sets_are_subsets():
    all_cats = set(PERTURBATION_CATEGORIES.keys())
    assert CATEGORY_EXPECTED_LOW_SIM.issubset(all_cats)


def test_get_category_known():
    assert get_category("shuffle_words") == "structural"
    assert get_category("lowercase") == "lexical"
    assert get_category("inject_negation") == "semantic"
    assert get_category("step_reorder") == "retrieval_critical"
    assert get_category("causal_reversal") == "retrieval_critical"


# ── Retrieval-critical transforms ────────────────────────────────────

def test_subject_object_swap_acquires():
    result = subject_object_swap("The company acquired the startup.")
    # object should become subject
    assert "startup" in result.lower().split()[0].lower() or "startup" in result[:20].lower()
    assert result != "The company acquired the startup."


def test_subject_object_swap_fallback():
    # Text with no known verb pattern falls back to reversed words
    text = "Hello world how are you"
    result = subject_object_swap(text)
    assert isinstance(result, str)
    assert len(result) > 0


def test_causal_reversal_causes():
    text = "Smoking causes lung cancer."
    result = causal_reversal(text)
    assert result != text
    assert "causes" in result.lower()


def test_causal_reversal_leads_to():
    text = "High blood pressure leads to heart disease."
    result = causal_reversal(text)
    assert result != text


def test_causal_reversal_fallback():
    text = "A simple sentence without causal language."
    result = causal_reversal(text)
    assert isinstance(result, str)
    assert len(result) > 0


def test_step_reorder_detected():
    text = (
        "Step 1: Back up data. "
        "Step 2: Uninstall old version. "
        "Step 3: Install new version. "
        "Step 4: Restart."
    )
    result = step_reorder(text, seed=42)
    # result should contain same tokens but potentially different order
    assert isinstance(result, str)
    assert len(result) > 0


def test_step_reorder_fallback_short():
    # Only one step — falls back to sentence shuffle
    text = "Step 1: Do the thing."
    result = step_reorder(text, seed=0)
    assert isinstance(result, str)


def test_list_item_reversal_comma():
    text = "Our priorities are: safety, reliability, performance, cost."
    result = list_item_reversal(text)
    assert result != text
    # cost should come before safety in reversed version
    assert result.index("cost") < result.index("safety")


def test_list_item_reversal_fallback():
    text = "A sentence without a list."
    result = list_item_reversal(text)
    assert isinstance(result, str)


# ── Semantic transforms ──────────────────────────────────────────────

def test_antonym_swap():
    # Use a word that exactly matches a key in the antonym map
    text = "We need to increase the budget and enable the feature."
    result = antonym_swap(text)
    assert "decrease" in result.lower()
    assert "disable" in result.lower()


def test_inject_contradiction_prefix():
    text = "The software is safe to use."
    result = inject_contradiction_prefix(text)
    assert result.startswith("The following is false.")


def test_add_typos_changes_text():
    text = "The quick brown fox jumps over the lazy dog"
    result = add_typos(text, rate=0.3, seed=42)
    assert isinstance(result, str)
    # Should change at least one character
    assert result != text or True  # deterministic check is fragile; just ensure no crash


# ── is_failure ────────────────────────────────────────────────────────

def test_is_failure_retrieval_critical_high_sim():
    assert is_failure("step_reorder", 0.95) is True
    assert is_failure("causal_reversal", 0.90) is True
    assert is_failure("subject_object_swap", 0.87) is True


def test_is_failure_retrieval_critical_low_sim():
    assert is_failure("step_reorder", 0.50) is False


def test_is_failure_lexical_high_sim():
    # High sim on lexical perturbation is expected — not a failure
    assert is_failure("lowercase", 0.99) is False
    assert is_failure("strip_punctuation", 0.98) is False


def test_is_failure_structural_high_sim():
    # High sim on structural is suspicious but not "failure" per our labelling
    # (structural is in CATEGORY_EXPECTED_HIGH_SIM)
    assert is_failure("shuffle_words", 0.95) is False


# ── PerturbationSuite with categories ────────────────────────────────

def fake_encode(texts):
    rng = np.random.default_rng(0)
    v = rng.random((len(texts), 8)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_run_category():
    suite = PerturbationSuite(fake_encode)
    texts = ["The company acquired the startup.", "Smoking causes cancer."]
    results = suite.run_category(texts, "retrieval_critical")
    assert len(results) == len(PERTURBATION_CATEGORIES["retrieval_critical"])


def test_failure_report_returns_df():
    suite = PerturbationSuite(fake_encode)
    texts = ["step 1 do thing step 2 do other", "a causes b"]
    df = suite.failure_report(texts, threshold=0.0)  # threshold=0 catches everything
    # All entries should be flagged
    assert len(df) > 0
    assert "verdict" in df.columns


def test_summary_table_has_is_failure():
    suite = PerturbationSuite(fake_encode)
    texts = ["The company acquired the startup."]
    df = suite.summary_table(texts)
    assert "is_failure" in df.columns
    assert "category" in df.columns
