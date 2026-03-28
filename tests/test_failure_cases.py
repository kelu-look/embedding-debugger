"""Tests for the curated failure cases dataset."""

import pytest
from demo.failure_cases import (
    ALL_FAILURE_CASES,
    FAILURE_CASE_CATEGORIES,
    get_failure_cases,
    failure_cases_dataframe,
    FailurePair,
)


def test_all_failure_cases_nonempty():
    assert len(ALL_FAILURE_CASES) > 0


def test_all_cases_have_required_fields():
    for case in ALL_FAILURE_CASES:
        assert isinstance(case, FailurePair)
        assert case.original
        assert case.adversarial
        assert case.category
        assert case.explanation
        assert 0.0 <= case.expected_sim <= 1.0
        assert 0.0 <= case.typical_sim <= 1.0


def test_all_cases_different():
    for case in ALL_FAILURE_CASES:
        assert case.original != case.adversarial, f"pair is identical: {case.original}"


def test_categories_covered():
    cats = {c.category for c in ALL_FAILURE_CASES}
    assert "subject_object_swap" in cats
    assert "causal_reversal" in cats
    assert "step_reorder" in cats
    assert "list_item_reversal" in cats


def test_get_failure_cases_all():
    cases = get_failure_cases("all")
    assert len(cases) == len(ALL_FAILURE_CASES)


def test_get_failure_cases_category():
    cases = get_failure_cases("causal_reversal")
    assert all(c.category == "causal_reversal" for c in cases)


def test_failure_cases_dataframe():
    df = failure_cases_dataframe()
    assert len(df) == len(ALL_FAILURE_CASES)
    assert "category" in df.columns
    assert "gap" in df.columns
    # gap = typical_sim - expected_sim; should be positive for all curated cases
    assert (df["gap"] >= 0).all()


def test_typical_sim_exceeds_expected():
    for case in ALL_FAILURE_CASES:
        assert case.typical_sim > case.expected_sim, (
            f"{case.category}: typical_sim={case.typical_sim} should exceed "
            f"expected_sim={case.expected_sim}"
        )
