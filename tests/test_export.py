"""Tests for the export module."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from embedding_debugger.export import (
    DebugReport,
    retrieval_report,
    perturbation_report,
)


def sample_df():
    return pd.DataFrame({
        "query": ["What is X?", "How to do Y?"],
        "rank": [1, 3],
        "score": [0.91, 0.62],
        "is_failure": [False, True],
    })


# ── DebugReport construction ─────────────────────────────────────────

def test_add_section_and_metadata():
    report = DebugReport(title="Test", model="mini", dataset="faq")
    report.add_metadata("recall@1", 0.85)
    report.add_section("Results", sample_df(), "Test description")
    assert report.metadata["recall@1"] == 0.85
    assert len(report.sections) == 1
    assert report.sections[0]["n_rows"] == 2


def test_to_dict():
    report = DebugReport(title="T", model="m", dataset="d")
    d = report.to_dict()
    assert "title" in d
    assert "sections" in d
    assert "metadata" in d


# ── JSON export ──────────────────────────────────────────────────────

def test_to_json_string():
    report = DebugReport(title="Test", model="mini", dataset="x")
    report.add_section("S1", sample_df())
    payload = report.to_json()
    parsed = json.loads(payload)
    assert parsed["title"] == "Test"
    assert len(parsed["sections"]) == 1


def test_to_json_file():
    report = DebugReport(title="T", model="m", dataset="d")
    report.add_section("S", sample_df())
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "report.json"
        report.to_json(p)
        assert p.exists()
        parsed = json.loads(p.read_text())
        assert "sections" in parsed


# ── CSV export ───────────────────────────────────────────────────────

def test_to_csv():
    report = DebugReport(title="T", model="m", dataset="d")
    report.add_section("S", sample_df())
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "out.csv"
        report.to_csv(p, section_idx=0)
        assert p.exists()
        df = pd.read_csv(p)
        assert len(df) == 2


def test_all_sections_to_csv():
    report = DebugReport(title="T", model="m", dataset="d")
    report.add_section("First section", sample_df())
    report.add_section("Second section", sample_df())
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = report.all_sections_to_csv(tmpdir)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()


# ── Markdown export ──────────────────────────────────────────────────

def test_to_markdown_contains_title():
    report = DebugReport(title="My Report", model="mini", dataset="faq")
    report.add_metadata("recall@1", 0.85)
    report.add_section("Results", sample_df(), "Description here")
    md = report.to_markdown()
    assert "# My Report" in md
    assert "recall@1" in md
    assert "Results" in md
    assert "Description here" in md


def test_to_markdown_file():
    report = DebugReport(title="T", model="m", dataset="d")
    report.add_section("S", sample_df())
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "report.md"
        report.to_markdown(p)
        assert p.exists()
        assert "# T" in p.read_text()


# ── Convenience constructors ─────────────────────────────────────────

def test_retrieval_report():
    report = retrieval_report(
        model="mini",
        dataset="faq",
        failures_df=sample_df(),
        metrics={"recall_at_1": 0.9, "mrr": 0.92},
    )
    assert "recall_at_1" in report.metadata
    assert len(report.sections) == 1


def test_perturbation_report():
    summary_df = pd.DataFrame({
        "perturbation": ["shuffle_words", "causal_reversal"],
        "category": ["structural", "retrieval_critical"],
        "mean_sim": [0.91, 0.88],
        "min_sim": [0.80, 0.75],
        "std_sim": [0.05, 0.04],
        "n": [10, 10],
        "is_failure": [False, True],
    })
    report = perturbation_report(model="mini", dataset="faq", summary_df=summary_df)
    assert report.metadata["n_failures"] == 1
