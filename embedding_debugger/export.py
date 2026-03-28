"""
Export utilities for Embedding Debugger.

Accumulate findings from any analysis module and export to:
  - JSON
  - CSV
  - Markdown report
  - Dict (for programmatic use)

Usage
-----
    report = DebugReport(title="FAQ retrieval audit", model="all-MiniLM-L6-v2")
    report.add_section("Retrieval failures", df_failures)
    report.add_section("Perturbation robustness", df_perturbation)
    report.add_metadata("recall@1", 0.85)
    report.to_markdown("report.md")
    report.to_json("report.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# DebugReport
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DebugReport:
    title: str
    model: str = "unknown"
    dataset: str = "unknown"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    metadata: Dict[str, Any] = field(default_factory=dict)
    sections: List[Dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_metadata(self, key: str, value: Any) -> "DebugReport":
        self.metadata[key] = value
        return self

    def add_section(
        self,
        title: str,
        df: pd.DataFrame,
        description: str = "",
    ) -> "DebugReport":
        self.sections.append({
            "title": title,
            "description": description,
            "rows": df.to_dict(orient="records"),
            "columns": list(df.columns),
            "n_rows": len(df),
        })
        return self

    def add_failure_cases(
        self,
        cases: List[Dict],
        section_title: str = "Failure Cases",
    ) -> "DebugReport":
        df = pd.DataFrame(cases)
        return self.add_section(section_title, df)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "model": self.model,
            "dataset": self.dataset,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "sections": self.sections,
        }

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, default=str)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_csv(self, path: Union[str, Path], section_idx: int = 0) -> None:
        """Export a single section to CSV."""
        if not self.sections:
            raise ValueError("No sections in report.")
        section = self.sections[section_idx]
        pd.DataFrame(section["rows"]).to_csv(path, index=False)

    def all_sections_to_csv(self, directory: Union[str, Path]) -> List[Path]:
        """Export each section as a separate CSV file."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, section in enumerate(self.sections):
            safe_title = re.sub(r"[^\w]+", "_", section["title"].lower())
            p = directory / f"{i:02d}_{safe_title}.csv"
            pd.DataFrame(section["rows"]).to_csv(p, index=False)
            paths.append(p)
        return paths

    def to_markdown(self, path: Optional[Union[str, Path]] = None) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"**Model:** {self.model}  ",
            f"**Dataset:** {self.dataset}  ",
            f"**Generated:** {self.created_at}",
            "",
        ]

        if self.metadata:
            lines += ["## Summary Metrics", ""]
            for k, v in self.metadata.items():
                v_fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
                lines.append(f"- **{k}:** {v_fmt}")
            lines.append("")

        for section in self.sections:
            lines += [f"## {section['title']}", ""]
            if section["description"]:
                lines += [section["description"], ""]
            df = pd.DataFrame(section["rows"])
            if len(df):
                lines.append(df.to_markdown(index=False))
            else:
                lines.append("_No rows._")
            lines.append("")

        md = "\n".join(lines)
        if path is not None:
            Path(path).write_text(md, encoding="utf-8")
        return md

    # ------------------------------------------------------------------
    # Streamlit helpers
    # ------------------------------------------------------------------

    def streamlit_download_buttons(self) -> None:
        """Render st.download_button widgets for JSON and Markdown exports."""
        import streamlit as st
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="⬇ Download JSON",
                data=self.to_json(),
                file_name=f"embedding_debug_{self.dataset}.json",
                mime="application/json",
            )
        with col2:
            st.download_button(
                label="⬇ Download Markdown report",
                data=self.to_markdown(),
                file_name=f"embedding_debug_{self.dataset}.md",
                mime="text/markdown",
            )


# ──────────────────────────────────────────────────────────────────────
# Convenience constructors
# ──────────────────────────────────────────────────────────────────────

def retrieval_report(
    model: str,
    dataset: str,
    failures_df: pd.DataFrame,
    metrics: Dict[str, float],
    perturbation_df: Optional[pd.DataFrame] = None,
) -> DebugReport:
    """Build a standard retrieval debug report."""
    report = DebugReport(
        title=f"Retrieval Debug Report — {dataset}",
        model=model,
        dataset=dataset,
    )
    for k, v in metrics.items():
        report.add_metadata(k, v)
    report.add_section(
        "Retrieval results",
        failures_df,
        "Per-query retrieval rank. Highlighted rows are failures (expected rank ≠ 0).",
    )
    if perturbation_df is not None:
        report.add_section(
            "Perturbation rank drift",
            perturbation_df,
            "How much retrieval rank shifts after each perturbation type.",
        )
    return report


def perturbation_report(
    model: str,
    dataset: str,
    summary_df: pd.DataFrame,
    failures_df: Optional[pd.DataFrame] = None,
) -> DebugReport:
    """Build a standard perturbation robustness report."""
    report = DebugReport(
        title=f"Perturbation Robustness Report — {dataset}",
        model=model,
        dataset=dataset,
    )
    n_failures = int(summary_df["is_failure"].sum()) if "is_failure" in summary_df.columns else 0
    report.add_metadata("n_perturbation_types", len(summary_df))
    report.add_metadata("n_failures", n_failures)
    report.add_section(
        "Perturbation summary",
        summary_df,
        "Mean cosine similarity after each perturbation. "
        "'is_failure' = True when a meaning-altering perturbation scores ≥ 0.85.",
    )
    if failures_df is not None and len(failures_df):
        report.add_section("Individual failure cases", failures_df)
    return report


# re-export for convenience
import re
