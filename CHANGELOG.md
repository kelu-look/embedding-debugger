# Changelog

All notable changes to **Embedding Debugger** are documented here.
This project follows [Semantic Versioning](https://semver.org/) and the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [0.1.0] — 2026-05-18

First public release. Embedding Debugger is a local-first toolkit for
debugging retrieval failures, perturbation sensitivity, neighborhood
instability, clustering, drift, and outliers in text embedding systems.

### Added
- **Retrieval Debugging**
  - FAISS-backed top-k retrieval inspector
  - Per-query failure analysis (wrong top-1, near-miss, low-margin)
  - Recall@k and MRR@k metrics
  - Rank-drift tracking under perturbations
- **Perturbation & Robustness**
  - 22 built-in perturbations across 4 categories: lexical, structural,
    semantic, retrieval-critical
  - Categories include ranking swaps, step reordering, causal reversal,
    negation, number changes, and list-item reversal
  - `is_failure` rule for similarity-vs-meaning mismatches
- **Geometry, Drift & Outliers**
  - KMeans clustering with elbow selection
  - 2D projections via PCA and UMAP
  - Outlier detection (LOF, Isolation Forest, centroid distance)
  - Cross-model neighborhood stability (RBO-ext, fixed to score 1.0 on
    identical lists)
  - Procrustes-aligned centroid drift
- **Exportable Reports**
  - `DebugReport` writer to Markdown, JSON, and CSV
  - One-shot `retrieval_report(...)` helper
- **Streamlit App**
  - Killer-demo page, retrieval, perturbation, clustering, comparison,
    drift, and outliers pages
- **CLI**
  - `python -m demo.killer_demo` for a one-command end-to-end run
  - `--model` flag to switch embedding backbones
- **Demo Datasets & Failure Cases**
  - 5 built-in datasets (no download required)
  - 15 hand-crafted adversarial pairs with documented explanations
- **Models**
  - Out-of-the-box support for SBERT, E5, GTE, and BGE families
- **Testing**
  - 78 unit tests, all passing, no model download required

### Known limitations
- CPU-first design; large corpora (>100k items) are slower without a GPU
  backbone for embedding extraction.
- UMAP and FAISS are optional at runtime but assumed installed via
  `requirements.txt`.
- The Streamlit UI is single-user; no multi-tenant auth.
- Drift metrics assume comparable tokenization across models; very
  different vocabularies may inflate apparent drift.
- The 22-perturbation suite is English-centric; multilingual coverage is
  on the roadmap.

### Notes
- License: MIT
- Python: 3.10+

[0.1.0]: https://github.com/kelu-look/embedding-debugger/releases/tag/v0.1.0
