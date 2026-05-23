# v0.1.0 — Local-first embedding failure debugger

**Embedding Debugger** is a local-first toolkit for debugging retrieval
failures, perturbation sensitivity, neighborhood instability, clustering,
drift, and outliers in text embedding systems.

If you ship embeddings into search or RAG, this gives you the
microscope you've been missing: queries that retrieve the wrong result
at high similarity, perturbations that should change meaning but don't,
and neighborhoods that quietly shift between models.

---

## Highlights

- **Killer demo in one command** — `python -m demo.killer_demo` walks
  query → retrieved wrong result → high similarity score →
  perturbation/rank drift → exportable debug report.
- **22 perturbations, 4 categories** — lexical, structural, semantic,
  and retrieval-critical (step reordering, causal reversal, negation,
  number changes, list-item reversal, subject-object swap).
- **Retrieval forensics** — FAISS top-k inspector, Recall@k, MRR@k, and
  rank-drift under perturbation.
- **Geometry & drift** — KMeans + PCA/UMAP, LOF / Isolation Forest /
  centroid-distance outliers, RBO-ext neighborhood stability, Procrustes
  centroid drift.
- **Exportable reports** — Markdown, JSON, and CSV via `DebugReport`.
- **Streamlit UI** — multi-page app for interactive debugging.
- **Local-first** — runs fully offline after model download; no API
  keys, no telemetry.
- **Tested** — 78 unit tests pass without any model download.

## What's in the box

- Models: SBERT, E5, GTE, BGE families out of the box.
- Datasets: 5 built-in toy datasets and 15 curated adversarial pairs.
- Reports: one-shot `retrieval_report(...)` helper.
- App: pages for Killer Demo, Retrieval, Perturbation, Clustering,
  Comparison, Drift, and Outliers.

## Install

```bash
git clone https://github.com/kelu-look/embedding-debugger.git
cd embedding-debugger
pip install -r requirements.txt
```

Python 3.10+. CPU-only works.

## Run

```bash
# Streamlit UI
streamlit run app/streamlit_app.py

# One-command killer demo (CLI)
python -m demo.killer_demo
python -m demo.killer_demo --model gte-small

# Tests
pytest tests/ -v
```

## Known limitations

- CPU-first; corpora >100k items are slower without GPU.
- UMAP and FAISS are required and installed via `requirements.txt`.
- The Streamlit UI is single-user.
- Perturbations are English-centric.
- Drift metrics assume comparable tokenization across compared models.

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for the full list of changes.

## License

MIT. Issues, PRs, and feedback welcome.
