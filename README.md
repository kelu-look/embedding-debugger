# 🔬 Embedding Debugger

**Local-first toolkit for analyzing text embedding behavior.**

Embedding models power search, RAG, and recommendations — but they fail in subtle,
hard-to-detect ways. Embedding Debugger gives you the tools to *see* those failures.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What it does

| Module | What you can diagnose |
|--------|----------------------|
| **Similarity Inspector** | Query a corpus; inspect ranked neighbors with scores |
| **Perturbation Lab** | Measure robustness to order swaps, word drops, casing, etc. |
| **Retrieval Debugger** | FAISS-based retrieval + failure / rank-drift analysis |
| **Cluster Geometry** | PCA / UMAP 2-D projection with KMeans clustering |
| **Model Comparison** | Compare SBERT / E5 / GTE: neighbor overlap, order sensitivity |
| **Drift Tracker** | Procrustes + RBO analysis between two embedding spaces |
| **Outlier Detector** | LOF / Isolation Forest / centroid-distance anomaly detection |

---

## Known failure modes demonstrated

### Order blindness

Most embedding models treat word-order-reversed text as nearly identical:

```
"The dog bit the man."  ↔  "The man bit the dog."   → cosine ≈ 0.96
"Alice loves Bob."      ↔  "Bob loves Alice."        → cosine ≈ 0.97
"Revenue exceeded expenses." ↔ "Expenses exceeded revenue." → cosine ≈ 0.95
```

Run `python -m demo.experiments` to reproduce on your machine.

### Retrieval rank collapse

Shuffling words in a query moves the correct answer from rank 1 to rank 10+ in ~30-50%
of cases depending on the model.

### Unstable neighborhoods

20-40% of a point's k-nearest-neighbors change when switching from SBERT to GTE,
even on identical text.

---

## Quick start

```bash
git clone https://github.com/yourusername/embedding-debugger
cd embedding-debugger

# Install dependencies
pip install -r requirements.txt

# Run CLI experiments (downloads models on first run, ~80 MB for MiniLM)
python -m demo.experiments

# Launch the Streamlit UI
streamlit run app/streamlit_app.py
```

---

## Installation (editable)

```bash
pip install -e .
```

---

## Supported models

| Short name | HuggingFace ID | Dim | Notes |
|-----------|----------------|-----|-------|
| `all-MiniLM-L6-v2` | sentence-transformers/all-MiniLM-L6-v2 | 384 | Fast, good baseline |
| `all-mpnet-base-v2` | sentence-transformers/all-mpnet-base-v2 | 768 | Stronger SBERT |
| `e5-small-v2` | intfloat/e5-small-v2 | 384 | E5 family (needs prefix) |
| `e5-base-v2` | intfloat/e5-base-v2 | 768 | |
| `gte-small` | thenlper/gte-small | 384 | GTE family |
| `gte-base` | thenlper/gte-base | 768 | |
| `bge-small-en-v1.5` | BAAI/bge-small-en-v1.5 | 384 | BGE family |
| `paraphrase-MiniLM-L6-v2` | sentence-transformers/paraphrase-MiniLM-L6-v2 | 384 | Paraphrase-tuned |

---

## Demo datasets (built-in, no download)

| Name | Size | Description |
|------|------|-------------|
| `faq` | 40 | FAQ Q&A pairs — Q→A retrieval |
| `order_blind` | 20 | Meaning-opposite word-order pairs |
| `news` | 33 | Headlines across 5 categories |
| `nli` | 24 | Entailment / contradiction / neutral |
| `products` | 20 | E-commerce product descriptions |

---

## Project structure

```
embedding-debugger/
├── embedding_debugger/       # Core library
│   ├── models.py             # EmbeddingModel: load + encode
│   ├── similarity.py         # Cosine, top-k, RBO
│   ├── perturbation.py       # 16 perturbation types
│   ├── clustering.py         # KMeans + PCA/UMAP
│   ├── retrieval.py          # FAISS + failure analysis
│   ├── drift.py              # Procrustes + neighborhood stability
│   ├── outliers.py           # LOF, Isolation Forest, centroid distance
│   └── utils.py              # Caching, display helpers
├── app/
│   ├── streamlit_app.py      # Entry point
│   └── pages/                # One module per UI page
├── demo/
│   ├── datasets.py           # 5 built-in datasets (no download)
│   └── experiments.py        # CLI experiments showing failure cases
└── tests/                    # pytest suite (no model download needed)
```

---

## Running tests

```bash
pytest tests/ -v
```

Tests use deterministic fake embedders — no model download required.

---

## Programmatic usage

```python
from embedding_debugger import EmbeddingModel, PerturbationSuite, RetrievalDebugger

# Load a model
model = EmbeddingModel("all-MiniLM-L6-v2")

# Encode texts
texts = ["The dog bit the man.", "The man bit the dog.", "A cat sat on the mat."]
vecs = model.encode(texts)

# Perturbation robustness
suite = PerturbationSuite(model.encode)
report = suite.summary_table(texts)
print(report)

# FAISS retrieval + failure analysis
answers = ["dogs bite humans", "humans bite dogs", "cats rest on mats"]
a_vecs = model.encode(answers)
debugger = RetrievalDebugger(answers, a_vecs)
result = debugger.retrieve(texts[0], vecs[0], k=3)
```

---

## Roadmap

- [ ] Token-level attention visualization
- [ ] Dataset upload in Streamlit (CSV / JSONL)
- [ ] Sentence-level order perturbation for longer documents
- [ ] BM25 vs dense retrieval comparison
- [ ] Automatic failure report export (PDF / Markdown)
- [ ] Support for OpenAI / Cohere embedding APIs

---

## License

MIT
