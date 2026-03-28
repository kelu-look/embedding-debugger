# 🧠 Embedding Debugger

**Debug what your embeddings are actually doing.**

A local-first toolkit for diagnosing retrieval failures, perturbation sensitivity,
and representation pathologies in text embeddings.

Instead of just visualizing embeddings, this tool helps you answer:

- Why did this query retrieve the wrong result?
- Why are two semantically different texts almost identical in embedding space?
- When do embeddings ignore order, structure, or meaning changes?
- How stable are neighborhoods under perturbations?
- How do different embedding models behave on the same data?

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-78%20passing-brightgreen)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

![Embedding Debugger demo](assets/demo.gif)

---

## 📊 Measured Results (4 models, 2026-03-28)

> Full details in [`results/model_benchmark.md`](results/model_benchmark.md).

### Order blindness — adversarial pair similarity
Mean cosine similarity between semantically-opposite pairs (lower = better; ideal = 0).

| Model | Mean sim ↓ | % pairs > 0.90 ↓ | FAQ R@1 †  | Pert. failures ↓ |
|:------|----------:|:----------------:|:----------:|:----------------:|
| all-MiniLM-L6-v2  | **0.986** | 100% | 95% | 8/8 |
| all-mpnet-base-v2 | **0.962** | 100% | 90% | **6/8** |
| e5-base-v2        | **0.993** | 100% | 85% | 8/8 |
| gte-base          | **0.992** | 100% | 95% | 8/8 |

† FAQ Recall@1 is confounded by dataset ambiguity (overlapping Q&A topics); all models reach Recall@5 = 100%. Do not interpret this column as a clean model ranking.

**Every model scores > 0.96 mean cosine on semantically-opposite pairs. This is not a bug in any specific model — it is a consequence of pooled-token architecture.**

### By failure category

| Model | subject_object_swap | causal_reversal | step_reorder | list_item_reversal |
|:------|:------------------:|:---------------:|:------------:|:-----------------:|
| all-MiniLM-L6-v2  | 0.989 | 0.978 | 0.986 | 0.995 |
| all-mpnet-base-v2 | 0.960 | 0.933 | 0.989 | 0.990 |
| e5-base-v2        | 0.993 | 0.988 | 0.999 | 0.998 |
| gte-base          | 0.995 | 0.985 | 0.992 | 0.998 |

### Neighborhood stability across model pairs
Mean RBO@10 — how much of each point's 10-nearest-neighbors is shared between models.

| Model A | Model B | Stability (1=identical) | % unstable texts |
|:--------|:--------|:-----------------------:|:----------------:|
| all-MiniLM-L6-v2 | all-mpnet-base-v2 | 0.713 | 0% |
| all-MiniLM-L6-v2 | e5-base-v2        | 0.681 | 0% |
| all-MiniLM-L6-v2 | gte-base          | 0.743 | 0% |
| all-mpnet-base-v2 | e5-base-v2       | 0.718 | 0% |
| all-mpnet-base-v2 | gte-base         | 0.709 | 0% |
| e5-base-v2        | gte-base         | 0.729 | **6.7%** |

~27% of each point's nearest neighbors change when switching between model families. Retrieval quality is model-dependent in ways invisible without measurement.

---

## 🚨 Why this matters

Modern systems — search, semantic retrieval, RAG — rely heavily on embeddings.

But embeddings often behave like order-invariant or bag-of-words representations, leading to:

- ❌ incorrect retrieval despite high similarity
- ❌ failure on ranking / procedural / causal text
- ❌ unstable or misleading nearest neighbors

Embedding Debugger helps you see and diagnose these failures directly.

---

## 🔍 Core Capabilities

### 1. Retrieval Debugging

- Inspect top-k nearest neighbors
- Analyze ranking errors and confusion cases
- Compute MRR@k, Recall@k
- Track rank drift under perturbations

### 2. Perturbation & Robustness

22 built-in perturbations across 4 categories:

| Category | Examples | High sim = ? |
|----------|---------|-------------|
| **lexical** | casing, punctuation, typos | ✅ expected |
| **structural** | word/sentence shuffle & reversal, truncation | reveals order blindness |
| **semantic** | negation injection, antonym swap, contradiction prefix | 🚨 failure |
| **retrieval_critical** | step reorder, causal reversal, subject-object swap, list inversion | 🚨 failure |

### 3. Geometry & Distribution

- Clustering (KMeans + elbow)
- 2D projection (PCA / UMAP)
- Outlier detection (LOF, Isolation Forest, centroid distance)
- Similarity distribution analysis

### 4. Drift & Model Comparison

- Compare embedding spaces across models or time
- Procrustes alignment + centroid drift
- Neighborhood stability analysis (RBO)
- Cross-model retrieval differences

---

## ⚡ Quickstart

### 1. Install

```bash
git clone https://github.com/yourname/embedding-debugger.git
cd embedding-debugger
pip install -r requirements.txt
```

### 2. Launch UI

```bash
streamlit run app/streamlit_app.py
```

Open the local URL in your browser. Start with **🎯 Killer Demo** for the full failure pipeline.

### 3. Run the killer demo (CLI)

```bash
python -m demo.killer_demo
# Try a different model:
python -m demo.killer_demo --model gte-small
```

This shows: order-sensitive failures → perturbation robustness → retrieval breakdowns → mitigation directions.
Output: console report + `killer_demo_report.md`.

---

## 🧪 Example: Retrieval Failure

**Query:** `"Step 1: boil water → Step 2: add pasta"`

**Retrieved:** `"Step 1: add pasta → Step 2: boil water"`

Despite reversed meaning:
- cosine similarity ≈ 0.99
- ranked as top result

> The embedding treats meaning-changing permutations as nearly identical.

More failure cases from the curated test set:

```
subject_object_swap:  "The company acquired the startup."
                    → "The startup acquired the company."   cosine: 0.93

causal_reversal:    "The power outage caused the server crash."
                  → "The server crash caused the power outage."  cosine: 0.95

step_reorder:       (5-step backup procedure, steps scrambled)   cosine: 0.97

list_item_reversal: "priorities: safety, reliability, performance, cost"
                  → "priorities: cost, performance, reliability, safety"  cosine: 0.96
```

---

## 🏗️ Project Structure

```
embedding_debugger/
  models.py        # EmbeddingModel: multi-model loader (SBERT, E5, GTE, BGE)
  similarity.py    # Cosine, top-k, RBO-ext neighborhood stability
  perturbation.py  # 22 perturbations across 4 categories
  retrieval.py     # FAISS + failure analysis + rank drift
  clustering.py    # KMeans + PCA/UMAP
  drift.py         # Procrustes + neighborhood stability
  outliers.py      # LOF, Isolation Forest, centroid distance
  export.py        # DebugReport → JSON / CSV / Markdown

app/
  streamlit_app.py         # 3-pillar navigation
  pages/killer_demo.py     # Full failure pipeline page
  pages/retrieval.py       # Pillar 1 — Retrieval Debugging
  pages/perturbation.py    # Pillar 2 — Perturbation & Robustness
  pages/clustering.py      # Pillar 3 — Geometry & Drift
  pages/comparison.py      # Model comparison
  pages/drift.py           # Drift tracker
  pages/outliers.py        # Outlier detector

demo/
  datasets.py      # 5 built-in datasets (no download)
  failure_cases.py # 15 curated adversarial pairs with explanations
  killer_demo.py   # One-command demo script
  experiments.py   # Research-style experiment runner
```

---

## 🧭 Design Principles

- **Local-first** — no API dependency, runs fully offline
- **Modular & extensible** — swap in any HuggingFace model, any dataset
- **Diagnostics over visualization** — every output answers a specific failure question
- **Reproducible** — deterministic tests, cached embeddings, export at every step
- **Focused on real failure modes** — built around documented, reproducible embedding pathologies

---

## 🛣️ Roadmap

- [ ] Retrieval failure report export (JSON / HTML) ✅ done
- [ ] Failure-case mining across datasets
- [ ] Structured perturbation benchmark
- [ ] RAG pipeline integration
- [ ] Token-level attribution / saliency
- [ ] BM25 vs dense retrieval comparison
- [ ] Custom dataset upload in UI (CSV / JSONL)

---

## 🤝 Use Cases

- Debugging embedding-based retrieval systems
- Evaluating robustness of embedding models before deployment
- Comparing SBERT / E5 / GTE behaviors on your data
- Analyzing dataset structure and representation drift
- Supporting research on embedding failures and order blindness

---

## Testing

```bash
pytest tests/ -v
# 78 tests, all passing, no model download required
```

Tests use deterministic fake embedders (numpy). Coverage includes all 22 perturbation
types, `is_failure` logic, FAISS retrieval, export, and the curated failure cases dataset.

---

## Export

```python
from embedding_debugger.export import DebugReport, retrieval_report

report = retrieval_report(
    model="all-MiniLM-L6-v2",
    dataset="faq",
    failures_df=df,
    metrics={"recall_at_1": 0.85, "mrr": 0.91},
)
report.to_markdown("report.md")
report.to_json("report.json")
report.to_csv("failures.csv")
```

---

## Supported models

| Name | HuggingFace ID | Dim |
|------|----------------|-----|
| `all-MiniLM-L6-v2` | sentence-transformers/all-MiniLM-L6-v2 | 384 |
| `all-mpnet-base-v2` | sentence-transformers/all-mpnet-base-v2 | 768 |
| `e5-small-v2` | intfloat/e5-small-v2 | 384 |
| `e5-base-v2` | intfloat/e5-base-v2 | 768 |
| `gte-small` | thenlper/gte-small | 384 |
| `gte-base` | thenlper/gte-base | 768 |
| `bge-small-en-v1.5` | BAAI/bge-small-en-v1.5 | 384 |
| `paraphrase-MiniLM-L6-v2` | sentence-transformers/paraphrase-MiniLM-L6-v2 | 384 |

Models are downloaded from HuggingFace on first use (~80–400 MB) and cached locally.

---

## Curated failure cases

`demo/failure_cases.py` — 15 hand-crafted adversarial pairs with documented explanations,
expected vs. typical cosine scores, and gap measurement:

| Category | Pairs | Typical cosine | Expected cosine |
|----------|-------|---------------|-----------------|
| subject_object_swap | 6 | ~0.95 | ~0.05 |
| causal_reversal | 4 | ~0.95 | ~0.10 |
| step_reorder | 3 | ~0.97 | ~0.02 |
| list_item_reversal | 2 | ~0.95 | ~0.00 |

---

## Mitigation directions

The root cause: sentence encoders pool all tokens into a single vector,
discarding positional and relational structure.

- **Chunked retrieval** — index at sentence/step level, not document level
- **Cross-encoder re-ranking** — use a model that attends to the full pair
- **Structural metadata** — store step numbers, dependency labels alongside vectors
- **Adversarial evaluation** — use `demo/failure_cases.py` as a test set before deployment

---

⭐ If you find this useful, star the repo and feel free to contribute!

> **GitHub description:** Debug retrieval failures and hidden behaviors in text embeddings.

---

## License

MIT
