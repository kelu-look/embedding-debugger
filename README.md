# 🔬 Embedding Debugger

**A debugger for embedding failures — not a visualization tool.**

Embedding models fail silently. They return wrong answers from RAG pipelines,
treat safety-critical procedures as interchangeable, and score semantically
opposite sentences at cosine > 0.95. This toolkit makes those failures visible,
measurable, and reproducible.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-78%20passing-brightgreen)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The core problem

```
Query:     "What are the steps to upgrade the software safely?"

Corpus doc A (CORRECT):
  Step 1: Back up your data.  Step 2: Close all applications.
  Step 3: Run the installer.  Step 4: Restart.  Step 5: Verify.

Corpus doc B (DANGEROUS — same steps, wrong order):
  Step 1: Restart.  Step 2: Run the installer.
  Step 3: Back up your data.  Step 4: Close all applications.  Step 5: Verify.

Cosine(A, B) = 0.97   ← model treats them as nearly identical
Score gap(A vs B) = 0.002  ← retrieval cannot distinguish them
```

Standard embedding models discard positional structure. A document with steps
in the **wrong order** — including dangerous or safety-critical order — embedds
identically to the correct version. This toolkit diagnoses exactly this class of failure.

---

## Three diagnostic pillars

### 🔴 Pillar 1 — Retrieval Debugging

Find where retrieval breaks at the query level.

- FAISS-based index with Recall@k, MRR scoring
- Per-query rank failure identification
- **Score gap analysis**: how close is the wrong answer to the right one?
- **Rank drift**: apply a perturbation and measure how much retrieval breaks

```python
from embedding_debugger import EmbeddingModel, RetrievalDebugger

model = EmbeddingModel("all-MiniLM-L6-v2")
a_vecs = model.encode(answers)
debugger = RetrievalDebugger(answers, a_vecs)

_, df = debugger.analyze_failures(questions, q_vecs, expected_indices, k=10)
print(f"Recall@1: {debugger.recall_at_k(df, 1):.1%}")
print(f"MRR@10:   {debugger.mrr_at_k(df, 10):.4f}")
```

---

### 🟡 Pillar 2 — Perturbation & Robustness

Stress-test models with 22 perturbations across 4 categories.

| Category | Examples | High sim = ? |
|----------|---------|-------------|
| **lexical** | casing, punctuation, typos | ✅ expected |
| **structural** | word/sentence shuffle | reveals order blindness |
| **semantic** | negation, antonyms, contradiction | 🚨 failure |
| **retrieval_critical** | step reorder, causal reversal, subject-object swap, list inversion | 🚨 failure |

```python
from embedding_debugger import PerturbationSuite

suite = PerturbationSuite(model.encode)

# Run all retrieval-critical perturbations
df = suite.summary_table(texts, perturbation_types=["step_reorder", "causal_reversal",
                                                      "subject_object_swap", "list_item_reversal"])
failures = df[df["is_failure"]]
print(f"{len(failures)} perturbation types scored ≥0.85 despite changing meaning")
```

**Retrieval-critical perturbations:**

```
subject_object_swap:  "The company acquired the startup."
                    → "The startup acquired the company."  cosine: 0.93

causal_reversal:    "The power outage caused the server crash."
                  → "The server crash caused the power outage."  cosine: 0.95

step_reorder:       (5-step backup procedure, steps scrambled)  cosine: 0.97

list_item_reversal: "priorities: safety, reliability, performance, cost"
                  → "priorities: cost, performance, reliability, safety"  cosine: 0.96
```

---

### 🔵 Pillar 3 — Geometry & Drift

Understand the embedding space structure and how it changes across models.

- PCA / UMAP 2-D projection with KMeans clustering
- **Neighborhood stability**: RBO score measuring how much k-NN changes across models
- **Procrustes alignment**: structural drift between two embedding spaces
- Outlier detection: LOF, Isolation Forest, centroid distance, consensus voting

```python
from embedding_debugger import DriftAnalyzer

da = DriftAnalyzer(texts, vecs_sbert, vecs_gte, name_a="SBERT", name_b="GTE")
report = da.report(k=10)
print(f"Neighborhood stability: {report.neighborhood_stability:.3f}")
# 0.6 = 40% of each point's 10-NN changes when switching SBERT → GTE
```

---

## Quick start

```bash
git clone https://github.com/yourusername/embedding-debugger
cd embedding-debugger
pip install -r requirements.txt
```

### One-command killer demo

```bash
python -m demo.killer_demo
# Optional: try a different model
python -m demo.killer_demo --model gte-small
```

This runs a 4-section narrative showing:
1. A query that retrieves a dangerous procedure as its top result
2. Proof of order blindness across all curated failure cases
3. Category-level robustness sweep
4. Summary with mitigation directions

Output: console report + `killer_demo_report.md`

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Navigate to **🎯 Killer Demo** first for the full failure pipeline.

---

## Testing

```bash
pytest tests/ -v
# 78 tests, all passing, no model download required
```

Tests use deterministic fake embedders (numpy random). The test suite covers:
- All 22 perturbation types
- Every category and `is_failure` logic
- FAISS retrieval with numpy fallback
- Export to JSON / CSV / Markdown
- Curated failure cases dataset

---

## Export

Every analysis page exports results as **JSON** and **Markdown**:

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

## Curated failure cases

`demo/failure_cases.py` contains hand-crafted semantically-opposite pairs
with documented explanations and expected vs. typical similarity scores:

| Category | # pairs | Typical cosine | Expected cosine |
|----------|---------|---------------|-----------------|
| subject_object_swap | 6 | ~0.95 | ~0.05 |
| causal_reversal | 4 | ~0.95 | ~0.10 |
| step_reorder | 3 | ~0.97 | ~0.02 |
| list_item_reversal | 2 | ~0.95 | ~0.00 |

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

---

## Project structure

```
embedding-debugger/
├── embedding_debugger/       # Core library
│   ├── models.py             # EmbeddingModel: multi-model loader
│   ├── similarity.py         # Cosine, top-k, RBO-ext
│   ├── perturbation.py       # 22 perturbations across 4 categories
│   ├── clustering.py         # KMeans + PCA/UMAP
│   ├── retrieval.py          # FAISS + failure + rank drift analysis
│   ├── drift.py              # Procrustes + neighborhood stability
│   ├── outliers.py           # LOF, Isolation Forest, centroid distance
│   ├── export.py             # DebugReport → JSON / CSV / Markdown
│   └── utils.py              # Caching, display helpers
├── app/
│   ├── streamlit_app.py      # 3-pillar navigation
│   └── pages/
│       ├── killer_demo.py    # Full failure pipeline page
│       ├── retrieval.py      # Pillar 1
│       ├── perturbation.py   # Pillar 2
│       ├── clustering.py     # Pillar 3
│       ├── comparison.py     # Model comparison
│       ├── drift.py          # Drift tracker
│       ├── outliers.py       # Outlier detector
│       └── similarity.py     # Similarity inspector
├── demo/
│   ├── datasets.py           # 5 built-in datasets (no download)
│   ├── failure_cases.py      # 15 curated adversarial pairs
│   ├── killer_demo.py        # One-command demo script
│   └── experiments.py        # Research-style experiment runner
└── tests/                    # 78 tests, no model download needed
```

---

## Mitigation directions (output from the killer demo)

The root cause is that sentence encoders pool all tokens into a single vector,
discarding positional and relational structure. Practical mitigations:

- **Chunked retrieval**: index at sentence/step level rather than document level
- **Cross-encoder re-ranking**: use a model that attends to full pair context
- **Structural metadata**: store step numbers and dependency labels alongside embeddings
- **Adversarial evaluation**: build a test set from `demo/failure_cases.py` and run it before deploying

---

## License

MIT
