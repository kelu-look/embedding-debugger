# Launch Assets — Embedding Debugger v0.1.0

Drop-in copy for the v0.1.0 announcement. All posts are written as an
independent open-source project. No external references implied.

---

## Twitter / X

**Option A (short, demo-led):**

> Shipping Embedding Debugger v0.1.0 🔬
>
> A local-first toolkit for debugging retrieval failures in text
> embeddings — wrong top-1 at high similarity, step reorder, causal
> reversal, drift, outliers.
>
> One command: `python -m demo.killer_demo`
>
> MIT • github.com/kelu-look/embedding-debugger

**Option B (problem-first):**

> Embeddings say "0.97 similarity" — and retrieve the wrong document.
>
> Embedding Debugger v0.1.0 is out: 22 perturbations, FAISS retrieval
> forensics, neighborhood stability, drift + outliers, exportable
> reports. Local-first, MIT.
>
> github.com/kelu-look/embedding-debugger

---

## LinkedIn

> **Releasing Embedding Debugger v0.1.0 — an open-source debugger for
> text embedding systems.**
>
> If you ship search or RAG on top of embeddings, you've probably seen
> this: a query retrieves the wrong document at 0.97 cosine similarity,
> a reordered procedure looks identical to the original, and neighbors
> quietly shuffle when you swap models.
>
> Embedding Debugger gives you a local-first microscope for those
> failures:
>
> • Retrieval forensics — FAISS top-k inspector, Recall@k, MRR@k, and
>   rank drift under perturbation.
> • 22 perturbations across lexical, structural, semantic, and
>   retrieval-critical categories (step reorder, causal reversal,
>   negation, number changes, list-item reversal).
> • Geometry, drift, and outliers — clustering, PCA/UMAP, LOF,
>   Isolation Forest, RBO neighborhood stability, Procrustes drift.
> • Exportable reports in Markdown / JSON / CSV.
> • Streamlit UI + one-command CLI demo.
>
> Runs fully offline. MIT licensed.
>
> Repo + killer demo: https://github.com/kelu-look/embedding-debugger

---

## Reddit — r/MachineLearning [P]

**Title:**

> [P] Embedding Debugger v0.1.0 — a local-first toolkit for debugging
> retrieval failures in text embeddings (FAISS, perturbations, drift,
> outliers)

**Body:**

> Hi r/MachineLearning,
>
> I just released v0.1.0 of **Embedding Debugger**, a local-first
> open-source toolkit for debugging text embedding behavior.
>
> The motivating problem: in production search and RAG, embeddings
> regularly produce "high-similarity wrong" retrievals. Two texts whose
> meaning differs (reversed causality, swapped subject/object,
> reordered steps, negation, number changes) often sit at >0.95 cosine
> similarity. Existing tooling visualizes embeddings, but doesn't tell
> you _why_ a specific query retrieved the wrong document.
>
> **What's in v0.1.0:**
>
> - **Retrieval Debugging** — FAISS top-k inspector, per-query failure
>   analysis, Recall@k, MRR@k, and rank drift under perturbation.
> - **Perturbation & Robustness** — 22 perturbations across 4
>   categories (lexical, structural, semantic, retrieval-critical).
> - **Geometry, Drift & Outliers** — KMeans + PCA/UMAP, LOF /
>   Isolation Forest / centroid distance, RBO neighborhood stability,
>   Procrustes centroid drift.
> - **Exportable Reports** — Markdown / JSON / CSV via `DebugReport`.
> - **Streamlit UI** with pages for each pillar, plus a
>   `python -m demo.killer_demo` one-command CLI flow.
> - **Local-first** — no API calls, runs on CPU. SBERT, E5, GTE, BGE
>   supported out of the box.
> - 78 unit tests, no model download required.
>
> **Killer demo (one command):**
>
> ```
> pip install -r requirements.txt
> python -m demo.killer_demo
> ```
>
> Walks query → retrieved wrong result → high similarity score →
> perturbation/rank drift → `killer_demo_report.md`.
>
> Repo: https://github.com/kelu-look/embedding-debugger
>
> Feedback, issues, and PRs welcome. MIT licensed.

---

## Demo captions (for GIFs / screenshots / video)

1. **High-similarity wrong retrieval.**
   *"Query asks for a forward procedure. Top-1 is the reversed
   procedure at cosine 0.97. Embedding Debugger shows the gap between
   semantic intent and what the index actually returned."*

2. **Perturbation sweep + rank drift.**
   *"Apply 22 perturbations across 4 categories — lexical, structural,
   semantic, retrieval-critical. Watch which ones change ranks and
   which ones the embedding ignores. Negation, step reorder, and
   causal reversal show up clearly."*

3. **Exportable debug report.**
   *"Every run exports a Markdown / JSON / CSV report — queries,
   retrieved items, similarity scores, perturbation outcomes, and
   failure tags. Drop it into a PR, a notebook, or a tracker."*
