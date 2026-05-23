# Embedding Debugger — Model Benchmark Results

**Date:** 2026-03-28  
**Models:** all-MiniLM-L6-v2, all-mpnet-base-v2, e5-base-v2, gte-base  
**Dataset:** curated failure cases (n=15 adversarial pairs) + FAQ Q&A (n=20)  

> All similarity values are cosine similarity on L2-normalised embeddings.
> Lower similarity on adversarial pairs = better (model is more order-sensitive).

---

## 1. Order Sensitivity — Adversarial Pair Similarity

Mean cosine similarity between semantically-opposite pairs.
**A good model should score near 0. Most score near 0.95.**

### Overall (all 15 curated pairs)

| Model             |   Mean sim ↓ |   Min sim |   Max sim | % pairs > 0.90 ↓   | % pairs > 0.85 ↓   |
|:------------------|-------------:|----------:|----------:|:-------------------|:-------------------|
| all-MiniLM-L6-v2  |       0.9859 |    0.9576 |    0.9959 | 100.0%             | 100.0%             |
| all-mpnet-base-v2 |       0.9623 |    0.9109 |    0.9963 | 100.0%             | 100.0%             |
| e5-base-v2        |       0.9933 |    0.9804 |    0.9991 | 100.0%             | 100.0%             |
| gte-base          |       0.992  |    0.9652 |    0.9987 | 100.0%             | 100.0%             |

### By perturbation category

| Model             |   subject_object_swap |   causal_reversal |   step_reorder |   list_item_reversal |
|:------------------|----------------------:|------------------:|---------------:|---------------------:|
| all-MiniLM-L6-v2  |                0.9886 |            0.9779 |         0.9855 |               0.9946 |
| all-mpnet-base-v2 |                0.9595 |            0.9327 |         0.9892 |               0.9896 |
| e5-base-v2        |                0.9925 |            0.9882 |         0.9985 |               0.9981 |
| gte-base          |                0.9947 |            0.9852 |         0.9918 |               0.9978 |

---

## 2. FAQ Retrieval — Q→A Matching

Query: question. Corpus: 20 answers. Correct = question matches its own answer at rank 1.

**Caveat:** The FAQ dataset has semantic overlap between some Q&A pairs (e.g. "How do I track my
shipment?" vs the "Where is my order?" answer both discuss tracking). The one failure shared by all
models ("track my shipment") is arguably a dataset ambiguity, not a model error. All models reach
Recall@5 = 100%, meaning the correct answer is always in the top 5. The Recall@1 differences should
not be interpreted as a clean model quality ranking on this dataset.

| Model             |   Recall@1 ↑ |   Recall@5 ↑ |   Recall@10 ↑ |   MRR@10 ↑ |
|:------------------|-------------:|-------------:|--------------:|-----------:|
| all-MiniLM-L6-v2  |         0.95 |            1 |             1 |     0.975  |
| all-mpnet-base-v2 |         0.90 |            1 |             1 |     0.950  |
| e5-base-v2        |         0.85 |            1 |             1 |     0.917  |
| gte-base          |         0.95 |            1 |             1 |     0.975  |

The lower Recall@1 for e5-base-v2 (3 failures vs 1 for MiniLM) is consistent with E5 being more
sensitive to semantic overlap between ambiguous FAQ pairs — not evidence that E5 is weaker overall.

---

## 3. Perturbation Robustness (Semantic + Retrieval-Critical)

**Failure** = mean cosine ≥ 0.85 after a meaning-altering perturbation.

| Model             | Failures / types tested ↓   | Failure rate ↓   |
|:------------------|:----------------------------|:-----------------|
| all-MiniLM-L6-v2  | 8 / 8                       | 100%             |
| all-mpnet-base-v2 | 6 / 8                       | 75%              |
| e5-base-v2        | 8 / 8                       | 100%             |
| gte-base          | 8 / 8                       | 100%             |

#### Per-perturbation breakdown (all-MiniLM-L6-v2)

| perturbation                |   mean_sim | failure   |
|:----------------------------|-----------:|:----------|
| subject_object_swap         |     0.9093 | 🚨        |
| causal_reversal             |     0.9847 | 🚨        |
| list_item_reversal          |     0.9947 | 🚨        |
| step_reorder                |     0.9969 | 🚨        |
| inject_contradiction_prefix |     0.8543 | 🚨        |
| inject_irrelevant_prefix    |     0.8879 | 🚨        |
| inject_negation             |     0.9124 | 🚨        |
| antonym_swap                |     0.9967 | 🚨        |

---

## 4. Neighborhood Stability Across Models

RBO-based k=10 neighborhood stability. 1.0 = identical neighbors, 0.0 = completely different.
**% unstable** = fraction of texts where stability < 0.5.

| model_a           | model_b           |   mean_stability |   min_stability |   pct_unstable |
|:------------------|:------------------|-----------------:|----------------:|---------------:|
| all-MiniLM-L6-v2  | all-mpnet-base-v2 |           0.7127 |          0.5338 |            0   |
| all-MiniLM-L6-v2  | e5-base-v2        |           0.6813 |          0.5047 |            0   |
| all-MiniLM-L6-v2  | gte-base          |           0.7432 |          0.6399 |            0   |
| all-mpnet-base-v2 | e5-base-v2        |           0.7175 |          0.5186 |            0   |
| all-mpnet-base-v2 | gte-base          |           0.7093 |          0.5328 |            0   |
| e5-base-v2        | gte-base          |           0.7293 |          0.4866 |            6.7 |

---

## Key Takeaways

- **Most order-sensitive model:** `all-mpnet-base-v2` (mean adversarial sim = 0.9623)
- **Best FAQ retrieval (Recall@1):** `all-MiniLM-L6-v2` (95.0%)
- **Most perturbation-robust:** `all-mpnet-base-v2` (6 failures)

**Universal finding:** All tested models score mean cosine > 0.85 on semantically-opposite pairs.
This is a property of the pooling architecture, not any specific model.
