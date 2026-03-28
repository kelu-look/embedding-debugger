"""
Curated failure-case datasets.

These are hand-crafted pairs where a naive embedding model is expected
to fail — assigning high cosine similarity to semantically opposite text.

Each dataset comes with:
  - a short narrative explaining *why* the failure matters
  - the pair of texts
  - the expected correct behaviour
  - a category (what class of failure this is)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class FailurePair:
    original: str
    adversarial: str
    category: str          # e.g. "causal_reversal", "step_reorder", "subject_object_swap"
    explanation: str       # why this matters in practice
    expected_sim: float    # what a good model *should* score
    typical_sim: float     # what standard models typically score (ballpark)


# ──────────────────────────────────────────────────────────────────────
# Subject-Object Swap
# ──────────────────────────────────────────────────────────────────────

SUBJECT_OBJECT_FAILURES: List[FailurePair] = [
    FailurePair(
        original="The dog bit the man.",
        adversarial="The man bit the dog.",
        category="subject_object_swap",
        explanation="Classic agency reversal. Different victim, different legal and medical implications.",
        expected_sim=0.0,
        typical_sim=0.95,
    ),
    FailurePair(
        original="The company acquired the startup.",
        adversarial="The startup acquired the company.",
        category="subject_object_swap",
        explanation="Opposite M&A direction. A RAG system would retrieve the wrong party as buyer.",
        expected_sim=0.1,
        typical_sim=0.93,
    ),
    FailurePair(
        original="The patient sued the doctor.",
        adversarial="The doctor sued the patient.",
        category="subject_object_swap",
        explanation="Legal direction reversal — who is the plaintiff vs. defendant.",
        expected_sim=0.0,
        typical_sim=0.94,
    ),
    FailurePair(
        original="Inflation causes interest rates to rise.",
        adversarial="Interest rates cause inflation to rise.",
        category="subject_object_swap",
        explanation="Economic causality — the Fed mechanism runs one way, not both.",
        expected_sim=0.1,
        typical_sim=0.96,
    ),
    FailurePair(
        original="Revenue exceeded expenses last quarter.",
        adversarial="Expenses exceeded revenue last quarter.",
        category="subject_object_swap",
        explanation="Profit vs loss — opposite financial outcome.",
        expected_sim=0.0,
        typical_sim=0.97,
    ),
    FailurePair(
        original="Alice loves Bob.",
        adversarial="Bob loves Alice.",
        category="subject_object_swap",
        explanation="Relationship direction matters in many knowledge-graph and QA tasks.",
        expected_sim=0.2,
        typical_sim=0.98,
    ),
]

# ──────────────────────────────────────────────────────────────────────
# Causal Reversal
# ──────────────────────────────────────────────────────────────────────

CAUSAL_REVERSAL_FAILURES: List[FailurePair] = [
    FailurePair(
        original="Smoking causes lung cancer.",
        adversarial="Lung cancer causes smoking.",
        category="causal_reversal",
        explanation="Reversed causal direction — medically nonsensical but embedded nearly identically.",
        expected_sim=0.0,
        typical_sim=0.94,
    ),
    FailurePair(
        original="High blood pressure leads to heart disease.",
        adversarial="Heart disease leads to high blood pressure.",
        category="causal_reversal",
        explanation="While some bidirectional relationship exists, the primary causal arrow is reversed.",
        expected_sim=0.3,
        typical_sim=0.96,
    ),
    FailurePair(
        original="The power outage caused the server to crash.",
        adversarial="The server crash caused the power outage.",
        category="causal_reversal",
        explanation="IT incident causality — root cause vs symptom — is completely reversed.",
        expected_sim=0.0,
        typical_sim=0.95,
    ),
    FailurePair(
        original="Deforestation causes climate change.",
        adversarial="Climate change causes deforestation.",
        category="causal_reversal",
        explanation="Both are partially true, but retrieving the wrong direction distorts policy framing.",
        expected_sim=0.4,
        typical_sim=0.97,
    ),
]

# ──────────────────────────────────────────────────────────────────────
# Step Reorder (procedural / instructional)
# ──────────────────────────────────────────────────────────────────────

STEP_REORDER_FAILURES: List[FailurePair] = [
    FailurePair(
        original=(
            "Step 1: Back up your data. "
            "Step 2: Uninstall the old version. "
            "Step 3: Install the new version. "
            "Step 4: Restore your data. "
            "Step 5: Verify the installation."
        ),
        adversarial=(
            "Step 1: Restore your data. "
            "Step 2: Install the new version. "
            "Step 3: Back up your data. "
            "Step 4: Uninstall the old version. "
            "Step 5: Verify the installation."
        ),
        category="step_reorder",
        explanation=(
            "Following this shuffled order would restore data that doesn't exist yet "
            "and install over an already-running system. A RAG model would return "
            "this as matching 'software upgrade procedure' at cosine > 0.95."
        ),
        expected_sim=0.1,
        typical_sim=0.97,
    ),
    FailurePair(
        original=(
            "Step 1: Preheat oven to 375°F. "
            "Step 2: Mix dry ingredients. "
            "Step 3: Add wet ingredients and stir. "
            "Step 4: Pour into pan. "
            "Step 5: Bake for 30 minutes."
        ),
        adversarial=(
            "Step 1: Bake for 30 minutes. "
            "Step 2: Mix dry ingredients. "
            "Step 3: Preheat oven to 375°F. "
            "Step 4: Add wet ingredients and stir. "
            "Step 5: Pour into pan."
        ),
        category="step_reorder",
        explanation=(
            "Baking before mixing produces nothing. Both documents contain identical "
            "vocabulary and will be retrieved interchangeably."
        ),
        expected_sim=0.0,
        typical_sim=0.96,
    ),
    FailurePair(
        original=(
            "Step 1: Turn off the circuit breaker. "
            "Step 2: Remove the old outlet. "
            "Step 3: Connect the new wires. "
            "Step 4: Attach the outlet to the wall. "
            "Step 5: Turn the circuit breaker back on."
        ),
        adversarial=(
            "Step 1: Connect the new wires. "
            "Step 2: Remove the old outlet. "
            "Step 3: Turn off the circuit breaker. "
            "Step 4: Attach the outlet to the wall. "
            "Step 5: Turn the circuit breaker back on."
        ),
        category="step_reorder",
        explanation=(
            "Connecting wires to a live circuit is a safety hazard. "
            "The original and adversarial document would be retrieved interchangeably."
        ),
        expected_sim=0.0,
        typical_sim=0.97,
    ),
]

# ──────────────────────────────────────────────────────────────────────
# List Inversion (ranking / priority)
# ──────────────────────────────────────────────────────────────────────

LIST_INVERSION_FAILURES: List[FailurePair] = [
    FailurePair(
        original="Our top priorities are: safety, reliability, performance, cost.",
        adversarial="Our top priorities are: cost, performance, reliability, safety.",
        category="list_item_reversal",
        explanation=(
            "Inverted priority list — exactly the opposite organizational values. "
            "A governance document retrieval system would conflate these."
        ),
        expected_sim=0.0,
        typical_sim=0.96,
    ),
    FailurePair(
        original="Recommended treatment order: surgery, chemotherapy, radiation, follow-up.",
        adversarial="Recommended treatment order: follow-up, radiation, chemotherapy, surgery.",
        category="list_item_reversal",
        explanation=(
            "Medical treatment protocol in reverse order. Embedding models treat "
            "this as near-identical, which is a patient safety risk."
        ),
        expected_sim=0.0,
        typical_sim=0.95,
    ),
]

# ──────────────────────────────────────────────────────────────────────
# Combined registry
# ──────────────────────────────────────────────────────────────────────

ALL_FAILURE_CASES: List[FailurePair] = (
    SUBJECT_OBJECT_FAILURES
    + CAUSAL_REVERSAL_FAILURES
    + STEP_REORDER_FAILURES
    + LIST_INVERSION_FAILURES
)

FAILURE_CASE_CATEGORIES = {
    "subject_object_swap": SUBJECT_OBJECT_FAILURES,
    "causal_reversal": CAUSAL_REVERSAL_FAILURES,
    "step_reorder": STEP_REORDER_FAILURES,
    "list_item_reversal": LIST_INVERSION_FAILURES,
}


def get_failure_cases(category: str = "all") -> List[FailurePair]:
    if category == "all":
        return ALL_FAILURE_CASES
    return FAILURE_CASE_CATEGORIES.get(category, [])


def failure_cases_dataframe() -> "pd.DataFrame":
    import pandas as pd
    rows = []
    for fp in ALL_FAILURE_CASES:
        rows.append({
            "category": fp.category,
            "original": fp.original,
            "adversarial": fp.adversarial,
            "explanation": fp.explanation,
            "expected_sim": fp.expected_sim,
            "typical_sim": fp.typical_sim,
            "gap": round(fp.typical_sim - fp.expected_sim, 2),
        })
    return pd.DataFrame(rows)
