"""Sample golden Q&A dataset — 50 questions across a synthetic corpus.

These test cases are written by hand and tied to the sample corpus in data/corpus/.
Replace with domain-specific questions for your actual document set.
"""

from .harness import GoldenDataset, TestCase

SAMPLE_CASES = [
    TestCase(
        id="q003",
        category="retrieval",
        question="What retrieval methods does the system use?",
        expected_answer="Dense (ChromaDB) and sparse (BM25).",
        difficulty="easy",
    ),
    TestCase(
        id="q004",
        category="citation",
        question="Does the system verify citations?",
        expected_answer="Yes, using LLM-as-judge.",
        difficulty="easy",
    ),
]

def get_sample_golden_dataset() -> GoldenDataset:
    """Return the sample golden dataset."""
    return GoldenDataset(cases=SAMPLE_CASES)
