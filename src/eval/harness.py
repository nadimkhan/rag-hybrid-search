from __future__ import annotations
"""Evaluation harness: golden Q&A dataset, automated metrics, and reporting."""


import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..config import ROOT_DIR, LLM_MODEL, LLM_BASE_URL, OPENROUTER_API_KEY


# ── Golden Q&A Dataset ────────────────────────────────────────────────────────

@dataclass
class TestCase:
    """A single golden Q&A test case."""

    id: str
    question: str
    expected_answer: str
    difficulty: str = "medium"
    requires_sources: bool = True
    notes: str = ""
    category: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class GoldenDataset:
    """Collection of test cases with load/save to JSON."""

    def __init__(self, cases: list[TestCase] | None = None):
        self.cases: list[TestCase] = cases or []

    def add(self, case: TestCase) -> None:
        self.cases.append(case)

    def filter_by(self, **kwargs) -> list[TestCase]:
        results = self.cases
        for key, val in kwargs.items():
            results = [c for c in results if getattr(c, key, None) == val]
        return results

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = [c.to_dict() for c in self.cases]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "GoldenDataset":
        with open(path) as f:
            raw = json.load(f)
        cases = [TestCase(**item) for item in raw]
        return cls(cases)


# ── Evaluation Result ──────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Result of running one test case through the pipeline."""

    test_id: str
    question: str
    answer: str
    confidence: float
    confidence_level: str
    citation_coverage: float
    citations_verified: bool
    faithfulness_score: float = 0.0
    llm_judge_score: float | None = None
    passed: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Eval Harness ──────────────────────────────────────────────────────────────

class EvalHarness:
    """Runs the full pipeline against a golden dataset and produces metrics."""

    def __init__(self, pipeline: Any, judge_model: str | None = None):
        self.pipeline = pipeline
        self.judge_model = judge_model or LLM_MODEL

    def run(self, dataset: GoldenDataset, verbose: bool = True) -> list[EvalResult]:
        results = []
        for i, case in enumerate(dataset.cases, 1):
            if verbose:
                print(f"[{i}/{len(dataset.cases)}] {case.question[:60]}...")
            try:
                result = self._run_one(case)
            except Exception as exc:
                result = EvalResult(
                    test_id=case.id,
                    question=case.question,
                    answer="",
                    confidence=0.0,
                    confidence_level="low",
                    citation_coverage=0.0,
                    citations_verified=False,
                    errors=[str(exc)],
                )
            results.append(result)
        return results

    def _run_one(self, case: TestCase) -> EvalResult:
        response = self.pipeline.ask(case.question)
        answer = response.get("answer", "")
        confidence = response.get("confidence", 0.0)
        confidence_level = response.get("confidence_level", "low")
        citation_coverage = response.get("citation_coverage", 0.0)
        citations_verified = response.get("citations_verified", False)

        faithfulness = 0.0
        if case.requires_sources:
            faithfulness = self._judge_faithfulness(case.question, answer)

        passed = (
            confidence >= 0.6
            and citation_coverage >= 0.5
            and (not case.requires_sources or citations_verified)
        )

        return EvalResult(
            test_id=case.id,
            question=case.question,
            answer=answer,
            confidence=confidence,
            confidence_level=confidence_level,
            citation_coverage=citation_coverage,
            citations_verified=citations_verified,
            faithfulness_score=faithfulness,
            passed=passed,
        )

    def _judge_faithfulness(self, question: str, answer: str) -> float:
        """Rate faithfulness 0-1 via OpenRouter."""
        import httpx

        prompt = (
            f"Question: {question}\n\n"
            f"Answer: {answer}\n\n"
            "Is the answer faithful to the question? "
            "Rate 0.0 (completely unfaithful) to 1.0 (perfectly faithful). "
            "Respond with only the numeric score."
        )
        headers = {"Content-Type": "application/json"}
        if OPENROUTER_API_KEY:
            headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
        payload = {
            "model": self.judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 10,
            "stream": False,
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return max(0.0, min(1.0, float(text.strip().split()[0])))
        except Exception:
            return 0.0


    @staticmethod
    def compute_metrics(results: list[EvalResult]) -> dict[str, Any]:
        total = len(results)
        if total == 0:
            return {"total": 0}

        passed = sum(1 for r in results if r.passed)
        avg_confidence = sum(r.confidence for r in results) / total
        avg_faithfulness = sum(r.faithfulness_score for r in results) / total
        avg_citation_coverage = sum(r.citation_coverage for r in results) / total
        verified = sum(1 for r in results if r.citations_verified)

        by_difficulty = {}
        for diff in ("easy", "medium", "hard"):
            subset = [r for r in results if _get_difficulty(r) == diff]
            if subset:
                by_difficulty[diff] = {
                    "count": len(subset),
                    "pass_rate": sum(1 for r in subset if r.passed) / len(subset),
                    "avg_confidence": sum(r.confidence for r in subset) / len(subset),
                }

        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total,
            "avg_confidence": avg_confidence,
            "avg_faithfulness": avg_faithfulness,
            "avg_citation_coverage": avg_citation_coverage,
            "citations_verified_rate": verified / total,
            "by_difficulty": by_difficulty,
        }

    @staticmethod
    def print_metrics(metrics: dict[str, Any]) -> None:
        print("\n" + "=" * 50)
        print("EVALUATION REPORT")
        print("=" * 50)
        print(f"Total cases           : {metrics['total']}")
        print(f"Pass rate             : {metrics['pass_rate']:.1%}")
        print(f"Avg confidence        : {metrics['avg_confidence']:.3f}")
        print(f"Avg faithfulness      : {metrics['avg_faithfulness']:.3f}")
        print(f"Citation coverage     : {metrics['avg_citation_coverage']:.1%}")
        print(f"Citations verified    : {metrics['citations_verified_rate']:.1%}")
        print("\nBy difficulty:")
        for diff, m in metrics.get("by_difficulty", {}).items():
            print(f"  {diff:8s}: pass={m['pass_rate']:.0%}  "
                  f"avg_conf={m['avg_confidence']:.3f}  (n={m['count']})")
        print("=" * 50)


def _get_difficulty(result: EvalResult) -> str:
    if result.confidence >= 0.8:
        return "easy"
    elif result.confidence >= 0.5:
        return "medium"
    return "hard"
