#!/usr/bin/env python3
"""Run the evaluation harness against the golden Q&A dataset.

Usage:
    python scripts/run_eval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.eval.dataset import get_sample_golden_dataset
from src.eval.harness import EvalHarness


def main():
    print("[*] Loading golden dataset...")
    dataset = get_sample_golden_dataset()
    print(f"[*] {len(dataset.cases)} test cases loaded")

    print("[*] Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    print("[*] Running evaluation (this will call the LLM for each test case)...")
    harness = EvalHarness(pipeline)
    results = harness.run(dataset)

    print("[*] Computing metrics...")
    metrics = harness.compute_metrics(results)
    harness.print_metrics(metrics)

    # Save results
    out_path = Path(__file__).parent.parent / "data" / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_path, "w") as f:
        json.dump({**metrics, "results": [r.to_dict() for r in results]}, f, indent=2)
    print(f"[+] Results saved to {out_path}")


if __name__ == "__main__":
    main()
