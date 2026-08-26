#!/usr/bin/env python3
"""Quick demo: ingest corpus, run a query, print answer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline


def main():
    pipeline = RAGPipeline()

    # Ingest
    corpus = Path(__file__).parent.parent / "data" / "corpus"
    if corpus.exists() and any(corpus.iterdir()):
        print("[*] Ingesting corpus...")
        stats = pipeline.ingest_directory(str(corpus))
        print(f"[+] {stats['chunks_indexed']} chunks indexed")
    else:
        print("[!] No corpus found. Run scripts/ingest_sample_docs.py first.")

    # Run sample queries
    questions = [
        "What is the retrieval system architecture?",
        "How does citation verification work?",
        "What is Reciprocal Rank Fusion?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = pipeline.ask(q)
        print(f"A: {result['answer'][:300]}")
        print(f"  confidence={result['confidence']:.2f}  "
              f"citations_verified={result['citations_verified']}")


if __name__ == "__main__":
    main()
