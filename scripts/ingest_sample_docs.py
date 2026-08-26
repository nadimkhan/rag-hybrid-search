#!/usr/bin/env python3
"""Ingest sample documents into the RAG pipeline.

Usage:
    python scripts/ingest_sample_docs.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.config import ROOT_DIR

CORPUS_DIR = ROOT_DIR / "data" / "corpus"


def main():
    print(f"[*] Ingesting sample corpus from {CORPUS_DIR}")
    pipeline = RAGPipeline()

    if not CORPUS_DIR.exists():
        print(f"[!] Corpus directory not found: {CORPUS_DIR}")
        print("[*] Creating sample corpus...")
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        create_sample_corpus(CORPUS_DIR)

    stats = pipeline.ingest_directory(str(CORPUS_DIR))
    print(f"[+] Ingested {stats['docs_processed']} documents, {stats['chunks_indexed']} chunks")


def create_sample_corpus(root: Path):
    """Write a small sample corpus for demo purposes."""
    docs = {
        "architecture.md": """# RAG Hybrid Search Architecture

## Overview
RAG Hybrid Search is a production-grade Retrieval-Augmented Generation pipeline that combines dense vector search with sparse BM25 keyword matching.

## Components
The system has five main components:

1. **Ingestion Pipeline** — loads documents, chunks them, embeds them, and indexes into ChromaDB and BM25.
2. **Dense Retriever** — uses ChromaDB with cosine similarity to find semantically similar chunks.
3. **Sparse Retriever** — uses BM25 for exact keyword matching, critical for technical terms.
4. **Fusion Layer** — combines dense and sparse rankings using Reciprocal Rank Fusion (RRF).
5. **Generator** — produces grounded answers with verified inline citations.

## Why Hybrid Search?
Dense semantic search misses exact technical terms like function names, config keys, and error codes.
BM25 catches exact keyword matches. RRF fusion combines both with configurable weights.
""",
        "retrieval.md": """# Retrieval System

## Dense Retrieval
ChromaDB stores embeddings and supports cosine similarity queries. We embed queries
using OpenAI's text-embedding-3-small model (or any OpenAI-compatible endpoint).

## Sparse Retrieval (BM25)
BM25 is a probabilistic ranking function used for keyword-based document retrieval.
It handles exact matches for technical terms, proper nouns, and configuration keys
that semantic search often misses.

## Reciprocal Rank Fusion (RRF)
RRF combines rankings from multiple retrieval methods without requiring score normalization.

Formula: score(d) = sum(w_r / (k + rank_r(d)) for all retrieval methods r)

Default k=60, weights: 0.7 dense / 0.3 sparse.

## Cross-Encoder Reranking
After fusion, a cross-encoder reranker scores query-chunk relevance in a second pass,
dramatically improving precision. We use an LLM-as-judge for reranking.
""",
        "generation.md": """# Generation and Citation System

## Grounded Generation
The generator produces answers using ONLY the retrieved context. The system prompt
instructs the model to cite every factual claim using bracketed references like [1].

## Citation Verification
After generation, every citation is verified using an LLM-as-judge.
A citation is marked verified when the judge's score >= 0.7.
Unverified citations are flagged, not hidden.

## Confidence Scoring
Confidence = 0.5 * retrieval_confidence + 0.5 * citation_coverage
- High: >= 0.85
- Medium: >= 0.60
- Low: < 0.60

## Unanswerable Handling
When retrieval confidence is too low, the system explicitly says it cannot answer
rather than hallucinating.
""",
        "evaluation.md": """# Evaluation Framework

## Golden Dataset
50+ hand-written Q&A pairs covering:
- Simple lookups (easy)
- Multi-hop reasoning (medium)
- Edge cases and failure modes (hard)
- Questions the corpus cannot answer

## Metrics
- **Faithfulness**: Are all claims grounded in retrieved context?
- **Answer correctness**: LLM-as-judge vs. golden answer
- **Retrieval relevance**: Were the right chunks retrieved?
- **Citation accuracy**: Do citations actually support their claims?

## Chunking Strategy Comparison
The eval framework runs the same suite across three chunking strategies:
fixed-size, recursive, and semantic. Results show which strategy wins per document type.
""",
    }

    for filename, content in docs.items():
        (root / filename).write_text(content, encoding="utf-8")
        print(f"  Created {filename}")


if __name__ == "__main__":
    main()
