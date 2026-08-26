# RAG Hybrid Search Architecture

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
