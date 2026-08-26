# Architecture Decision Records

## ADR-001: Hybrid Retrieval over Dense-Only

**Context:** Dense vector search is strong on semantic similarity but weak on exact technical terms (function names, config keys, error codes).

**Decision:** Combine ChromaDB dense retrieval with BM25 sparse retrieval using Reciprocal Rank Fusion with configurable weights (default 0.7/0.3 dense/sparse).

**Consequences:** Improved recall on technical docs. Slightly higher latency from running two retrievals. Configurable weights let us tune per corpus.
