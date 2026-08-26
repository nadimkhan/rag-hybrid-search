# Retrieval System

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
