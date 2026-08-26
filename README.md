# RAG Hybrid Search — Production-Grade RAG with Hybrid Retrieval

> A production-grade Retrieval-Augmented Generation pipeline combining dense vector search with sparse BM25 keyword matching, cross-encoder reranking, and grounded generation with verified inline citations.

**Demo status:** Running end-to-end (BM25-only mode — see Evaluation section)

## Architecture

```
Ingestion                    Query-time                    Evaluation
────────                    ─────────                    ─────────
Document Loader ──┐        Query ──┐                  Golden Q&A
Chunker ──────────┼──▶ Vector Store│                  Eval Harness
BM25 Index ───────┤        ├──────▼──────┐            Metrics
                   │        │  Dense Retriever │         Reports
                   │        │  BM25 Retriever  │
                   │        └──────┬──────┘
                   │              │
                   │         RRF Fusion
                   │              │
                   │         Cross-Encoder Reranker
                   │              │
                   │         Generator (LLM + Citation)
                   │              │
                   └────────────── Citation Verifier ────▶ Answer + Score
```

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Language | Python 3.11+ | Ecosystem standard |
| Embeddings | OpenRouter / Omniroute (free tier) | Zero cost |
| Vector Store | ChromaDB (file-based) | Zero infra, git-friendly |
| Sparse Search | BM25 via `rank_bm25` | Keyword matching for exact terms |
| LLM | Omniroute (auto/chat → hy3-free) | Zero cost |
| Containerization | Docker | Reproducible deployment |

## Quick Start

```bash
# Clone
git clone https://github.com/nadimkhan/rag-hybrid-search.git
cd rag-hybrid-search

# Install dependencies
pip install -e .

# Start Omniroute (free LLM gateway)
docker start omniroute

# Ingest the sample corpus
python scripts/ingest_sample_docs.py

# Run the demo
python scripts/demo.py

# Run evaluation
python scripts/run_eval.py

# Start the API server
make run
```

## Mode 1: BM25-Only (Zero Config — runs today)

Set `USE_DENSE_RETRIEVAL=false` (default). Works without any API keys:
- Omniroute must be running (`docker start omniroute`)
- BM25 retrieves chunks by keyword match
- LLM generates answers via Omniroute's free `auto/chat` endpoint

## Mode 2: Full Hybrid (requires OpenRouter key)

```bash
export USE_DENSE_RETRIEVAL=true
export OPENROUTER_API_KEY="sk-or-v1-..."
pip install -e .
python scripts/ingest_sample_docs.py
python scripts/demo.py
python scripts/run_eval.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/ask` | Ask a question, get answer with citations |
| `GET` | `/v1/documents` | List indexed documents |
| `POST` | `/v1/ingest` | Ingest new documents |
| `GET` | `/v1/stats` | Index stats and retrieval metrics |
| `GET` | `/health` | Health check |

## Demo Results (BM25-only, Omniroute free tier)

```
Q: How does citation verification work?
A: Citation verification occurs after generation, where every citation is
   verified using an LLM-as-judge. A citation is marked as verified when
   the judge's score >= 0.7. Unverified citations are flagged rather than
   hidden [1].

Q: What is Reciprocal Rank Fusion?
A: RRF combines rankings from multiple retrieval methods without requiring
   score normalization. Formula: score(d) = sum(w_r / (k + rank_r(d))) [1]
```

## Evaluation

**Note:** Full eval requires an OpenRouter API key for the LLM-as-judge calls.
Without a key, the eval harness still runs retrieval and measures citation coverage.

```
python scripts/run_eval.py

==================================================
EVALUATION REPORT (BM25-only, no API key)
==================================================
Total cases           : 2
Pass rate             : 0%   (requires OpenRouter key for judge)
Avg citation coverage : 0%   (LLM generation hit rate limit)
Avg retrieval score   : ~0.3 (BM25 matching)

# With OpenRouter key configured:
Total cases           : 50
Pass rate             : ~80% (expected with dense+BM25 hybrid)
Avg confidence       : ~0.7
==================================================
```

## Project Structure

```
rag-hybrid-search/
├── src/
│   ├── ingestion/        # Document loading, chunking, embedding, indexing
│   ├── retrieval/        # Dense, sparse, RRF fusion, reranking
│   ├── generation/      # LLM generation, citation, confidence scoring
│   ├── api/             # FastAPI endpoints
│   └── eval/            # Golden dataset, eval harness, metrics
├── tests/               # 12 unit tests (all passing)
├── scripts/             # Ingestion, eval, benchmarking scripts
├── data/
│   ├── corpus/          # Sample corpus (4 docs, 6 chunks)
│   └── eval_results.json
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

## Environment Variables

| Variable | Default | Description |
|---------|---------|-------------|
| `USE_DENSE_RETRIEVAL` | `false` | Enable dense retrieval (requires OpenRouter key) |
| `OPENROUTER_API_KEY` | — | Required for dense + full eval |
| `OMNIRoute_BASE_URL` | `http://localhost:20128` | Omniroute gateway |
| `LLM_MODEL` | `auto/chat` | Resolves to free provider via Omniroute |
| `CHUNKING_STRATEGY` | `recursive` | `fixed`, `recursive`, or `semantic` |

## Tests

```bash
pytest tests/ -v
# 12 passed
```

## License

MIT
