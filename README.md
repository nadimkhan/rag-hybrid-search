# RAG Hybrid Search

**Ask questions about your own documents. Get answers with citations.**

A search engine that reads your files, understands them, and answers your questions in plain English — with links to exactly which document each fact came from.

---

## What is this for?

You have a folder of documents. Contracts, manuals, codebases, research papers, notes. You know the information is in there but searching it means opening files and reading through them.

This turns that folder into a searchable Q&A system.

**Real examples:**

- **Job search** — ingest your resume + job listings. Ask "which of my projects match a Senior AI Engineer role?"
- **Codebase Q&A** — ingest your GitHub repos. Ask "how does the transcription pipeline work?"
- **Contract review** — ingest an NDA. Ask "who owns the IP after termination?"
- **Research** — ingest papers and notes. Ask "what did we decide about the chunking strategy?"

Traditional keyword search finds files that *might* contain your answer. RAG finds the specific passage, reads it, and gives you a direct answer — with a citation so you can open the source and verify it.

---

## How it works

```
You ask: "How does citation verification work?"

                    ┌──────────────────────────────┐
                    │  1. BM25 retrieves chunks    │
Your question ─────▶│     matching keywords        │
                    │  2. LLM reads the chunks   │
                    │  3. Answer is generated     │
                    │  4. Every [N] citation is   │
                    │     verified against source │
                    └──────────────────────────────┘

Answer: "Citation verification occurs after generation,
where every citation is verified using an LLM-as-judge.
A citation is marked verified when score >= 0.7 [1]."
```

**Why hybrid retrieval?** Vector search finds semantically similar text. BM25 finds exact keyword matches. Technical terms like `getUserById`, `config.yaml`, and error codes get missed by vectors but caught by BM25. This uses both.

---

## Installation

### Prerequisites

- Python 3.11+
- Docker (for Omniroute, the free LLM gateway)
- 2GB RAM minimum

### Step 1 — Clone and install

```bash
git clone https://github.com/nadimkhan/rag-hybrid-search.git
cd rag-hybrid-search
pip install -e .
```

### Step 2 — Start Omniroute (free LLM, no API key needed)

```bash
docker start omniroute
```

If you don't have Omniroute, install it first:
```bash
docker pull ghcr.io/oscontext/omniroute:latest
docker run -d --name omniroute -p 20128:20128 ghcr.io/oscontext/omniroute:latest
```

### Step 3 — Ingest your documents

```bash
# Ingest the sample corpus
python scripts/ingest_sample_docs.py

# Or ingest your own documents
python scripts/ingest_sample_docs.py --path /path/to/your/docs
```

The ingest step reads all `.txt`, `.md`, `.pdf` files in the folder, splits them into chunks, and builds a searchable index.

### Step 4 — Ask questions

```bash
python scripts/demo.py
```

Example output:
```
Q: How does citation verification work?
A: Citation verification occurs after generation, where every
   citation is verified using an LLM-as-judge. A citation is
   marked as verified when the judge's score >= 0.7 [1].

Q: What is Reciprocal Rank Fusion?
A: RRF combines rankings from multiple retrieval methods without
   requiring score normalization. Formula: score(d) = sum(w_r / (k + rank_r(d))) [1]
```

---

## Running the API server

For programmatic access or building a frontend on top:

```bash
make run
# or
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Then query it:

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What retrieval methods does the system use?"}'
```

Response:
```json
{
  "answer": "The system uses dense retrieval via ChromaDB and sparse retrieval via BM25...",
  "confidence": 0.75,
  "confidence_level": "medium",
  "citations_verified": true,
  "chunks_retrieved": 3
}
```

---

## Two modes

### Zero-config mode (runs today, no API key)

Uses BM25 for retrieval + Omniroute free tier for generation. Everything works without an API key.

```bash
# Default — no setup needed
python scripts/demo.py
```

### Full hybrid mode (requires OpenRouter API key)

Adds dense vector retrieval for better semantic matching. Sign up at [openrouter.ai](https://openrouter.ai) (free tier available).

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export USE_DENSE_RETRIEVAL=true
python scripts/ingest_sample_docs.py
python scripts/demo.py
```

| Feature | Zero-config (BM25) | Full hybrid (BM25 + vectors) |
|---------|---------------------|-------------------------------|
| Keyword matching | Yes | Yes |
| Semantic similarity | No | Yes |
| Exact technical terms | Excellent | Excellent |
| API key required | No | Yes (free tier works) |

---

## Project structure

```
rag-hybrid-search/
├── src/
│   ├── ingestion/       # Load docs → chunk → embed → index
│   ├── retrieval/       # Dense (ChromaDB) + sparse (BM25) + RRF fusion
│   ├── generation/      # LLM answer + citation parsing + confidence score
│   ├── api/            # FastAPI endpoints
│   └── eval/           # Golden Q&A dataset + automated metrics
├── scripts/
│   ├── ingest_sample_docs.py   # Ingest documents
│   ├── demo.py                # Run interactive demo
│   └── run_eval.py            # Run evaluation harness
├── data/
│   ├── corpus/         # Sample documents
│   └── bm25_index.json # Built automatically after ingest
└── tests/              # Unit tests
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `USE_DENSE_RETRIEVAL` | `false` | Enable dense vector retrieval |
| `OPENROUTER_API_KEY` | — | For dense retrieval + full eval |
| `OMNIRoute_BASE_URL` | `http://localhost:20128` | Omniroute gateway |
| `LLM_MODEL` | `auto/chat` | LLM model (free via Omniroute) |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `DENSE_WEIGHT` | `0.7` | RRF weight for dense results |
| `SPARSE_WEIGHT` | `0.3` | RRF weight for BM25 results |

---

## Evaluation

Run the automated test suite against your corpus:

```bash
python scripts/run_eval.py
```

The harness measures:
- **Faithfulness** — are claims actually supported by the source?
- **Citation accuracy** — do citations point to the right document?
- **Retrieval relevance** — were the right chunks retrieved?
- **Confidence calibration** — does the confidence score match reality?

---

## Tests

```bash
pytest tests/ -v
# 12 passed
```

---

## License

MIT
