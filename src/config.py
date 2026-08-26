"""Configuration for the RAG pipeline.

Environment variables:
    OPENROUTER_API_KEY    — OpenRouter API key (for direct API calls)
    OMNIRoute_BASE_URL  — Omniroute base URL (default: http://localhost:20128)
    OMNIRoute_API_KEY   — Omniroute API key (if using Omniroute gateway)
    EMBEDDING_MODEL      — Embedding model name (default: text-embedding-3-small)
    LLM_MODEL            — LLM model for generation
                              (default: auto/chat — resolves to a free provider via Omniroute)
    USE_DENSE_RETRIEVAL — Set to "true" to enable dense retrieval (requires embeddings)
    CHROMA_DB_PATH      — Path to ChromaDB persistent storage
    BM25_INDEX_PATH     — Path to BM25 serialized index
    LOG_LEVEL           — Logging level (default: INFO)
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH",  str(DATA_DIR / "chromadb"))
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", str(DATA_DIR / "bm25_index.json"))

# ── API Gateways ────────────────────────────────────────────────────────────────
# Omniroute (local gateway with free providers — auto/chat works on port 20128)
OMNIRoute_BASE_URL = os.getenv("OMNIRoute_BASE_URL", "http://localhost:20128")
OMNIRoute_API_KEY  = os.getenv("OMNIRoute_API_KEY", "")

# Direct OpenRouter (set OPENROUTER_API_KEY if you have one)
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Use Omniroute if it's running, otherwise fall back to direct OpenRouter
def _resolve_base_url() -> str:
    import subprocess
    r = subprocess.run(
        ["curl", "-s", "-m", "2", "-o", "/dev/null", "-w", "%{http_code}",
         f"{OMNIRoute_BASE_URL}/v1/models"],
        capture_output=True, text=True
    )
    if r.stdout.strip() == "200":
        return OMNIRoute_BASE_URL
    return OPENROUTER_BASE_URL

def _resolve_api_key() -> str:
    if OMNIRoute_API_KEY:
        return OMNIRoute_API_KEY
    return OPENROUTER_API_KEY

# ── Embeddings ────────────────────────────────────────────────────────────────
# NOTE: Omniroute currently has no free embedding provider.
# If USE_DENSE_RETRIEVAL=true, you must provide OPENROUTER_API_KEY for direct
# OpenRouter embeddings, or configure an embeddings provider in Omniroute.
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL",     "text-embedding-3-small")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
USE_DENSE_RETRIEVAL  = os.getenv("USE_DENSE_RETRIEVAL", "false").lower() == "true"

# ── LLM ──────────────────────────────────────────────────────────────────────
# Default: auto/chat via Omniroute resolves to hy3-free (zero cost)
LLM_MODEL   = os.getenv("LLM_MODEL", "auto/chat")
LLM_BASE_URL = _resolve_base_url()
LLM_API_KEY  = _resolve_api_key()

# ── Retrieval ─────────────────────────────────────────────────────────────────
DENSE_TOP_K   = int(os.getenv("DENSE_TOP_K",   "10"))
SPARSE_TOP_K  = int(os.getenv("SPARSE_TOP_K",  "10"))
RERANK_TOP_K  = int(os.getenv("RERANK_TOP_K",  "5"))
RRF_K         = int(os.getenv("RRF_K",         "60"))
DENSE_WEIGHT  = float(os.getenv("DENSE_WEIGHT",  "0.7"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.3"))
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.95"))

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE",       "800"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP",    "150"))
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive")

# ── Generation ────────────────────────────────────────────────────────────────
MAX_TOKENS               = int(os.getenv("MAX_TOKENS",               "1024"))
TEMPERATURE              = float(os.getenv("TEMPERATURE",              "0.0"))
CITATION_VERIFY_THRESHOLD = float(os.getenv("CITATION_VERIFY_THRESHOLD", "0.7"))
CONFIDENCE_THRESHOLD_HIGH   = float(os.getenv("CONFIDENCE_THRESHOLD_HIGH",   "0.85"))
CONFIDENCE_THRESHOLD_MEDIUM = float(os.getenv("CONFIDENCE_THRESHOLD_MEDIUM",  "0.60"))

# ── Logging / Server ──────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
API_HOST  = os.getenv("API_HOST",  "0.0.0.0")
API_PORT  = int(os.getenv("API_PORT",  "8000"))
