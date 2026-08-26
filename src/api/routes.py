"""FastAPI routes for the RAG API."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Request/Response models ──────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    collection: str = "chunks"
    include_citations: bool = True


class CitationModel(BaseModel):
    marker: str
    chunk_id: str
    text: str
    verified: bool
    verification_score: float


class AnswerResponse(BaseModel):
    answer: str
    question: str
    confidence: float
    confidence_level: str
    citation_coverage: float
    citations_verified: bool
    citations: list[CitationModel]
    latency_ms: float
    chunks_retrieved: int


class IngestRequest(BaseModel):
    collection: str = "chunks"
    glob_pattern: str = "*"


class StatsResponse(BaseModel):
    collection: str
    total_chunks: int
    total_documents: int


# ── API Factory ──────────────────────────────────────────────────────────────

def create_app(pipeline) -> FastAPI:
    """Build the FastAPI app with the RAG pipeline injected."""

    app = FastAPI(
        title="RAG Hybrid Search API",
        description="Production-grade RAG with hybrid dense + BM25 retrieval, reranking, and citation verification.",
        version="0.1.0",
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/ask", response_model=AnswerResponse)
    async def ask(req: AskRequest):
        start = time.monotonic()
        try:
            result = pipeline.ask(
                question=req.question,
                collection=req.collection,
                include_citations=req.include_citations,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        latency_ms = (time.monotonic() - start) * 1000

        return AnswerResponse(
            answer=result.get("answer", ""),
            question=result.get("question", req.question),
            confidence=result.get("confidence", 0.0),
            confidence_level=result.get("confidence_level", "low"),
            citation_coverage=result.get("citation_coverage", 0.0),
            citations_verified=result.get("citations_verified", False),
            citations=[
                CitationModel(
                    marker=c.marker,
                    chunk_id=c.chunk_id,
                    text=c.text[:100],
                    verified=c.verified,
                    verification_score=c.verification_score,
                )
                for c in result.get("citations", [])
            ],
            latency_ms=round(latency_ms, 2),
            chunks_retrieved=len(result.get("chunks", [])),
        )

    @app.post("/v1/ingest")
    async def ingest(
        directory: str = Form(...),
        collection: str = Form("chunks"),
        glob_pattern: str = Form("*"),
    ):
        try:
            stats = pipeline.ingest_directory(directory, collection, glob_pattern)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return {"status": "ok", **stats}

    @app.get("/v1/documents", response_model=list[dict])
    async def list_documents(collection: str = "chunks"):
        try:
            docs = pipeline.list_documents(collection)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return docs

    @app.get("/v1/stats", response_model=StatsResponse)
    async def stats(collection: str = "chunks"):
        try:
            s = pipeline.get_stats(collection)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return StatsResponse(**s)

    return app
