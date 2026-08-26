"""Full RAG pipeline: retrieval → fusion → rerank → generate → cite."""

from __future__ import annotations

from typing import Any

from .ingestion.loader import DocumentLoader
from .ingestion.chunker import ChunkPipeline
from .ingestion.embedder import ChunkEmbedder
from .ingestion.indexer import IngestionPipeline
from .retrieval.dense import DenseRetriever
from .retrieval.sparse import SparseRetriever, BM25Index
from .retrieval.fusion import reciprocal_rank_fusion
from .retrieval.reranker import CrossEncoderReranker
from .generation.generator import Generator
from .config import (
    USE_DENSE_RETRIEVAL,
    DENSE_TOP_K, SPARSE_TOP_K, RERANK_TOP_K,
    RRF_K, DENSE_WEIGHT, SPARSE_WEIGHT,
)


class RAGPipeline:
    """Full RAG pipeline.

    Supports BM25-only mode (default — no embedding provider needed) and
    dense+BM25 hybrid mode (set USE_DENSE_RETRIEVAL=true + OpenRouter API key).

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest_directory("/path/to/docs")
        result = pipeline.ask("What is this about?")
    """

    def __init__(
        self,
        collection: str = "chunks",
        embedder: ChunkEmbedder | None = None,
    ):
        self.collection = collection
        self.use_dense = USE_DENSE_RETRIEVAL
        self.embedder = embedder or ChunkEmbedder()
        self.ingestion = IngestionPipeline(embedder=self.embedder)
        self.dense = DenseRetriever(collection_name=collection, embedder=self.embedder) if self.use_dense else None
        self.sparse = SparseRetriever()
        self.reranker = CrossEncoderReranker()
        self.generator = Generator()

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_directory(
        self,
        directory: str,
        collection: str | None = None,
        glob_pattern: str = "*",
    ) -> dict[str, Any]:
        collection = collection or self.collection
        return self.ingestion.ingest_directory(directory, collection, glob_pattern)

    def ingest_documents(self, documents: list, collection: str | None = None) -> dict[str, Any]:
        collection = collection or self.collection
        return self.ingestion.ingest_documents(documents, collection)

    # ── Query ───────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        collection: str | None = None,
        include_citations: bool = True,
        dense_k: int = DENSE_TOP_K,
        sparse_k: int = SPARSE_TOP_K,
        rerank_k: int = RERANK_TOP_K,
    ) -> dict[str, Any]:
        """Ask a question and get a grounded answer with citations."""
        collection = collection or self.collection

        # BM25 retrieval (always available)
        sparse_results = self.sparse.retrieve(question, top_k=sparse_k)

        if self.use_dense and self.dense is not None:
            # Hybrid: dense + sparse → RRF → rerank
            dense_results = self.dense.retrieve(question, top_k=dense_k)
            fused = reciprocal_rank_fusion(
                dense_results,
                sparse_results,
                k=RRF_K,
                dense_weight=DENSE_WEIGHT,
                sparse_weight=SPARSE_WEIGHT,
            )
        else:
            # BM25-only: use sparse results as fusion results
            from .retrieval.fusion import FusionResult
            fused = [
                FusionResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    source=r.source,
                    section_heading=r.section_heading,
                    page_number=None,
                    sparse_rank=i,
                    dense_rank=9999,
                    sparse_score=r.score,
                    dense_score=0.0,
                    rrf_score=1.0 / (60 + i + 1),
                )
                for i, r in enumerate(sparse_results)
            ]

        # Cross-encoder reranking
        reranked = self.reranker.rerank(question, fused)

        # Generate
        answer = self.generator.generate(
            question=question,
            retrieved_chunks=reranked,
            include_citations=include_citations,
        )

        citations_verified = any(c.verified for c in answer.citations) if answer.citations else False

        return {
            "answer": answer.text,
            "question": question,
            "confidence": answer.confidence,
            "confidence_level": answer.confidence_level,
            "citation_coverage": answer.citation_coverage,
            "citations_verified": citations_verified,
            "citations": answer.citations,
            "chunks": reranked,
            "unanswerable": answer.unanswerable,
        }

    # ── Stats ─────────────────────────────────────────────────────────────

    def list_documents(self, collection: str | None = None) -> list[dict]:
        collection = collection or self.collection
        try:
            coll = self.ingestion.get_collection(collection)
            all_items = coll.get(include=["metadatas"])
            sources = list(dict.fromkeys(m.get("source", "") for m in all_items["metadatas"]))
            return [{"source": s} for s in sources]
        except Exception:
            return []

    def get_stats(self, collection: str | None = None) -> dict[str, Any]:
        collection = collection or self.collection
        try:
            coll = self.ingestion.get_collection(collection)
            count = coll.count()
            all_items = coll.get(include=["metadatas"])
            sources = list(dict.fromkeys(m.get("source", "") for m in all_items["metadatas"]))
            return {
                "collection": collection,
                "total_chunks": count,
                "total_documents": len(sources),
            }
        except Exception:
            return {"collection": collection, "total_chunks": 0, "total_documents": 0}
