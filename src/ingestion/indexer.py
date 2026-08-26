"""Full ingestion pipeline: load → chunk → embed → index (ChromaDB + BM25)."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np

from .loader import Document, DocumentLoader
from .chunker import Chunk, ChunkPipeline
from .embedder import ChunkEmbedder, cosine_similarity
from ..config import (
    CHROMA_DB_PATH, BM25_INDEX_PATH, DEDUP_SIMILARITY_THRESHOLD,
)
from ..retrieval.sparse import BM25Index


@dataclass
class IndexedChunk:
    """A chunk stored in ChromaDB."""

    chunk: Chunk
    embedding: np.ndarray
    chunk_id: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.chunk.text,
            "source": self.chunk.source,
            "section_heading": self.chunk.section_heading,
            "page_number": self.chunk.page_number,
            "char_count": self.chunk.char_count,
            "chunking_strategy": self.chunk.chunking_strategy,
            "metadata": {
                "source": self.chunk.source,
                "section_heading": self.chunk.section_heading,
                "page_number": self.chunk.page_number,
                "char_count": self.chunk.char_count,
                "chunking_strategy": self.chunk.chunking_strategy,
            },
        }


class IngestionPipeline:
    """Full ingestion pipeline: load → chunk → embed → index."""

    def __init__(
        self,
        embedder: ChunkEmbedder | None = None,
        chunker: ChunkPipeline | None = None,
    ):
        self.embedder = embedder or ChunkEmbedder()
        self.chunker = chunker or ChunkPipeline()
        self._chroma_client: chromadb.PersistentClient | None = None
        self._bm25_index: BM25Index | None = None

    @property
    def chroma_client(self) -> chromadb.PersistentClient:
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(
                path=str(Path(CHROMA_DB_PATH).resolve()),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._chroma_client

    def get_collection(self, name: str = "chunks"):
        return self.chroma_client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_documents(
        self,
        documents: list[Document],
        collection_name: str = "chunks",
        skip_dedup: bool = False,
    ) -> dict[str, Any]:
        """Ingest documents and return stats.

        When dense embeddings are unavailable (USE_DENSE_RETRIEVAL=false),
        only the BM25 index is built and ChromaDB is skipped.
        """
        chunks = self.chunker.chunk_documents(documents)
        if not skip_dedup and len(chunks) > 1:
            chunks = self._deduplicate(chunks)

        if not chunks:
            return {"chunks_indexed": 0, "docs_processed": len(documents)}

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)

        # Check if embeddings are available (dense retrieval enabled + provider working)
        has_embeddings = any(e is not None for e in embeddings)

        if has_embeddings:
            indexed = [
                IndexedChunk(chunk=c, embedding=e, chunk_id=f"chunk_{i:06d}")
                for i, (c, e) in enumerate(zip(chunks, embeddings))
                if e is not None
            ]
            coll = self.get_collection(collection_name)
            coll.add(
                ids=[ic.chunk_id for ic in indexed],
                embeddings=[ic.embedding.tolist() for ic in indexed],
                documents=[ic.chunk.text for ic in indexed],
                metadatas=[ic.to_dict()["metadata"] for ic in indexed],
            )
            chunks_indexed = len(indexed)
        else:
            # Dense retrieval unavailable — store metadata only in ChromaDB (no vectors)
            # BM25 will handle retrieval; ChromaDB just tracks documents
            coll = self.get_collection(collection_name)
            ids = [f"chunk_{i:06d}" for i in range(len(chunks))]
            coll.add(
                ids=ids,
                documents=[c.text for c in chunks],
                metadatas=[
                    {
                        "source": str(c.source) if c.source else "",
                        "section_heading": str(c.section_heading) if c.section_heading else "",
                        "page_number": int(c.page_number) if c.page_number is not None else 0,
                        "char_count": int(c.char_count) if c.char_count else 0,
                        "chunking_strategy": str(c.chunking_strategy) if c.chunking_strategy else "",
                    }
                    for c in chunks
                ],
            )
            chunks_indexed = len(chunks)
            print("[WARN] Dense embeddings unavailable — using BM25-only retrieval mode")

        # Always build BM25 index
        self._bm25_index = BM25Index()
        self._bm25_index.build(texts, metadata=[{"text": c.text, "section_heading": getattr(c, "section_heading", "")} for c in chunks])
        Path(BM25_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        self._bm25_index.save(BM25_INDEX_PATH)

        return {"chunks_indexed": chunks_indexed, "docs_processed": len(documents)}

    def ingest_directory(
        self,
        root: str | Path,
        collection_name: str = "chunks",
        glob_pattern: str = "*",
    ) -> dict[str, Any]:
        docs = DocumentLoader().load_dir(root, glob_pattern)
        return self.ingest_documents(docs, collection_name)

    def _deduplicate(self, chunks: list[Chunk]) -> list[Chunk]:
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        keep: list[Chunk] = []
        for chunk, emb in zip(chunks, embeddings):
            is_dup = any(
                cosine_similarity(emb, self.embedder.embed_one(k.text)) > DEDUP_SIMILARITY_THRESHOLD
                for k in keep
            )
            if not is_dup:
                keep.append(chunk)
        dropped = len(chunks) - len(keep)
        if dropped:
            print(f"[INFO] Dedup: dropped {dropped}/{len(chunks)} near-duplicate chunks")
        return keep

    @property
    def bm25_index(self) -> BM25Index | None:
        if self._bm25_index is None and Path(BM25_INDEX_PATH).exists():
            bm = BM25Index()
            bm.load(BM25_INDEX_PATH)
            self._bm25_index = bm
        return self._bm25_index
