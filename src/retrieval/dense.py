"""Dense (vector) retrieval via ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np

from ..config import CHROMA_DB_PATH, DENSE_TOP_K
from ..ingestion.embedder import ChunkEmbedder

if TYPE_CHECKING:
    from ..ingestion.chunker import Chunk


@dataclass
class RetrievedChunk:
    """A chunk retrieved from ChromaDB."""

    chunk_id: str
    text: str
    source: str
    section_heading: str = ""
    page_number: int | None = None
    score: float = 0.0   # cosine similarity
    rerank_score: float | None = None

    @classmethod
    def from_result(cls, result: dict, idx: int) -> "RetrievedChunk":
        meta = result["metadatas"][idx]
        dist = result.get("distances", [0.0])[idx]
        score = 1.0 - dist if dist is not None else 0.0
        return cls(
            chunk_id=result["ids"][idx],
            text=result["documents"][idx],
            source=meta.get("source", ""),
            section_heading=meta.get("section_heading", ""),
            page_number=meta.get("page_number"),
            score=score,
        )


class DenseRetriever:
    """Dense retrieval via ChromaDB cosine similarity search."""

    def __init__(
        self,
        collection_name: str = "chunks",
        embedder: ChunkEmbedder | None = None,
        top_k: int = DENSE_TOP_K,
    ):
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedder = embedder or ChunkEmbedder()
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.top_k
        query_emb = self.embedder.embed_one(query)
        coll = self._client.get_collection(self.collection_name)
        results = coll.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [RetrievedChunk.from_result(results, i) for i in range(len(results["ids"][0]))]
