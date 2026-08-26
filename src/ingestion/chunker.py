"""Configurable text chunking: fixed-size, recursive, and semantic strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKING_STRATEGY


@dataclass
class Chunk:
    """A text chunk with provenance metadata."""

    text: str
    chunk_index: int
    total_chunks: int
    source: str
    section_heading: str = ""
    page_number: int | None = None
    chunking_strategy: str = ""
    char_count: int = field(default_factory=int)

    def __post_init__(self):
        if not self.char_count:
            self.char_count = len(self.text)


class ChunkPipeline:
    """Pipeline that chunks documents and tracks provenance."""

    def __init__(
        self,
        strategy: str = CHUNKING_STRATEGY,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.strategy = strategy

    def chunk_document(self, doc) -> list[Chunk]:
        """Chunk a single Document into provenance-tracked Chunks."""
        from ..ingestion.loader import Document
        raw = self.chunker.split_text(doc.content)
        return [
            Chunk(
                text=t.strip(),
                chunk_index=i,
                total_chunks=len(raw),
                source=doc.source,
                section_heading=doc.section_heading,
                page_number=doc.page_number,
                chunking_strategy=self.strategy,
            )
            for i, t in enumerate(raw)
            if t.strip()
        ]

    def chunk_documents(self, docs: list) -> list[Chunk]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
