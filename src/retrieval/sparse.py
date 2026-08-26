"""Sparse (BM25) retrieval using rank_bm25."""

from __future__ import annotations
import re

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from ..config import BM25_INDEX_PATH, SPARSE_TOP_K


@dataclass
class BM25Result:
    chunk_id: str
    text: str
    source: str = ""
    section_heading: str = ""
    score: float = 0.0


class BM25Index:
    """Build and query a BM25 index over text chunks."""

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._chunks: list[str] = []
        self._metadata: list[dict] = []

    def build(self, chunks: list[str], metadata: list[dict] | None = None) -> None:
        """Build BM25 index from a list of text chunks."""
        def tokenize(text: str) -> list[str]:
            text = text.lower()
            text = re.sub(r"[!?.,;:\"'()\[\]{}]", " ", text)
            return [t for t in text.split() if t]
        tokenized = [tokenize(c) for c in chunks]
        self._index = BM25Okapi(tokenized)
        self._chunks = chunks
        self._metadata = metadata or [{} for _ in chunks]

    def query(self, query: str, top_k: int = SPARSE_TOP_K) -> list[BM25Result]:
        """Query the BM25 index and return top-k results."""
        if self._index is None:
            return []
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            BM25Result(
                chunk_id=f"chunk_{i:06d}",
                text=self._chunks[i],
                source=self._metadata[i].get("source", ""),
                section_heading=self._metadata[i].get("section_heading", ""),
                score=float(scores[i]),
            )
            for i in top_indices
            if scores[i] > 0
        ]

    def save(self, path: str | Path) -> None:
        """Serialize the BM25 index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # rank_bm25 doesn't have a native save; pickle the indexer + metadata
        import pickle
        data = {
            "chunks": self._chunks,
            "metadata": self._metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[INFO] BM25 index saved to {path} ({len(self._chunks)} chunks)")

    def load(self, path: str | Path) -> None:
        """Load a BM25 index from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._metadata = data["metadata"]
        tokenized = [c.lower().split() for c in self._chunks]
        self._index = BM25Okapi(tokenized)
        print(f"[INFO] BM25 index loaded from {path} ({len(self._chunks)} chunks)")


class SparseRetriever:
    """Sparse retrieval using BM25 keyword matching.

    Automatically loads the persisted BM25 index from disk if available.
    """

    def __init__(self, top_k: int = SPARSE_TOP_K, index_path: str | None = None):
        self.top_k = top_k
        self._index: BM25Index | None = None
        index_path = index_path or BM25_INDEX_PATH
        if Path(index_path).exists():
            try:
                self._index = BM25Index()
                self._index.load(index_path)
                print(f"[INFO] BM25 index loaded from {index_path}")
            except Exception as exc:
                print(f"[WARN] Could not load BM25 index from {index_path}: {exc}")

    @property
    def index(self) -> BM25Index | None:
        return self._index

    @index.setter
    def index(self, value: BM25Index) -> None:
        self._index = value

    def retrieve(self, query: str, top_k: int | None = None) -> list[BM25Result]:
        if self._index is None:
            print("[WARN] BM25 index not loaded — run IngestionPipeline first")
            return []
        return self._index.query(query, top_k or self.top_k)
