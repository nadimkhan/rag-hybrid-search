"""Embed text chunks via Omniroute (local) or direct OpenRouter.

When USE_DENSE_RETRIEVAL=false (default), the embedder returns None to
disable dense retrieval and fall back to BM25-only mode.
"""

from __future__ import annotations

import os
import numpy as np

from ..config import (
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    LLM_BASE_URL, LLM_API_KEY,
    USE_DENSE_RETRIEVAL, OPENROUTER_API_KEY,
)


class ChunkEmbedder:
    """Embed texts using Omniroute (or direct OpenRouter).

    Requires USE_DENSE_RETRIEVAL=true and a working embeddings provider.
    Falls back to returning None when no provider is available.
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        self.model = model
        self.batch_size = batch_size
        self.base_url = LLM_BASE_URL.rstrip("/")
        self.api_key = LLM_API_KEY or OPENROUTER_API_KEY

        if not USE_DENSE_RETRIEVAL:
            self._available = False
            return

        # Quick availability check
        import httpx
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": ["health check"]},
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                )
            self._available = resp.status_code == 200
        except Exception:
            self._available = False

        if not self._available:
            print("[WARN] Embeddings unavailable — falling back to BM25-only retrieval. "
                  "Set USE_DENSE_RETRIEVAL=true and provide OPENROUTER_API_KEY to enable dense.")

    def embed(self, texts: list[str]) -> list[np.ndarray | None]:
        """Embed a list of texts, or return None list if embeddings unavailable."""
        if not getattr(self, "_available", False):
            return [None] * len(texts)

        import httpx
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json={"model": self.model, "input": batch}, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            results.extend(np.array(d["embedding"]) for d in data["data"])
        return results

    def embed_one(self, text: str) -> np.ndarray | None:
        """Embed a single text, or return None if unavailable."""
        [vec] = self.embed([text])
        return vec


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (norm + 1e-12)) if norm > 0 else 0.0
