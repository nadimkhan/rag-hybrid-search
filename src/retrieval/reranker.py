"""Cross-encoder reranking: precision boost after RRF fusion via LLM-as-judge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import RERANK_TOP_K, LLM_MODEL, LLM_BASE_URL, LLM_API_KEY


@dataclass
class RerankedResult:
    """A result after cross-encoder reranking."""

    chunk_id: str
    text: str
    source: str
    section_heading: str = ""
    page_number: int | None = None
    fusion_score: float = 0.0
    rerank_score: float = 0.0


class CrossEncoderReranker:
    """Rerank fused results using an LLM-as-judge.

    Sends query + each candidate chunk to the LLM for relevance scoring.
    Keeps top_k after reranking (default k=5).
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        top_k: int = RERANK_TOP_K,
        batch_size: int = 5,
    ):
        self.model = model
        self.top_k = top_k
        self.batch_size = batch_size
        self.base_url = LLM_BASE_URL.rstrip("/")
        self.api_key = LLM_API_KEY or ""

    def rerank(
        self,
        query: str,
        candidates: list,
    ) -> list[RerankedResult]:
        """Rerank candidate chunks by relevance to the query using an LLM judge."""
        if not candidates:
            return []

        # LLM-as-judge scoring in batches
        scores: dict[str, float] = {}
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i : i + self.batch_size]
            batch_scores = self._score_batch(query, batch)
            scores.update(batch_scores)

        # Sort by rerank score descending, keep top_k
        sorted_candidates = sorted(
            [
                RerankedResult(
                    chunk_id=getattr(c, "chunk_id", f"chunk_{i}"),
                    text=getattr(c, "text", ""),
                    source=getattr(c, "source", ""),
                    section_heading=getattr(c, "section_heading", ""),
                    page_number=getattr(c, "page_number", None),
                    fusion_score=getattr(c, "rrf_score", 0.0),
                    rerank_score=scores.get(getattr(c, "chunk_id", f"chunk_{i}"), 0.0),
                )
                for i, c in enumerate(candidates)
            ],
            key=lambda x: x.rerank_score,
            reverse=True,
        )
        return sorted_candidates[: self.top_k]

    def _score_batch(self, query: str, batch: list) -> dict[str, float]:
        """Score a batch of chunks for relevance to the query.

        When LLM is unavailable (no API key or connection fails),
        falls back to using the fusion RRF score as the rerank score.
        """
        # Build prompt
        prompt = (
            f"Query: {query}\n\n"
            "Rate the relevance of each chunk below from 0.0 (irrelevant) to 1.0 (highly relevant). "
            "Respond ONLY with a JSON object mapping chunk number to score: "
            '{"0": 0.9, "1": 0.3}'
        )
        chunk_texts = "\n".join(
            f"[{i}] {getattr(c, 'text', '')[:300]}"
            for i, c in enumerate(batch)
        )
        full_prompt = f"{prompt}\n\n{chunk_texts}"

        # Skip LLM call if no API key configured
        if not self.api_key:
            return {
                str(i): getattr(c, "rrf_score", getattr(c, "fusion_score", 0.0))
                for i, c in enumerate(batch)
            }

        try:
            import httpx
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": full_prompt}],
                        "temperature": 0.0,
                        "max_tokens": 200,
                        "stream": False,
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            import json, re
            text = re.sub(r"```json\s*|\s*```", "", text).strip()
            scores = json.loads(text)
            return {str(k): float(v) for k, v in scores.items()}
        except Exception:
            # Fall back to fusion score on any failure
            return {
                str(i): getattr(c, "rrf_score", getattr(c, "fusion_score", 0.0))
                for i, c in enumerate(batch)
            }
