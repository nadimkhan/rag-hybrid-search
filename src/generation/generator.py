"""LLM generation with grounded answers, inline citations, and confidence scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import (
    LLM_MODEL, MAX_TOKENS, TEMPERATURE, CITATION_VERIFY_THRESHOLD,
    LLM_BASE_URL, OPENROUTER_API_KEY,
    CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_MEDIUM,
)


@dataclass
class Citation:
    """A single inline citation from the answer."""

    marker: str
    chunk_id: str
    text: str
    verified: bool = False
    verification_score: float = 0.0


@dataclass
class Answer:
    """A generated answer with citations and confidence."""

    text: str
    question: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    confidence_level: str = "low"
    retrieval_confidence: float = 0.0
    citation_coverage: float = 0.0
    answered: bool = True
    unanswerable: bool = False
    raw_response: Any = None

    @property
    def has_verified_citations(self) -> bool:
        return any(c.verified for c in self.citations)


class Generator:
    """Grounded LLM generation via OpenRouter with inline citations and confidence scoring.

    Environment:
        LLM_MODEL       — model name (default: google/gemini-2.0-flash-thinking-exp-01-21)
        LLM_BASE_URL   — base URL (default: https://openrouter.ai/api/v1)
        MAX_TOKENS     — max response tokens (default: 1024)
        TEMPERATURE    — sampling temperature (default: 0.0)
        CITATION_VERIFY_THRESHOLD — min score to mark citation verified (default: 0.7)
        OPENROUTER_API_KEY — your OpenRouter API key
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        verify_threshold: float = CITATION_VERIFY_THRESHOLD,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verify_threshold = verify_threshold
        self.base_url = LLM_BASE_URL.rstrip("/")
        self.api_key = OPENROUTER_API_KEY or ""

    def generate(
        self,
        question: str,
        retrieved_chunks: list,
        include_citations: bool = True,
    ) -> Answer:
        """Generate a grounded answer from retrieved chunks with inline citations."""

        if not retrieved_chunks:
            return Answer(
                text="I don't have enough context to answer this question based on the available documents.",
                question=question,
                answered=False,
                unanswerable=True,
                confidence=0.0,
                confidence_level="low",
            )

        # Build numbered context blocks
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_blocks.append(
                f"[{i}] Source: {getattr(chunk, 'source', '')}\n{getattr(chunk, 'text', '')[:500]}"
            )
        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a precise technical assistant. Answer the user's question using ONLY the "
            "provided context blocks. Each block is cited with a number in brackets like [1]. "
            "Inline citations are REQUIRED — after every factual claim, include [N] where N is the "
            "number of the source block that supports it. "
            "If the context does not contain enough information to answer, say so explicitly. "
            "Do NOT hallucinate. Do NOT cite sources that don't support the claim."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"

        try:
            response_text = self._call_llm(system_prompt, user_prompt)
        except Exception as exc:
            return Answer(
                text=f"Generation failed: {exc}",
                question=question,
                confidence=0.0,
                confidence_level="low",
            )

        # Parse citations
        citations = self._parse_citations(response_text, retrieved_chunks)

        # Verify citations
        if include_citations and citations:
            citations = self._verify_citations(question, response_text, citations, retrieved_chunks)

        # Score confidence
        confidence, retrieval_conf, citation_cov = self._score_confidence(
            retrieved_chunks, citations
        )
        confidence_level = (
            "high" if confidence >= CONFIDENCE_THRESHOLD_HIGH
            else "medium" if confidence >= CONFIDENCE_THRESHOLD_MEDIUM
            else "low"
        )

        return Answer(
            text=response_text,
            question=question,
            citations=citations,
            confidence=confidence,
            confidence_level=confidence_level,
            retrieval_confidence=retrieval_conf,
            citation_coverage=citation_cov,
            answered=True,
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make a chat completion call via OpenRouter's OpenAI-compatible API."""
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
                "stream": False,
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]

    def _parse_citations(self, text: str, chunks: list) -> list[Citation]:
        """Extract bracketed citation markers and map to chunks."""
        import re
        citation_map: dict[str, int] = {}
        for i, chunk in enumerate(chunks, 1):
            marker = f"[{i}]"
            if marker in text:
                citation_map[marker] = i - 1

        citations = []
        for marker, idx in citation_map.items():
            citations.append(Citation(
                marker=marker,
                chunk_id=getattr(chunks[idx], "chunk_id", f"chunk_{idx}"),
                text=getattr(chunks[idx], "text", "")[:200],
            ))
        return citations

    def _verify_citations(
        self,
        question: str,
        answer: str,
        citations: list[Citation],
        chunks: list,
    ) -> list[Citation]:
        """Verify each citation using LLM-as-judge via OpenRouter."""

        for cit in citations:
            chunk = next(
                (c for c in chunks if getattr(c, "chunk_id", "") == cit.chunk_id),
                None,
            )
            if not chunk:
                continue

            verify_prompt = (
                f"Question: {question}\n\n"
                f"Answer: {answer}\n\n"
                f"Cited source [{cit.marker}]:\n{cit.text}\n\n"
                "Does this source actually support a factual claim in the answer above? "
                "Rate your confidence that this citation is valid on a scale of 0.0 to 1.0. "
                "Respond with only the numeric score."
            )

            try:
                score_text = self._call_llm("", verify_prompt).strip()
                score = float(score_text.split()[0])
            except Exception:
                score = 0.0

            cit.verification_score = score
            cit.verified = score >= self.verify_threshold

        return citations

    def _score_confidence(
        self,
        chunks: list,
        citations: list[Citation],
    ) -> tuple[float, float, float]:
        """Compute composite confidence score."""
        if chunks:
            scores = [getattr(c, "rerank_score", getattr(c, "score", 0.0)) for c in chunks]
            retrieval_conf = sum(s for s in scores if s) / len(scores)
        else:
            retrieval_conf = 0.0

        if citations:
            verified = sum(1 for c in citations if c.verified)
            citation_cov = verified / len(citations)
        else:
            citation_cov = 0.0

        confidence = 0.5 * retrieval_conf + 0.5 * citation_cov
        return confidence, retrieval_conf, citation_cov
