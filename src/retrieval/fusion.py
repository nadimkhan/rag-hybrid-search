"""Reciprocal Rank Fusion (RRF) for combining dense + sparse retrieval results."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import RRF_K, DENSE_WEIGHT, SPARSE_WEIGHT


@dataclass
class FusionResult:
    """A result from RRF fusion."""

    chunk_id: str
    text: str
    source: str
    section_heading: str = ""
    page_number: int | None = None
    dense_rank: int = 9999
    sparse_rank: int = 9999
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0

    @property
    def is_boosted_by_sparse(self) -> bool:
        return self.sparse_rank < self.dense_rank


def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    k: int = RRF_K,
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
) -> list[FusionResult]:
    """Fuse dense and sparse rankings using weighted RRF.

    Weighted RRF: score = sum(w / (k + rank))
    """

    # Index dense/sparse results by chunk_id for O(1) lookup
    dense_by_id = {getattr(r, "chunk_id", str(id(r))): r for r in dense_results}
    sparse_by_id = {getattr(r, "chunk_id", str(id(r))): r for r in sparse_results}

    # Union of all chunk IDs, preserving order (deduped)
    seen: dict = {}
    for r in dense_results + sparse_results:
        cid = getattr(r, "chunk_id", str(id(r)))
        if cid not in seen:
            seen[cid] = True

    all_ids = list(seen.keys())

    # Compute RRF scores
    scored: dict[str, float] = {}
    for cid in all_ids:
        d_rank = next((i for i, r in enumerate(dense_results)
                       if getattr(r, "chunk_id", str(id(r))) == cid), None)
        s_rank = next((i for i, r in enumerate(sparse_results)
                       if getattr(r, "chunk_id", str(id(r))) == cid), None)

        d_score = getattr(dense_by_id.get(cid), "score", 0.0) if cid in dense_by_id else 0.0
        s_score = getattr(sparse_by_id.get(cid), "score", 0.0) if cid in sparse_by_id else 0.0

        rrf = 0.0
        if d_rank is not None:
            rrf += dense_weight * (1.0 / (k + d_rank + 1))
        if s_rank is not None:
            rrf += sparse_weight * (1.0 / (k + s_rank + 1))

        scored[cid] = rrf

    # Sort descending
    ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)

    results: list[FusionResult] = []
    for rrf_score, (cid, _) in enumerate(ranked):
        d = dense_by_id.get(cid)
        s = sparse_by_id.get(cid)

        d_rank = next((i for i, r in enumerate(dense_results)
                       if getattr(r, "chunk_id", str(id(r))) == cid), None)
        s_rank = next((i for i, r in enumerate(sparse_results)
                       if getattr(r, "chunk_id", str(id(r))) == cid), None)

        results.append(FusionResult(
            chunk_id=cid,
            text=getattr(d, "text", "") or getattr(s, "text", ""),
            source=getattr(d, "source", "") or getattr(s, "source", ""),
            section_heading=getattr(d, "section_heading", "") or getattr(s, "section_heading", ""),
            page_number=getattr(d, "page_number", None) or getattr(s, "page_number", None),
            dense_rank=d_rank if d_rank is not None else 9999,
            sparse_rank=s_rank if s_rank is not None else 9999,
            dense_score=d_score,
            sparse_score=s_score,
            rrf_score=rrf_score,
        ))

    return results
