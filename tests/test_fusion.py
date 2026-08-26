"""Tests for RRF fusion."""

import pytest
from src.retrieval.fusion import reciprocal_rank_fusion


class MockDenseResult:
    def __init__(self, chunk_id, score):
        self.chunk_id = chunk_id
        self.score = score
        self.text = f"text_{chunk_id}"
        self.source = f"source_{chunk_id}"
        self.section_heading = ""
        self.page_number = None


class MockSparseResult:
    def __init__(self, chunk_id, score):
        self.chunk_id = chunk_id
        self.score = score
        self.text = f"text_{chunk_id}"
        self.source = f"source_{chunk_id}"
        self.section_heading = ""
        self.page_number = None


def test_fusion_returns_both_sources():
    dense = [
        MockDenseResult("a", 0.9),
        MockDenseResult("b", 0.8),
        MockDenseResult("c", 0.7),
    ]
    sparse = [
        MockSparseResult("b", 1.0),
        MockSparseResult("c", 0.9),
        MockSparseResult("d", 0.8),
    ]
    results = reciprocal_rank_fusion(dense, sparse)
    ids = [r.chunk_id for r in results]
    assert "b" in ids
    assert "c" in ids
    assert results[0].rrf_score <= results[-1].rrf_score  # sorted desc by score
    # 'b' appears in both — should rank high
    b_result = next(r for r in results if r.chunk_id == "b")
    assert b_result.dense_rank == 1  # rank 1 in dense (0-indexed)
    assert b_result.sparse_rank == 0  # rank 0 in sparse


def test_fusion_empty_sparse():
    results = reciprocal_rank_fusion([], [MockSparseResult("x", 1.0)])
    assert len(results) == 1
    assert results[0].chunk_id == "x"


def test_fusion_empty_dense():
    results = reciprocal_rank_fusion([MockDenseResult("y", 0.5)], [])
    assert len(results) == 1
    assert results[0].chunk_id == "y"
