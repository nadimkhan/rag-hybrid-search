"""Tests for configuration loading."""

import pytest
from src.config import (
    DENSE_WEIGHT, SPARSE_WEIGHT, RRF_K,
    CHUNK_SIZE, CHUNK_OVERLAP,
)


def test_default_weights_sum_to_one():
    assert abs(DENSE_WEIGHT + SPARSE_WEIGHT - 1.0) < 1e-6


def test_rrf_k_is_positive():
    assert RRF_K > 0


def test_chunk_size_reasonable():
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP >= 0
    assert CHUNK_OVERLAP < CHUNK_SIZE
