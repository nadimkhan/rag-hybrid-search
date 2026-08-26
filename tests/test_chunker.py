"""Tests for text chunking."""

import pytest
from src.ingestion.chunker import ChunkPipeline
from src.ingestion.loader import Document


def test_chunks_are_trimmed():
    pipeline = ChunkPipeline()
    doc = Document(
        id="test123",
        content="  Hello world this is a test.  \n\n  More content here.  ",
        source="test.txt",
    )
    chunks = pipeline.chunk_document(doc)
    assert all(c.text == c.text.strip() for c in chunks)
    assert all(c.char_count > 0 for c in chunks)


def test_empty_content_returns_nothing():
    pipeline = ChunkPipeline()
    doc = Document(id="test", content="   ", source="test.txt")
    chunks = pipeline.chunk_document(doc)
    assert len(chunks) == 0
