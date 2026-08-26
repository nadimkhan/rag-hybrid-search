"""Tests for document loader."""

import pytest
from pathlib import Path
from src.ingestion.loader import DocumentLoader, Document


def test_load_text(tmp_path):
    doc_file = tmp_path / "test.txt"
    doc_file.write_text("Hello world", encoding="utf-8")
    loader = DocumentLoader()
    doc = loader.load(doc_file)
    assert doc.content == "Hello world"
    assert doc.source == str(doc_file)


def test_load_markdown_extracts_heading(tmp_path):
    doc_file = tmp_path / "test.md"
    doc_file.write_text("# My Heading\n\nContent here.", encoding="utf-8")
    loader = DocumentLoader()
    doc = loader.load(doc_file)
    assert doc.section_heading == "My Heading"


def test_load_dir_filters_by_extension(tmp_path):
    (tmp_path / "a.txt").write_text("text", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Doc", encoding="utf-8")
    (tmp_path / "c.py").write_text("print(1)", encoding="utf-8")  # should be excluded
    loader = DocumentLoader()
    docs = loader.load_dir(tmp_path)
    sources = [d.source for d in docs]
    assert any("a.txt" in s for s in sources)
    assert any("b.md" in s for s in sources)
    assert not any("c.py" in s for s in sources)
    assert len(docs) == 2
