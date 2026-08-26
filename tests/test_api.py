"""Basic API smoke test."""

import pytest


def test_import_app():
    from src.api.main import create_app
    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline()
    app = create_app(pipeline)
    assert app.title == "RAG Hybrid Search API"
