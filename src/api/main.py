"""FastAPI app entry point."""

from src.pipeline import RAGPipeline
from src.api.routes import create_app


def make_app() -> any:
    """Build the app with a fresh pipeline."""
    pipeline = RAGPipeline()
    return create_app(pipeline)


app = make_app()
