FROM python:3.11-slim

WORKDIR /app

# Install system deps for ChromaDB / BM25
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/corpus/ ./data/corpus/

ENV PYTHONPATH=/app
ENV CHROMA_DB_PATH=/app/data/chromadb
ENV BM25_INDEX_PATH=/app/data/bm25_index.json

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
