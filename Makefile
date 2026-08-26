.PHONY: install run eval test clean

install:
	pip install -e .

ingest:
	python scripts/ingest_sample_docs.py

ask:
	python scripts/demo.py

eval:
	python scripts/run_eval.py

run:
	python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf data/chromadb data/bm25_index.json data/eval_results.json
