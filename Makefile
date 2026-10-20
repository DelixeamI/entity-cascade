.PHONY: up down test lint fetch index eval record tune demo

CONFIG ?= configs/default.yaml
RUN := uv run

up:
	docker compose up -d qdrant

down:
	docker compose down

test:
	$(RUN) pytest -q

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy

fetch:
	$(RUN) python scripts/fetch_data.py --config $(CONFIG)

# Build the dense index in a running Qdrant (QDRANT_URL, default http://localhost:6333).
index:
	$(RUN) python -m entity_cascade.eval.run index --config $(CONFIG)

# Full evaluation: reports/results.md + reports/*.png. Needs no API keys (RecordedLLM cassettes).
eval:
	$(RUN) python scripts/fetch_data.py --config configs/amazon_google.yaml
	$(RUN) python -m entity_cascade.eval.run eval --config $(CONFIG) --secondary configs/amazon_google.yaml --readme

# Re-record LLM cassettes against a real provider (see configs/default.yaml: llm.*).
record:
	$(RUN) python -m entity_cascade.eval.run record --config $(CONFIG)

# Val only: record the arbiter on the val gray zone for every prompt version and choose the
# prompt version and llm.min_confidence by the rule in the config. Does not touch test.
tune:
	$(RUN) python -m entity_cascade.eval.run tune --config $(CONFIG)

demo:
	$(RUN) python -m entity_cascade.cli match --config $(CONFIG) --demo 100 --llm recorded --output out/demo_result.csv
