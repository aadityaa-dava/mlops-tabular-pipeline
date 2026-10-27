.PHONY: help setup test lint format repro mlflow-ui serve

help:
	@echo "make setup      install dependencies and the pinned Python (uv sync)"
	@echo "make test       run the unit tests"
	@echo "make lint       ruff lint and format check (no changes)"
	@echo "make format     apply ruff fixes and formatting"
	@echo "make repro      rebuild the pipeline (dvc repro): merge, validate, train"
	@echo "make mlflow-ui  browse runs at http://localhost:5050"

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff check --fix . && uv run ruff format .

repro:
	uv run dvc repro

mlflow-ui:
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5050

serve:
	uv run uvicorn mlops_pipeline.api:create_app --factory --port 8000
