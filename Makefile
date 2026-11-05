.PHONY: setup fixtures ingest index eval test lint app streamlit all

PYTHON := .venv/Scripts/python.exe

setup:
	uv sync --extra dev --extra app --extra eval --extra tables

fixtures:
	$(PYTHON) scripts/make_fixtures.py

ingest: fixtures
	$(PYTHON) -m mmrag.cli ingest data/raw

index: ingest
	$(PYTHON) -m mmrag.cli index

eval: index
	$(PYTHON) -m mmrag.cli eval

test:
	uv run --extra dev pytest

lint:
	uv run --extra dev ruff check .
	uv run --extra dev ruff format --check .
	uv run --extra dev mypy src/mmrag

app:
	$(PYTHON) scripts/run_app.py

streamlit:
	uv run --extra app streamlit run src/mmrag/app/streamlit_app.py

all: lint test fixtures ingest index eval
