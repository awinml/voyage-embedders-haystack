.PHONY: sync test test-cov cov-report cov lint-typing lint-style lint-fmt lint-all example-text-embedder example-doc-embedder example-semantic-search example-reranker test-examples clean help

help:
	@echo "Available make targets:"
	@echo "  make sync                     - Sync project and install dependencies"
	@echo "  make test                     - Run unit tests"
	@echo "  make test-cov                 - Run tests with coverage collection"
	@echo "  make cov-report               - Generate coverage reports (xml, html)"
	@echo "  make cov                      - Run tests and generate coverage reports"
	@echo "  make lint-typing              - Type check with mypy"
	@echo "  make lint-style               - Lint with ruff and black (check only)"
	@echo "  make lint-fmt                 - Format code and lint with auto-fixes"
	@echo "  make lint-all                 - Run formatting, linting, and type checking"
	@echo "  make example-text-embedder    - Run text embedder example"
	@echo "  make example-doc-embedder     - Run document embedder example"
	@echo "  make example-semantic-search  - Run semantic search example"
	@echo "  make example-reranker         - Run reranker example"
	@echo "  make test-examples            - Run all examples"
	@echo "  make clean                    - Clean build artifacts and cache"

sync:
	uv sync --all-extras

test:
	uv run pytest tests

test-cov:
	uv run coverage run -m pytest tests

cov-report:
	- uv run coverage combine
	uv run coverage xml
	uv run coverage html

cov: test-cov cov-report

lint-typing:
	uv run mypy --install-types --non-interactive --explicit-package-bases src/ tests

lint-style:
	uv run ruff check .
	uv run black --check --diff .

lint-fmt:
	uv run black .
	uv run ruff check --fix .
	uv run black --check --diff .

lint-all: lint-fmt lint-typing

example-text-embedder:
	uv run python examples/text_embedder_example.py

example-doc-embedder:
	uv run python examples/document_embedder_example.py

example-semantic-search:
	uv run python examples/semantic_search_pipeline_example.py

example-reranker:
	uv run python examples/reranker_example.py

test-examples: example-text-embedder example-doc-embedder example-semantic-search example-reranker

clean:
	rm -rf .coverage coverage.xml htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
