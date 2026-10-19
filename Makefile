.PHONY: check test lint

check: lint test
	python scripts/check_repo.py

test:
	python -m pytest -q

lint:
	ruff check src tests scripts
