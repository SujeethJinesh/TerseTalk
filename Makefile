PY := $(shell command -v python3.12 >/dev/null 2>&1 && echo python3.12 || echo python3)
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
BLACK := $(VENV)/bin/black

.PHONY: all setup lint test dev fmt clean

all: setup lint test

setup:
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install pytest ruff black

lint:
	$(RUFF) check .
	$(BLACK) --check .

fmt:
	$(BLACK) .
	$(RUFF) check --fix .

test:
	PYTHONPATH=. $(PYTEST)

dev:
	@echo "Dev placeholder: research runners coming soon (see RESEARCH_PROPOSAL.md)."

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .pytest_cache
