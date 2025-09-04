PY := python3
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
	$(PYTHON) src/primes.py 10

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .pytest_cache
