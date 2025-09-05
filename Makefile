.PHONY: install test smoke help

install:
	python -m pip install -U pip
	python -m pip install -e .
	python -m pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest -q

smoke:
	python scripts/repro_smoke.py

help:
	@echo "Targets: install | test | smoke"
