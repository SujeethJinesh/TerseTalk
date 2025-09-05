.PHONY: install test smoke

install:
	python -m pip install -e .
	python -m pip install -r requirements-dev.txt

test:
	pytest -q

smoke:
	python scripts/repro_smoke.py
