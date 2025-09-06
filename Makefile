.PHONY: install test smoke analyze help

install:
	python -m pip install -U pip
	python -m pip install -e .
	python -m pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest -q

smoke:
	python scripts/repro_smoke.py

analyze:
	python scripts/analyze_v05.py --indir results --outdir results/figures

help:
	@echo "Targets: install | test | smoke | analyze"
