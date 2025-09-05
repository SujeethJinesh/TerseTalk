Read through your AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the AGENTS.md. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update AGENTS.md and ask CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly.. Here are more detailed instructions.

### PR Summary

PR‑01 — Repository Scaffold & CLI Skeleton

Role: You are a senior engineer implementing PR‑01 for the TerseTalk project, immediately after PR‑00 (reproducibility) was merged.

Goal (from spec): Provide a minimal, clean repository scaffold and a CLI runner skeleton so that:

make install works reliably

python scripts/run_v05.py --help prints

(nice‑to‑have) --version prints package version and --dry-run emits a JSON config snapshot

Strict scope: Do not implement later PRs (protocol/benchmarks/etc.). Keep dependencies minimal for fast install.

Requirements & guardrails

Python ≥ 3.10

Preserve PR‑00 files and tests; do not remove or break them.

Keep new runtime deps minimal (only click and tqdm for now).

Add package versioning (tersetalk/\_version.py) and expose **version**.

Add .gitignore and a concise README.md.

Update Makefile so make install installs both runtime and dev requirements.

Provide a CLI skeleton at scripts/run_v05.py using click with these options (stub only, no real execution):

--task {hotpotqa,gsm8k} (default hotpotqa)

--system {tersetalk,freeform,llmlingua} (default tersetalk)

--n (default 100)

--seed (default 0)

--caps (default '{"f":30,"p":20,"q":30}')

--model (default mistral)

--out (default results)

--dry-run / --execute (default --dry-run) → prints parsed config JSON and exits 0

--version

Use set_global_seed from PR‑00 to populate deterministic defaults inside the printed JSON.

Create/Update the following files exactly

⚠️ Only add/modify what’s listed. Keep PR‑00 tests intact. Where a file already exists, replace it with the contents below.

1. .gitignore

# Byte-compiled / cache

**pycache**/
_.py[cod]
_.pyo
_.pyd
_.so
_.dll
_.dylib

# Packaging / build

build/
dist/
\*.egg-info/
.eggs/

# Virtual envs

.venv/
venv/
.env
.envrc

# IDE / OS

.vscode/
.idea/
.DS_Store

# Test / coverage

.pytest_cache/
.coverage
htmlcov/

# Jupyter

.ipynb_checkpoints/

# Project outputs

results/
figures/

2. requirements.txt (runtime, minimal for PR‑01)
   click>=8.1.3
   tqdm>=4.66.0

3. README.md

# TerseTalk — PR-01 Scaffold

This repository contains the TerseTalk project. PR‑01 adds a minimal repository scaffold and a CLI skeleton.

## Quickstart

````bash
make install
python scripts/run_v05.py --help
python scripts/run_v05.py --version
python scripts/run_v05.py --task hotpotqa --system tersetalk --n 5 --seed 123 --dry-run

Notes

PR‑00 reproducibility utilities live in tersetalk/reproducibility.py.

The CLI here is a stub. Execution paths (datasets, models, protocol) arrive in later PRs.


### 4) Update `Makefile` (replace file)

```make
.PHONY: install test smoke help

install:
\tpython -m pip install -U pip
\tpython -m pip install -e .
\tpython -m pip install -r requirements.txt -r requirements-dev.txt

test:
\tpytest -q

smoke:
\tpython scripts/repro_smoke.py

help:
\t@echo "Targets: install | test | smoke"

5) Update pyproject.toml to declare package discovery
[build-system]
requires = ["setuptools>=65", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "tersetalk"
version = "0.0.2"
description = "TerseTalk reproducibility + scaffold (PR-00/PR-01)"
requires-python = ">=3.10"
dependencies = []

[tool.setuptools.packages.find]
where = ["."]
include = ["tersetalk*"]

[tool.pytest.ini_options]
addopts = "-q"
pythonpath = ["."]

6) tersetalk/_version.py (new)
__version__ = "0.0.2"

7) Update tersetalk/__init__.py (replace file)
from ._version import __version__

__all__ = ["__version__", "reproducibility"]


(PR‑00’s reproducibility module remains importable as tersetalk.reproducibility.)

8) scripts/run_v05.py (new CLI skeleton)
from __future__ import annotations

import json
import sys
import click

from tersetalk._version import __version__
from tersetalk.reproducibility import set_global_seed

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--task", type=click.Choice(["hotpotqa", "gsm8k"]), default="hotpotqa", show_default=True, help="Benchmark task to run.")
@click.option("--system", type=click.Choice(["tersetalk", "freeform", "llmlingua"]), default="tersetalk", show_default=True, help="System variant to run.")
@click.option("--n", default=100, show_default=True, help="Number of examples.")
@click.option("--seed", default=0, show_default=True, help="Global random seed.")
@click.option("--caps", default='{"f":30,"p":20,"q":30}', show_default=True, help="Soft caps JSON for tags.")
@click.option("--model", default="mistral", show_default=True, help="Model name (placeholder).")
@click.option("--out", default="results", show_default=True, help="Output directory.")
@click.option("--dry-run/--execute", default=True, show_default=True, help="Dry-run prints parsed config JSON and exits 0.")
@click.version_option(version=__version__, prog_name="tersetalk v0.5 runner")
def main(task, system, n, seed, caps, model, out, dry_run):
    """
    TerseTalk v0.5 Runner (PR-01 scaffold)

    This command provides a CLI skeleton only. Use --dry-run (default) to print
    the parsed configuration. Execution paths are implemented in later PRs.
    """
    try:
        parsed_caps = json.loads(caps)
        if not isinstance(parsed_caps, dict):
            raise ValueError
    except Exception:
        click.echo(
            'Error: --caps must be a JSON object, e.g. \'{"f":30,"p":20,"q":30}\'',
            err=True,
        )
        sys.exit(2)

    defaults = set_global_seed(int(seed))
    cfg = {
        "task": task,
        "system": system,
        "n": int(n),
        "seed": int(seed),
        "caps": parsed_caps,
        "model": model,
        "out": out,
        "defaults": defaults,
        "mode": "dry-run" if dry_run else "execute",
    }

    click.echo(json.dumps(cfg, indent=2))

    if dry_run:
        sys.exit(0)

    # Execution path intentionally unimplemented in PR-01
    click.echo("Execution mode is not implemented in PR-01.", err=True)
    sys.exit(0)


if __name__ == "__main__":
    main()

9) New tests: tests/test_cli_scaffold.py
from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_cmd(args: list[str]):
    return subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True)

def test_help_exits_zero_and_shows_options():
    r = run_cmd([sys.executable, "scripts/run_v05.py", "--help"])
    assert r.returncode == 0
    # Click may print to stdout
    out = r.stdout + r.stderr
    assert "--task" in out and "--system" in out and "--dry-run" in out

def test_version_flag_prints_version():
    r = run_cmd([sys.executable, "scripts/run_v05.py", "--version"])
    assert r.returncode == 0
    out = (r.stdout + r.stderr).lower()
    assert "tersetalk v0.5 runner" in out or "version" in out

def test_dry_run_json_shape_is_valid():
    r = run_cmd([
        sys.executable, "scripts/run_v05.py",
        "--task", "hotpotqa",
        "--system", "tersetalk",
        "--n", "3",
        "--seed", "123",
        "--caps", '{"f":30,"p":20,"q":30}',
        "--dry-run"
    ])
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["task"] == "hotpotqa"
    assert data["system"] == "tersetalk"
    assert data["n"] == 3
    assert data["seed"] == 123
    assert data["caps"]["f"] == 30
    assert data["defaults"]["seed"] == 123

What to run (and what to paste as evidence in the PR)

Install

make install


Tests

make test


Help & Version

python scripts/run_v05.py --help
python scripts/run_v05.py --version


Dry‑run sample

python scripts/run_v05.py --task hotpotqa --system tersetalk --n 5 --seed 123 --dry-run


Acceptance evidence to paste in PR description:

The full pytest summary (should be all green, including PR‑00 tests).

The --help output snippet showing options.

The --version output line.

The JSON blob from the dry‑run command.

Commit message
PR-01: Repository scaffold & CLI skeleton

- Add .gitignore, minimal requirements.txt, concise README
- Update Makefile to install runtime + dev deps
- Add package versioning (tersetalk/_version.py), expose __version__
- Update pyproject to ensure package discovery
- Introduce scripts/run_v05.py Click-based CLI with --help/--version/--dry-run
- Add CLI tests; preserve PR-00 reproducibility utilities and tests
- DoD: `make install` works; `python scripts/run_v05.py --help` prints
````
