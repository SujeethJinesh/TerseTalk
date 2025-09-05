# TerseTalk — PR-01 Scaffold

This repository contains the TerseTalk project. PR‑01 adds a minimal repository scaffold and a CLI skeleton.

## Quickstart

```bash
make install
python scripts/run_v05.py --help
python scripts/run_v05.py --version
python scripts/run_v05.py --task hotpotqa --system tersetalk --n 5 --seed 123 --dry-run
```

Notes

PR‑00 reproducibility utilities live in `tersetalk/reproducibility.py`.

The CLI here is a stub. Execution paths (datasets, models, protocol) arrive in later PRs.

## PR-02 quick smoke

```bash
# Mixed format + overflow demo
python scripts/jsonl_guard.py --caps '{"f":5,"q":5}' <<'EOF'
["r","M"]
{"f":"alpha beta gamma delta epsilon zeta eta"}
["q","W","please compare the following two long things"]
plain text line (mixed!)
EOF
```
