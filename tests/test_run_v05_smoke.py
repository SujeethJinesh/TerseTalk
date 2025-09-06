from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_v05_synth_echo(tmp_path: Path):
  script = Path("scripts") / "run_v05.py"
  assert script.exists(), "run_v05.py not found"

  outdir = tmp_path / "results"
  cmd = [
    sys.executable,
    str(script),
    "--task",
    "synth",
    "--system",
    "tersetalk",
    "--n",
    "3",
    "--seed",
    "0",
    "--model",
    "echo",
    "--out",
    str(outdir),
    "--save-every",
    "1",
    "--sp",
    "jaccard",
  ]
  proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
  assert proc.returncode == 0, proc.stderr

  runs = list((outdir / "synth_tersetalk").glob("*"))
  assert runs, "No run directory created"
  # pick a timestamp dir (exclude latest)
  rundir = [p for p in runs if p.is_dir() and p.name != "latest"][0]

  raw = rundir / "raw_outputs.jsonl"
  cfg = rundir / "config.json"
  summ = rundir / "summary.json"

  assert cfg.exists() and cfg.read_text().strip(), "config missing/empty"
  assert raw.exists() and raw.read_text().strip(), "raw_outputs.jsonl missing/empty"
  assert summ.exists() and summ.read_text().strip(), "summary.json missing/empty"

  s = json.loads(summ.read_text())
  assert "num_examples" in s and s["num_examples"] == 3

