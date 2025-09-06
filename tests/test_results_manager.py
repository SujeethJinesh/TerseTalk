from __future__ import annotations

import json
from pathlib import Path

from tersetalk.results_manager import ResultsManager


def test_get_run_dir_and_latest_mirror(tmp_path: Path):
  rm = ResultsManager(base_dir=tmp_path / "results")
  run_dir = rm.get_run_dir("expA", timestamp=False, run_id="2025-01-01-00-00-00")

  assert run_dir.exists() and run_dir.is_dir()
  exp_dir = run_dir.parent
  latest = exp_dir / "latest"
  assert latest.exists()

  cfg_path = rm.save_config(run_dir, {"a": 1})
  assert cfg_path.exists()
  assert (latest / "config.json").exists()
  data = json.loads((latest / "config.json").read_text())
  assert data["a"] == 1


def test_jsonl_and_csv_append(tmp_path: Path):
  rm = ResultsManager(base_dir=tmp_path / "results")
  run_dir = rm.get_run_dir("expB", timestamp=False, run_id="2025-01-01-00-00-01")
  latest = run_dir.parent / "latest"

  p = rm.append_jsonl(run_dir, "raw_outputs.jsonl", {"k": "v1"})
  rm.append_jsonl(run_dir, "raw_outputs.jsonl", {"k": "v2"})
  lines = p.read_text().strip().splitlines()
  assert len(lines) == 2
  latest_lines = (latest / "raw_outputs.jsonl").read_text().strip().splitlines()
  assert len(latest_lines) == 2

  c = rm.append_csv_row(run_dir, "metrics.csv", {"id": 1, "acc": 0.8})
  rm.append_csv_row(run_dir, "metrics.csv", {"id": 2, "acc": 0.9})
  text = c.read_text().strip().splitlines()
  assert len(text) == 3
  latest_text = (latest / "metrics.csv").read_text().strip().splitlines()
  assert len(latest_text) == 3


def test_cleanup_keeps_n_newest(tmp_path: Path):
  rm = ResultsManager(base_dir=tmp_path / "results")

  exp = "expC"
  r1 = rm.get_run_dir(exp, timestamp=False, run_id="2025-01-01-00-00-00")
  r2 = rm.get_run_dir(exp, timestamp=False, run_id="2025-02-01-00-00-00")
  r3 = rm.get_run_dir(exp, timestamp=False, run_id="2025-03-01-00-00-00")

  removed = rm.cleanup_old_runs(exp, keep_last_n=2)
  kept = sorted([p.name for p in (r3.parent.iterdir()) if p.is_dir() and p.name != "latest"]) 

  assert "2025-01-01-00-00-00" in {p.name for p in removed}
  assert "2025-02-01-00-00-00" in kept
  assert "2025-03-01-00-00-00" in kept

