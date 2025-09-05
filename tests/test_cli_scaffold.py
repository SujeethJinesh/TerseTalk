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
  out = r.stdout + r.stderr
  assert "--task" in out and "--system" in out and "--dry-run" in out


def test_version_flag_prints_version():
  r = run_cmd([sys.executable, "scripts/run_v05.py", "--version"])
  assert r.returncode == 0
  out = (r.stdout + r.stderr).lower()
  assert "tersetalk v0.5 runner" in out or "version" in out


def test_dry_run_json_shape_is_valid():
  r = run_cmd(
    [
      sys.executable,
      "scripts/run_v05.py",
      "--task",
      "hotpotqa",
      "--system",
      "tersetalk",
      "--n",
      "3",
      "--seed",
      "123",
      "--caps",
      '{"f":30,"p":20,"q":30}',
      "--dry-run",
    ]
  )
  assert r.returncode == 0
  data = json.loads(r.stdout)
  assert data["task"] == "hotpotqa"
  assert data["system"] == "tersetalk"
  assert data["n"] == 3
  assert data["seed"] == 123
  assert data["caps"]["f"] == 30
  assert data["defaults"]["seed"] == 123

