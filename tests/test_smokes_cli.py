from __future__ import annotations

import os
import subprocess
import sys


def _run(cmd: list[str]) -> tuple[int, str]:
  p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  return p.returncode, p.stdout


def test_pipeline_smoke_echo_model_offline():
  env = os.environ.copy()
  env["TERSETALK_OFFLINE_DATA"] = "1"
  code, out = _run([sys.executable, "-m", "scripts.pipeline_smoke", "--task", "hotpotqa", "--seed", "1", "--offline", "--model", "echo"])
  assert code == 0
  assert "manager_jsonl" in out and "worker_jsonl" in out


def test_pipeline_smoke_real_model_skips_by_default():
  env = os.environ.copy()
  env.pop("RUN_REAL_MODEL", None)
  code, out = _run([sys.executable, "-m", "scripts.pipeline_smoke", "--task", "hotpotqa", "--seed", "1", "--offline", "--model", "real"])
  assert code == 0
  assert "SKIP: real model not enabled" in out


def test_baselines_smoke_real_model_skips_by_default():
  env = os.environ.copy()
  env.pop("RUN_REAL_MODEL", None)
  code, out = _run([sys.executable, "-m", "scripts.baselines_smoke", "--task", "hotpotqa", "--seed", "1", "--offline", "--model", "real"])
  assert code == 0
  assert "SKIP: real model not enabled" in out

