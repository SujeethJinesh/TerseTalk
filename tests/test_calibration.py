from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(args: list[str]) -> Tuple[int, str, str]:
  p = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True)
  return p.returncode, p.stdout, p.stderr


def test_calibration_writes_yaml_and_schema(tmp_path: Path):
  outp = tmp_path / "calib.yaml"
  code, out, err = _run(
    [
      PY,
      "scripts/calibrate_caps.py",
      "--n",
      "24",
      "--seed",
      "123",
      "--out",
      str(outp),
      "--density-min",
      "0.6",
      "--summarizers",
      "extractive",
      "--gate-modes",
      "off,on",
      "--token-budgets",
      "120,240",
    ]
  )
  assert code == 0, err
  assert outp.exists() and outp.read_text().strip()
  data = json.loads(outp.read_text())
  assert "best" in data and "grid_evaluations" in data and data["n"] == 24
  best = data["best"]
  assert "spec" in best and "metrics" in best
  assert isinstance(best["metrics"]["avg_est_tokens"], (int, float))
  assert 0.0 <= best["metrics"]["avg_density"] <= 1.0


def test_determinism_same_seed_same_output(tmp_path: Path):
  out1 = tmp_path / "a.yaml"
  out2 = tmp_path / "b.yaml"
  cmd = [
    PY,
    "scripts/calibrate_caps.py",
    "--n",
    "30",
    "--seed",
    "7",
    "--out",
    str(out1),
    "--density-min",
    "0.7",
    "--summarizers",
    "extractive",
    "--gate-modes",
    "off,on",
    "--token-budgets",
    "150,300",
  ]
  code, out, err = _run(cmd)
  assert code == 0, err
  cmd2 = cmd.copy()
  cmd2[cmd2.index(str(out1))] = str(out2)
  code2, outb, errb = _run(cmd2)
  assert code2 == 0, errb
  assert out1.read_text() == out2.read_text()


def test_gate_affects_routing_with_fake_ll2(tmp_path: Path, monkeypatch):
  """
  Use TERSETALK_FAKE_LL2_COMPRESS to ensure some routing to freeform when gate is on.
  """
  outp = tmp_path / "calib.yaml"
  monkeypatch.setenv("TERSETALK_FAKE_LL2_COMPRESS", "80")
  code, out, err = _run(
    [
      PY,
      "scripts/calibrate_caps.py",
      "--n",
      "20",
      "--seed",
      "42",
      "--out",
      str(outp),
      "--density-min",
      "0.5",
      "--summarizers",
      "extractive",
      "--gate-modes",
      "off,on",
      "--token-budgets",
      "100,120",
    ]
  )
  assert code == 0, err
  data = json.loads(outp.read_text())
  assert any(
    ev["spec"]["gate_enabled"] and ev["metrics"]["routed_freeform_frac"] > 0.0
    for ev in data["grid_evaluations"]
  )

