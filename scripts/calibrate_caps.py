from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

# Ensure package import for direct execution
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from tersetalk.calibration import (
  default_caps_grid,
  save_calibration_yaml,
  sweep_grid,
)


def _parse_caps_grid(s: str) -> List[dict]:
  try:
    val = json.loads(s)
    if isinstance(val, list) and all(isinstance(x, dict) for x in val):
      return val
  except Exception:
    pass
  raise SystemExit(
    "Error: --caps-grid must be a JSON list of objects, e.g. ' [{\"f\":20,\"p\":15,\"q\":20},{\"f\":30,\"p\":20,\"q\":30}]'"
  )


def _csv_list(s: str) -> List[str]:
  return [x.strip() for x in s.split(",") if x.strip()]


def _csv_ints(s: str) -> List[int]:
  try:
    return [int(x) for x in _csv_list(s)]
  except Exception as e:
    raise SystemExit(f"Error parsing integer list: {e}") from e


def main() -> None:
  ap = argparse.ArgumentParser(description="PR-H2: Calibration sweep for caps/summarizer/gate")
  ap.add_argument("--n", type=int, default=50, help="Number of synthetic examples")
  ap.add_argument("--seed", type=int, default=0, help="Random seed for determinism")
  ap.add_argument("--out", type=str, default="configs/calibration.yaml", help="Output YAML path")
  ap.add_argument("--density-min", type=float, default=0.75, help="Min avg density required")
  ap.add_argument("--caps-grid", type=str, default="", help="JSON list of caps dicts (overrides defaults)")
  ap.add_argument("--summarizers", type=str, default="extractive", help="Comma list among {extractive,llmlingua}")
  ap.add_argument(
    "--deref-policies",
    type=str,
    default="never,conditional,always",
    help="Comma list (placeholder)",
  )
  ap.add_argument("--gate-modes", type=str, default="off,on", help="Comma list among {off,on}")
  ap.add_argument("--token-budgets", type=str, default="400,600,800", help="Comma list of integers")
  args = ap.parse_args()

  caps_grid = _parse_caps_grid(args.caps_grid) if args.caps_grid else default_caps_grid()
  summarizers = _csv_list(args.summarizers)
  deref_pols = _csv_list(args.deref_policies)
  gate_modes = [g.lower() in ("on", "true", "1", "yes") for g in _csv_list(args.gate_modes)]
  budgets = _csv_ints(args.token_budgets)

  report = sweep_grid(
    n=int(args.n),
    seed=int(args.seed),
    caps_grid=caps_grid,
    summarizers=summarizers,  # type: ignore[arg-type]
    deref_policies=deref_pols,  # type: ignore[arg-type]
    gate_modes=gate_modes,
    token_budgets=budgets,
    density_min=float(args.density_min),
  )
  out_path = save_calibration_yaml(report, args.out)

  # Also print a compact JSON summary to stdout (helpful for CI)
  best = report["best"]
  summary = {
    "out_path": str(out_path),
    "n": report["n"],
    "seed": report["seed"],
    "density_min": report["density_min"],
    "best_spec": best["spec"],
    "best_metrics": best["metrics"],
  }
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()

