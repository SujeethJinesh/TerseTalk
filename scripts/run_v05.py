from __future__ import annotations

import json
import os
import sys
import click

# Ensure project root on path for direct script runs
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from tersetalk._version import __version__
from tersetalk.reproducibility import set_global_seed

# Optional hybrid import (dry-run preview only)
try:  # pragma: no cover
  from tersetalk.hybrid_gate import GateCfg, gate_choose_protocol
except Exception:  # pragma: no cover
  GateCfg = None  # type: ignore
  gate_choose_protocol = None  # type: ignore

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"]) 


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
  "--task",
  type=click.Choice(["hotpotqa", "gsm8k"]),
  default="hotpotqa",
  show_default=True,
  help="Benchmark task to run.",
)
@click.option(
  "--system",
  type=click.Choice(["tersetalk", "freeform", "llmlingua"]),
  default="tersetalk",
  show_default=True,
  help="System variant to run.",
)
@click.option("--n", default=100, show_default=True, help="Number of examples.")
@click.option("--seed", default=0, show_default=True, help="Global random seed.")
@click.option(
  "--caps",
  default='{"f":30,"p":20,"q":30}',
  show_default=True,
  help="Soft caps JSON for tags.",
)
@click.option("--model", default="mistral", show_default=True, help="Model name (placeholder).")
@click.option("--out", default="results", show_default=True, help="Output directory.")
@click.option(
  "--dry-run/--execute",
  default=True,
  show_default=True,
  help="Dry-run prints parsed config JSON and exits 0.",
)
@click.option("--hybrid/--no-hybrid", default=False, show_default=True, help="Include hybrid gate decision in dry-run output if probe texts are provided.")
@click.option("--token-budget", default=600, show_default=True, help="Token budget for hybrid gate (dry-run only).")
@click.option("--gate-jsonl-probe", default=None, help="JSONL probe text for the gate (dry-run only).")
@click.option("--gate-freeform-probe", default=None, help="Free-form probe text for the gate (dry-run only).")
@click.version_option(version=__version__, prog_name="tersetalk v0.5 runner")
def main(task, system, n, seed, caps, model, out, dry_run, hybrid, token_budget, gate_jsonl_probe, gate_freeform_probe):
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
      "Error: --caps must be a JSON object, e.g. '{\"f\":30,\"p\":20,\"q\":30}'",
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

  # Optional gate preview for dry-run
  gate_obj = None
  if (
    dry_run
    and hybrid
    and gate_jsonl_probe
    and gate_freeform_probe
    and GateCfg is not None
    and gate_choose_protocol is not None
  ):
    try:
      gcfg = GateCfg(token_budget=int(token_budget))  # type: ignore
      gate_obj = gate_choose_protocol(gate_jsonl_probe, gate_freeform_probe, gcfg)  # type: ignore
    except Exception as e:  # keep dry-run robust
      gate_obj = {"error": str(e)}
  cfg["gate"] = gate_obj

  click.echo(json.dumps(cfg, indent=2))

  if dry_run:
    sys.exit(0)

  # Execution path intentionally unimplemented in PR-01
  click.echo("Execution mode is not implemented in PR-01.", err=True)
  sys.exit(0)


if __name__ == "__main__":
  main()
