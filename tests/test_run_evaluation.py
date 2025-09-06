from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_evaluation_dryrun(tmp_path: Path):
    # Run the evaluation in dry-run (echo) and ensure summary exists
    script = Path("scripts") / "run_evaluation.py"
    assert script.exists(), "scripts/run_evaluation.py missing"

    outdir = tmp_path / "results"
    cmd = [sys.executable, str(script), "--task", "hotpotqa", "--systems", "tersetalk", "--dry-run", "--out", str(outdir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # Find summary.json under results/evaluation/hotpotqa
    base = outdir / "hotpotqa"
    assert base.exists(), f"missing results dir: {base}"
    summaries = list(base.rglob("summary.json"))
    assert summaries, "summary.json not produced"
