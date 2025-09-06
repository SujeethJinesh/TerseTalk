from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_analyze_smoke_min(tmp_path: Path):
    # Create a minimal faux run: two systems + summary.json
    run = tmp_path / "results" / "evaluation" / "hotpotqa" / "2025-01-01-00-00-00"
    run.mkdir(parents=True)
    (run / "freeform.jsonl").write_text('{"tokens": 100, "correct": true, "status":"success"}\n')
    (run / "tersetalk_f30_p20_q30.jsonl").write_text('{"tokens_total": 60, "correct": true, "status":"success", "overflow_count":0}\n')
    (run / "summary.json").write_text('{"freeform":{"n_total":1,"n_successful":1,"avg_tokens":100,"accuracy":1.0,"compliance_rate":1.0}}')

    outdir = tmp_path / "figs"
    cmd = [sys.executable, "scripts/analyze_v05.py", "--indir", str(tmp_path / "results"), "--outdir", str(outdir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    assert (outdir / "pareto_points.csv").exists()
    assert (outdir / "pareto_frontier.pdf").exists()

