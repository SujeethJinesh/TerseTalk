from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_significance_smoke(tmp_path: Path):
    d = tmp_path / "runs"
    d.mkdir(parents=True)

    # Minimal aligned rows: freeform worse tokens than tersetalk; identical accuracy; hybrid ~ ll2
    (d / "freeform.jsonl").write_text('{"tokens":100,"correct":true,"status":"success"}\n')
    (d / "tersetalk_baseline.jsonl").write_text('{"tokens":60,"correct":true,"status":"success"}\n')
    (d / "llmlingua.jsonl").write_text('{"tokens":70,"correct":true,"status":"success"}\n')
    (d / "hybrid_budget_600.jsonl").write_text('{"tokens":65,"correct":true,"status":"success"}\n')

    cmd = [
        sys.executable,
        "scripts/run_significance.py",
        "--results-dir",
        str(d),
        "--boots",
        "2000",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    out_path = d / "significance_tests.json"
    assert out_path.exists(), "significance_tests.json not created"
    report = json.loads(out_path.read_text())
    assert "token_reduction" in report and report["token_reduction"]["n"] == 1
    assert "hybrid_noninferiority" in report

