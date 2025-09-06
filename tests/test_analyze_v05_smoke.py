from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_analyze_v05_smoke(tmp_path: Path):
    # 1) Produce a tiny offline run with EchoModel on the synth task
    run_script = Path("scripts") / "run_v05.py"
    assert run_script.exists(), "scripts/run_v05.py not found"

    results_dir = tmp_path / "results"
    cmd_run = [
        sys.executable, str(run_script),
        "--task", "synth",
        "--system", "tersetalk",
        "--n", "3",
        "--seed", "0",
        "--model", "echo",
        "--out", str(results_dir),
        "--save-every", "1",
        "--sp", "jaccard",
    ]
    proc_run = subprocess.run(cmd_run, capture_output=True, text=True, timeout=120)
    assert proc_run.returncode == 0, f"run_v05 failed: {proc_run.stderr}"

    # 2) Analyze
    analyze_script = Path("scripts") / "analyze_v05.py"
    assert analyze_script.exists(), "scripts/analyze_v05.py not found"

    figures_dir = tmp_path / "figures"
    cmd_an = [
        sys.executable, str(analyze_script),
        "--indir", str(results_dir),
        "--outdir", str(figures_dir),
        "--task", "synth",          # focus the test
        "--format", "pdf",
        "--no-annotate"
    ]
    proc_an = subprocess.run(cmd_an, capture_output=True, text=True, timeout=120)
    assert proc_an.returncode == 0, f"analyze_v05 failed: {proc_an.stderr}"

    # 3) Artifacts
    by_run = figures_dir / "by_run.csv"
    assert by_run.exists() and by_run.read_text().strip(), "by_run.csv missing/empty"

    pareto = figures_dir / "pareto_synth.pdf"
    assert pareto.exists(), "pareto_synth.pdf not created"

