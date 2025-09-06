Read through your @AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated @RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the @RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the @AGENTS.md. To ask claude code for a review, you can do something like `echo "Please ensure you understand @RESEARCH_PROPOSAL.md. I am implementing PR-X. Please review the following files I created: @README.md, @xxx. Ensure that my implementation is properly aligned with the goals of the project. Also here is some additional data on the runs I gathered, please critique it as if you're a senior data scientist and ensure that I'm not cheating on the results, lying, or misrepresenting things" | claude -p --dangerously-skip-permissions --model opus`. Please ensure you are indeed calling it like `claude -p --dangerously-skip-permissions --model opus` and ensure you get both the code review, and the data review, and an additional PI review about the state of the project with Claude and yourself. You must be truthful and honest, and address all the points Claude makes (though keep away from making fakes or mocks). If you need to create fakes or mocks for debugging, delete them afterwards so as to not confuse fakes and mocks for actual results. You must only present results from real model runs as much as possible for our project. It's imperative. Ensure that you are going back and incorporating feedback from claude and yourself as necessary. You must continue this loop until both you and claude agree with each other and give yourselfs both approval without nits. After that you should push your changes up for review. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your @AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the @RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the @RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update @AGENTS.md and ask @CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly. Before you merge the PR, you must wait for my approval. Also at the end I want you to outline any risks we are seeing in this project, are our expectations aligned with how the progress is going? Will our project succeed and achieve the baselines we expect using your best judgement? Absolutely make sure you're reporting results truthfully and honestly. Avoid fake, mocked, or other non genuine results. You should also analyze the results we get for each run and determine if they meet our figures of merit, when you report back, it's crucial to include that analysis (e.g. compression amount, failure rate, latency, etc.). We should be aiming to properly fix things and run proper evaluations. If we do have any expected goals or outcomes (e.g. >= 10x on xyz) and they aren't achieved, then explain why, but do not lie or cheat and use drastically contrived inefficient metrics. It's important to generally be comparing to a standard implementation. Please also report the results on real data runs, it's important that we start gathering as much data on real runs as possible now so it can be reviewed by Claude AND YOU. If you see there are problems or optimizations, don't hesitate to suggest them as we collect more data. Here are more detailed instructions for PR implementation.

### PR Summary

PR‑16 — Statistical Significance & Non‑Inferiority (offline‑safe)

Goal (≤250 LOC total):
Add a tiny stats helper and a one‑shot CLI to produce paper‑grade significance outputs without SciPy (stdlib + numpy only). We report:

Token reduction significance (paired, per‑item % reduction).

Quality preservation (TerseTalk − Free‑form accuracy, CI).

Hybrid non‑inferiority vs LLMLingua (δ = 0.02 absolute accuracy).

Outputs are JSON + human‑readable console lines.

Scope & constraints

No new heavy deps: only numpy (already present).

No pandas, no SciPy; use bootstrap for CIs & one‑sided p-values.

Works on JSONL files emitted by PR‑14 (e.g., tersetalk_baseline.jsonl, freeform.jsonl, llmlingua.jsonl, hybrid_budget_600.jsonl).

Robust to missing fields (tokens_total vs tokens, missing correct → treated as 0.0).

Pairing is by row index (truncate to the common length).

Files (NEW)

tersetalk/statistics.py — small, reusable bootstrap & tests.

scripts/run_significance.py — CLI that loads JSONL, runs tests, writes results.

Implementation

1. tersetalk/statistics.py (NEW, ~90 LOC)

# tersetalk/statistics.py

from **future** import annotations
import numpy as np
from typing import List, Tuple, Dict

def \_rng(seed: int | None = None):
return np.random.default_rng(seed)

def bootstrap*ci_diff(a: List[float], b: List[float],
n_boot: int = 5000, conf: float = 0.95,
paired: bool = True, seed: int | None = None
) -> Tuple[float, float, float]:
"""Bootstrap CI for mean(a - b). Paired by index by default."""
a, b = np.asarray(a, float), np.asarray(b, float)
n = min(len(a), len(b))
a, b = a[:n], b[:n]
if n == 0:
return float("nan"), float("nan"), float("nan")
r = \_rng(seed)
diffs = []
if paired:
idx = np.arange(n)
for * in range(n*boot):
s = r.choice(idx, size=n, replace=True)
diffs.append(np.mean(a[s] - b[s]))
else:
for * in range(n_boot):
s1 = r.choice(a, size=n, replace=True)
s2 = r.choice(b, size=n, replace=True)
diffs.append(np.mean(s1 - s2))
diffs = np.asarray(diffs)
mean = float(np.mean(a - b))
alpha = (1.0 - conf) / 2.0
lo = float(np.quantile(diffs, alpha))
hi = float(np.quantile(diffs, 1 - alpha))
return mean, lo, hi

def percent_reduction(before: List[float], after: List[float]) -> List[float]:
"""Per-item % reduction: (before - after) / max(before, eps)."""
eps = 1e-9
before, after = np.asarray(before, float), np.asarray(after, float)
n = min(len(before), len(after))
if n == 0:
return []
b = before[:n]
a = after[:n]
return ((b - a) / np.maximum(b, eps)).tolist()

def bootstrap*mean_ci(x: List[float], n_boot: int = 5000,
conf: float = 0.95, seed: int | None = None
) -> Tuple[float, float, float]:
"""Bootstrap CI for mean(x)."""
x = np.asarray(x, float)
if len(x) == 0:
return float("nan"), float("nan"), float("nan")
r = \_rng(seed)
means = []
for * in range(n_boot):
s = r.choice(x, size=len(x), replace=True)
means.append(np.mean(s))
means = np.asarray(means)
alpha = (1.0 - conf) / 2.0
return float(np.mean(x)), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))

def one*sided_p_gt_zero(x: List[float], n_boot: int = 10000, seed: int | None = None) -> float:
"""Approximate one-sided p-value that mean(x) <= 0 via bootstrap."""
x = np.asarray(x, float)
if len(x) == 0:
return float("nan")
r = \_rng(seed)
cnt = 0
for * in range(n_boot):
s = r.choice(x, size=len(x), replace=True)
if np.mean(s) <= 0.0:
cnt += 1
return cnt / n_boot

def noninferiority(treatment_acc: List[float], control_acc: List[float],
delta: float = 0.02, conf: float = 0.95,
seed: int | None = None) -> Dict[str, float | bool]:
"""
H0: treat - control < -δ; H1: treat - control >= -δ.
Pass if CI_lower > -δ.
"""
mean, lo, hi = bootstrap_ci_diff(treatment_acc, control_acc, conf=conf, paired=True, seed=seed)
return {
"mean_difference": float(mean),
"ci_lower": float(lo),
"ci_upper": float(hi),
"delta": float(delta),
"is_noninferior": bool(lo > -delta),
}

2. scripts/run_significance.py (NEW, ~130–150 LOC)

# scripts/run_significance.py

import json, sys
from pathlib import Path
from typing import Tuple, List
import click
import numpy as np
from tersetalk.statistics import (
percent_reduction, bootstrap_mean_ci,
bootstrap_ci_diff, one_sided_p_gt_zero, noninferiority,
)

def load_tokens_acc(path: Path) -> Tuple[List[float], List[float]]:
toks, acc = [], []
if not path.exists():
return toks, acc
with path.open() as f:
for line in f:
line = line.strip()
if not line:
continue
try:
r = json.loads(line)
except Exception:
continue
if r.get("status") == "error":
continue
tok = r.get("tokens_total", r.get("tokens", None))
if tok is not None:
toks.append(float(tok)) # default 0.0 if unknown correctness
acc.append(1.0 if r.get("correct") else 0.0)
return toks, acc

@click.command()
@click.option("--results-dir", type=Path, required=True)
@click.option("--terse-file", default="tersetalk_baseline.jsonl")
@click.option("--free-file", default="freeform.jsonl")
@click.option("--ll2-file", default="llmlingua.jsonl")
@click.option("--hybrid-file",default="hybrid_budget_600.jsonl")
@click.option("--confidence", default=0.95, show_default=True)
@click.option("--boots", default=5000, show_default=True)
@click.option("--out", default="significance_tests.json", show_default=True)
def main(results_dir, terse_file, free_file, ll2_file, hybrid_file, confidence, boots, out):
"""Run significance tests for token reduction, quality, and non-inferiority."""
terse_t, terse_a = load_tokens_acc(results_dir / terse_file)
free_t, free_a = load_tokens_acc(results_dir / free_file)
ll2_t, ll2_a = load_tokens_acc(results_dir / ll2_file)
hyb_t, hyb_a = load_tokens_acc(results_dir / hybrid_file)

    report = {}

    # 1) Token reduction: per-item % reduction (freeform → tersetalk), paired
    pct = percent_reduction(free_t, terse_t)
    mean_pct, lo_pct, hi_pct = bootstrap_mean_ci(pct, n_boot=boots, conf=confidence)
    p_one_sided = one_sided_p_gt_zero(pct, n_boot=max(boots, 5000))
    report["token_reduction"] = {
        "mean_reduction_pct": float(mean_pct),
        "ci_lower": float(lo_pct),
        "ci_upper": float(hi_pct),
        "p_one_sided_gt0": float(p_one_sided),
        "n": len(pct)
    }

    # 2) Quality preservation: accuracy(terse) - accuracy(freeform)
    mean_q, lo_q, hi_q = bootstrap_ci_diff(terse_a, free_a, n_boot=boots, conf=confidence, paired=True)
    report["quality_preservation"] = {
        "mean_diff": float(mean_q),
        "ci_lower": float(lo_q),
        "ci_upper": float(hi_q),
        "n": int(min(len(terse_a), len(free_a)))
    }

    # 3) Hybrid non-inferiority vs LLMLingua
    report["hybrid_noninferiority"] = noninferiority(hyb_a, ll2_a, delta=0.02, conf=confidence)

    # Print concise summary
    print("\n" + "="*60)
    print("SIGNIFICANCE RESULTS")
    print("="*60)
    tr = report["token_reduction"]
    print(f"Token Reduction (free→terse): {tr['mean_reduction_pct']:.1%} "
          f"[{tr['ci_lower']:.1%}, {tr['ci_upper']:.1%}]  "
          f"p(one‑sided>0)={tr['p_one_sided_gt0']:.4f}  n={tr['n']}")
    qp = report["quality_preservation"]
    print(f"Quality Δ (terse‑free): {qp['mean_diff']:.3f} "
          f"[{qp['ci_lower']:.3f}, {qp['ci_upper']:.3f}]  n={qp['n']}")
    hi = report["hybrid_noninferiority"]
    print(f"Hybrid vs LLMLingua Non‑Inferiority (δ=0.02): "
          f"{'PASS' if hi['is_noninferior'] else 'FAIL'}; "
          f"Δ={hi['mean_difference']:.3f}, CI=[{hi['ci_lower']:.3f},{hi['ci_upper']:.3f}]")

    # Save full JSON
    out_path = results_dir / out
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved: {out_path}")

if **name** == "**main**":
main()

Tests (NEW, tiny)

tests/test_statistics_smoke.py (~35 LOC)

from pathlib import Path
import subprocess, sys, json

def test_significance_smoke(tmp_path: Path):
d = tmp_path / "runs"
d.mkdir() # Minimal aligned rows: freeform worse tokens, same accuracy; hybrid ~ ll2
(d / "freeform.jsonl").write_text('{"tokens":100,"correct":true,"status":"success"}\n')
(d / "tersetalk_baseline.jsonl").write_text('{"tokens":60,"correct":true,"status":"success"}\n')
(d / "llmlingua.jsonl").write_text('{"tokens":70,"correct":true,"status":"success"}\n')
(d / "hybrid_budget_600.jsonl").write_text('{"tokens":65,"correct":true,"status":"success"}\n')

    cmd = [sys.executable, "scripts/run_significance.py", "--results-dir", str(d), "--boots", "2000"]
    subprocess.run(cmd, check=True)
    out = json.loads((d / "significance_tests.json").read_text())
    assert "token_reduction" in out and out["token_reduction"]["n"] == 1
    assert "hybrid_noninferiority" in out

Definition of Done

python scripts/run_significance.py --results-dir results/evaluation/TASK/seed_42:

Prints three lines (token reduction, quality Δ, non‑inferiority PASS/FAIL).

Writes significance_tests.json with:

token_reduction: mean_reduction_pct, ci_lower, ci_upper, p_one_sided_gt0, n

quality_preservation: mean_diff, ci_lower, ci_upper, n

hybrid_noninferiority: mean_difference, ci_lower, ci_upper, delta, is_noninferior

Handles missing files gracefully (n=0 → NaNs, still emits JSON).

Keeps total new code ≤250 LOC (library + script + smoke test, comments minimal).

No SciPy; offline‑safe.

Out of scope

Multiple‑comparison corrections, Bayesian tests, or stratified analysis by task subsets.

Fancy tables/plots (handled by PR‑15).

Branch name: pr-16-significance
Commit msg: PR-16: Bootstrap significance + non-inferiority CLI (numpy-only, offline-safe)
