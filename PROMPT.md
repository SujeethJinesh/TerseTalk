Read through your @AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated @RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the @RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the @AGENTS.md. To ask claude code for a review, you can do something like `echo "Please ensure you understand @RESEARCH_PROPOSAL.md. I am implementing PR-X. Please review the following files I created: @README.md, @xxx. Ensure that my implementation is properly aligned with the goals of the project. Also here is some additional data on the runs I gathered, please critique it as if you're a senior data scientist and ensure that I'm not cheating on the results, lying, or misrepresenting things" | claude -p --dangerously-skip-permissions --model opus`. Please ensure you are indeed calling it like `claude -p --dangerously-skip-permissions --model opus` and ensure you get both the code review, and the data review, and an additional PI review about the state of the project with Claude and yourself. You must be truthful and honest, and address all the points Claude makes (though keep away from making fakes or mocks). If you need to create fakes or mocks for debugging, delete them afterwards so as to not confuse fakes and mocks for actual results. You must only present results from real model runs as much as possible for our project. It's imperative. Ensure that you are going back and incorporating feedback from claude and yourself as necessary. You must continue this loop until both you and claude agree with each other and give yourselfs both approval without nits. After that you should push your changes up for review. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your @AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the @RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the @RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update @AGENTS.md and ask @CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly. Before you merge the PR, you must wait for my approval. Also at the end I want you to outline any risks we are seeing in this project, are our expectations aligned with how the progress is going? Will our project succeed and achieve the baselines we expect using your best judgement? Absolutely make sure you're reporting results truthfully and honestly. Avoid fake, mocked, or other non genuine results. You should also analyze the results we get for each run and determine if they meet our figures of merit, when you report back, it's crucial to include that analysis (e.g. compression amount, failure rate, latency, etc.). We should be aiming to properly fix things and run proper evaluations. If we do have any expected goals or outcomes (e.g. >= 10x on xyz) and they aren't achieved, then explain why, but do not lie or cheat and use drastically contrived inefficient metrics. It's important to generally be comparing to a standard implementation. Please also report the results on real data runs, it's important that we start gathering as much data on real runs as possible now so it can be reviewed by Claude AND YOU. If you see there are problems or optimizations, don't hesitate to suggest them as we collect more data. Here are more detailed instructions for PR implementation.

### PR Summary

PR‑15 — Analysis Polish & Metrics Provenance (plots + Pareto + deterministic ablations)

Goal (≤250 LOC total):
Upgrade the analysis layer so we can (a) compute & plot Pareto frontiers robustly, (b) render deterministic cap ablations, and (c) enrich summary.json with provenance (tokenizer/SP method, timestamp, git hash). Keep it offline‑safe, stdlib+numpy+matplotlib only (no pandas). Headless by default.

Why this now

Pairs with PR‑14’s run_evaluation.py outputs to make paper‑grade figures.

Provenance ensures results are audit‑ready (artifact badge friendly).

Deterministic ordering avoids “figure churn” in CI.

Scope & constraints

Files touched

scripts/analyze_v05.py (UPDATE, main changes)

tests/test_analyze_min.py (NEW, tiny smoke)

No new heavy deps: stdlib (+ json, csv, pathlib, sys, subprocess) + numpy, matplotlib (Agg).

Works on a single run dir (with \*.jsonl + summary.json) or a parent tree of runs (recursively discovers).

Deterministic: fixed ablation order aggressive → baseline → relaxed → very_relaxed.

Graceful when missing fields: warn to stderr, skip plot segment, still emit files.

Required implementation

1. Update scripts/analyze_v05.py

Add a small CLI with three switches and defaults that are CI‑safe:

# scripts/analyze_v05.py (UPDATE)

# Headless plots for CI

import matplotlib
matplotlib.use("Agg")

import json, csv, sys, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import click

CAP_ORDER = ["aggressive","baseline","relaxed","very_relaxed"]

@click.command()
@click.option('--indir', type=Path, required=True, help='Root dir with results')
@click.option('--outdir', type=Path, required=True, help='Dir to write figures/CSVs')
@click.option('--enrich-provenance/--no-enrich-provenance', default=True)
@click.option('--pareto/--no-pareto', default=True)
@click.option('--ablation/--no-ablation', default=True)
def main(indir, outdir, enrich_provenance, pareto, ablation):
outdir.mkdir(parents=True, exist_ok=True)
runs = discover_runs(indir)
sys.stderr.write(f"[analyze] discovered {len(runs)} run(s) under {indir}\n")
if not runs:
sys.stderr.write("[analyze] no runs found; exiting\n")
return

    # Prefer the most recent run folder if multiple; still allow tree aggregation for Pareto table
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    sys.stderr.write(f"[analyze] using latest run for ablation: {latest}\n")

    if pareto:
        pts = collect_points_from_tree(runs)
        save_pareto_csv(pts, outdir / "pareto_points.csv")
        plot_pareto(pts, outdir / "pareto_frontier.pdf")

    if ablation:
        rows = collect_cap_ablation(latest)
        if rows:
            save_ablation_csv(rows, outdir / "ablation_caps.csv")
            plot_ablation(rows, outdir / "ablation_caps.pdf")
        else:
            sys.stderr.write("[analyze] no cap ablation files found in latest run\n")

    if enrich_provenance:
        # Find a summary.json near the latest run
        summary_path = latest / "summary.json"
        if summary_path.exists():
            enrich_summary_with_provenance(summary_path)
        else:
            sys.stderr.write(f"[analyze] summary.json missing at {summary_path}\n")

def discover_runs(root: Path):
"""Return list of directories that contain summary.json."""
return sorted({p.parent for p in root.rglob("summary.json")})

def list*system_files(run_dir: Path):
"""Return mapping {name -> path} for all *.jsonl files in a run dir."""
return {p.stem: p for p in run*dir.glob("*.jsonl")}

def load_rows(jsonl_path: Path):
with jsonl_path.open() as f:
for line in f:
line = line.strip()
if not line: continue
try:
yield json.loads(line)
except Exception:
continue

def summarize_file(jsonl_path: Path):
"""Compute avg_tokens, accuracy, overflow_rate (if available) for one system file."""
toks, accs, overflows = [], [], []
for r in load_rows(jsonl_path):
if r.get('status') == 'error':
continue
toks.append(r.get('tokens_total') or r.get('tokens') or 0)
accs.append(1.0 if r.get('correct') else 0.0) # Overflow can be per-example or absent
if 'overflow_count' in r:
overflows.append(r['overflow_count'])
if not toks:
return None
avg_tokens = float(np.mean(toks))
accuracy = float(np.mean(accs))
overflow_rate = float(np.mean(overflows)) if overflows else float('nan')
return {'avg_tokens': avg_tokens, 'accuracy': accuracy, 'overflow_rate': overflow_rate}

def collect_points_from_tree(runs):
"""Aggregate all systems across all discovered runs into Pareto points."""
points = []
skipped = 0
for run in runs:
for name, path in list_system_files(run).items():
s = summarize_file(path)
if not s:
skipped += 1
continue
points.append({'system': name, 'tokens': s['avg_tokens'], 'accuracy': s['accuracy']})
sys.stderr.write(f"[analyze] pareto: {len(points)} point(s); skipped {skipped} file(s)\n") # Dedup by (system, tokens, accuracy) keeping best (lowest tokens at highest accuracy)
uniq = {}
for p in points:
key = (p['system'], round(p['tokens'], 4), round(p['accuracy'], 4))
uniq[key] = p
return list(uniq.values())

def pareto_frontier(points):
"""
Min tokens, max accuracy.
Mark each point with 'is_pareto' True/False.
""" # Sort for stable behavior
pts = sorted(points, key=lambda x: (x['tokens'], -x['accuracy']))
for i, p in enumerate(pts):
p['is_pareto'] = True
for j, q in enumerate(pts):
if j == i: continue # q dominates p if: tokens <= and accuracy >= with at least one strict
if (q['tokens'] <= p['tokens'] and q['accuracy'] >= p['accuracy']) and \
 (q['tokens'] < p['tokens'] or q['accuracy'] > p['accuracy']):
p['is_pareto'] = False
break
return pts

def save_pareto_csv(points, path: Path):
pts = pareto_frontier(points)
with path.open('w', newline='') as f:
w = csv.DictWriter(f, fieldnames=['system','tokens','accuracy','is_pareto'])
w.writeheader()
for p in pts: w.writerow(p)

def plot_pareto(points, pdf_path: Path):
pts = pareto_frontier(points)
fig, ax = plt.subplots(figsize=(8,5))
for p in pts:
m = 'o' if p['is_pareto'] else 'x'
ax.scatter(p['tokens'], p['accuracy'], marker=m) # Label frontier only to reduce clutter
if p['is_pareto']:
ax.annotate(p['system'], (p['tokens'], p['accuracy']), xytext=(3,3), textcoords='offset points', fontsize=8) # Connect frontier curve (sorted by tokens)
front = [p for p in pts if p['is_pareto']]
front.sort(key=lambda x: x['tokens'])
if front:
ax.plot([p['tokens'] for p in front], [p['accuracy'] for p in front], linestyle='--', linewidth=1)
ax.set_xlabel('Total Tokens (lower is better)')
ax.set_ylabel('Task Accuracy (higher is better)')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(pdf_path)

def collect*cap_ablation(run_dir: Path):
"""Return deterministic rows for tersetalk*{cap}.jsonl in CAP*ORDER."""
rows = []
for cap in CAP_ORDER:
path = run_dir / f"tersetalk*{cap}.jsonl"
if not path.exists():
sys.stderr.write(f"[analyze] missing {path.name}; skipping\n")
continue
s = summarize_file(path)
if not s:
continue
if np.isnan(s['overflow_rate']):
sys.stderr.write(f"[analyze] overflow_rate NaN for {cap}\n")
rows.append({'name': cap,
'avg_cap': {'aggressive':20,'baseline':30,'relaxed':50,'very_relaxed':100}[cap],
'tokens': s['avg_tokens'],
'accuracy': s['accuracy'],
'overflow_rate': s['overflow_rate']})
return rows

def save_ablation_csv(rows, path: Path):
with path.open('w', newline='') as f:
w = csv.DictWriter(f, fieldnames=['name','avg_cap','tokens','accuracy','overflow_rate'])
w.writeheader()
for r in rows: w.writerow(r)

def plot_ablation(rows, pdf_path: Path): # rows already deterministic order
xs_cap = [r['avg_cap'] for r in rows]
ys_tok = [r['tokens'] for r in rows]
xs_over = [r['overflow_rate'] for r in rows if not np.isnan(r['overflow_rate'])]
ys_acc = [r['accuracy'] for r in rows if not np.isnan(r['overflow_rate'])]
fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(xs_cap, ys_tok, 'o-')
ax1.set_xlabel('Average Cap Size (tokens)')
ax1.set_ylabel('Total Tokens Used')
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(pdf_path) # single‑page: keep concise

def enrich_summary_with_provenance(summary_path: Path):
summary = json.loads(summary_path.read_text())
for name, stats in summary.items():
stats['tokens_method'] = 'tiktoken' if tiktoken_available() else 'heuristic'
stats['sp_method'] = 'bertscore' if bertscore_available() else 'jaccard'
stats['timestamp'] = datetime.now().isoformat()
stats['version'] = get_repo_version()
summary_path.write_text(json.dumps(summary, indent=2))
sys.stderr.write(f"[analyze] enriched provenance in {summary_path}\n")

def tiktoken_available():
try:
import tiktoken # noqa: F401
return True
except Exception:
return False

def bertscore_available():
try:
import bert_score # noqa: F401
return True
except Exception:
return False

def get_repo_version():
try:
out = subprocess.check_output(['git','rev-parse','--short','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
return out or "unknown"
except Exception:
return "unknown"

if **name** == "**main**":
main()

Notes:

Reads any run layout produced by PR‑14 (e.g., hybrid*budget_600.jsonl, freeform.jsonl, tersetalk*{cap}.jsonl).

Writes pareto_points.csv, pareto_frontier.pdf, ablation_caps.{csv,pdf}.

Logs discovered run count + missing files to stderr.

2. Tiny smoke test tests/test_analyze_min.py (NEW)

# tests/test_analyze_min.py

import sys, subprocess, pathlib

def test_analyze_smoke(tmp_path): # Create a tiny faux run: two systems + summary.json
run = tmp_path / "results" / "evaluation" / "hotpotqa" / "2025-01-01-00-00-00"
run.mkdir(parents=True)
(run / "freeform.jsonl").write_text('{"tokens": 100, "correct": true, "status":"success"}\n')
(run / "tersetalk_baseline.jsonl").write_text('{"tokens": 60, "correct": true, "status":"success", "overflow_count":0}\n')
(run / "summary.json").write_text('{"freeform":{"n_total":1,"n_successful":1,"avg_tokens":100,"accuracy":1.0,"compliance_rate":1.0}}')
outdir = tmp_path / "figs"

    cmd = [sys.executable, "scripts/analyze_v05.py",
           "--indir", str(tmp_path / "results"),
           "--outdir", str(outdir)]
    subprocess.run(cmd, check=True)

    assert (outdir / "pareto_points.csv").exists()
    assert (outdir / "pareto_frontier.pdf").exists()

Definition of Done (acceptance)

python scripts/analyze_v05.py --indir results/evaluation --outdir figures:

Prints discovered run count and latest run used to stderr.

Emits:

figures/pareto_points.csv (cols: system,tokens,accuracy,is_pareto)

figures/pareto_frontier.pdf

If cap files present: figures/ablation_caps.csv, figures/ablation_caps.pdf

If all overflow_rate are NaN → warns to stderr and still saves tokens plot.

--no-pareto or --no-ablation skips corresponding artifacts.

--enrich-provenance updates a detected summary.json in latest run with:

tokens_method, sp_method, timestamp, version (git short hash or "unknown").

CI/headless safe (Agg backend), no pandas, no network required for smoke.

Out of scope / non‑goals

Significance tests (handled in PR‑16).

Multi‑page, styled figures (keep simple and legible).

Parsing latency distributions (that’s PR‑12D optional).

Branch name suggestion: pr-15-analysis-provenance
Commit message: PR-15: Pareto + deterministic cap ablations and provenance; headless plots; CSV artifacts
