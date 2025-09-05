Read through your AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the AGENTS.md. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update AGENTS.md and ask CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly. Before you merge the PR, you must wait for my approval. Also at the end I want you to outline any risks we are seeing in this project, are our expectations aligned with how the progress is going? Will our project succeed and achieve the baselines we expect using your best judgement? Here are more detailed instructions for PR implementation.

### PR Summary

PR‑H2 — calibrate_caps.py + library

Role: You are a senior engineer implementing PR‑H2 right after PR‑H1.
Goal (from spec): Sweep {caps, summarizer, deref_policy, gate_on/off, token_budget} on a 50‑example shard; select the best combo; write it to configs/calibration.yaml.
Constraints:

Offline, stdlib‑only (llmlingua optional but not required).

Deterministic given the same seed and inputs.

Do not implement deref behavior yet—only treat deref_policy as a recorded parameter.

Keep prior PR tests green.

What “best” means (explicit policy for this PR):
We minimize avg_est_tokens (estimated outgoing tokens per example) subject to avg_density ≥ density_min (default 0.75). If no candidate meets the threshold, choose the one with highest avg_density, breaking ties by lower avg_est_tokens.

Scoring per example (offline & cheap):

Build a Manager JSONL (synthetic) → validate & overflow with the selected caps and summarizer.

Convert validated JSONL to a free‑form probe via jsonl_to_prose.

If gate_on=True, run the Hybrid Gate (PR‑H1) with token_budget on the validated JSONL vs the free‑form probe.

If route = "tersetalk" → tokens = estimate_tokens(validated_jsonl).

If route = "freeform_llmlingua" → tokens = returned "ll2" projection (if any); (by gate design this only happens when it fits budget).

Record density from validator stats, plus routed path.

Aggregate across examples: avg_est_tokens, avg_density, routed_freeform_frac, avg_overflow_rate.

Deliverables

tersetalk/calibration.py — library with the sweep + selection logic (deterministic).

scripts/calibrate_caps.py — CLI wrapper that prints a compact JSON report and writes configs/calibration.yaml (JSON is valid YAML).

Tests: tests/test_calibration.py ensuring determinism, schema correctness, and gate‑aware behavior (using gate’s FAKE env override).

Add configs/ to .gitignore.

Create/Update the following files exactly

1. Update .gitignore (append at the end)

# Calibration outputs

configs/

2. tersetalk/calibration.py (new)
   from **future** import annotations

import json
import math
import random
import string
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from tersetalk.reproducibility import set_global_seed
from tersetalk.protocol_jsonl import JSONLValidator
from tersetalk.summarization import Summarizer
from tersetalk.memory import MemoryStore
from tersetalk.hybrid_gate import GateCfg, gate_choose_protocol, estimate_tokens

# --------------------------

# Synthetic shard generator

# --------------------------

\_LOREM = (
"alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
"omicron pi rho sigma tau upsilon phi chi psi omega"
).split()

def _rand_words(rng: random.Random, lo: int, hi: int) -> str:
n = rng.randint(lo, hi)
return " ".join(rng.choice(\_LOREM) for _ in range(n)).strip()

def \_synth_example(rng: random.Random, idx: int) -> Dict:
"""
Produce a single synthetic Manager task with variability to trigger overflow.
Deterministic for a fixed RNG state.
"""
goal = f"Compare entities and return the earlier or smaller value (case {idx})." # Alternate between date-like and description-like facts
if idx % 3 == 0:
f1 = f"Item A: 200{rng.randint(0,9)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
f2 = f"Item B: 199{rng.randint(0,9)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
else: # Longish facts to force caps
f1 = \_rand_words(rng, 12, 28)
f2 = \_rand_words(rng, 6, 18)

    # Optional third fact to vary density
    facts = [f1, f2]
    if idx % 4 == 0:
        facts.append(_rand_words(rng, 10, 20))

    question = "Which is earlier or smaller? Provide only the answer."
    # Compose JSONL (lenient mix to exercise normalizer)
    lines: List[str] = [
        '["r","M"]',
        json.dumps(["g", goal]),
    ]
    for f in facts:
        # mix array/object forms
        if rng.random() < 0.5:
            lines.append(json.dumps(["f", f]))
        else:
            lines.append(json.dumps({"f": f}))
    # occasional assumptions/plans to change tag mix
    if rng.random() < 0.5:
        lines.append(json.dumps(["u", "Use ISO dates if dates are present."]))
    if rng.random() < 0.4:
        lines.append(json.dumps(["p", "Compare A and B; output one token."]))
    lines.append(json.dumps(["q", "W", question]))
    return {"jsonl": "\n".join(lines)}

def synth_shard(n: int, seed: int) -> List[Dict]:
rng = random.Random(seed)
return [_synth_example(rng, i) for i in range(n)]

# --------------------------

# Grid + scoring

# --------------------------

Caps = Dict[str, int]
SummMethod = Literal["extractive", "llmlingua"]
DerefPolicy = Literal["never", "conditional", "always"] # placeholder for PR-H4

def default_caps_grid() -> List[Caps]:
return [
{"f": 20, "p": 15, "q": 20, "g": 30, "u": 20, "t": 50}, # aggressive
{"f": 30, "p": 20, "q": 30, "g": 30, "u": 20, "t": 50}, # baseline
{"f": 50, "p": 40, "q": 50, "g": 40, "u": 25, "t": 60}, # relaxed
{"f": 100, "p": 80, "q": 100, "g": 60, "u": 30, "t": 80}, # very relaxed
]

@dataclass(frozen=True)
class CalibSpec:
caps: Caps
summarizer: SummMethod
deref_policy: DerefPolicy
gate_enabled: bool
token_budget: int

@dataclass
class CalibMetrics:
avg_est_tokens: float
avg_density: float
avg_overflow_rate: float
routed_freeform_frac: float
n: int

@dataclass
class CalibEval:
spec: CalibSpec
metrics: CalibMetrics

def evaluate_spec_on_shard(shard: List[Dict], spec: CalibSpec, seed: int) -> CalibEval:
"""
Evaluate one calibration spec on a synthetic shard.
Deterministic for fixed (shard, spec, seed).
""" # Ensure deterministic behavior of any randomness in summarizer/validator code paths.
set_global_seed(seed)

    summarizer = Summarizer(method=spec.summarizer)

    densities: List[float] = []
    token_estimates: List[int] = []
    overflow_rates: List[float] = []
    routed_freeform = 0
    gate_cfg = GateCfg(token_budget=spec.token_budget)

    for ex in shard:
        memory = MemoryStore()
        validator = JSONLValidator(caps=spec.caps, memory=memory, summarizer=summarizer)

        # Normalize/overflow with the selected caps/summarizer
        validated_jsonl, stats = validator.validate_and_overflow(ex["jsonl"])

        # Free-form probe
        freeform = validator.jsonl_to_prose(validated_jsonl)

        # Gate decision
        if spec.gate_enabled:
            decision = gate_choose_protocol(validated_jsonl, freeform, gate_cfg)
            route = decision["route"]
            if route == "freeform_llmlingua":
                routed_freeform += 1
                ll2 = decision["est_tokens"].get("ll2")
                # ll2 must exist for the route to be freeform in our gate logic
                token_estimates.append(int(ll2))
            else:
                token_estimates.append(estimate_tokens(validated_jsonl))
        else:
            # No gate: always TerseTalk tokens post-validation
            token_estimates.append(estimate_tokens(validated_jsonl))

        densities.append(float(stats["density"]))
        of_count = stats["overflow"]["count"]
        total_lines = max(1, stats["lines_total"])
        overflow_rates.append(of_count / total_lines)

        # Reset memory between tasks (explicitly, though validator has no shared state)
        memory.reset()

    n = len(shard)
    metrics = CalibMetrics(
        avg_est_tokens=sum(token_estimates) / n if n else 0.0,
        avg_density=sum(densities) / n if n else 0.0,
        avg_overflow_rate=sum(overflow_rates) / n if n else 0.0,
        routed_freeform_frac=(routed_freeform / n) if n else 0.0,
        n=n,
    )
    return CalibEval(spec=spec, metrics=metrics)

def sweep_grid(
n: int,
seed: int,
caps_grid: List[Caps],
summarizers: List[SummMethod],
deref_policies: List[DerefPolicy],
gate_modes: List[bool],
token_budgets: List[int],
density_min: float = 0.75,
) -> Dict:
"""
Run a full sweep over the grid and return a deterministic report dict:
{
"n": int, "seed": int, "density_min": float,
"grid_evaluations": [ { "spec": {...}, "metrics": {...} }, ... ],
"best": { "spec": {...}, "metrics": {...} }
}
"""
shard = synth_shard(n=n, seed=seed)
evals: List[CalibEval] = []

    # Deterministic iteration order
    for caps in caps_grid:
        for sm in summarizers:
            for dp in deref_policies:
                for gate_on in gate_modes:
                    for budget in token_budgets:
                        spec = CalibSpec(
                            caps=caps,
                            summarizer=sm,
                            deref_policy=dp,
                            gate_enabled=gate_on,
                            token_budget=int(budget),
                        )
                        ev = evaluate_spec_on_shard(shard, spec, seed=seed)
                        evals.append(ev)

    # Selection: filter by density_min; pick lowest avg_est_tokens
    def _rank_key(ev: CalibEval) -> Tuple[float, float, float]:
        # Lower tokens better; higher density better; lower freeform frac better (tie-breaker)
        return (ev.metrics.avg_est_tokens, -ev.metrics.avg_density, ev.metrics.routed_freeform_frac)

    feasible = [ev for ev in evals if ev.metrics.avg_density >= density_min]
    chosen: CalibEval
    if feasible:
        feasible.sort(key=_rank_key)
        chosen = feasible[0]
    else:
        # No candidate meets density; pick by highest density then lowest tokens
        evals.sort(key=lambda ev: (-ev.metrics.avg_density, ev.metrics.avg_est_tokens))
        chosen = evals[0]

    # Deterministic JSON-like report (JSON is valid YAML)
    def _ev_to_dict(ev: CalibEval) -> Dict:
        return {"spec": asdict(ev.spec), "metrics": asdict(ev.metrics)}

    report = {
        "n": n,
        "seed": seed,
        "density_min": density_min,
        "grid_evaluations": [_ev_to_dict(ev) for ev in evals],
        "best": _ev_to_dict(chosen),
    }
    return report

def save_calibration_yaml(report: Dict, out_path: str | Path) -> Path:
"""
Write the report as JSON (which is valid YAML 1.2) to out_path.
Returns the Path. Deterministic content (no timestamps).
"""
p = Path(out_path)
p.parent.mkdir(parents=True, exist_ok=True) # JSON subset of YAML => valid .yaml
text = json.dumps(report, indent=2, sort_keys=True)
p.write_text(text, encoding="utf-8")
return p

3. scripts/calibrate_caps.py (new)
   from **future** import annotations

import argparse
import json
import sys
from typing import List

from tersetalk.calibration import (
default_caps_grid,
sweep_grid,
save_calibration_yaml,
)

def \_parse_caps_grid(s: str) -> List[dict]:
try:
val = json.loads(s)
if isinstance(val, list) and all(isinstance(x, dict) for x in val):
return val
except Exception:
pass
raise SystemExit("Error: --caps-grid must be a JSON list of objects, e.g. "
"'[{\"f\":20,\"p\":15,\"q\":20},{\"f\":30,\"p\":20,\"q\":30}]'")

def \_csv_list(s: str) -> List[str]:
return [x.strip() for x in s.split(",") if x.strip()]

def \_csv_ints(s: str) -> List[int]:
try:
return [int(x) for x in _csv_list(s)]
except Exception as e:
raise SystemExit(f"Error parsing integer list: {e}") from e

def main():
ap = argparse.ArgumentParser(description="PR-H2: Calibration sweep for caps/summarizer/gate")
ap.add_argument("--n", type=int, default=50, help="Number of synthetic examples")
ap.add_argument("--seed", type=int, default=0, help="Random seed for determinism")
ap.add_argument("--out", type=str, default="configs/calibration.yaml", help="Output YAML path")
ap.add_argument("--density-min", type=float, default=0.75, help="Min avg density required")
ap.add_argument("--caps-grid", type=str, default="", help="JSON list of caps dicts (overrides defaults)")
ap.add_argument("--summarizers", type=str, default="extractive", help="Comma list among {extractive,llmlingua}")
ap.add_argument("--deref-policies", type=str, default="never,conditional,always", help="Comma list (placeholder)")
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
        summarizers=summarizers,              # type: ignore[arg-type]
        deref_policies=deref_pols,            # type: ignore[arg-type]
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

if **name** == "**main**":
main()

4. tests/test_calibration.py (new)
   from **future** import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(**file**).resolve().parents[1]
PY = sys.executable

def \_run(args: list[str]) -> Tuple[int, str, str]:
p = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True)
return p.returncode, p.stdout, p.stderr

def test_calibration_writes_yaml_and_schema(tmp_path: Path):
outp = tmp_path / "calib.yaml"
code, out, err = \_run([
PY, "scripts/calibrate_caps.py",
"--n", "24",
"--seed", "123",
"--out", str(outp),
"--density-min", "0.6",
"--summarizers", "extractive", # keep tests offline
"--gate-modes", "off,on",
"--token-budgets", "120,240",
])
assert code == 0, err
assert outp.exists() and outp.read_text().strip() # File is JSON (valid YAML) → parse to verify schema
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
PY, "scripts/calibrate_caps.py",
"--n", "30",
"--seed", "7",
"--out", str(out1),
"--density-min", "0.7",
"--summarizers", "extractive",
"--gate-modes", "off,on",
"--token-budgets", "150,300",
]
code, out, err = \_run(cmd)
assert code == 0, err # rerun with identical args to a different file
cmd2 = cmd.copy()
cmd2[-3] = str(out2) # replace outfile
code2, outb, errb = \_run(cmd2)
assert code2 == 0, errb # Deterministic content
assert out1.read_text() == out2.read_text()

def test_gate_affects_routing_with_fake_ll2(tmp_path: Path, monkeypatch):
"""
Use TERSETALK_FAKE_LL2_COMPRESS to ensure some routing to freeform when gate is on.
"""
outp = tmp_path / "calib.yaml"
monkeypatch.setenv("TERSETALK_FAKE_LL2_COMPRESS", "80")
code, out, err = \_run([
PY, "scripts/calibrate_caps.py",
"--n", "20",
"--seed", "42",
"--out", str(outp),
"--density-min", "0.5",
"--summarizers", "extractive",
"--gate-modes", "off,on",
"--token-budgets", "100,120",
])
assert code == 0, err
data = json.loads(outp.read_text()) # At least one evaluated spec with gate enabled should have routed_freeform_frac > 0.0
assert any(
ev["spec"]["gate_enabled"] and ev["metrics"]["routed_freeform_frac"] > 0.0
for ev in data["grid_evaluations"]
)

What to run (and what to paste as evidence in the PR)

Install (unchanged)

make install

Run tests

make test

Calibrate with defaults (50 examples)

python scripts/calibrate_caps.py

Calibrate with custom grid + gate preview (offline, with fake LL2 projection)

export TERSETALK_FAKE_LL2_COMPRESS=90
python scripts/calibrate_caps.py \
 --n 50 --seed 123 --density-min 0.7 \
 --summarizers extractive \
 --gate-modes off,on \
 --token-budgets 400,600,800
unset TERSETALK_FAKE_LL2_COMPRESS

Acceptance evidence to paste in the PR description:

✅ pytest summary (all green).

✅ Sample of the printed JSON summary from calibrate_caps.py showing best_spec, best_metrics, and out_path.

✅ The saved configs/calibration.yaml (JSON-as-YAML) with fields:

n, seed, density_min

grid_evaluations (array of spec+metrics)

best.spec.caps, best.spec.summarizer, best.spec.gate_enabled, best.spec.token_budget

best.metrics.avg_est_tokens, best.metrics.avg_density, best.metrics.routed_freeform_frac

Commit message
PR-H2: Calibration sweep for caps/summarizer/gate → configs/calibration.yaml

- Add tersetalk/calibration.py:

  - Deterministic synthetic shard generator
  - Grid sweep across {caps, summarizer, deref_policy (placeholder), gate on/off, token_budget}
  - Per-spec evaluation using JSONLValidator, Summarizer, MemoryStore, and Hybrid Gate
  - Selection: minimize avg_est_tokens subject to avg_density ≥ density_min (fallback to highest density)
  - save_calibration_yaml() writes JSON (valid YAML) deterministically

- Add scripts/calibrate_caps.py CLI:

  - Configurable grid via flags; prints compact JSON summary and writes configs/calibration.yaml

- Add tests/test_calibration.py:

  - Schema + file creation
  - Determinism (same seed → identical file)
  - Gate effect using TERSETALK_FAKE_LL2_COMPRESS

- Update .gitignore to ignore configs/
