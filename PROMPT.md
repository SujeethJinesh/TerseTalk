Read through your AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the AGENTS.md. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update AGENTS.md and ask CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly. Before you merge the PR, you must wait for my approval. Also at the end I want you to outline any risks we are seeing in this project, are our expectations aligned with how the progress is going? Will our project succeed and achieve the baselines we expect using your best judgement? Absolutely make sure you're reporting results truthfully and honestly. Avoid fake, mocked, or other non genuine results. We should be aiming to properly fix things and run proper evaluations. Here are more detailed instructions for PR implementation.

### PR Summary

PR‑H3 — noninferiority.py + smoke CLI

Role: You are a senior engineer implementing PR‑H3 immediately after PR‑H2 merged.

Goal (from spec):
Provide a simple analysis helper that, given per‑example correctness arrays for Hybrid and LLMLingua, runs a one‑sided non‑inferiority test with margin δ (default 0.02) and confidence 95% using a paired bootstrap over example indices. Return a decision flag and CI bounds. Ship a small CLI and tests.

Decision rule (one‑sided NI):
Let d = acc(Hybrid) − acc(LLML). We claim non‑inferiority if the 95% lower bound of d is greater than −δ. (Null: d ≤ −δ; Reject null if LB₀.₉₅(d) > −δ.)

Constraints & expectations

Stdlib only (no numpy/scipy).

Deterministic given the same seed and inputs.

Paired bootstrap over example indices (resample N indices with replacement).

Fast enough for tests: default n_boot=1000, tests may use n_boot=400.

Input validation: same length, non‑empty, values in {0,1} / {False,True}.

Provide both one‑sided LB (95%) and a two‑sided 95% CI for visibility (percentiles 2.5%/97.5%).

Keep previous tests green.

Deliverables

tersetalk/noninferiority.py — library with the bootstrap and NI decision.

scripts/noninferiority_smoke.py — CLI to generate synthetic outcomes or (optionally) read two JSON arrays and print the NI report.

tests/test_noninferiority.py — deterministic, offline tests on synthetic data.

Update tersetalk/**init**.py to export noninferiority.

Create/Update the following files exactly

1. tersetalk/noninferiority.py (new)
   from **future** import annotations

import json
import random
from dataclasses import dataclass, asdict
from typing import Iterable, List, Tuple, Dict

def \_to01(x) -> int:
if isinstance(x, bool):
return 1 if x else 0
if isinstance(x, (int,)):
if x in (0, 1):
return int(x)
raise ValueError("Outcomes must be 0/1 or bool.")

def \_validate_inputs(hybrid: Iterable, lingua: Iterable) -> Tuple[List[int], List[int]]:
h = [ _to01(v) for v in list(hybrid) ]
l = [ _to01(v) for v in list(lingua) ]
if len(h) == 0 or len(l) == 0 or len(h) != len(l):
raise ValueError("Hybrid and LLMLingua arrays must be non-empty and equal-length.")
return h, l

def \_acc(arr: List[int]) -> float:
return sum(arr) / len(arr) if arr else 0.0

def \_acc_subset(arr: List[int], idxs: List[int]) -> float:
if not idxs:
return 0.0
s = 0
for i in idxs:
s += arr[i]
return s / len(idxs)

def \_percentile(sorted_vals: List[float], q: float) -> float:
"""
Simple percentile on a sorted list; q in [0,1]. Linear interpolation.
"""
if not sorted_vals:
return 0.0
n = len(sorted_vals)
if n == 1:
return sorted_vals[0]
pos = q _ (n - 1)
lo = int(pos)
hi = min(lo + 1, n - 1)
frac = pos - lo
return sorted_vals[lo] _ (1 - frac) + sorted_vals[hi] \* frac

@dataclass
class NIReport:
n: int
alpha: float
delta: float
acc_hybrid: float
acc_llml: float
diff: float
lb_one_sided_95: float
ci2_lower_95: float
ci2_upper_95: float
method: str
n_boot: int
seed: int
noninferior: bool

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["decision"] = "non-inferior" if self.noninferior else "fail-to-demonstrate"
        return d

def paired_bootstrap_diff(
hybrid: Iterable,
lingua: Iterable,
n_boot: int = 1000,
alpha: float = 0.05,
seed: int = 0
) -> NIReport:
"""
Paired bootstrap of the accuracy difference d = acc(H) - acc(L).
Returns an NIReport with two-sided 95% CI and the one-sided 95% lower bound.
"""
h, l = \_validate_inputs(hybrid, lingua)
n = len(h)
rng = random.Random(seed)

    base_h = _acc(h)
    base_l = _acc(l)
    base_d = base_h - base_l

    diffs: List[float] = []
    idxs = list(range(n))
    for _ in range(max(1, n_boot)):
        # resample paired indices with replacement
        sample = [ idxs[rng.randrange(n)] for _ in range(n) ]
        dh = _acc_subset(h, sample)
        dl = _acc_subset(l, sample)
        diffs.append(dh - dl)

    diffs.sort()
    # two-sided 95% CI (2.5%, 97.5%)
    ci2_lo = _percentile(diffs, 0.025)
    ci2_hi = _percentile(diffs, 0.975)
    # one-sided lower 95% bound (5%)
    lb = _percentile(diffs, alpha)

    return NIReport(
        n=n,
        alpha=alpha,
        delta=0.02,              # default margin; caller may override in decision step
        acc_hybrid=base_h,
        acc_llml=base_l,
        diff=base_d,
        lb_one_sided_95=lb,
        ci2_lower_95=ci2_lo,
        ci2_upper_95=ci2_hi,
        method="paired-bootstrap",
        n_boot=n_boot,
        seed=seed,
        noninferior=False,       # filled by noninferiority_test
    )

def noninferiority_test(
hybrid: Iterable,
lingua: Iterable,
delta: float = 0.02,
alpha: float = 0.05,
n_boot: int = 1000,
seed: int = 0
) -> Dict:
"""
One-sided non-inferiority: H0: d <= -delta; HA: d > -delta.
We declare non-inferiority if lb_one_sided_95(d) > -delta.
Returns a JSON-serializable dict (NIReport + decision).
"""
rep = paired_bootstrap_diff(hybrid, lingua, n_boot=n_boot, alpha=alpha, seed=seed)
decision = rep.lb_one_sided_95 > (-delta)
rep.delta = float(delta)
rep.noninferior = bool(decision)
return rep.to_dict()

2. scripts/noninferiority_smoke.py (new)
   from **future** import annotations

import argparse
import json
import sys
import random
from typing import List

from tersetalk.noninferiority import noninferiority_test

def _gen_bernoulli(n: int, p: float, seed: int) -> List[int]:
rng = random.Random(seed)
return [1 if rng.random() < p else 0 for _ in range(n)]

def main():
ap = argparse.ArgumentParser(description="PR-H3: One-sided non-inferiority smoke")
ap.add_argument("--n", type=int, default=500, help="Number of examples")
ap.add_argument("--p-hybrid", type=float, default=0.79, help="Hybrid accuracy (synthetic)")
ap.add_argument("--p-ll", type=float, default=0.78, help="LLMLingua accuracy (synthetic)")
ap.add_argument("--delta", type=float, default=0.02, help="Non-inferiority margin (absolute)")
ap.add_argument("--alpha", type=float, default=0.05, help="One-sided alpha")
ap.add_argument("--seed", type=int, default=0, help="Random seed")
ap.add_argument("--n-boot", type=int, default=800, help="Bootstrap replicates")
ap.add_argument("--from-json", type=str, default="", help="Path to JSON file with {'hybrid':[0/1...],'ll':[0/1...]}")
args = ap.parse_args()

    if args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        hybrid = payload["hybrid"]
        ll = payload["ll"]
    else:
        # Deterministic paired generation: use different seeds to introduce slight correlation if desired
        hybrid = _gen_bernoulli(args.n, args.p_hybrid, args.seed)
        ll = _gen_bernoulli(args.n, args.p_ll, args.seed + 1)

    report = noninferiority_test(
        hybrid, ll, delta=args.delta, alpha=args.alpha, n_boot=args.n_boot, seed=args.seed
    )
    print(json.dumps(report, indent=2))

if **name** == "**main**":
main()

3. tests/test_noninferiority.py (new)
   from **future** import annotations

import json
import random

from tersetalk.noninferiority import noninferiority_test, paired_bootstrap_diff

def _gen(n, p, seed):
rng = random.Random(seed)
return [1 if rng.random() < p else 0 for _ in range(n)]

def test_schema_and_determinism_small_bootstrap():
n = 300
seed = 7
h = \_gen(n, 0.79, seed)
l = \_gen(n, 0.78, seed + 1)

    rep1 = noninferiority_test(h, l, delta=0.02, n_boot=400, seed=seed)
    rep2 = noninferiority_test(h, l, delta=0.02, n_boot=400, seed=seed)
    assert rep1 == rep2
    # schema fields present
    for k in ["n","alpha","delta","acc_hybrid","acc_llml","diff",
              "lb_one_sided_95","ci2_lower_95","ci2_upper_95",
              "method","n_boot","seed","noninferior","decision"]:
        assert k in rep1

def test_noninferiority_pass_case(): # Hybrid slightly better than LL; should pass NI at delta=0.02
n = 500
seed = 11
h = \_gen(n, 0.80, seed)
l = \_gen(n, 0.78, seed + 1)
rep = noninferiority_test(h, l, delta=0.02, n_boot=400, seed=seed)
assert rep["noninferior"] is True
assert rep["lb_one_sided_95"] > -0.02

def test_noninferiority_fail_case(): # Hybrid meaningfully worse; should fail NI
n = 400
seed = 21
h = \_gen(n, 0.73, seed)
l = \_gen(n, 0.78, seed + 1)
rep = noninferiority_test(h, l, delta=0.02, n_boot=400, seed=seed)
assert rep["noninferior"] is False # Typically LB well below -delta
assert rep["lb_one_sided_95"] <= -0.02

def test_paired_bootstrap_reporting_consistency():
n = 200
seed = 3
h = \_gen(n, 0.76, seed)
l = \_gen(n, 0.76, seed + 1)
rpt = paired_bootstrap_diff(h, l, n_boot=300, seed=seed)
assert abs(rpt.acc_hybrid - rpt.acc_llml) <= 0.1 # sanity
assert rpt.ci2_lower_95 <= rpt.ci2_upper_95

4. Update tersetalk/**init**.py (replace file)
   from .\_version import **version**

**all** = [
"__version__",
"reproducibility",
"protocol_jsonl",
"structured",
"memory",
"summarization",
"hybrid_gate",
"noninferiority",
]

What to run (and what to paste as evidence in the PR)

Install (unchanged)

make install

Run tests

make test

Smoke: NI on synthetic accuracies (should PASS)

python scripts/noninferiority_smoke.py --n 500 --p-hybrid 0.80 --p-ll 0.78 --delta 0.02 --seed 42 --n-boot 800

Smoke: NI on synthetic accuracies (should FAIL)

python scripts/noninferiority_smoke.py --n 400 --p-hybrid 0.73 --p-ll 0.78 --delta 0.02 --seed 7 --n-boot 600

Acceptance evidence to paste in the PR description:

✅ pytest summary (all green).

✅ Two smoke JSON snippets:

A pass with "noninferior": true and "lb_one_sided_95" > -0.02.

A fail with "noninferior": false and "lb_one_sided_95" <= -0.02.

✅ Verification that the report includes fields: n, alpha, delta, acc_hybrid, acc_llml, diff, lb_one_sided_95, ci2_lower_95, ci2_upper_95, method, n_boot, seed, noninferior, decision.

Commit message
PR-H3: One-sided non-inferiority analysis (paired bootstrap)

- Add tersetalk/noninferiority.py:
  - Paired bootstrap for d = acc(H) - acc(L)
  - One-sided lower 95% bound and two-sided 95% CI
  - noninferiority_test() returns JSON-serializable report and decision
- Add scripts/noninferiority_smoke.py for synthetic or JSON-supplied outcomes
- Add tests covering determinism, pass/fail cases, and schema
- Export 'noninferiority' in package **init**
- Stdlib-only, offline, deterministic with seed
