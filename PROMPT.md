Read through your AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the AGENTS.md. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update AGENTS.md and ask CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly. Before you merge the PR, you must wait for my approval. Also at the end I want you to outline any risks we are seeing in this project, are our expectations aligned with how the progress is going? Will our project succeed and achieve the baselines we expect using your best judgement? Here are more detailed instructions for PR implementation.

### PR Summary

PR‑H1 — Hybrid Gate

Role: You are a senior engineer implementing PR‑H1 after PR‑04 merged.

Goal (from spec):
Create a lightweight hybrid gate that, for a single turn, routes to:

"tersetalk" if the estimated tokens for the JSONL message are within budget; else

"freeform_llmlingua" iff a projected LLMLingua compression meets the budget; else

fallback to "tersetalk" (with overflow).

Provide a small CLI to demo decisions and extend the main runner to surface gate info in --dry-run. Keep everything offline‑safe; llmlingua is optional.

Key behaviors & constraints

Stdlib‑only for this PR (llmlingua optional; import‑guard).

Token estimator: reuse the repo’s heuristic (≈ len(text)//4).

Gate returns a JSON‑serializable dict with: route, est_tokens, and a notes field for traceability.

If llmlingua is not installed or errors, treat projection as unknown and choose "tersetalk".

Provide an escape hatch for tests: an env var TERSETALK_FAKE_LL2_COMPRESS=<int> that simulates the projected token count (no llmlingua needed).

Do not implement full pipeline execution.

DoD (Definition of Done)

tersetalk/hybrid_gate.py with:

@dataclass GateCfg(token_budget:int=600, use_ll2_tags: tuple[str,...]=("f","p","q"))

estimate_tokens(text:str)->int

project_ll2_tokens(prompt:str, budget:int)->int|None
Tries llmlingua; honors TERSETALK_FAKE_LL2_COMPRESS; returns None on failure/unavailable.

gate_choose_protocol(manager_jsonl:str, freeform_prompt:str, cfg:GateCfg)->dict

scripts/hybrid_gate_smoke.py: CLI that prints a decision JSON for provided inputs/budget.

Extend scripts/run_v05.py (dry‑run only) with optional flags:

--hybrid/--no-hybrid (default: --no-hybrid)

--token-budget <int> (default 600)

--gate-jsonl-probe <str> and --gate-freeform-probe <str> (optional; if provided, include a gate object in the printed JSON).

Keep existing behavior intact; no execution mode changes.

Tests (tests/test_hybrid_gate.py) covering:

Routes "tersetalk" when JSONL ≤ budget.

Routes "freeform_llmlingua" when JSONL > budget and a simulated llmlingua projection fits.

Falls back to "tersetalk" when projection is unavailable/None.

estimate_tokens sanity.

All tests pass without llmlingua installed and without network.

Create/Update the following files exactly

Keep earlier PRs’ files intact. Only add/modify what’s listed.

1. tersetalk/hybrid_gate.py (new)
   from **future** import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class GateCfg:
"""
Configuration for the per-turn hybrid gate.

    token_budget: maximum allowed tokens for the outgoing message.
    use_ll2_tags: reserved for future fine-grain control of which tags to compress;
                  kept here to match the spec, not used in PR-H1 logic.
    """
    token_budget: int = 600
    use_ll2_tags: tuple[str, ...] = ("f", "p", "q")

def estimate_tokens(text: str) -> int:
"""
Cheap token estimator used repo-wide (~4 chars ≈ 1 token).
Non-negative and monotonic w.r.t len(text).
"""
if not text:
return 0 # round up slightly to be conservative
return max(0, (len(text) + 3) // 4)

def \_fake_ll2_env_projection() -> Optional[int]:
"""
Testing escape hatch: if TERSETALK_FAKE_LL2_COMPRESS is set,
return that integer (simulates projected token count).
"""
val = os.environ.get("TERSETALK_FAKE_LL2_COMPRESS")
if val is None:
return None
try:
return int(val)
except Exception:
return None

def project_ll2_tokens(prompt: str, budget: int) -> Optional[int]:
"""
Try to project the compressed token length using llmlingua (if available).
Returns an integer token count or None if unavailable/error.

    Honors TERSETALK_FAKE_LL2_COMPRESS to enable offline tests without the package.
    """
    fake = _fake_ll2_env_projection()
    if fake is not None:
        return fake

    try:
        from llmlingua import PromptCompressor  # type: ignore
    except Exception:
        return None

    try:
        comp = PromptCompressor()
        # target_token sets a goal, but we just need the returned size estimate.
        res = comp.compress(prompt, target_token=budget)
        # Common keys used by llmlingua returns
        for key in ("compressed_tokens", "target_token", "projected_tokens", "tokens"):
            v = res.get(key)
            if isinstance(v, int) and v >= 0:
                return v
        # Fall back to measuring compressed string if provided
        for k in ("compressed_prompt", "compressed_text"):
            s = res.get(k)
            if isinstance(s, str):
                return estimate_tokens(s)
        return None
    except Exception:
        return None

def gate_choose_protocol(manager_jsonl: str, freeform_prompt: str, cfg: GateCfg) -> Dict:
"""
Decide which path to use for the current turn.

    Strategy:
      1) If JSONL estimate <= budget → "tersetalk".
      2) Else try llmlingua projection on freeform; if <= budget → "freeform_llmlingua".
      3) Else → "tersetalk" (with overflow).

    Returns a dict:
    {
      "route": "tersetalk" | "freeform_llmlingua",
      "est_tokens": {"jsonl": int, "ll2": Optional[int]},
      "notes": str
    }
    """
    t_jsonl = estimate_tokens(manager_jsonl)
    if t_jsonl <= cfg.token_budget:
        return {
            "route": "tersetalk",
            "est_tokens": {"jsonl": t_jsonl, "ll2": None},
            "notes": "JSONL within budget; chose tersetalk."
        }

    t_ll2 = project_ll2_tokens(freeform_prompt, cfg.token_budget)
    if t_ll2 is not None and t_ll2 <= cfg.token_budget:
        return {
            "route": "freeform_llmlingua",
            "est_tokens": {"jsonl": t_jsonl, "ll2": t_ll2},
            "notes": "JSONL over budget; LLMLingua projection fits -> freeform_llmlingua."
        }

    return {
        "route": "tersetalk",
        "est_tokens": {"jsonl": t_jsonl, "ll2": t_ll2},
        "notes": "JSONL over budget; LLMLingua unavailable or still over -> tersetalk (overflow as needed)."
    }

2. scripts/hybrid_gate_smoke.py (new)
   from **future** import annotations

import argparse
import json
from tersetalk.hybrid_gate import GateCfg, gate_choose_protocol

EXAMPLE_JSONL = '["r","M"]\n["g","Compare dates"]\n["f","Event A: 2001-07-16"]\n["f","Event B: 1999-05-02"]\n["q","W","Which is earlier?"]'
EXAMPLE_FREEFORM = "Role: Manager\nGoal: Compare dates\nFacts: Event A: 2001-07-16; Event B: 1999-05-02\nQuestion: Which is earlier?"

def main():
ap = argparse.ArgumentParser(description="PR-H1 hybrid gate smoke tool")
ap.add_argument("--jsonl", default=EXAMPLE_JSONL, help="JSONL probe text")
ap.add_argument("--freeform", default=EXAMPLE_FREEFORM, help="Free-form probe text")
ap.add_argument("--token-budget", type=int, default=600, help="Token budget for gate")
args = ap.parse_args()

    cfg = GateCfg(token_budget=args.token_budget)
    decision = gate_choose_protocol(args.jsonl, args.freeform, cfg)
    print(json.dumps({"cfg": cfg.__dict__, "decision": decision}, indent=2))

if **name** == "**main**":
main()

3. Update scripts/run_v05.py to surface optional gate info (replace the file with this version)

Keep existing CLI options and behavior. Add optional hybrid flags that only affect dry‑run output by including a gate object if probes are provided.

from **future** import annotations

import json
import sys
import click

from tersetalk.\_version import **version**
from tersetalk.reproducibility import set_global_seed

# Optional hybrid import guarded so tests remain offline-safe

try:
from tersetalk.hybrid_gate import GateCfg, gate_choose_protocol
except Exception: # pragma: no cover
GateCfg = None
gate_choose_protocol = None

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--task", type=click.Choice(["hotpotqa", "gsm8k"]), default="hotpotqa", show_default=True, help="Benchmark task to run.")
@click.option("--system", type=click.Choice(["tersetalk", "freeform", "llmlingua"]), default="tersetalk", show_default=True, help="System variant to run.")
@click.option("--n", default=100, show_default=True, help="Number of examples.")
@click.option("--seed", default=0, show_default=True, help="Global random seed.")
@click.option("--caps", default='{"f":30,"p":20,"q":30}', show_default=True, help="Soft caps JSON for tags.")
@click.option("--model", default="mistral", show_default=True, help="Model name (placeholder).")
@click.option("--out", default="results", show_default=True, help="Output directory.")
@click.option("--dry-run/--execute", default=True, show_default=True, help="Dry-run prints parsed config JSON and exits 0.")

# --- Hybrid gate optional probes (dry-run only) ---

@click.option("--hybrid/--no-hybrid", default=False, show_default=True, help="Include hybrid gate decision in dry-run output if probe texts are provided.")
@click.option("--token-budget", default=600, show_default=True, help="Token budget for hybrid gate (dry-run only).")
@click.option("--gate-jsonl-probe", default=None, help="JSONL probe text for the gate (dry-run only).")
@click.option("--gate-freeform-probe", default=None, help="Free-form probe text for the gate (dry-run only).")
@click.version_option(version=**version**, prog_name="tersetalk v0.5 runner")
def main(task, system, n, seed, caps, model, out, dry_run, hybrid, token_budget, gate_jsonl_probe, gate_freeform_probe):
"""
TerseTalk v0.5 Runner (PR-01 scaffold + PR-H1 gate in dry-run)

    This command provides a CLI skeleton only. Use --dry-run (default) to print
    the parsed configuration. Execution paths are implemented in later PRs.
    """
    try:
        parsed_caps = json.loads(caps)
        if not isinstance(parsed_caps, dict):
            raise ValueError
    except Exception:
        click.echo(
            'Error: --caps must be a JSON object, e.g. \'{"f":30,"p":20,"q":30}\'',
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
    if dry_run and hybrid and gate_jsonl_probe and gate_freeform_probe and GateCfg and gate_choose_protocol:
        try:
            gcfg = GateCfg(token_budget=int(token_budget))
            gate_obj = gate_choose_protocol(gate_jsonl_probe, gate_freeform_probe, gcfg)
        except Exception as e:  # keep dry-run robust
            gate_obj = {"error": str(e)}
    cfg["gate"] = gate_obj

    click.echo(json.dumps(cfg, indent=2))

    if dry_run:
        sys.exit(0)

    # Execution path intentionally unimplemented at this stage
    click.echo("Execution mode is not implemented yet.", err=True)
    sys.exit(0)

if **name** == "**main**":
main()

4. tests/test_hybrid_gate.py (new)
   from **future** import annotations

import os
from contextlib import contextmanager

from tersetalk.hybrid_gate import GateCfg, estimate_tokens, gate_choose_protocol

JSONL_SHORT = '["r","M"]\n["g","Short goal"]\n["q","W","?"]'
FREEFORM_LONG = "Goal: " + ("alpha beta gamma " \* 300)

@contextmanager
def fake_ll2_projection(value: int | None):
"""
Sets TERSETALK_FAKE_LL2_COMPRESS to simulate llmlingua projected tokens.
Pass None to clear it.
"""
old = os.environ.get("TERSETALK_FAKE_LL2_COMPRESS")
try:
if value is None:
os.environ.pop("TERSETALK_FAKE_LL2_COMPRESS", None)
else:
os.environ["TERSETALK_FAKE_LL2_COMPRESS"] = str(value)
yield
finally:
if old is None:
os.environ.pop("TERSETALK_FAKE_LL2_COMPRESS", None)
else:
os.environ["TERSETALK_FAKE_LL2_COMPRESS"] = old

def test_estimate_tokens_monotonic_and_nonnegative():
assert estimate_tokens("") == 0
a = estimate_tokens("abcd") # ~1
b = estimate_tokens("abcd" \* 10)
assert a >= 0 and b >= a and b > 0

def test_gate_routes_tersetalk_when_within_budget():
cfg = GateCfg(token_budget=50)
decision = gate_choose_protocol(JSONL_SHORT, FREEFORM_LONG, cfg)
assert decision["route"] == "tersetalk"
assert decision["est_tokens"]["jsonl"] <= 50

def test_gate_routes_freeform_llmlingua_when_projection_fits(): # Force JSONL to exceed budget while LL2 projection fits
jsonl = '["g","' + ("x" * 2000) + '"]' # very long jsonl payload
freeform = "alpha beta gamma " \* 200
cfg = GateCfg(token_budget=120)

    with fake_ll2_projection(100):  # simulate LL2 compressed within budget
        decision = gate_choose_protocol(jsonl, freeform, cfg)
        assert decision["route"] == "freeform_llmlingua"
        assert decision["est_tokens"]["ll2"] == 100

def test_gate_falls_back_to_tersetalk_when_projection_unavailable(): # Long JSONL, LL2 unavailable -> tersetalk
jsonl = '["g","' + ("x" * 2000) + '"]'
freeform = "alpha beta gamma " \* 200
cfg = GateCfg(token_budget=120)

    with fake_ll2_projection(None):  # ensure no fake projection
        decision = gate_choose_protocol(jsonl, freeform, cfg)
        assert decision["route"] == "tersetalk"
        # ll2 estimate may be None in this case
        assert decision["est_tokens"]["jsonl"] > cfg.token_budget

5. Update tersetalk/**init**.py to export hybrid_gate (replace file)
   from .\_version import **version**

**all** = [
"__version__",
"reproducibility",
"protocol_jsonl",
"structured",
"memory",
"summarization",
"hybrid_gate",
]

What to run (and what to paste as evidence in the PR)

Install

make install

Run tests

make test

Smoke: simple decision (default examples)

python scripts/hybrid_gate_smoke.py --token-budget 120

Smoke: force LL2 route (offline via env var)

export TERSETALK_FAKE_LL2_COMPRESS=100
python scripts/hybrid_gate_smoke.py --token-budget 120 \
 --jsonl "$([ -f /bin/yes ] && yes x | head -c 4000 | sed 's/.*/["g","&"]/' || echo '["g","x..."]')" \
  --freeform "$(python - <<'PY'\nprint('alpha beta gamma ' \* 200)\nPY)"
unset TERSETALK_FAKE_LL2_COMPRESS

Runner dry‑run with gate preview

python scripts/run_v05.py --dry-run --hybrid \
 --token-budget 120 \
 --gate-jsonl-probe '["g","xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"]' \
 --gate-freeform-probe 'Goal: many x; Question: ?'

Acceptance evidence to paste in PR description:

✅ pytest summary (all green).

✅ hybrid_gate_smoke.py JSON showing a decision with route, est_tokens, and notes.

✅ A second smoke where TERSETALK_FAKE_LL2_COMPRESS causes routing to "freeform_llmlingua".

✅ run_v05.py --dry-run output includes a gate object when hybrid probes are given.

Commit message
PR-H1: Per-turn hybrid gate (TerseTalk vs Free-form+LLMLingua)

- Add tersetalk/hybrid_gate.py with GateCfg, estimate_tokens, project_ll2_tokens (llmlingua optional), and gate_choose_protocol
- Provide scripts/hybrid_gate_smoke.py to demo decisions
- Extend scripts/run_v05.py to optionally include gate preview in --dry-run via probes and token budget
- Add tests for routing logic (within budget, ll2-projection fit, unavailable projection) and estimator sanity
- No new hard dependencies; llmlingua path is import-guarded; tests pass offline
