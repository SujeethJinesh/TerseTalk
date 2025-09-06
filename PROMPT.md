Read through your AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the AGENTS.md. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update AGENTS.md and ask CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly. Before you merge the PR, you must wait for my approval. Also at the end I want you to outline any risks we are seeing in this project, are our expectations aligned with how the progress is going? Will our project succeed and achieve the baselines we expect using your best judgement? Absolutely make sure you're reporting results truthfully and honestly. Avoid fake, mocked, or other non genuine results. We should be aiming to properly fix things and run proper evaluations. Here are more detailed instructions for PR implementation.

### PR Summary

PR‑H4 — protocol_handler.py + flags

Role: You are a senior engineer implementing PR‑H4 right after PR‑H3.
Goal (from spec): Plug LLMLingua into three touchpoints with toggles and counters; expose flags via the main runner; produce auditable logs/metrics.

Three touchpoints (all toggleable):

Pre‑overflow compression (--preoverflow-ll2): When a tag payload exceeds its cap, attempt LLMLingua compression first; if it fits, avoid overflow.

Overflow summarization (--overflow-ll2): When overflow is needed, summaries are generated with Summarizer(method="llmlingua") (falls back silently if LLMLingua is absent); ["o",..., "llmlingua"] encodes the chosen method.

Dereference compression (--deref-ll2, --deref-policy {never,conditional,always}): Resolve ["d","M#id"] into inline facts; optionally compress the dereferenced content before re‑injection; re‑validate caps.

Key constraints

Stdlib‑only in this PR; LLMLingua path import‑guarded.

Deterministic behavior; provide fake env hatches for offline CI:

TERSETALK_FAKE_LL2_TEXT=1 → simulate LL2 by hard‑trimming to target_tokens\*4 chars (word‑aware, ellipsis).

Backwards compatible; all prior tests must stay green.

Deliverables

tersetalk/protocol_handler.py — ProtocolHandler and config/dataclasses; counters + processing.

scripts/protocol_handler_smoke.py — CLI to run handler on stdin or a synthetic example; prints counters + pre/post stats.

Runner update (scripts/run_v05.py): add flags --preoverflow-ll2, --overflow-ll2, --deref-ll2, --deref-policy, and an optional --protocol-demo section in dry‑run output showing one processed synthetic example’s counters & stats.

Tests: tests/test_protocol_handler.py verifying the three touchpoints and counters using offline fakes.

Update tersetalk/**init**.py to export protocol_handler.

Create/Update the following files exactly

1. tersetalk/protocol_handler.py (new)
   from **future** import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Literal

from tersetalk.protocol_jsonl import JSONLValidator
from tersetalk.summarization import Summarizer
from tersetalk.memory import MemoryStore

# -------------------------

# Config & result types

# -------------------------

DerefPolicy = Literal["never", "conditional", "always"]

@dataclass
class PHConfig:
caps: Dict[str, int]
summarizer_method: Literal["extractive", "llmlingua"] = "extractive"
preoverflow_ll2: bool = False
overflow_ll2: bool = False
deref_ll2: bool = False
deref_policy: DerefPolicy = "never"

@dataclass
class PHOutcome:
validated_jsonl: str
stats_before: Dict
post_deref_jsonl: Optional[str]
stats_after: Optional[Dict]
counters: Dict[str, Dict[str, int]]

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

# -------------------------

# Internal helpers

# -------------------------

def \_tok_est(text: str) -> int:
return max(0, (len(text) + 3) // 4)

def \_hard_trim_to_tokens(text: str, target_tokens: int) -> str:
max_chars = max(1, target_tokens \* 4)
if len(text) <= max_chars:
return text
cut = text[:max_chars] # try to cut on last space for readability
sp = cut.rfind(" ")
if sp >= 16:
cut = cut[:sp]
return cut.rstrip() + "..."

def \_ll2_compress_text(text: str, target_tokens: int) -> Optional[str]:
"""
Try LLMLingua compression. If unavailable and TERSETALK_FAKE_LL2_TEXT=1,
simulate by hard trimming to target token budget. Else return None.
""" # Fake path for offline CI
if os.environ.get("TERSETALK_FAKE_LL2_TEXT", ""):
return \_hard_trim_to_tokens(text, target_tokens)

    try:
        from llmlingua import PromptCompressor  # type: ignore
    except Exception:
        return None

    try:
        comp = PromptCompressor()
        res = comp.compress(text, target_token=int(target_tokens))
        cand = (
            res.get("compressed_prompt")
            or res.get("compressed_text")
            or res.get("text")
            or res.get("prompt")
        )
        if not isinstance(cand, str) or not cand.strip():
            return None
        # Ensure we don't exceed the target; final enforce
        return _hard_trim_to_tokens(cand.strip(), target_tokens)
    except Exception:
        return None

def \_is_textual_tag(tag: str) -> bool:
return tag in ("f", "p", "g", "u", "t", "q")

# -------------------------

# Protocol Handler

# -------------------------

class ProtocolHandler:
"""
Coordinates the three LLMLingua touchpoints around the JSONLValidator: 1) Pre-overflow compression on long payloads 2) Overflow summarization method selection 3) Dereference (d:M#) injection and optional compression
"""

    def __init__(self, cfg: PHConfig) -> None:
        self.cfg = cfg

    def _counters_init(self) -> Dict[str, Dict[str, int]]:
        return {
            "preoverflow": {"attempted": 0, "succeeded": 0, "ll2_used": 0, "ll2_unavailable": 0},
            "overflow": {"count": 0, "method_extractive": 0, "method_llmlingua": 0},
            "deref": {"attempted": 0, "injected": 0, "ll2_compressed": 0},
        }

    def _normalize_lines(self, jsonl: str, validator: JSONLValidator) -> List[List]:
        out: List[List] = []
        for raw in [ln for ln in jsonl.splitlines() if ln.strip()]:
            arr = validator.normalize_line(raw)
            if not isinstance(arr, list) or not arr:
                arr = ["t", raw.strip()]
            out.append(arr)
        return out

    def _preoverflow_pass(self, lines: List[List], counters: Dict[str, Dict[str, int]]) -> List[List]:
        """
        Attempt to compress over-cap payloads in-place, before validator overflow logic runs.
        For 'q', payload is at index 2; for others, payload at index 1.
        """
        if not self.cfg.preoverflow_ll2:
            return lines

        caps = self.cfg.caps
        new_lines: List[List] = []
        for arr in lines:
            tag = arr[0] if arr else "t"
            if _is_textual_tag(tag):
                if tag == "q":
                    role = arr[1] if len(arr) > 1 else "W"
                    text = arr[2] if len(arr) > 2 else ""
                    cap = caps.get("q")
                    if isinstance(text, str) and cap is not None and _tok_est(text) > cap:
                        counters["preoverflow"]["attempted"] += 1
                        comp = _ll2_compress_text(text, cap)
                        if comp is not None:
                            counters["preoverflow"]["ll2_used"] += 1
                            if _tok_est(comp) <= cap:
                                counters["preoverflow"]["succeeded"] += 1
                                new_lines.append(["q", role, comp])
                                continue
                        else:
                            counters["preoverflow"]["ll2_unavailable"] += 1
                    new_lines.append(["q", role, text])
                else:
                    text = arr[1] if len(arr) > 1 else ""
                    cap = caps.get(tag)
                    if isinstance(text, str) and cap is not None and _tok_est(text) > cap:
                        counters["preoverflow"]["attempted"] += 1
                        comp = _ll2_compress_text(text, cap)
                        if comp is not None:
                            counters["preoverflow"]["ll2_used"] += 1
                            if _tok_est(comp) <= cap:
                                counters["preoverflow"]["succeeded"] += 1
                                # Keep 'f' inline pointer position (2) unused here; validator may add overflow later if needed
                                new_lines.append([tag, comp] + (arr[2:] if len(arr) > 2 else []))
                                continue
                        else:
                            counters["preoverflow"]["ll2_unavailable"] += 1
                    new_lines.append(arr)
            else:
                new_lines.append(arr)
        return new_lines

    def _overflow_counters(self, stats: Dict, counters: Dict[str, Dict[str, int]], method: str) -> None:
        cnt = int(stats.get("overflow", {}).get("count", 0))
        counters["overflow"]["count"] += cnt
        if method == "llmlingua":
            counters["overflow"]["method_llmlingua"] += cnt
        else:
            counters["overflow"]["method_extractive"] += cnt

    def _deref_inject(self, jsonl: str, memory: MemoryStore, counters: Dict[str, Dict[str, int]]) -> str:
        """
        Replace each ["d","M#id"] with an inline ["f", <(maybe compressed) text>].
        Then return a new JSONL string (array form).
        """
        out_lines: List[str] = []
        for raw in [ln for ln in jsonl.splitlines() if ln.strip()]:
            arr = None
            try:
                arr = json.loads(raw)
            except Exception:
                out_lines.append(raw)
                continue
            if not isinstance(arr, list) or not arr:
                out_lines.append(raw)
                continue
            tag = arr[0]
            if tag != "d" or len(arr) < 2:
                out_lines.append(json.dumps(arr))
                continue

            counters["deref"]["attempted"] += 1
            mid = arr[1]
            text = memory.get(mid)
            if not isinstance(text, str):
                out_lines.append(json.dumps(arr))
                continue

            if self.cfg.deref_ll2:
                # Target cap for injected fact = 'f' cap (fallback to 30)
                cap = int(self.cfg.caps.get("f", 30))
                comp = _ll2_compress_text(text, cap)
                if comp is not None:
                    counters["deref"]["ll2_compressed"] += 1
                    text = comp

            counters["deref"]["injected"] += 1
            out_lines.append(json.dumps(["f", text]))
        return "\n".join(out_lines)

    # --------- Public API ---------

    def process(self, manager_jsonl: str, memory: Optional[MemoryStore] = None) -> PHOutcome:
        """
        Run pre-overflow compression, validate (with chosen summarizer method),
        then optional dereference injection + revalidation.
        """
        memory = memory or MemoryStore()
        counters = self._counters_init()

        # Normalization pass
        tmp_validator = JSONLValidator(caps=self.cfg.caps, memory=memory, summarizer=Summarizer(method="extractive"))
        norm_lines = self._normalize_lines(manager_jsonl, tmp_validator)

        # Pre-overflow compression (in-place), if enabled
        pre_lines = self._preoverflow_pass(norm_lines, counters)

        # Overflow summarization method selection
        ov_method = "llmlingua" if self.cfg.overflow_ll2 else self.cfg.summarizer_method
        validator = JSONLValidator(caps=self.cfg.caps, memory=memory, summarizer=Summarizer(method=ov_method))

        # Validate + overflow
        pre_jsonl = "\n".join(json.dumps(l) for l in pre_lines)
        validated_jsonl, stats_before = validator.validate_and_overflow(pre_jsonl)
        self._overflow_counters(stats_before, counters, method=ov_method)

        # Dereference policy
        post_jsonl: Optional[str] = None
        stats_after: Optional[Dict] = None
        if self.cfg.deref_policy in ("conditional", "always"):
            post_jsonl = self._deref_inject(validated_jsonl, memory, counters)
            # Re-validate after injection to enforce caps (still using same summarizer method)
            validated2, stats2 = validator.validate_and_overflow(post_jsonl)
            post_jsonl, stats_after = validated2, stats2
            # Count overflow post-deref toward the same overflow bucket
            self._overflow_counters(stats2, counters, method=ov_method)

        return PHOutcome(
            validated_jsonl=validated_jsonl,
            stats_before=stats_before,
            post_deref_jsonl=post_jsonl,
            stats_after=stats_after,
            counters=counters,
        )

2. scripts/protocol_handler_smoke.py (new)
   from **future** import annotations

import argparse
import json
import sys

from tersetalk.protocol_handler import PHConfig, ProtocolHandler
from tersetalk.memory import MemoryStore
from tersetalk.calibration import synth_shard

def main():
ap = argparse.ArgumentParser(description="PR-H4: Protocol Handler smoke tool")
ap.add_argument("--input", default="-", help="JSONL input or '-' for synthetic")
ap.add_argument("--seed", type=int, default=0, help="Seed for synthetic example")
ap.add_argument("--caps", default='{"f":30,"p":20,"q":30,"g":30,"u":20,"t":50}')
ap.add_argument("--summarizer", choices=["extractive","llmlingua"], default="extractive")
ap.add_argument("--preoverflow-ll2", action="store_true")
ap.add_argument("--overflow-ll2", action="store_true")
ap.add_argument("--deref-ll2", action="store_true")
ap.add_argument("--deref-policy", choices=["never","conditional","always"], default="never")
args = ap.parse_args()

    if args.input == "-":
        ex = synth_shard(1, args.seed)[0]
        mgr = ex["jsonl"]
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            mgr = f.read()

    caps = json.loads(args.caps)
    cfg = PHConfig(
        caps=caps,
        summarizer_method=args.summarizer,
        preoverflow_ll2=args.preoverflow_ll2,
        overflow_ll2=args.overflow_ll2,
        deref_ll2=args.deref_ll2,
        deref_policy=args.deref_policy,
    )
    handler = ProtocolHandler(cfg)
    out = handler.process(mgr, memory=MemoryStore())
    print(json.dumps(out.to_dict(), indent=2))

if **name** == "**main**":
main()

3. Update scripts/run_v05.py to add flags + optional demo (replace file)

Keep prior options; add the four LL2 toggles, deref policy, and an optional mini demo in dry‑run (no execution mode changes).

from **future** import annotations

import json
import sys
import click

from tersetalk.\_version import **version**
from tersetalk.reproducibility import set_global_seed

# Optional imports (guarded)

try:
from tersetalk.hybrid_gate import GateCfg, gate_choose_protocol
except Exception: # pragma: no cover
GateCfg = None
gate_choose_protocol = None

try:
from tersetalk.protocol_handler import PHConfig, ProtocolHandler
from tersetalk.calibration import synth_shard
from tersetalk.memory import MemoryStore
except Exception: # pragma: no cover
PHConfig = None
ProtocolHandler = None
synth_shard = None
MemoryStore = None

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--task", type=click.Choice(["hotpotqa", "gsm8k"]), default="hotpotqa", show_default=True)
@click.option("--system", type=click.Choice(["tersetalk", "freeform", "llmlingua"]), default="tersetalk", show_default=True)
@click.option("--n", default=100, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--caps", default='{"f":30,"p":20,"q":30}', show_default=True)
@click.option("--model", default="mistral", show_default=True)
@click.option("--out", default="results", show_default=True)
@click.option("--dry-run/--execute", default=True, show_default=True)

# --- Hybrid gate (PR-H1) ---

@click.option("--hybrid/--no-hybrid", default=False, show_default=True)
@click.option("--token-budget", default=600, show_default=True)
@click.option("--gate-jsonl-probe", default=None)
@click.option("--gate-freeform-probe", default=None)

# --- Protocol handler toggles (PR-H4) ---

@click.option("--preoverflow-ll2/--no-preoverflow-ll2", default=False, show_default=True)
@click.option("--overflow-ll2/--no-overflow-ll2", default=False, show_default=True)
@click.option("--deref-ll2/--no-deref-ll2", default=False, show_default=True)
@click.option("--deref-policy", type=click.Choice(["never","conditional","always"]), default="never", show_default=True)
@click.option("--protocol-demo/--no-protocol-demo", default=False, show_default=True, help="Include a one-sample protocol demo in dry-run.")
@click.version_option(version=**version**, prog_name="tersetalk v0.5 runner")
def main(task, system, n, seed, caps, model, out, dry_run,
hybrid, token_budget, gate_jsonl_probe, gate_freeform_probe,
preoverflow_ll2, overflow_ll2, deref_ll2, deref_policy, protocol_demo):
"""
TerseTalk v0.5 Runner (scaffold + PR-H1 gate + PR-H4 protocol handler demo in dry-run)
"""
try:
parsed_caps = json.loads(caps)
if not isinstance(parsed_caps, dict):
raise ValueError
except Exception:
click.echo('Error: --caps must be a JSON object, e.g. \'{"f":30,"p":20,"q":30}\'', err=True)
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
        "handler_flags": {
            "preoverflow_ll2": preoverflow_ll2,
            "overflow_ll2": overflow_ll2,
            "deref_ll2": deref_ll2,
            "deref_policy": deref_policy,
        }
    }

    # Optional gate preview (dry-run)
    gate_obj = None
    if dry_run and hybrid and gate_jsonl_probe and gate_freeform_probe and GateCfg and gate_choose_protocol:
        try:
            gcfg = GateCfg(token_budget=int(token_budget))
            gate_obj = gate_choose_protocol(gate_jsonl_probe, gate_freeform_probe, gcfg)
        except Exception as e:
            gate_obj = {"error": str(e)}
    cfg["gate"] = gate_obj

    # Optional protocol handler demo (dry-run)
    proto_demo = None
    if dry_run and protocol_demo and PHConfig and ProtocolHandler and synth_shard and MemoryStore:
        try:
            sample = synth_shard(1, int(seed))[0]["jsonl"]
            phcfg = PHConfig(
                caps=parsed_caps,
                summarizer_method="extractive",
                preoverflow_ll2=preoverflow_ll2,
                overflow_ll2=overflow_ll2,
                deref_ll2=deref_ll2,
                deref_policy=deref_policy,
            )
            ph = ProtocolHandler(phcfg)
            outcome = ph.process(sample, memory=MemoryStore())
            proto_demo = {
                "validated_jsonl": outcome.validated_jsonl,
                "stats_before": outcome.stats_before,
                "post_deref_jsonl": outcome.post_deref_jsonl,
                "stats_after": outcome.stats_after,
                "counters": outcome.counters,
            }
        except Exception as e:
            proto_demo = {"error": str(e)}
    cfg["protocol_demo"] = proto_demo

    click.echo(json.dumps(cfg, indent=2))

    if dry_run:
        sys.exit(0)

    click.echo("Execution mode is not implemented yet.", err=True)
    sys.exit(0)

if **name** == "**main**":
main()

4. tests/test_protocol_handler.py (new)
   from **future** import annotations

import json
import os

from tersetalk.protocol_handler import PHConfig, ProtocolHandler
from tersetalk.memory import MemoryStore

def \_mk_long(n=200):
return " ".join(["alpha","beta","gamma","delta","epsilon","zeta","eta","theta"] \* (n // 8))

def test_preoverflow_ll2_avoids_overflow_when_fake_enabled(monkeypatch): # Enable fake LL2 text compression
monkeypatch.setenv("TERSETALK_FAKE_LL2_TEXT", "1")

    caps = {"f": 10, "q": 8, "p": 20, "g": 30, "u": 20, "t": 50}
    long_fact = _mk_long(120)
    mgr = '\n'.join([
        '["r","M"]',
        json.dumps(["f", long_fact]),
        json.dumps(["q", "W", _mk_long(80)]),
    ])
    cfg = PHConfig(caps=caps, preoverflow_ll2=True, overflow_ll2=False, deref_ll2=False, deref_policy="never")
    out = ProtocolHandler(cfg).process(mgr, memory=MemoryStore())

    # No overflow lines expected because preoverflow compressed to within caps
    assert all(not ln.startswith('["o"') for ln in out.validated_jsonl.splitlines())
    assert out.counters["preoverflow"]["attempted"] >= 1
    assert out.counters["preoverflow"]["succeeded"] >= 1
    assert out.counters["preoverflow"]["ll2_used"] >= 1

def test_overflow_summary_method_switches_to_llmlingua_without_pkg(): # No FAKE needed here; summarizer(method="llmlingua") labels 'o' lines accordingly.
caps = {"f": 5, "q": 5, "p": 20, "g": 30, "u": 20, "t": 50}
mgr = '\n'.join([
'["r","M"]',
json.dumps(["f", _mk_long(120)]),
json.dumps(["q", "W", _mk_long(100)]),
])
cfg = PHConfig(caps=caps, preoverflow_ll2=False, overflow_ll2=True, deref_ll2=False, deref_policy="never")
out = ProtocolHandler(cfg).process(mgr, memory=MemoryStore())

    # Expect overflow lines with method == "llmlingua"
    o_lines = [json.loads(ln) for ln in out.validated_jsonl.splitlines() if ln.startswith('["o"')]
    assert len(o_lines) >= 1
    assert all(arr[3] == "llmlingua" for arr in o_lines)
    # Counters reflect overflow counts
    assert out.counters["overflow"]["count"] >= 1
    assert out.counters["overflow"]["method_llmlingua"] >= 1

def test_deref_injection_and_optional_ll2_compression(monkeypatch):
monkeypatch.setenv("TERSETALK_FAKE_LL2_TEXT", "1")

    # Put a long text in memory and reference it via ["d","M#1"]
    mem = MemoryStore()
    mid = mem.put(_mk_long(160))  # e.g., "M#1"
    caps = {"f": 12, "q": 20, "p": 20, "g": 30, "u": 20, "t": 50}
    mgr = '\n'.join([
        '["r","M"]',
        json.dumps(["d", mid]),
    ])
    cfg = PHConfig(caps=caps, preoverflow_ll2=False, overflow_ll2=False, deref_ll2=True, deref_policy="always")
    out = ProtocolHandler(cfg).process(mgr, memory=mem)

    assert out.post_deref_jsonl is not None
    # 'd' lines replaced by 'f' lines
    assert any(ln.startswith('["f"') for ln in out.post_deref_jsonl.splitlines())
    assert all(not ln.startswith('["d"') for ln in out.post_deref_jsonl.splitlines())
    # Counters confirm deref happened and compression attempted
    assert out.counters["deref"]["attempted"] >= 1
    assert out.counters["deref"]["injected"] >= 1
    assert out.counters["deref"]["ll2_compressed"] >= 1

5. Update tersetalk/**init**.py to export the handler (replace file)
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
"protocol_handler",
]

What to run (and what to paste as evidence in the PR)

Install (unchanged)

make install

Run tests

make test

Smoke: pre‑overflow compression (offline via fake)

export TERSETALK_FAKE_LL2_TEXT=1
python scripts/protocol_handler_smoke.py --preoverflow-ll2 --deref-policy never <<'EOF'
["r","M"]
["f","alpha beta gamma delta epsilon zeta eta theta " \
 "iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega " \
 "alpha beta gamma delta epsilon zeta eta theta iota kappa"]
["q","W","please compress this question to fit the cap without overflowing using ll2"]
EOF
unset TERSETALK_FAKE_LL2_TEXT

Smoke: overflow summarization set to llmlingua (labels method)

python scripts/protocol_handler_smoke.py --overflow-ll2 --caps '{"f":6,"q":6,"p":20,"g":30,"u":20,"t":50}' <<'EOF'
["r","M"]
["f","alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"]
["q","W","explain the error with as few words as possible but this is long"]
EOF

Smoke: dereference injection with ll2 compression (offline via fake)

export TERSETALK_FAKE_LL2_TEXT=1
python - <<'PY'
from tersetalk.memory import MemoryStore
mem = MemoryStore()
mid = mem.put(" ".join(["alpha beta gamma delta epsilon zeta eta theta"]\*20))
print('["r","M"]')
print(f'["d","{mid}"]')
PY | python scripts/protocol_handler_smoke.py --deref-ll2 --deref-policy always
unset TERSETALK_FAKE_LL2_TEXT

Runner dry‑run with protocol demo

export TERSETALK_FAKE_LL2_TEXT=1
python scripts/run_v05.py --dry-run --protocol-demo \
 --preoverflow-ll2 --overflow-ll2 --deref-ll2 --deref-policy always
unset TERSETALK_FAKE_LL2_TEXT

Acceptance evidence to paste in the PR description:

✅ pytest summary (all green).

✅ protocol_handler_smoke.py JSON snippets illustrating:

pre‑overflow success (counters.preoverflow.succeeded > 0) and no ["o",...] lines,

overflow summaries with "method": "llmlingua" in ["o",...],

dereference injection replacing ["d",...] with ["f", ...] and counters.deref.ll2_compressed > 0.

✅ run_v05.py --dry-run --protocol-demo output includes a protocol_demo object with counters, stats_before, and (if applicable) stats_after.

Commit message
PR-H4: Protocol Handler — LLMLingua touchpoints + toggles

- Add tersetalk/protocol_handler.py with PHConfig and ProtocolHandler:

  - Pre-overflow LLMLingua compression (optional) to avoid overflow
  - Overflow summarization method switch (extractive ↔ llmlingua)
  - Dereference (d:M#) injection with optional LL2 compression and revalidation
  - Structured counters for all touchpoints

- Add scripts/protocol_handler_smoke.py for end-to-end inspection on stdin or synthetic inputs
- Extend scripts/run_v05.py with --preoverflow-ll2, --overflow-ll2, --deref-ll2, --deref-policy and --protocol-demo (dry-run)
- Add tests covering preoverflow success with offline fake, llmlingua summary labeling, and deref injection/compression
- Export protocol_handler in package **init**
- LLMLingua optional; deterministic offline fallbacks provided via TERSETALK_FAKE_LL2_TEXT
