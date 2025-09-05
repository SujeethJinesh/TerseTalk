Read through your AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the AGENTS.md. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update AGENTS.md and ask CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly.. Here are more detailed instructions.

### PR Summary

PR‑02 — JSONL Protocol & Validator

Role: You are a senior engineer implementing PR‑02 for the TerseTalk project, right after PR‑01 merged.

Goal (from spec):
Create tersetalk/protocol_jsonl.py with a JSONLValidator that:

Accepts JSONL/NDJSON strings.

Detects mixed format lines.

Normalizes lenient object lines to the canonical array form.

Enforces soft caps per tag; when exceeded, emits an overflow pointer via M#<id> and an ["o", ...] line.

Tracks overflow frequency and reports density proxy: density = 1.0 - (overflow_count / total_lines).

Provides jsonl_to_prose(...) for SP references.

Returns (validated_jsonl_str, stats_dict).

Strict scope for PR‑02:

Do not implement PR‑02S (Instructor), PR‑03 (full MemoryStore), or PR‑04 (Summarizer).

Use an internal lightweight memory (incremental M#1, M#2, …) just for PR‑02 tests and the CLI. The real MemoryStore arrives in PR‑03.

Keep dependencies stdlib‑only. No new pip packages.

DoD (Definition of Done):

Mixed format detection works and returns (is_mixed: bool, line_index: int).

Normalization converts objects to arrays per the spec examples.

Caps are enforced; long payloads create an overflow summary, a memory pointer (M#k), and an ["o", summary, "M#k", "extractive"] line.

Stats include overflow.count, overflow.per_tag, overflow.rate, and density.

Tests pass; CLI guard script runs and demonstrates behavior.

Create/Update the following files exactly

Keep all PR‑00 and PR‑01 files working. Only add/modify what’s listed below.

1. tersetalk/protocol_jsonl.py (new)
   from **future** import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

# --- Protocol Tags (single letters) -----------------------------

TAGS: Dict[str, str] = {
"r": "role", # ["r","M"|"W"|"C"]
"g": "subgoal", # ["g","<short text>"]
"f": "fact", # ["f","<text>","M#id"?]
"u": "assumption", # ["u","<short text>"]
"p": "plan_step", # ["p","<short text>"]
"q": "question", # ["q","M|W|C","<question>"]
"d": "delta_ref", # ["d","M#id"]
"v": "verdict", # ["v","A"|"R"|"E"]
"o": "overflow", # ["o","<summary>","M#ref","extractive"]
"t": "free_text", # ["t","<very short>"]
"x": "context" # ["x","<key>","<val>"]
}

# --- Minimal memory interface for PR-02 only --------------------

class SupportsPut(Protocol):
def put(self, text: str) -> str: ...

class \_LocalMemory:
"""
PR-02-only, lightweight in-module memory to mint M# ids.
PR-03 will provide a full MemoryStore; this is intentionally minimal.
"""
def **init**(self) -> None:
self.\_store: Dict[str, str] = {}
self.\_ctr: int = 0

    def put(self, text: str) -> str:
        self._ctr += 1
        mid = f"M#{self._ctr}"
        self._store[mid] = text
        return mid

# --- Helper ------------------------------------------------------

def \_token_estimate(text: str) -> int:
"""Cheap token estimator used throughout this repo."""
return max(0, (len(text) + 3) // 4)

# --- JSONL Validator --------------------------------------------

@dataclass
class JSONLValidator:
"""
Validate + normalize JSONL, enforce soft caps, and create overflow lines.

    caps: per-tag target token caps. Missing keys fall back to defaults.
    memory: object with .put(text)->"M#k" used to store overflow text.
    """
    caps: Dict[str, int] = field(default_factory=dict)
    memory: Optional[SupportsPut] = None

    def __post_init__(self) -> None:
        default_caps = {"f": 30, "p": 20, "q": 30, "g": 30, "u": 20, "t": 50}
        merged = dict(default_caps)
        merged.update(self.caps or {})
        self.caps = merged
        self.memory = self.memory or _LocalMemory()
        self.overflow_freq: Dict[str, int] = {}

    # ---------- Detection ----------
    def detect_format_break(self, output: str) -> Tuple[bool, int]:
        """
        Returns (is_mixed, line_number_of_break).
        Mixed if any non-empty line does not start with '[' or '{'.
        """
        lines = [ln for ln in output.splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            s = line.lstrip()
            if not (s.startswith("[") or s.startswith("{")):
                return True, i
        return False, -1

    # ---------- Normalization ----------
    def normalize_line(self, raw: str) -> List[Any]:
        """
        Convert a lenient object form to the canonical array form.
        Unknown structures become ["t", "<stringified>"].
        """
        s = raw.strip()
        if not s:
            return ["t", ""]
        if s[0] == "[":
            arr = json.loads(s)
            if isinstance(arr, list):
                return arr
            return ["t", json.dumps(arr, separators=(",", ":"))]
        if s[0] == "{":
            obj = json.loads(s)
            # If explicit 'tag'
            if "tag" in obj:
                tag = obj.get("tag")
                if tag == "f":
                    text = obj.get("text") or obj.get("value") or obj.get("f") or ""
                    ref = obj.get("ref") or obj.get("id")
                    return ["f", str(text)] + ([str(ref)] if ref else [])
                if tag == "q":
                    who = obj.get("role") or obj.get("to") or obj.get("who") or "W"
                    text = obj.get("text") or obj.get("question") or ""
                    return ["q", str(who), str(text)]
                if tag == "r":
                    role = obj.get("role") or obj.get("r") or "M"
                    return ["r", str(role)]
                if tag in TAGS:
                    # generic object: prefer 'text' or tag-named key
                    v = obj.get("text", obj.get(tag, ""))
                    if isinstance(v, list):
                        return [tag] + v
                    return [tag, str(v)]
                # unknown tag → t
                return ["t", json.dumps(obj, separators=(",", ":"))]

            # No explicit 'tag': look for single-letter key
            for k in TAGS.keys():
                if k in obj:
                    v = obj[k]
                    if isinstance(v, list):
                        return [k] + v
                    return [k, str(v)]
            # Fall back
            return ["t", json.dumps(obj, separators=(",", ":"))]

        # Fallback for raw text lines (not valid JSON) → 't'
        return ["t", s]

    # ---------- (De)serialization helpers ----------
    def jsonl_to_prose(self, lines: str) -> str:
        """Convert canonical array-lines to simple prose (for SP reference)."""
        prose: List[str] = []
        for ln in [ln for ln in lines.splitlines() if ln.strip()]:
            try:
                arr = json.loads(ln)
            except Exception:
                continue
            if not isinstance(arr, list) or not arr:
                continue
            tag = arr[0]
            if tag == "g":
                prose.append(f"Goal: {arr[1] if len(arr)>1 else ''}")
            elif tag == "f":
                prose.append(f"Fact: {arr[1] if len(arr)>1 else ''}")
            elif tag == "u":
                prose.append(f"Assumption: {arr[1] if len(arr)>1 else ''}")
            elif tag == "p":
                prose.append(f"Plan: {arr[1] if len(arr)>1 else ''}")
            elif tag == "q":
                who = arr[1] if len(arr) > 1 else ""
                txt = arr[2] if len(arr) > 2 else ""
                prose.append(f"Question ({who}): {txt}")
            elif tag == "v":
                prose.append(f"Verdict: {arr[1] if len(arr)>1 else ''}")
            elif tag == "o":
                mid = arr[2] if len(arr) > 2 else ""
                prose.append(f"Overflow: {arr[1] if len(arr)>1 else ''} [{mid}]")
            elif tag == "d":
                prose.append(f"Ref: {arr[1] if len(arr)>1 else ''}")
            elif tag == "r":
                prose.append(f"Role: {arr[1] if len(arr)>1 else ''}")
            elif tag == "t":
                prose.append(f"Note: {arr[1] if len(arr)>1 else ''}")
            elif tag == "x":
                key = arr[1] if len(arr) > 1 else ""
                val = arr[2] if len(arr) > 2 else ""
                prose.append(f"Meta: {key}={val}")
        return "\n".join(prose)

    # ---------- Internal: summarization stub ----------
    def _summarize(self, text: str, target_tokens: int) -> str:
        """
        PR-02 stub: simple word-capped summary with ellipsis.
        PR-04 will provide a proper summarization module.
        """
        words = text.strip().split()
        if len(words) <= target_tokens:
            return text
        return " ".join(words[:target_tokens]) + "..."

    # ---------- Core: validation + overflow ----------
    def validate_and_overflow(self, jsonl: str) -> Tuple[str, Dict[str, Any]]:
        """
        Validate JSONL, normalize to arrays, enforce caps, and emit overflow lines.
        Returns (validated_jsonl_str, stats_dict).
        """
        out_lines: List[str] = []
        self.overflow_freq.clear()

        raw_lines = [ln for ln in jsonl.splitlines() if ln.strip()]
        for raw in raw_lines:
            arr = self.normalize_line(raw)
            if not isinstance(arr, list) or not arr:
                arr = ["t", json.dumps(raw)]

            tag = arr[0]
            if tag not in TAGS:
                # Unknown tag → coerce to free_text
                arr = ["t", json.dumps(arr, separators=(",", ":"))]
                tag = "t"

            # Enforce caps on textual payloads; q has (role, text)
            cap = self.caps.get(tag)
            if tag == "q":
                role = arr[1] if len(arr) > 1 else "W"
                text = arr[2] if len(arr) > 2 else ""
                if isinstance(text, str) and cap is not None and _token_estimate(text) > cap:
                    summary = self._summarize(text, cap)
                    mid = self.memory.put(text)  # type: ignore[union-attr]
                    self.overflow_freq[tag] = self.overflow_freq.get(tag, 0) + 1
                    out_lines.append(json.dumps(["q", role, summary]))
                    out_lines.append(json.dumps(["o", summary, mid, "extractive"]))
                else:
                    out_lines.append(json.dumps(["q", role, text]))
                continue

            if tag in ("f", "p", "g", "u", "t"):
                text = arr[1] if len(arr) > 1 else ""
                if isinstance(text, str) and cap is not None and _token_estimate(text) > cap:
                    summary = self._summarize(text, cap)
                    mid = self.memory.put(text)  # type: ignore[union-attr]
                    self.overflow_freq[tag] = self.overflow_freq.get(tag, 0) + 1
                    # For 'f', attach inline M# pointer as third element; for others, only summary text.
                    new_line = ["f", summary, mid] if tag == "f" else [tag, summary]
                    out_lines.append(json.dumps(new_line))
                    out_lines.append(json.dumps(["o", summary, mid, "extractive"]))
                else:
                    out_lines.append(json.dumps(arr))
                continue

            # Pass-through for non-textual or control tags ('r','d','v','x','o')
            out_lines.append(json.dumps(arr))

        total_lines = len(out_lines)
        overflow_count = sum(self.overflow_freq.values())
        rate = overflow_count / total_lines if total_lines else 0.0
        density = 1.0 - rate

        stats = {
            "lines_total": total_lines,
            "overflow": {
                "count": overflow_count,
                "per_tag": {k: v for k, v in self.overflow_freq.items() if v},
                "rate": rate,
            },
            "density": density,
        }
        return "\n".join(out_lines), stats

    # ---------- Public utility ----------
    def estimate_tokens(self, text: str) -> int:
        return _token_estimate(text)

2. scripts/jsonl_guard.py (new)
   from **future** import annotations

import argparse
import json
import sys
from tersetalk.protocol_jsonl import JSONLValidator

def main():
ap = argparse.ArgumentParser(description="TerseTalk JSONL guard (PR-02)")
ap.add_argument("--caps", type=str, default='{"f":30,"p":20,"q":30}',
help='Per-tag caps JSON, e.g. \'{"f":20,"q":25}\'')
ap.add_argument("--fail-on-mixed", action="store_true",
help="Exit nonzero if mixed format detected.")
ap.add_argument("--input", type=str, default="-",
help="Path to JSONL file or '-' for stdin.")
args = ap.parse_args()

    try:
        caps = json.loads(args.caps)
        if not isinstance(caps, dict):
            raise ValueError
    except Exception:
        print("Error: --caps must be a JSON object.", file=sys.stderr)
        sys.exit(2)

    if args.input == "-":
        data = sys.stdin.read()
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            data = f.read()

    validator = JSONLValidator(caps=caps)
    mixed, idx = validator.detect_format_break(data)
    if mixed and args.fail_on_mixed:
        print(json.dumps({"mixed_format": True, "break_line": idx}), file=sys.stderr)
        sys.exit(3)

    out, stats = validator.validate_and_overflow(data)
    res = {"mixed_format": mixed, "break_line": idx, "stats": stats, "out": out}
    print(json.dumps(res, indent=2))

if **name** == "**main**":
main()

3. Update tersetalk/**init**.py to export the new module (replace file)
   from .\_version import **version**

# Keep reproducibility (PR-00) import path working

**all** = ["__version__", "reproducibility", "protocol_jsonl"]

(Do not import heavy symbols at package import time; tests import directly from module files.)

4. tests/test_protocol_jsonl.py (new)
   from **future** import annotations

import json
import re
from tersetalk.protocol_jsonl import JSONLValidator

def test_detect_format_break():
v = JSONLValidator()
s = '["f","ok"]\nnot json\n{"g":"x"}'
mixed, idx = v.detect_format_break(s)
assert mixed is True and idx == 1
s2 = '["f","ok"]\n{"g":"x"}'
mixed2, idx2 = v.detect_format_break(s2)
assert mixed2 is False and idx2 == -1

def test_normalize_object_to_array():
v = JSONLValidator()
a = v.normalize_line('{"f":"Event A: 2001"}')
assert a == ["f", "Event A: 2001"]
b = v.normalize_line('{"tag":"f","text":"Event A","id":"M#12"}')
assert b == ["f", "Event A", "M#12"]
c = v.normalize_line('{"tag":"q","role":"W","text":"Which is earlier?"}')
assert c == ["q", "W", "Which is earlier?"]

def test_validate_and_overflow_creates_o_lines_and_pointers():
v = JSONLValidator(caps={"f": 5, "q": 5})
long_fact = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
long_q = "please compare the following two things and return the earlier"
raw = "\n".join([
'["r","M"]',
json.dumps(["f", long_fact]),
json.dumps(["q", "W", long_q]),
'["g","short goal"]'
])
out, stats = v.validate_and_overflow(raw)
lines = out.splitlines()

    # Expect o-lines present
    o_lines = [ln for ln in lines if ln.strip().startswith('["o"')]
    assert len(o_lines) >= 2

    # Fact line should be summarized and may include inline M# pointer as third field
    fact_lines = [json.loads(ln) for ln in lines if ln.startswith('["f"')]
    assert len(fact_lines) == 1
    fact_arr = fact_lines[0]
    assert fact_arr[0] == "f"
    if len(fact_arr) >= 3:
        assert re.match(r"^M#\d+$", fact_arr[2])  # has pointer

    # q-line summarized
    q_lines = [json.loads(ln) for ln in lines if ln.startswith('["q"')]
    assert len(q_lines) == 1
    assert q_lines[0][0] == "q" and isinstance(q_lines[0][2], str)

    # Stats must include density proxy
    assert "density" in stats and 0.0 <= stats["density"] <= 1.0
    assert stats["overflow"]["count"] >= 2
    assert stats["overflow"]["rate"] == stats["overflow"]["count"] / stats["lines_total"]

def test*jsonl_to_prose_roundtrip_signal():
v = JSONLValidator()
js = '\n'.join([
'["r","M"]',
'["g","Compare dates"]',
'["f","Event A: 2001-01-01"]',
'["q","W","Which is earlier?"]'
])
out, * = v.validate_and_overflow(js)
prose = v.jsonl_to_prose(out)
assert "Goal:" in prose and "Fact:" in prose and "Question (W):" in prose

5. Optional: add a smoke sample to the README (append)

Append this section to the bottom of README.md:

## PR-02 quick smoke

````bash
# Mixed format + overflow demo
python scripts/jsonl_guard.py --caps '{"f":5,"q":5}' <<'EOF'
["r","M"]
{"f":"alpha beta gamma delta epsilon zeta eta"}
["q","W","please compare the following two long things"]
plain text line (mixed!)
EOF


---

## What to run (and what to paste as evidence in the PR)

1) **Install (unchanged)**
```bash
make install


Run tests

make test


Quick smoke via CLI

python scripts/jsonl_guard.py --caps '{"f":5,"q":5}' <<'EOF'
["r","M"]
{"f":"alpha beta gamma delta epsilon zeta eta"}
["q","W","please compare the following two long things"]
EOF


Acceptance evidence to paste in PR description:

✅ pytest summary (all green).

✅ jsonl_guard.py JSON output snippet showing:

mixed_format (true/false) and break_line index correct for mixed inputs.

stats.density and stats.overflow.{count,per_tag,rate} present.

out contains:

summarized ["f", "...", "M#k"] (pointer present for facts),

summarized ["q", "W", "..."],

at least two ["o", "...", "M#k", "extractive"] lines.

Commit message
PR-02: JSONL protocol & validator

- Implement tersetalk/protocol_jsonl.py with JSONLValidator:
  * mixed-format detection, lenient→canonical normalization
  * soft-cap enforcement for tags (f,p,g,u,q,t)
  * overflow handling with M# pointers and ["o", ...] records
  * density proxy and overflow stats reporting
  * jsonl_to_prose() and estimate_tokens() utilities
- Add scripts/jsonl_guard.py for CLI validation and overflow demo
- Add tests for detection, normalization, overflow behavior, and prose export
- Export module in tersetalk.__init__
- DoD: tests pass; CLI demonstrates caps + overflow; density metric reported
````
