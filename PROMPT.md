Read through your AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the AGENTS.md. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update AGENTS.md and ask CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly. Before you merge the PR, you must wait for my approval. Also at the end I want you to outline any risks we are seeing in this project, are our expectations aligned with how the progress is going? Will our project succeed and achieve the baselines we expect using your best judgement? Here are more detailed instructions for PR implementation.

### PR Summary

PR‑03 — Memory Store

Role: You are a senior engineer implementing PR‑03 for the TerseTalk project, immediately after PR‑02S merged.

Goal (from spec):
Create a bounded MemoryStore that mints and manages M#<id> references for overflowed content. Support put/get/reset/stats, oldest‑first eviction (by last access time), and deterministic id minting. Provide a smoke CLI and tests. Do not implement summarization, datasets, or the full pipeline yet.

Deliverables:

tersetalk/memory.py — MemoryStore with MAX_ENTRIES=10_000, put/get/reset/stats, and eviction.

scripts/memory_smoke.py — CLI: (a) store & fetch demo; (b) validator demo with MemoryStore injection.

Tests: tests/test_memory_store.py validating eviction, id pattern, reset, stats, and validator integration.

Update tersetalk/**init**.py to export memory.

Strict scope:

Stdlib‑only for this PR. No new pip deps beyond what’s already in the repo.

Do not change PR‑02 validator behavior except by supplying a real memory instance to it in tests/CLI.

Keep all prior tests green.

DoD (Definition of Done):

Oldest‑first eviction works (based on last access time).

put() returns unique refs matching ^M#\d+$.

get() returns the stored text and refreshes last‑access time.

reset() clears state and id counter.

stats() returns {"entries": int, "bytes": int, "oldest": float|None}.

Validator integration: overflow lines reference M#k that are retrievable from MemoryStore.

Tests pass; smoke CLI shows working behavior.

Create/Update the following files exactly

Keep all PR‑00/PR‑01/PR‑02/PR‑02S files intact and working. Only add/modify what’s listed.

1. tersetalk/memory.py (new)
   from **future** import annotations

import time
from typing import Dict, Optional

class MemoryStore:
"""
Bounded key-value store for overflowed text, referenced by ids like "M#23".

    Eviction policy: oldest-by-last-access (access time min).
    Ids are minted monotonically per process/session and reset by reset().
    """
    MAX_ENTRIES: int = 10_000

    def __init__(self) -> None:
        self.store: Dict[str, str] = {}
        self.access_times: Dict[str, float] = {}
        self.counter: int = 0

    # -------- Core API --------
    def put(self, text: str) -> str:
        """
        Store text and return an M# reference.
        Evicts the oldest-by-last-access entry if at capacity.
        """
        if len(self.store) >= self.MAX_ENTRIES:
            self._evict_oldest()

        self.counter += 1
        mid = f"M#{self.counter}"
        # Ensure we only store strings; callers may pass other types
        text = text if isinstance(text, str) else str(text)
        self.store[mid] = text
        self.access_times[mid] = time.time()
        return mid

    def get(self, mid: str) -> Optional[str]:
        """
        Retrieve text by M# reference. Updates last-access time on hit.
        Returns None if the id is not present.
        """
        if mid in self.store:
            self.access_times[mid] = time.time()
            return self.store[mid]
        return None

    def reset(self) -> None:
        """Clear all entries and reset the id counter."""
        self.store.clear()
        self.access_times.clear()
        self.counter = 0

    def stats(self) -> Dict[str, Optional[float]]:
        """
        Memory usage statistics:
        - entries: number of keys
        - bytes: sum of len(text) across entries (ASCII/UTF-8 char count)
        - oldest: earliest last-access timestamp (epoch seconds) or None
        """
        oldest = None
        if self.access_times:
            oldest = min(self.access_times.values())
        return {
            "entries": len(self.store),
            "bytes": sum(len(v) for v in self.store.values()),
            "oldest": oldest,
        }

    # -------- Helpers --------
    def _evict_oldest(self) -> None:
        """Remove the id with the smallest last-access time."""
        if not self.access_times:
            return
        # Select the (id, ts) pair with the smallest timestamp
        oldest_id = min(self.access_times.items(), key=lambda kv: kv[1])[0]
        # Remove from both maps
        self.access_times.pop(oldest_id, None)
        self.store.pop(oldest_id, None)

    # Convenience for tests / introspection
    def __len__(self) -> int:  # pragma: no cover
        return len(self.store)

2. scripts/memory_smoke.py (new)
   from **future** import annotations

import argparse
import json
import sys

from tersetalk.memory import MemoryStore
from tersetalk.protocol_jsonl import JSONLValidator

def do_store(args):
mem = MemoryStore()
ids = [mem.put(s) for s in args.items]
fetched = {mid: mem.get(mid) for mid in ids}
out = {
"mode": "store",
"minted_ids": ids,
"fetched": fetched,
"stats": mem.stats(),
}
print(json.dumps(out, indent=2))

def do_validate(args):
try:
caps = json.loads(args.caps)
if not isinstance(caps, dict):
raise ValueError
except Exception:
print("Error: --caps must be a JSON object", file=sys.stderr)
sys.exit(2)

    data = sys.stdin.read() if args.input == "-" else open(args.input, "r", encoding="utf-8").read()
    mem = MemoryStore()
    validator = JSONLValidator(caps=caps, memory=mem)
    mixed, idx = validator.detect_format_break(data)
    out, stats = validator.validate_and_overflow(data)
    res = {
        "mode": "validate",
        "mixed_format": mixed,
        "break_line": idx,
        "validated_jsonl": out,
        "validator_stats": stats,
        "memory_stats": mem.stats(),
        "o_refs_retrievable": _check_o_refs(out, mem),
    }
    print(json.dumps(res, indent=2))

def \_check_o_refs(jsonl_out: str, mem: MemoryStore) -> bool:
"""Verify that every M# in overflow lines can be dereferenced."""
ok = True
for line in [ln for ln in jsonl_out.splitlines() if ln.strip()]:
if line.startswith('["o"'):
try:
arr = json.loads(line)
except Exception:
continue
if len(arr) >= 3:
mid = arr[2]
if not isinstance(mem.get(mid), str):
ok = False
return ok

def main():
ap = argparse.ArgumentParser(description="PR-03 MemoryStore smoke utility")
sub = ap.add_subparsers(dest="cmd", required=True)

    p_store = sub.add_parser("store", help="Mint M# ids for provided items and fetch them back.")
    p_store.add_argument("items", nargs="+", help="Strings to store")
    p_store.set_defaults(func=do_store)

    p_val = sub.add_parser("validate", help="Validate JSONL with real MemoryStore and show stats.")
    p_val.add_argument("--caps", default='{"f":30,"p":20,"q":30}',
                       help='Per-tag caps JSON, e.g. \'{"f":20,"q":25}\'')
    p_val.add_argument("--input", default="-", help="Path to JSONL file or '-' for stdin.")
    p_val.set_defaults(func=do_validate)

    args = ap.parse_args()
    args.func(args)

if **name** == "**main**":
main()

3. tests/test_memory_store.py (new)
   from **future** import annotations

import json
import re
import time

from tersetalk.memory import MemoryStore
from tersetalk.protocol_jsonl import JSONLValidator

def test_put_get_and_id_pattern():
mem = MemoryStore()
m1 = mem.put("foo")
m2 = mem.put("bar")
assert m1 != m2
assert re.match(r"^M#\d+$", m1)
assert mem.get(m1) == "foo"
assert mem.get(m2) == "bar"
st = mem.stats()
assert st["entries"] == 2
assert st["bytes"] >= len("foo") + len("bar")

def test*oldest_by_last_access_eviction():
mem = MemoryStore()
mem.MAX_ENTRIES = 2 # shrink capacity for test
a = mem.put("A") # oldest access initially
time.sleep(0.002)
b = mem.put("B") # Touch A so B becomes the oldest by last-access
* = mem.get(a)
time.sleep(0.002)
c = mem.put("C") # should evict B (oldest by last access)
assert mem.get(b) is None
assert mem.get(a) == "A"
assert mem.get(c) == "C"
assert mem.stats()["entries"] == 2

def test_reset_clears_and_resets_counter():
mem = MemoryStore()
m1 = mem.put("X")
mem.reset()
assert mem.get(m1) is None
assert mem.stats()["entries"] == 0
m2 = mem.put("Y")
assert m2 == "M#1" # counter restarted

def test_validator_integration_with_real_memory():
mem = MemoryStore()
caps = {"f": 5, "q": 5}
v = JSONLValidator(caps=caps, memory=mem)
long_fact = "alpha beta gamma delta epsilon zeta eta theta"
long_q = "please compare the following two things and return the earlier now"
raw = "\n".join([
'["r","M"]',
json.dumps(["f", long_fact]),
json.dumps(["q", "W", long_q]),
'["g","short goal"]'
])
out, stats = v.validate_and_overflow(raw) # Must have at least two overflow lines
o_lines = [ln for ln in out.splitlines() if ln.startswith('["o"')]
assert len(o_lines) >= 2 # Every overflow M# must be retrievable from memory
for ln in o_lines:
arr = json.loads(ln)
mid = arr[2]
assert isinstance(mem.get(mid), str)

    # Memory should have stored the long items
    mstats = mem.stats()
    assert mstats["entries"] >= 2
    assert "density" in stats and 0.0 <= stats["density"] <= 1.0

4. Update tersetalk/**init**.py (replace file)
   from .\_version import **version**

**all** = ["__version__", "reproducibility", "protocol_jsonl", "structured", "memory"]

What to run (and what to paste as evidence in the PR)

Install (unchanged)

make install

Run tests

make test

Smoke: store/fetch

python scripts/memory_smoke.py store "alpha" "beta" "gamma"

Smoke: validator with real MemoryStore (forces overflow)

python scripts/memory_smoke.py validate --caps '{"f":5,"q":5}' <<'EOF'
["r","M"]
["f","alpha beta gamma delta epsilon zeta"]
["q","W","please compare these very long things and return earlier"]
EOF

Acceptance evidence to paste in PR description:

✅ pytest summary (all tests green).

✅ Output from store showing minted_ids like ["M#1","M#2","M#3"], fetched mapping, and plausible stats.

✅ Output from validate showing:

"mixed_format": false, "o_refs_retrievable": true

"validator_stats" with overflow.count ≥ 2 and a density ∈ [0,1]

"memory_stats.entries" ≥ 2

Commit message
PR-03: Bounded MemoryStore (M#) with oldest-first eviction

- Implement tersetalk/memory.py:
  - put/get/reset/stats and oldest-by-last-access eviction
  - deterministic M# id minting; counter resets on reset()
- Add scripts/memory_smoke.py for store/fetch and validator demos
- Add tests validating eviction, id pattern, reset, stats, and validator integration
- Export 'memory' in package **init**
- DoD: tests pass; smoke outputs show retrievable M# refs and correct stats
