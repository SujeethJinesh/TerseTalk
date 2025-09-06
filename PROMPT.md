Read through your AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the AGENTS.md. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update AGENTS.md and ask CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly. Before you merge the PR, you must wait for my approval. Also at the end I want you to outline any risks we are seeing in this project, are our expectations aligned with how the progress is going? Will our project succeed and achieve the baselines we expect using your best judgement? Absolutely make sure you're reporting results truthfully and honestly. Avoid fake, mocked, or other non genuine results. You should also analyze the results we get for each run and determine if they meet our figures of merit, when you report back, it's crucial to include that analysis (e.g. compression amount, failure rate, latency, etc.). We should be aiming to properly fix things and run proper evaluations. Here are more detailed instructions for PR implementation.

### PR Summary

PR‑MB — Microbenchmarks

Role: You are a senior engineer implementing the microbenchmark suite after PR‑H4 merged.

Goals (from spec):

Provide measurable “10× somewhere” via MB‑1.

Provide ≥5× speedup in MB‑2.

Report bytes‑on‑wire & SerDe timings in MB‑3 (no strict ratio requirement), keeping optional formats import‑guarded.

Constraints:

No network. No extra deps required. (simdjson, msgpack, protobuf may be measured if present but must be optional and import‑guarded.)

Benchmarks must finish in seconds on CI; use parameterized sizes and a fast mode.

Tests must be deterministic and succeed on modest CI hardware.

Deliverables:

benchmarks/tag_extraction.py — MB‑1 implementation

benchmarks/streaming_boundaries.py — MB‑2 implementation

benchmarks/serde_bytes.py — MB‑3 implementation (optional formats guarded)

benchmarks/run_all.py — CLI runner that prints a JSON summary (fast by default)

benchmarks/**init**.py — empty file to make it a package

tests/test_benchmarks.py — asserts MB‑1 ≥10× and MB‑2 ≥5× on reduced sizes; checks MB‑3 byte ratio

(Optional nicety) scripts/benchmarks_run_all.py — thin wrapper to run the suite with CLI flags

Create/Update these files exactly

1. benchmarks/**init**.py (new)

# Package marker for benchmarks

2. benchmarks/tag_extraction.py (new)
   from **future** import annotations

import json
import random
import re
import time
from typing import Dict, List, Tuple

TAGS = ["f", "g", "p", "u", "q"] # fact, goal, plan, assumption, question
WORDS = (
"alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
"omicron pi rho sigma tau upsilon phi chi psi omega"
).split()

def _rand_words(rng: random.Random, lo: int, hi: int) -> str:
n = rng.randint(lo, hi)
return " ".join(rng.choice(WORDS) for _ in range(n))

def make_corpora(n: int = 40_000, seed: int = 0) -> Tuple[List[str], List[str]]:
"""
Build parallel corpora: - jsonl_lines: canonical array form, e.g., ["f","..."] or ["q","W","...?"] - free_lines: free-form English with labeled fields like `fact: ...;`
"""
rng = random.Random(seed)
jsonl_lines: List[str] = []
free_lines: List[str] = []

    for i in range(n):
        tag = rng.choice(TAGS)
        if tag == "q":
            text = _rand_words(rng, 10, 28)
            jsonl_lines.append(json.dumps(["q", "W", text]))
            free_lines.append(f"role: manager; question: {text}; please answer.")
        else:
            text = _rand_words(rng, 12, 24)
            jsonl_lines.append(json.dumps([tag, text]))
            # Expand to increase regex work vs. typed JSONL first-char path
            name = {
                "f": "fact",
                "g": "goal",
                "p": "plan",
                "u": "assumption",
            }[tag]
            free_lines.append(f"{name}: {text}; notes: {_rand_words(rng, 5, 12)}.")

    return jsonl_lines, free_lines

# ---- Extraction methods ----

def extract_tag_jsonl_fast(line: str) -> str | None:
"""
Fast-path tag extraction for canonical JSONL array form:
line like ["f","text"] or ["q","W","text"] → tag is the 2nd char after leading [".
Falls back to json.loads on mismatch.
"""
s = line.lstrip() # common canonical: ["x",...
if len(s) >= 4 and s[0] == "[" and s[1] == '"' and s[3] == '"':
return s[2]
try:
arr = json.loads(s)
return arr[0] if isinstance(arr, list) and arr else None
except Exception:
return None

# compiled once (realistic baseline)

COMPILED = re.compile(
r"\b(?P<tag>fact|goal|plan|assumption|question)\b\s*:\s*",
re.IGNORECASE,
)

def extract_tag_freeform_compiled(line: str) -> str | None:
m = COMPILED.search(line)
if not m:
return None
t = m.group("tag").lower()
return t[0] if t else None

def extract_tag_freeform_uncompiled(line: str) -> str | None:
"""
Intentionally heavier baseline (naive code path seen in many prototypes):
recompiles the regex per call. This is the 'worst practice' baseline.
"""
m = re.search(
r"\b(?P<tag>fact|goal|plan|assumption|question)\b\s*:\s*",
line,
re.IGNORECASE,
)
if not m:
return None
t = m.group("tag").lower()
return t[0] if t else None

def \_timeit(fn, data: List[str]) -> float:
start = time.perf_counter()
sink = 0
for x in data:
t = fn(x) # cheap sink to avoid dead-code elimination
sink += 1 if t else 0
end = time.perf_counter() # small anti-optim variable read
if sink < 0:
print("impossible", sink)
return end - start

def benchmark_tag_extraction(n: int = 40_000, seed: int = 0) -> Dict:
jsonl_lines, free_lines = make_corpora(n=n, seed=seed)

    t_jsonl = _timeit(extract_tag_jsonl_fast, jsonl_lines)
    t_free_compiled = _timeit(extract_tag_freeform_compiled, free_lines)
    t_free_uncompiled = _timeit(extract_tag_freeform_uncompiled, free_lines)

    return {
        "n": n,
        "jsonl_seconds": t_jsonl,
        "freeform_compiled_seconds": t_free_compiled,
        "freeform_uncompiled_seconds": t_free_uncompiled,
        "speedup_vs_compiled": (t_free_compiled / t_jsonl) if t_jsonl > 0 else float("inf"),
        "speedup_vs_uncompiled": (t_free_uncompiled / t_jsonl) if t_jsonl > 0 else float("inf"),
    }

3. benchmarks/streaming_boundaries.py (new)
   from **future** import annotations

import random
import re
import time
from typing import Dict, Tuple, List

from .tag_extraction import WORDS, TAGS, \_rand_words

def make_streams(n_msgs: int = 40_000, seed: int = 0) -> Tuple[str, str]:
"""
Build two streams of concatenated messages: - jsonl_stream: one JSON value per line (O(1) boundary detection) - free_stream: multi-sentence prose with punctuation, abbreviations
"""
rng = random.Random(seed) # JSONL stream: a variety of short lines (reuse from tag gen)
jsonl_lines: List[str] = []
for i in range(n_msgs):
tag = rng.choice(TAGS)
if tag == "q":
text = \_rand_words(rng, 8, 18)
jsonl_lines.append(f'["q","W","{text}"]')
else:
text = \_rand_words(rng, 8, 18)
jsonl_lines.append(f'["{tag}","{text}"]')
jsonl_stream = "\n".join(jsonl_lines)

    # Free-form stream: approximate sentence boundaries with tricky cases
    abbrs = ["e.g.", "i.e.", "Dr.", "Mr.", "Ms.", "vs."]
    sentences: List[str] = []
    for i in range(n_msgs):
        s1 = f"{_rand_words(rng, 5, 12)}."
        s2 = f"{rng.choice(abbrs)} {_rand_words(rng, 4, 9)}."
        s3 = f"{_rand_words(rng, 4, 9)}?"
        s4 = f"{_rand_words(rng, 4, 9)}!"
        sentences.append(" ".join([s1, s2, s3, s4]))
    free_stream = " ".join(sentences)
    return jsonl_stream, free_stream

# JSONL O(1) boundary detector: scan for '\n'

def count_jsonl_boundaries(stream: str) -> int:
if not stream:
return 0
cnt = 0
pos = 0
while True:
i = stream.find("\n", pos)
if i == -1:
break
cnt += 1
pos = i + 1 # last line (if not ending with newline)
if stream[-1] != "\n":
cnt += 1
return cnt

# Free-form sentence boundary with heuristic regex (heavy)

BOUNDARY_RE = re.compile(
r'(?<!\b(?:e\.g|i\.e|mr|mrs|ms|dr|vs))' # negative lookbehind on common abbrev
r'(?<=[.!?])\s+' # end punctuation + space
r'(?=["\(\[]?[A-Z0-9])', # next sentence starts with capital/number
re.IGNORECASE
)

def count_freeform_sentences(stream: str) -> int: # Count matches as boundaries; add 1 for the last fragment
if not stream:
return 0
return len(BOUNDARY_RE.findall(stream)) + 1

def \_timeit(fn, arg: str) -> float:
start = time.perf_counter()
val = fn(arg)
end = time.perf_counter() # lightweight sink
if val < 0:
print("impossible", val)
return end - start

def benchmark_streaming(n_msgs: int = 40_000, seed: int = 0) -> Dict:
j_stream, f_stream = make_streams(n_msgs=n_msgs, seed=seed)
t_jsonl = \_timeit(count_jsonl_boundaries, j_stream)
t_free = \_timeit(count_freeform_sentences, f_stream)
return {
"n_msgs": n_msgs,
"jsonl_seconds": t_jsonl,
"freeform_seconds": t_free,
"speedup": (t_free / t_jsonl) if t_jsonl > 0 else float("inf"),
}

4. benchmarks/serde_bytes.py (new)
   from **future** import annotations

import json
import random
import time
from typing import Dict, List, Tuple, Any

from .tag_extraction import WORDS, \_rand_words

def make_messages(n: int = 5_000, seed: int = 0) -> List[Dict[str, Any]]:
rng = random.Random(seed)
msgs = []
for i in range(n):
goal = f"Compare values; case {i}."
facts = [_rand_words(rng, 8, 16), _rand_words(rng, 6, 12)]
if i % 4 == 0:
facts.append(\_rand_words(rng, 6, 12))
q = "Which is earlier?"
msgs.append({"role": "M", "goal": goal, "facts": facts, "q_to": "W", "question": q})
return msgs

def serialize_jsonl(msgs: List[Dict[str, Any]]) -> str:
lines: List[str] = []
for m in msgs:
lines.append(json.dumps(["r", m["role"]]))
lines.append(json.dumps(["g", m["goal"]]))
for f in m["facts"]:
lines.append(json.dumps(["f", f]))
lines.append(json.dumps(["q", m["q_to"], m["question"]]))
return "\n".join(lines)

def serialize_freeform(msgs: List[Dict[str, Any]]) -> str:
parts: List[str] = []
for m in msgs:
parts.append(f"Role: {m['role']}\n")
parts.append(f"Goal: {m['goal']}\n")
for f in m["facts"]:
parts.append(f"Fact: {f}\n")
parts.append(f"Question({m['q_to']}): {m['question']}\n")
parts.append("\n")
return "".join(parts)

def maybe_msgpack(msgs: List[Dict[str, Any]]) -> Tuple[bool, bytes, float]:
try:
import msgpack # type: ignore
except Exception:
return False, b"", 0.0
start = time.perf_counter()
blob = msgpack.packb(msgs, use_bin_type=True)
t = time.perf_counter() - start
return True, blob, t

def maybe_protobuf(msgs: List[Dict[str, Any]]) -> Tuple[bool, bytes, float]:
"""
Guarded placeholder: skip real schema to keep PR dependency-free.
"""
return False, b"", 0.0

def benchmark_serde_bytes(n: int = 5_000, seed: int = 0) -> Dict:
msgs = make_messages(n=n, seed=seed)

    t0 = time.perf_counter()
    jsonl = serialize_jsonl(msgs)
    t_jsonl = time.perf_counter() - t0

    t1 = time.perf_counter()
    free = serialize_freeform(msgs)
    t_free = time.perf_counter() - t1

    b_jsonl = len(jsonl.encode("utf-8"))
    b_free = len(free.encode("utf-8"))

    has_msgpack, blob_mp, t_mp = maybe_msgpack(msgs)
    res = {
        "n_msgs": n,
        "jsonl_seconds": t_jsonl,
        "freeform_seconds": t_free,
        "jsonl_bytes": b_jsonl,
        "freeform_bytes": b_free,
        "bytes_ratio_jsonl_over_freeform": (b_jsonl / b_free) if b_free > 0 else 0.0,
    }
    if has_msgpack:
        res.update({
            "msgpack_bytes": len(blob_mp),
            "msgpack_seconds": t_mp,
            "bytes_ratio_msgpack_over_jsonl": (len(blob_mp) / b_jsonl) if b_jsonl > 0 else 0.0,
        })
    return res

5. benchmarks/run_all.py (new)
   from **future** import annotations

import argparse
import json

from .tag_extraction import benchmark_tag_extraction
from .streaming_boundaries import benchmark_streaming
from .serde_bytes import benchmark_serde_bytes

def main():
ap = argparse.ArgumentParser(description="TerseTalk Microbenchmark Suite")
ap.add_argument("--fast", action="store_true", help="Use smaller sizes (CI-friendly).")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

    if args.fast:
        n_tag = 20_000
        n_stream = 20_000
        n_serde = 2_000
    else:
        n_tag = 40_000
        n_stream = 40_000
        n_serde = 5_000

    mb1 = benchmark_tag_extraction(n=n_tag, seed=args.seed)
    mb2 = benchmark_streaming(n_msgs=n_stream, seed=args.seed)
    mb3 = benchmark_serde_bytes(n=n_serde, seed=args.seed)

    report = {
        "MB1_tag_extraction": mb1,
        "MB2_streaming_boundaries": mb2,
        "MB3_serde_bytes": mb3,
        "notes": "Times are wall-clock seconds (time.perf_counter). JSON is valid YAML.",
    }
    print(json.dumps(report, indent=2))

if **name** == "**main**":
main()

6. scripts/benchmarks_run_all.py (new, thin wrapper)
   from **future** import annotations

import subprocess
import sys

if **name** == "**main**":
sys.exit(subprocess.call([sys.executable, "-m", "benchmarks.run_all", "--fast"]))

7. tests/test_benchmarks.py (new)
   from **future** import annotations

from benchmarks.tag_extraction import benchmark_tag_extraction
from benchmarks.streaming_boundaries import benchmark_streaming
from benchmarks.serde_bytes import benchmark_serde_bytes

def test_mb1_tag_extraction_speedup_ge_10x(): # Smaller n for CI; uncompiled baseline should be dramatically slower
res = benchmark_tag_extraction(n=20_000, seed=123)
assert res["speedup_vs_uncompiled"] >= 10.0, res # compiled baseline may vary; ensure it's at least >1.5× for sanity
assert res["speedup_vs_compiled"] > 1.5, res

def test_mb2_streaming_boundaries_speedup_ge_5x():
res = benchmark_streaming(n_msgs=20_000, seed=7)
assert res["speedup"] >= 5.0, res

def test_mb3_jsonl_bytes_significantly_smaller():
res = benchmark_serde_bytes(n=2_000, seed=42) # JSONL should be at least 30% smaller than a labeled free-form equivalent
assert res["bytes_ratio_jsonl_over_freeform"] <= 0.70, res

What to run (and what to paste as evidence in the PR)

Run tests

make test

Run the suite (fast mode)

python -m benchmarks.run_all --fast

# or

python scripts/benchmarks_run_all.py

Acceptance evidence to paste in the PR description:

✅ pytest summary (all green).

✅ benchmarks.run_all JSON showing:

MB‑1: speedup_vs_uncompiled ≥ 10 (and speedup_vs_compiled > 1.5)

MB‑2: speedup ≥ 5

MB‑3: bytes_ratio_jsonl_over_freeform ≤ 0.70

(If present) Optional fields for MsgPack in MB‑3.

Commit message
PR-MB: Microbenchmark Suite (MB‑1/MB‑2/MB‑3) + CI tests

- MB‑1 (tag_extraction): JSONL typed fast‑path vs free‑form regex (compiled/uncompiled)
- MB‑2 (streaming): O(1) newline boundary detection vs heuristic sentence regex
- MB‑3 (SerDe & bytes): JSONL vs free‑form; optional msgpack guarded
- Add benchmarks/run_all.py and scripts/benchmarks_run_all.py
- Deterministic, stdlib‑only; tests assert ≥10× (MB‑1), ≥5× (MB‑2), and JSONL ≤70% bytes (MB‑3)
