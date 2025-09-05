Read through your AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the AGENTS.md. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update AGENTS.md and ask CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly. Before you merge the PR, you must wait for my approval. Also at the end I want you to outline any risks we are seeing in this project, are our expectations aligned with how the progress is going? Will our project succeed and achieve the baselines we expect using your best judgement? Here are more detailed instructions for PR implementation.

### PR Summary

PR‑04 — Summarization Module

Role: You are a senior engineer implementing PR‑04 right after PR‑03 merged.

Goal (from spec):
Create Summarizer with extractive (default) and llmlingua (optional) methods; integrate it into JSONLValidator so overflow summaries come from Summarizer instead of the PR‑02 stub. Provide a smoke CLI and tests. Do not implement later PRs (hybrid gate, datasets, pipeline, etc.).

Key behaviors:

Extractive summarization (stdlib only): sentence‑scoring with simple TF‑IDF‑like weighting + position bonus; greedily select sentences under a target_tokens budget (via the existing 4‑chars≈1‑token heuristic).

LLMLingua path (optional): if llmlingua is importable, use it; otherwise fallback to the extractive method. Tests must not require llmlingua.

Validator wiring: JSONLValidator must accept a summarizer instance and use it to produce summaries for over‑cap payloads on ["f"], ["p"], ["g"], ["u"], ["t"], and ["q"] (the ["q"] text field). Overflow records’ "method" field should reflect the selected summarizer method (e.g., "extractive" or "llmlingua").

Do not break prior PRs. Default behavior = extractive summarizer if none provided.

DoD (Definition of Done):

tersetalk/summarization.py provides Summarizer(method="extractive"|"llmlingua") with:

.summarize(text: str, tag: str, target_tokens: int = 20) -> str

Extractive: regex sentence split + TF/IDF‑like scoring + position bonus; greedy selection under token budget; final hard trim if needed; returns non‑empty string for non‑empty input.

LLMLingua: import‑guarded; on any failure uses extractive.

tersetalk/protocol_jsonl.py:

JSONLValidator accepts summarizer: Optional[Summarizer] (new), defaulting to Summarizer("extractive").

All overflow summaries come from self.summarizer.summarize(...).

["o", summary, "M#k", "<method>"] uses self.summarizer.method.

scripts/summarize_smoke.py:

CLI that reads text, summarizes with chosen method + target tokens; prints JSON (orig tokens, summary tokens, summary text).

scripts/jsonl_guard.py:

Add --summarizer {extractive,llmlingua} and pass a Summarizer into JSONLValidator.

Tests:

Summarizer extractive returns shorter/equal token count than input for long text and contains salient tokens.

Summarizer llmlingua path does not crash when llmlingua is absent (falls back).

Validator integration: overflow summaries exist, ["o", ..., method] matches the selected method, and M# pointers dereference via MemoryStore.

All tests pass locally without llmlingua/network.

Create/Update the following files exactly

Keep all previous files working; only add/modify what's listed.

1. tersetalk/summarization.py (new)
   from **future** import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Dict

def \_token_estimate(text: str) -> int: # Same heuristic used elsewhere: ~4 chars ≈ 1 token
return max(0, (len(text) + 3) // 4)

\_STOPWORDS = {
"a","an","the","and","or","but","if","then","else","for","to","of","in","on","at",
"by","with","from","as","that","this","these","those","be","is","are","was","were",
"it","its","into","over","under","about","after","before","so","we","you","they",
"he","she","him","her","them","I","my","your","our","their","not","no","yes","do",
"does","did","done","can","could","may","might","must","should","would","will"
}

def \_norm_ws(s: str) -> str:
return re.sub(r"\s+", " ", s).strip()

def \_split_sentences(text: str) -> List[str]:
text = \_norm_ws(text)
if not text:
return [] # Split on ., !, ? followed by space or end. Keep punctuation attached.
parts = re.split(r"(?<=[.!?])\s+", text)
out = [p.strip() for p in parts if p.strip()]
if not out:
out = [text]
return out

def \_words(text: str) -> List[str]:
return re.findall(r"[A-Za-z0-9']+", text.lower())

@dataclass
class Summarizer:
"""
Lightweight summarizer with two modes: - method="extractive" (default): stdlib-only scoring - method="llmlingua": optional; falls back to extractive if not available
"""
method: str = "extractive"

    def summarize(self, text: str, tag: str, target_tokens: int = 20) -> str:
        text = _norm_ws(text)
        if not text:
            return ""
        if target_tokens <= 0:
            return text[:1]

        if self.method == "llmlingua":
            out = self._llmlingua_summary(text, target_tokens)
            if out is not None:
                return out

        # Default / fallback
        return self._extractive_summary(text, target_tokens)

    # --------- Methods ---------
    def _extractive_summary(self, text: str, target_tokens: int) -> str:
        sents = _split_sentences(text)
        if len(sents) == 1:
            # Single sentence; trim hard if needed
            return self._hard_trim(sents[0], target_tokens)

        # Build TF and DF
        sent_tokens: List[List[str]] = []
        df: Dict[str, int] = {}
        for s in sents:
            toks = [w for w in _words(s) if w not in _STOPWORDS]
            if not toks:
                toks = _words(s)  # fallback keep all
            sent_tokens.append(toks)
            for w in set(toks):
                df[w] = df.get(w, 0) + 1

        N = len(sents)
        idf: Dict[str, float] = {}
        for w, d in df.items():
            idf[w] = math.log((N + 1) / (1 + d)) + 1.0  # smoothed

        # Score sentences: sum(TF * IDF)/sqrt(len) + position bonus
        scores: List[float] = []
        for i, toks in enumerate(sent_tokens):
            if toks:
                # TF counts
                tf: Dict[str, int] = {}
                for w in toks:
                    tf[w] = tf.get(w, 0) + 1
                max_tf = max(tf.values())
                core = sum((tf[w] / max_tf) * idf.get(w, 0.0) for w in tf)
                core /= math.sqrt(len(toks) + 1e-6)
            else:
                core = 0.0
            pos_bonus = 0.1 * (1.0 - (i / max(1, N - 1))) if N > 1 else 0.0
            scores.append(core + pos_bonus)

        order = sorted(range(N), key=lambda i: scores[i], reverse=True)

        # Greedy selection under token budget
        chosen: List[int] = []
        acc_text = ""
        for idx in order:
            cand = (acc_text + " " + sents[idx]).strip() if acc_text else sents[idx]
            if _token_estimate(cand) <= target_tokens:
                chosen.append(idx)
                acc_text = cand

        # If nothing fits, take the best sentence and trim it
        if not chosen:
            best = sents[order[0]]
            return self._hard_trim(best, target_tokens)

        # Present in original order for readability
        chosen.sort()
        summary = " ".join(sents[i] for i in chosen)
        return self._hard_trim(summary, target_tokens)

    def _hard_trim(self, text: str, target_tokens: int) -> str:
        if _token_estimate(text) <= target_tokens:
            return text
        # Trim to approx token budget with word boundary and ellipsis
        max_chars = max(1, target_tokens * 4)
        if len(text) <= max_chars:
            return text
        # Cut at last space before limit if available
        cut = text[:max_chars]
        space = cut.rfind(" ")
        if space >= 20:  # avoid microscopic fragments
            cut = cut[:space]
        return cut.rstrip() + "..."

    def _llmlingua_summary(self, text: str, target_tokens: int) -> str | None:
        """
        Attempt LLMLingua compression; fallback to extractive on any error/import failure.
        """
        try:
            from llmlingua import PromptCompressor  # type: ignore
        except Exception:
            return None
        try:
            comp = PromptCompressor()
            result = comp.compress(text, target_token=target_tokens)
            cand = (
                result.get("compressed_prompt")
                or result.get("compressed_text")
                or result.get("prompt")
                or result.get("text")
            )
            if not isinstance(cand, str) or not cand.strip():
                return None
            # Ensure we don't exceed the target; hard-trim if necessary
            return self._hard_trim(cand.strip(), target_tokens)
        except Exception:
            return None

2. Update tersetalk/protocol_jsonl.py to wire the Summarizer

Replace only the parts indicated; keep everything else intact.

Add import at top:

from tersetalk.summarization import Summarizer

Modify JSONLValidator dataclass signature to include summarizer:

@dataclass
class JSONLValidator:
caps: Dict[str, int] = field(default_factory=dict)
memory: Optional[SupportsPut] = None
summarizer: Optional[Summarizer] = None

In **post_init**, set default summarizer if missing:

def **post_init**(self) -> None:
default_caps = {"f": 30, "p": 20, "q": 30, "g": 30, "u": 20, "t": 50}
merged = dict(default_caps)
merged.update(self.caps or {})
self.caps = merged
self.memory = self.memory or \_LocalMemory()
self.overflow_freq: Dict[str, int] = {}
self.summarizer = self.summarizer or Summarizer(method="extractive")

Replace every call to the PR‑02 stub summary with the new summarizer
In both the "q" branch and the generic textual branch ("f","p","g","u","t"), replace:

summary = self.\_summarize(text, cap)
...
out_lines.append(json.dumps(["o", summary, mid, "extractive"]))

with:

summary = self.summarizer.summarize(text, tag, cap)
method = getattr(self.summarizer, "method", "extractive")
...
out_lines.append(json.dumps(["o", summary, mid, method]))

(Optional clean-up): remove the old \_summarize stub (or leave a shim that calls self.summarizer.summarize to avoid any accidental references).

3. scripts/summarize_smoke.py (new)
   from **future** import annotations

import argparse
import json
import sys

from tersetalk.summarization import Summarizer

def main():
ap = argparse.ArgumentParser(description="PR-04 Summarizer smoke tool")
ap.add_argument("--method", choices=["extractive", "llmlingua"], default="extractive")
ap.add_argument("--target-tokens", type=int, default=20)
ap.add_argument("--input", default="-", help="Path or '-' for stdin")
args = ap.parse_args()

    text = sys.stdin.read() if args.input == "-" else open(args.input, "r", encoding="utf-8").read()
    s = Summarizer(method=args.method)
    summary = s.summarize(text, tag="t", target_tokens=args.target_tokens)

    def est(t: str) -> int:
        return max(0, (len(t) + 3) // 4)

    print(json.dumps({
        "method": args.method,
        "target_tokens": args.target_tokens,
        "orig_tokens_est": est(text),
        "summary_tokens_est": est(summary),
        "summary": summary
    }, indent=2))

if **name** == "**main**":
main()

4. Update scripts/jsonl_guard.py to accept --summarizer

Replace the file’s contents with the version below (only differences are imports + arg + validator wiring).

from **future** import annotations

import argparse
import json
import sys
from tersetalk.protocol_jsonl import JSONLValidator
from tersetalk.summarization import Summarizer

def main():
ap = argparse.ArgumentParser(description="TerseTalk JSONL guard (PR-02+PR-04)")
ap.add_argument("--caps", type=str, default='{"f":30,"p":20,"q":30}',
help='Per-tag caps JSON, e.g. \'{"f":20,"q":25}\'')
ap.add_argument("--summarizer", choices=["extractive","llmlingua"], default="extractive",
help="Summarizer method for overflow summaries.")
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

    validator = JSONLValidator(caps=caps, summarizer=Summarizer(method=args.summarizer))
    mixed, idx = validator.detect_format_break(data)
    if mixed and args.fail_on_mixed:
        print(json.dumps({"mixed_format": True, "break_line": idx}), file=sys.stderr)
        sys.exit(3)

    out, stats = validator.validate_and_overflow(data)
    res = {"mixed_format": mixed, "break_line": idx, "stats": stats, "out": out}
    print(json.dumps(res, indent=2))

if **name** == "**main**":
main()

5. Update tersetalk/**init**.py to export summarization (replace file)
   from .\_version import **version**

**all** = ["__version__", "reproducibility", "protocol_jsonl", "structured", "memory", "summarization"]

6. tests/test_summarization.py (new)
   from **future** import annotations

from tersetalk.summarization import Summarizer

def test_extractive_produces_shorter_or_equal_and_keeps_keywords():
text = (
"Alpha beta gamma delta. Gamma delta epsilon zeta eta. "
"This sentence is filler. Alpha appears again with beta and gamma."
)
s = Summarizer(method="extractive")
out = s.summarize(text, tag="t", target_tokens=10)
assert isinstance(out, str) and len(out) > 0 # Should be <= target tokens estimate
def est(t: str) -> int: return (len(t) + 3)//4
assert est(out) <= 10 or est(text) <= 10 # Keep at least one salient token
lowered = out.lower()
assert ("alpha" in lowered) or ("gamma" in lowered) or ("delta" in lowered)

def test_llmlingua_path_falls_back_if_unavailable():
text = " ".join(["alpha beta gamma delta epsilon zeta"] \* 5)
s = Summarizer(method="llmlingua")
out = s.summarize(text, tag="f", target_tokens=8)
assert isinstance(out, str) and len(out) > 0 # Should not be longer than original
def est(t: str) -> int: return (len(t) + 3)//4
assert est(out) <= est(text)

7. tests/test_validator_with_summarizer.py (new)
   from **future** import annotations

import json
import re

from tersetalk.protocol_jsonl import JSONLValidator
from tersetalk.memory import MemoryStore
from tersetalk.summarization import Summarizer

def test_validator_overflow_uses_summarizer_and_sets_method_field():
mem = MemoryStore()
caps = {"f": 5, "q": 5}
v = JSONLValidator(caps=caps, memory=mem, summarizer=Summarizer(method="extractive"))
raw = "\n".join([
'["r","M"]',
json.dumps(["f", "alpha beta gamma delta epsilon zeta eta"]),
json.dumps(["q", "W", "please compare these very long things and return earlier"]),
'["g","short goal"]'
])
out, stats = v.validate_and_overflow(raw)

    # At least two overflow lines; method should match summarizer
    o_lines = [json.loads(ln) for ln in out.splitlines() if ln.startswith('["o"')]
    assert len(o_lines) >= 2
    for arr in o_lines:
        assert arr[0] == "o"
        # check M# format
        assert re.match(r"^M#\d+$", arr[2])
        # method reflects selected summarizer
        assert arr[3] == "extractive"

    # Each M# dereferences
    for arr in o_lines:
        assert isinstance(mem.get(arr[2]), str)

    # Density present
    assert "density" in stats and 0.0 <= stats["density"] <= 1.0

def test_validator_with_llmlingua_selection_does_not_crash_without_pkg():
mem = MemoryStore()
v = JSONLValidator(caps={"f": 5}, memory=mem, summarizer=Summarizer(method="llmlingua"))
raw = json.dumps(["f", "alpha beta gamma delta epsilon zeta eta"])
out, stats = v.validate_and_overflow(raw) # At least one overflow line present
assert any(ln.startswith('["o"') for ln in out.splitlines())

What to run (and what to paste as evidence in the PR)

Install (unchanged)

make install

Run tests

make test

Summarizer smoke (extractive)

python scripts/summarize_smoke.py --method extractive --target-tokens 12 <<'EOF'
The Mars Climate Orbiter was lost due to a metric-imperial mismatch. NASA later updated procedures to prevent similar failures. The incident occurred in 1999.
EOF

Guard with summarizer selection (forces overflow)

python scripts/jsonl_guard.py --summarizer extractive --caps '{"f":6,"q":6}' <<'EOF'
["r","M"]
["f","The Mars Climate Orbiter was lost due to a metric-imperial mismatch. Engineers reported root causes in follow-up documents."]
["q","W","Explain the failure succinctly and list the year."]
EOF

Acceptance evidence to paste in the PR description:

✅ pytest summary (all green).

✅ Smoke JSON from summarize_smoke.py showing: method, orig_tokens_est, summary_tokens_est (≤), and summary.

✅ Guard JSON snippet showing:

stats.overflow.count ≥ 1 and out containing ["o", ..., "M#k", "extractive"] (or "llmlingua" if chosen),

M# pointers retrievable (you can also run the PR‑03 memory smoke if desired).

Commit message
PR-04: Summarization module + validator integration

- Add tersetalk/summarization.py with Summarizer(method={"extractive","llmlingua"})
  - stdlib extractive summarization (sentence scoring + position bonus)
  - optional llmlingua path (import-guarded) with safe fallback
- Wire Summarizer into JSONLValidator; overflow lines now include selected method
- Add scripts/summarize_smoke.py to demo summarization
- Enhance scripts/jsonl_guard.py with --summarizer flag
- Add tests for extractive behavior, llmlingua fallback, and validator integration
- All tests pass without llmlingua/network; default summarizer=extractive
