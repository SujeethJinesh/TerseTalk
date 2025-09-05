Read through your AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the AGENTS.md. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update AGENTS.md and ask CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly. Before you merge the PR, you must wait for my approval. Here are more detailed instructions.

### PR Summary

PR‑02S — Structured Output with Instructor

Role: You are a senior engineer implementing PR‑02S for the TerseTalk project, immediately after PR‑02 merged.

Goal (from spec):
Provide Pydantic models for TerseTalk lines and a structured generation harness using Instructor + Pydantic. Keep it offline‑safe by default (Echo). This PR does not call datasets or run full experiments.

Deliverables:

tersetalk/structured.py — Pydantic models for all tags; generic TerseTalkLine (Instructor‑friendly); converters to specific typed lines; Instructor harness + Echo fallback; JSONL helpers.

scripts/structured_demo.py — CLI to generate lines via Echo or Instructor, convert to JSONL, and run through JSONLValidator.

Tests that verify model validation, conversion, JSONL emission, and validator interop (no network required).

Update runtime deps to include pydantic, instructor, openai (installed via requirements.txt; tests never require a running server).

Strict scope:

Do not implement PR‑03 MemoryStore, PR‑04 Summarizer, or PR‑05 pipeline.

No dataset/model evaluation; just schema + harness + demo.

Instructor calls are optional and off by default; Echo path is used in tests.

DoD (Definition of Done):

All tags (r,g,f,u,p,q,d,v,o,t,x) have typed Pydantic models with to_array() converters.

TerseTalkLine (generic) + converter to typed lines.

Instructor harness compiles (imports guarded); Echo is default.

Demo CLI prints a JSON blob with: mode, lines_jsonl, validator.stats, and compliance_ok=true (no mixed format).

Tests pass locally with no network.

Create/Update these files exactly

Keep PR‑00/PR‑01/PR‑02 files intact. Only add/modify the listed files.

1. Update requirements.txt (append these runtime deps)
   pydantic>=2.0.0
   instructor>=1.0.0
   openai>=1.3.0

2. Update tersetalk/**init**.py to export structured (replace file)
   from .\_version import **version**

**all** = ["__version__", "reproducibility", "protocol_jsonl", "structured"]

3. tersetalk/structured.py (new)
   from **future** import annotations

import json
import os
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError

# -----------------------------

# Generic line for Instructor

# -----------------------------

class TerseTalkLine(BaseModel):
"""
Generic typed line that is easy for Instructor to produce.
Use convert_to_specific(...) to map into strongly-typed models.
"""
tag: Literal["r", "g", "f", "u", "p", "q", "d", "v", "o", "t", "x"]
payload: List[str]

    def to_array(self) -> List[str]:
        return [self.tag] + self.payload

# -----------------------------

# Strongly-typed line models

# -----------------------------

RefStr = Annotated[str, Field(pattern=r"^M#\d+$")]

class \_BaseLine(BaseModel):
def to_array(self) -> List[str]: # pragma: no cover - overridden
raise NotImplementedError

class RoleLine(\_BaseLine):
tag: Literal["r"] = "r"
role: Literal["M", "W", "C"]

    def to_array(self) -> List[str]:
        return ["r", self.role]

class GoalLine(\_BaseLine):
tag: Literal["g"] = "g"
text: str

    def to_array(self) -> List[str]:
        return ["g", self.text]

class FactLine(\_BaseLine):
tag: Literal["f"] = "f"
text: str
ref: Optional[RefStr] = None # Optional inline M# pointer

    def to_array(self) -> List[str]:
        arr = ["f", self.text]
        if self.ref:
            arr.append(self.ref)
        return arr

class AssumptionLine(\_BaseLine):
tag: Literal["u"] = "u"
text: str

    def to_array(self) -> List[str]:
        return ["u", self.text]

class PlanLine(\_BaseLine):
tag: Literal["p"] = "p"
text: str

    def to_array(self) -> List[str]:
        return ["p", self.text]

class QuestionLine(\_BaseLine):
tag: Literal["q"] = "q"
to: Literal["M", "W", "C"]
text: str

    def to_array(self) -> List[str]:
        return ["q", self.to, self.text]

class DeltaRefLine(\_BaseLine):
tag: Literal["d"] = "d"
ref: RefStr

    def to_array(self) -> List[str]:
        return ["d", self.ref]

class VerdictLine(\_BaseLine):
tag: Literal["v"] = "v"
verdict: Literal["A", "R", "E"] # Accept / Revise / Escalate

    def to_array(self) -> List[str]:
        return ["v", self.verdict]

class OverflowLine(\_BaseLine):
tag: Literal["o"] = "o"
summary: str
ref: RefStr
method: Literal["extractive", "llmlingua"] = "extractive"

    def to_array(self) -> List[str]:
        return ["o", self.summary, self.ref, self.method]

class FreeTextLine(\_BaseLine):
tag: Literal["t"] = "t"
text: str

    def to_array(self) -> List[str]:
        return ["t", self.text]

class ContextLine(\_BaseLine):
tag: Literal["x"] = "x"
key: str
value: str

    def to_array(self) -> List[str]:
        return ["x", self.key, self.value]

AnyTypedLine = Union[
RoleLine, GoalLine, FactLine, AssumptionLine, PlanLine,
QuestionLine, DeltaRefLine, VerdictLine, OverflowLine, FreeTextLine, ContextLine
]

# -----------------------------

# Converters

# -----------------------------

def terse_to_specific(lines: List[TerseTalkLine]) -> List[AnyTypedLine]:
"""
Convert generic TerseTalkLine objects into specific typed models.
Unknown shapes are mapped to FreeTextLine.
"""
out: List[AnyTypedLine] = []
for ln in lines:
t = ln.tag
pl = ln.payload or []
try:
if t == "r":
out.append(RoleLine(role=(pl[0] if pl else "M")))
elif t == "g":
out.append(GoalLine(text=(pl[0] if pl else "")))
elif t == "f":
text = pl[0] if pl else ""
ref = pl[1] if len(pl) > 1 else None
out.append(FactLine(text=text, ref=ref))
elif t == "u":
out.append(AssumptionLine(text=(pl[0] if pl else "")))
elif t == "p":
out.append(PlanLine(text=(pl[0] if pl else "")))
elif t == "q":
to = pl[0] if pl else "W"
text = pl[1] if len(pl) > 1 else ""
out.append(QuestionLine(to=to, text=text))
elif t == "d":
out.append(DeltaRefLine(ref=(pl[0] if pl else "M#1")))
elif t == "v":
out.append(VerdictLine(verdict=(pl[0] if pl else "A")))
elif t == "o":
summary = pl[0] if pl else ""
ref = pl[1] if len(pl) > 1 else "M#1"
method = pl[2] if len(pl) > 2 else "extractive"
out.append(OverflowLine(summary=summary, ref=ref, method=method)) # type: ignore[arg-type]
elif t == "t":
out.append(FreeTextLine(text=(pl[0] if pl else "")))
elif t == "x":
key = pl[0] if pl else ""
val = pl[1] if len(pl) > 1 else ""
out.append(ContextLine(key=key, value=val))
else:
out.append(FreeTextLine(text=json.dumps(ln.to_array())))
except ValidationError:
out.append(FreeTextLine(text=json.dumps(ln.to_array())))
return out

def lines_to_jsonl(lines: List[AnyTypedLine]) -> str:
"""Emit newline-delimited JSON (canonical array form)."""
return "\n".join(json.dumps(l.to_array(), ensure_ascii=False) for l in lines)

# -----------------------------

# Structured generation harness

# -----------------------------

class EchoGenerator:
"""
Offline-safe, deterministic stub for CI and tests.
"""
def generate(self, goal: str, facts: List[str], question: str) -> List[AnyTypedLine]:
lines: List[AnyTypedLine] = [
RoleLine(role="M"),
GoalLine(text=goal),
]
for f in facts:
lines.append(FactLine(text=f))
lines.append(QuestionLine(to="W", text=question))
return lines

class InstructorGenerator:
"""
Optional: requires 'instructor' and 'openai' installed and a server
speaking the OpenAI API (e.g., Ollama at http://localhost:11434/v1).
"""
def **init**(
self,
model: str = "mistral",
base_url: str = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
api_key: str = os.environ.get("OPENAI_API_KEY", "ollama"),
max_retries: int = 2,
) -> None:
try:
import instructor # type: ignore
from openai import OpenAI # type: ignore
except Exception as e: # pragma: no cover - import guarded
raise RuntimeError(
"InstructorGenerator requires 'instructor' and 'openai' to be installed."
) from e

        self._instructor = instructor
        self._OpenAI = OpenAI
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_retries = max_retries
        self._client = self._instructor.patch(
            self._OpenAI(base_url=self.base_url, api_key=self.api_key)
        )

    def generate(self, system_prompt: str, user_prompt: str) -> List[AnyTypedLine]:
        """
        Ask the model to produce a list of generic TerseTalkLine objects,
        then convert to specific typed lines.
        """
        result: List[TerseTalkLine] = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=List[TerseTalkLine],  # Typed magic by Instructor
            max_retries=self.max_retries,
        )
        return terse_to_specific(result)

4. scripts/structured_demo.py (new)
   from **future** import annotations

import argparse
import json
import sys

from tersetalk.structured import (
EchoGenerator,
InstructorGenerator,
lines_to_jsonl,
GoalLine, FactLine, QuestionLine, RoleLine, AnyTypedLine
)
from tersetalk.protocol_jsonl import JSONLValidator

def main():
ap = argparse.ArgumentParser(description="PR-02S Structured generation demo")
ap.add_argument("--mode", choices=["echo", "instructor"], default="echo",
help="Generation mode (default: echo)")
ap.add_argument("--model", default="mistral", help="Model name for Instructor mode")
ap.add_argument("--goal", default="Compare dates of two events; return the earlier.",
help="Goal text")
ap.add_argument("--facts", nargs="\*", default=["Event A: 2001-07-16", "Event B: 1999-05-02"],
help="Facts list")
ap.add_argument("--question", default="Which is earlier?", help="Question text")
ap.add_argument("--caps", default='{"f":30,"p":20,"q":30}', help="Caps JSON for validator")
args = ap.parse_args()

    # Produce lines
    if args.mode == "echo":
        gen = EchoGenerator()
        lines = gen.generate(args.goal, args.facts, args.question)
        sys_prompt = "You are a Worker."
        usr_prompt = "Echo generator used (offline)."
    else:
        try:
            gen = InstructorGenerator(model=args.model)
        except Exception as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            sys.exit(2)
        sys_prompt = "You are a Manager. Produce concise TerseTalk lines."
        # In practice you'd describe the schema; we keep it brief for the demo.
        usr_prompt = (
            f"Goal: {args.goal}\nFacts: {args.facts}\n"
            f'Question: {args.question}\nRespond as a list of TerseTalkLine items.'
        )
        lines = gen.generate(sys_prompt, usr_prompt)

    # JSONL + validation
    jsonl = lines_to_jsonl(lines)
    try:
        caps = json.loads(args.caps)
        if not isinstance(caps, dict):
            raise ValueError
    except Exception:
        print("Error: --caps must be JSON object", file=sys.stderr)
        sys.exit(2)

    validator = JSONLValidator(caps=caps)
    mixed, idx = validator.detect_format_break(jsonl)
    out, stats = validator.validate_and_overflow(jsonl)

    res = {
        "mode": args.mode,
        "system_prompt": sys_prompt,
        "user_prompt": usr_prompt,
        "lines_count": len(lines),
        "compliance_ok": (not mixed),
        "break_line": idx,
        "lines_jsonl": jsonl,
        "validated_jsonl": out,
        "validator_stats": stats,
    }
    print(json.dumps(res, indent=2))

if **name** == "**main**":
main()

5. tests/test_structured_models.py (new)
   from **future** import annotations

import json

from tersetalk.structured import (
TerseTalkLine, terse_to_specific, lines_to_jsonl,
RoleLine, GoalLine, FactLine, QuestionLine, FreeTextLine, EchoGenerator
)
from tersetalk.protocol_jsonl import JSONLValidator

def test_specific_models_to_array_and_jsonl():
lines = [
RoleLine(role="M"),
GoalLine(text="Compare dates"),
FactLine(text="Event A: 2001-01-01"),
QuestionLine(to="W", text="Which is earlier?"),
]
js = lines_to_jsonl(lines).splitlines()
assert json.loads(js[0]) == ["r", "M"]
assert json.loads(js[1])[0] == "g"
assert json.loads(js[3]) == ["q", "W", "Which is earlier?"]

def test_generic_to_specific_conversion():
raw = [
TerseTalkLine(tag="r", payload=["M"]),
TerseTalkLine(tag="f", payload=["some fact", "M#12"]),
TerseTalkLine(tag="q", payload=["W", "a question"]),
TerseTalkLine(tag="x", payload=["k", "v"]),
]
out = terse_to_specific(raw)
assert isinstance(out[0], RoleLine)
assert isinstance(out[1], FactLine) and out[1].ref == "M#12"
assert isinstance(out[2], QuestionLine)
assert isinstance(out[3], FreeTextLine.**bases**[0]) or out[3].to_array()[0] in ("x", "t")

def test_validator_interop_and_overflow(): # Use small caps to force overflow on fact/question text
validator = JSONLValidator(caps={"f": 5, "q": 5})
gen = EchoGenerator()
lines = gen.generate(
"A very long goal that will not overflow in PR-02S",
["alpha beta gamma delta epsilon zeta", "short"],
"please expand on the long comparison question now"
)
jsonl = lines_to_jsonl(lines)
mixed, idx = validator.detect_format_break(jsonl)
assert mixed is False and idx == -1
out, stats = validator.validate_and_overflow(jsonl)
assert stats["overflow"]["count"] >= 1
assert "density" in stats and 0.0 <= stats["density"] <= 1.0

What to run (and what to paste as evidence in the PR)

Install (unchanged)

make install

Run tests

make test

Quick demo (offline Echo)

python scripts/structured_demo.py --mode echo --goal "Compare dates" --facts "A: 2001-07-16" "B: 1999-05-02" --question "Which is earlier?"

(Optional, if Ollama/OpenAI‑compatible server is running)

export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
python scripts/structured_demo.py --mode instructor --model mistral

Acceptance evidence to paste in PR description:

✅ The full pytest summary (all tests green).

✅ A snippet of the Echo demo JSON showing:

"compliance_ok": true and "break_line": -1

"lines_jsonl" in canonical array form

"validator_stats" including overflow.count, overflow.per_tag, density

✅ (Optional) An Instructor demo JSON (if you ran it) proving typed lines were produced and validated.

Commit message
PR-02S: Structured output with Instructor (CORE)

- Add tersetalk/structured.py with:
  - Pydantic models for all TerseTalk tags
  - Generic TerseTalkLine (Instructor-friendly) + converter to typed lines
  - EchoGenerator (offline-safe) and InstructorGenerator (Ollama/OpenAI-compatible)
  - JSONL emission helper
- Add scripts/structured_demo.py to generate lines (echo or instructor) and validate via PR-02 JSONLValidator
- Add tests for model conversion, JSONL emission, and validator interop
- Update requirements to include pydantic, instructor, openai
- DoD: CLI prints compliance+stats; tests pass without network
