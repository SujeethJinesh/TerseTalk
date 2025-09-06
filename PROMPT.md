Read through your AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the AGENTS.md. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update AGENTS.md and ask CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly. Before you merge the PR, you must wait for my approval. Also at the end I want you to outline any risks we are seeing in this project, are our expectations aligned with how the progress is going? Will our project succeed and achieve the baselines we expect using your best judgement? Absolutely make sure you're reporting results truthfully and honestly. Avoid fake, mocked, or other non genuine results. You should also analyze the results we get for each run and determine if they meet our figures of merit, when you report back, it's crucial to include that analysis (e.g. compression amount, failure rate, latency, etc.). We should be aiming to properly fix things and run proper evaluations. If we do have any expected goals or outcomes (e.g. >= 10x on xyz) and they aren't achieved, then explain why, but do not lie or cheat and use drastically contrived inefficient metrics. It's important to generally be comparing to a standard implementation. Here are more detailed instructions for PR implementation.

### PR Summary

PR‑05 — model_io.py + smoke + tests

Role: You are a senior engineer implementing PR‑05 immediately after PR‑MB merged.

Goal (from spec):
Provide a clean ModelClient that can:

call_jsonl_strict(...) → List[TerseTalkLine] using Instructor to guarantee schema‑valid outputs (Pydantic from PR‑02S).

call_text(...) → str for free‑form prompts (used by baselines later).

Ship an EchoModel for deterministic, offline CI.

Include helpers to dump JSONL strings from typed lines.

Constraints:

Use OpenAI Python SDK + Instructor patched client, pointing at an Ollama OpenAI‑compatible base URL (default http://localhost:11434/v1).

Tests must pass offline using EchoModel.

Real endpoint smoke is optional and guarded by an env var (no CI dependency).

Keep prior tests green.

DoD (Definition of Done):

tersetalk/model_io.py implements ModelClient, EchoModel, and JSONL helpers.

scripts/model_smoke.py exercises both call_jsonl_strict and call_text (echo by default; real if flagged).

tests/test_model_io.py verifies Echo behavior, JSONL dumping, and (optionally) a real call when opt‑in env is set.

Export model_io in tersetalk/**init**.py.

No network required for tests.

Create/Update the following files exactly

1. tersetalk/model_io.py (new)
   from **future** import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import instructor
from openai import OpenAI
from pydantic import BaseModel # only for typing clarity in signatures

from tersetalk.structured import TerseTalkLine

# ---------------------------

# Configuration & utilities

# ---------------------------

@dataclass
class ModelCfg:
"""
Minimal model client configuration. - base_url: OpenAI-compatible endpoint (Ollama recommended) - api_key: required by OpenAI client but ignored by local Ollama; default 'ollama' - model: model name available on the server (e.g., 'mistral' or 'mistral:instruct')
"""
base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
api_key: str = os.environ.get("OLLAMA_API_KEY", "ollama")
model: str = os.environ.get("OLLAMA_MODEL", "mistral")

def \_build_instructor_client(cfg: ModelCfg):
"""
Patch the OpenAI client with Instructor so response_model returns pydantic objects.
"""
raw = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
return instructor.patch(raw)

def dump_jsonl(lines: List[TerseTalkLine]) -> str:
"""
Convert a list of TerseTalkLine into canonical JSONL strings:
each line is ["<tag>", ...payload...]
"""
arrs = [ [ln.tag, *ln.payload] for ln in lines ]
return "\n".join(json.dumps(a, ensure_ascii=False) for a in arrs)

# ---------------------------

# Model clients

# ---------------------------

class ModelClient:
"""
Real client that talks to an OpenAI-compatible endpoint (e.g., Ollama),
guaranteeing structured outputs via Instructor+Pydantic.
"""

    def __init__(self, cfg: Optional[ModelCfg] = None):
        self.cfg = cfg or ModelCfg()
        self.client = _build_instructor_client(self.cfg)
        self.model = self.cfg.model

    # Structured (typed) JSONL output using Instructor
    def call_jsonl_strict(
        self,
        system: str,
        user_prompt: str,
        max_tokens: int = 256,
        retries: int = 2,
    ) -> List[TerseTalkLine]:
        """
        Returns a list of TerseTalkLine objects parsed/validated by Instructor.
        NOTE: Relies on the model cooperating with the instruction. Instructor
        will retry/coerce within reason, then raise on failure.
        """
        result: List[TerseTalkLine] = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            response_model=List[TerseTalkLine],  # <-- magic: returns typed objects
            max_retries=retries,
            # Instructor forwards extra kwargs as needed; keep minimal for portability
        )
        return result

    # Free-form text (baseline support)
    def call_text(
        self,
        system: str,
        user_prompt: str,
        max_tokens: int = 512,
    ) -> str:
        """
        Returns raw assistant text. Keep as a simple baseline utility.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        # When response_model is NOT used, Instructor returns a normal OpenAI object
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            # Defensive fallback
            return ""

class EchoModel(ModelClient):
"""
Offline deterministic client for CI. - call_jsonl_strict returns a fixed valid TerseTalk line. - call_text returns a simple echoed sentence.
"""

    def __init__(self, cfg: Optional[ModelCfg] = None):
        # Do not initialize a real HTTP client in echo mode
        self.cfg = cfg or ModelCfg()
        self.client = None
        self.model = "echo"

    def call_jsonl_strict(
        self,
        system: str,
        user_prompt: str,
        max_tokens: int = 256,
        retries: int = 2,
    ) -> List[TerseTalkLine]:
        return [TerseTalkLine(tag="g", payload=["This is an echoed goal."])]

    def call_text(
        self,
        system: str,
        user_prompt: str,
        max_tokens: int = 512,
    ) -> str:
        return "ECHO: hello from EchoModel"

2. scripts/model_smoke.py (new)
   from **future** import annotations

import argparse
import json
import os

from tersetalk.model_io import ModelClient, EchoModel, ModelCfg, dump_jsonl

SYSTEM_JSONL = (
"You output compact, typed JSONL lines for the TerseTalk protocol. "
"Each line is an array: ['r'|'g'|'f'|'u'|'p'|'q'|'d'|'v'|'o'|'t'|'x', ...]. "
"Return 2-4 lines that set a role and a short goal."
)
USER_JSONL = "Create a manager role and a concise subgoal about comparing two dates."

SYSTEM_TEXT = "You are a concise assistant."
USER_TEXT = "Say one short sentence about efficiency."

def main():
ap = argparse.ArgumentParser(description="PR-05: Model I/O smoke tool")
ap.add_argument("--mode", choices=["echo", "real"], default=os.environ.get("MODEL_CLIENT", "echo"))
ap.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "mistral"))
ap.add_argument("--api-key", default=os.environ.get("OLLAMA_API_KEY", "ollama"))
args = ap.parse_args()

    if args.mode == "real":
        client = ModelClient(ModelCfg(base_url=args.base_url, api_key=args.api_key, model=args.model))
    else:
        client = EchoModel()

    # Structured call
    try:
        lines = client.call_jsonl_strict(SYSTEM_JSONL, USER_JSONL, max_tokens=200)
        jsonl = dump_jsonl(lines)
    except Exception as e:
        lines = []
        jsonl = f"<error: {e}>"

    # Free-form call
    try:
        text = client.call_text(SYSTEM_TEXT, USER_TEXT, max_tokens=50)
    except Exception as e:
        text = f"<error: {e}>"

    out = {
        "mode": args.mode,
        "structured_lines": [ {"tag": ln.tag, "payload": ln.payload} for ln in lines ],
        "structured_jsonl": jsonl,
        "freeform_text": text,
    }
    print(json.dumps(out, indent=2))

if **name** == "**main**":
main()

3. tests/test_model_io.py (new)
   from **future** import annotations

import os

from tersetalk.model_io import EchoModel, ModelClient, ModelCfg, dump_jsonl
from tersetalk.structured import TerseTalkLine

def test_echo_jsonl_and_text_offline():
client = EchoModel()
lines = client.call_jsonl_strict("sys", "user")
assert isinstance(lines, list) and len(lines) == 1
assert isinstance(lines[0], TerseTalkLine)
assert lines[0].tag == "g"
assert isinstance(lines[0].payload, list)

    text = client.call_text("sys", "user")
    assert text.startswith("ECHO:")

def test_dump_jsonl_helper_roundtrip():
lines = [TerseTalkLine(tag="r", payload=["M"]),
TerseTalkLine(tag="g", payload=["Compare two dates."])]
s = dump_jsonl(lines) # Should contain both tags in order
assert s.splitlines()[0].startswith('["r"')
assert s.splitlines()[1].startswith('["g"')

def test_real_call_optional_smoke():
"""
Optional: If RUN_REAL_OLLAMA=1, attempt a tiny real call.
This is skipped in CI by default.
"""
if os.environ.get("RUN_REAL_OLLAMA") != "1":
return

    cfg = ModelCfg(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
        model=os.environ.get("OLLAMA_MODEL", "mistral"),
    )
    client = ModelClient(cfg)
    # Real model might or might not follow spec perfectly; Instructor will try.
    lines = client.call_jsonl_strict(
        "Output a single TerseTalk line with tag 'g' and a very short goal.",
        "One short goal only.",
        max_tokens=64,
    )
    assert isinstance(lines, list) and len(lines) >= 1
    assert isinstance(lines[0], TerseTalkLine)

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
"protocol_handler",
"model_io",
]

What to run (and what to paste as evidence in the PR)

Run tests (offline, Echo model)

make test

Smoke (Echo mode, no network)

python scripts/model_smoke.py --mode echo

Optional: Real Ollama smoke (manual, not CI)

# Ensure: ollama serve (and model pulled, e.g., `ollama pull mistral`)

RUN_REAL_OLLAMA=1 \
OLLAMA_MODEL=mistral \
OLLAMA_BASE_URL=http://localhost:11434/v1 \
python scripts/model_smoke.py --mode real --model mistral --base-url http://localhost:11434/v1

Acceptance evidence to paste in the PR description:

✅ pytest summary (all green).

✅ model_smoke.py --mode echo JSON showing structured_lines with a {"tag":"g",...} and freeform_text starting with "ECHO:".

(Optional) A real‑endpoint smoke JSON showing non‑empty structured_lines and freeform_text when Ollama is running.

Commit message
PR-05: Model I/O via Instructor + Ollama (with EchoModel)

- Add tersetalk/model_io.py:
  - ModelCfg and ModelClient using OpenAI SDK + Instructor
  - call_jsonl_strict() → List[TerseTalkLine] (typed)
  - call_text() → str (free-form baseline)
  - EchoModel for offline deterministic CI
  - dump_jsonl() helper for canonical JSONL output
- Add scripts/model_smoke.py for quick local smoke (echo/real)
- Add tests/test_model_io.py (offline by default; optional real smoke via RUN_REAL_OLLAMA=1)
- Export model_io in package **init**
