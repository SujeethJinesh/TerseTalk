from __future__ import annotations

import json
import re

from tersetalk.protocol_jsonl import JSONLValidator
from tersetalk.memory import MemoryStore
from tersetalk.summarization import Summarizer


def test_validator_overflow_uses_summarizer_and_sets_method_field():
  mem = MemoryStore()
  caps = {"f": 5, "q": 5}
  v = JSONLValidator(caps=caps, memory=mem, summarizer=Summarizer(method="extractive"))
  raw = "\n".join(
    [
      '["r","M"]',
      json.dumps(["f", "alpha beta gamma delta epsilon zeta eta"]),
      json.dumps(["q", "W", "please compare these very long things and return earlier"]),
      '["g","short goal"]',
    ]
  )
  out, stats = v.validate_and_overflow(raw)

  o_lines = [json.loads(ln) for ln in out.splitlines() if ln.startswith('["o"')]
  assert len(o_lines) >= 2
  for arr in o_lines:
    assert arr[0] == "o"
    assert re.match(r"^M#\d+$", arr[2])
    assert arr[3] == "extractive"
    assert isinstance(mem.get(arr[2]), str)
  assert "density" in stats and 0.0 <= stats["density"] <= 1.0


def test_validator_with_llmlingua_selection_does_not_crash_without_pkg():
  mem = MemoryStore()
  v = JSONLValidator(caps={"f": 5}, memory=mem, summarizer=Summarizer(method="llmlingua"))
  raw = json.dumps(["f", "alpha beta gamma delta epsilon zeta eta"])
  out, stats = v.validate_and_overflow(raw)
  assert any(ln.startswith('["o"') for ln in out.splitlines())

