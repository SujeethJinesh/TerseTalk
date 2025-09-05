from __future__ import annotations

import json

from tersetalk.structured import (
  TerseTalkLine,
  terse_to_specific,
  lines_to_jsonl,
  RoleLine,
  GoalLine,
  FactLine,
  QuestionLine,
  FreeTextLine,
  EchoGenerator,
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
  assert isinstance(out[1], FactLine) and getattr(out[1], "ref", None) == "M#12"
  assert isinstance(out[2], QuestionLine)
  # The last line may be context or a safe fallback to free text
  assert out[3].to_array()[0] in ("x", "t")


def test_validator_interop_and_overflow():
  # Use small caps to force overflow on fact/question text
  validator = JSONLValidator(caps={"f": 5, "q": 5})
  gen = EchoGenerator()
  lines = gen.generate(
    "A very long goal that will not overflow in PR-02S",
    ["alpha beta gamma delta epsilon zeta", "short"],
    "please expand on the long comparison question now",
  )
  jsonl = lines_to_jsonl(lines)
  mixed, idx = validator.detect_format_break(jsonl)
  assert mixed is False and idx == -1
  out, stats = validator.validate_and_overflow(jsonl)
  assert stats["overflow"]["count"] >= 1
  assert "density" in stats and 0.0 <= stats["density"] <= 1.0

