from __future__ import annotations

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
  raw = "\n".join(
    [
      '["r","M"]',
      json.dumps(["f", long_fact]),
      json.dumps(["q", "W", long_q]),
      '["g","short goal"]',
    ]
  )
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


def test_jsonl_to_prose_roundtrip_signal():
  v = JSONLValidator()
  js = "\n".join([
    '["r","M"]',
    '["g","Compare dates"]',
    '["f","Event A: 2001-01-01"]',
    '["q","W","Which is earlier?"]',
  ])
  out, _ = v.validate_and_overflow(js)
  prose = v.jsonl_to_prose(out)
  assert "Goal:" in prose and "Fact:" in prose and "Question (W):" in prose

