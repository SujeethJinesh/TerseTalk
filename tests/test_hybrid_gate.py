from __future__ import annotations

import os
from contextlib import contextmanager

from tersetalk.hybrid_gate import GateCfg, estimate_tokens, gate_choose_protocol

JSONL_SHORT = '["r","M"]\n["g","Short goal"]\n["q","W","?"]'
FREEFORM_LONG = "Goal: " + ("alpha beta gamma " * 300)


@contextmanager
def fake_ll2_projection(value: int | None):
  """
  Sets TERSETALK_FAKE_LL2_COMPRESS to simulate llmlingua projected tokens.
  Pass None to clear it.
  """
  old = os.environ.get("TERSETALK_FAKE_LL2_COMPRESS")
  try:
    if value is None:
      os.environ.pop("TERSETALK_FAKE_LL2_COMPRESS", None)
    else:
      os.environ["TERSETALK_FAKE_LL2_COMPRESS"] = str(value)
    yield
  finally:
    if old is None:
      os.environ.pop("TERSETALK_FAKE_LL2_COMPRESS", None)
    else:
      os.environ["TERSETALK_FAKE_LL2_COMPRESS"] = old


def test_estimate_tokens_monotonic_and_nonnegative():
  assert estimate_tokens("") == 0
  a = estimate_tokens("abcd")  # ~1
  b = estimate_tokens("abcd" * 10)
  assert a >= 0 and b >= a and b > 0


def test_gate_routes_tersetalk_when_within_budget():
  cfg = GateCfg(token_budget=50)
  decision = gate_choose_protocol(JSONL_SHORT, FREEFORM_LONG, cfg)
  assert decision["route"] == "tersetalk"
  assert decision["est_tokens"]["jsonl"] <= 50


def test_gate_routes_freeform_llmlingua_when_projection_fits():
  # Force JSONL to exceed budget while LL2 projection fits
  jsonl = '["g","' + ("x" * 2000) + '"]'
  freeform = "alpha beta gamma " * 200
  cfg = GateCfg(token_budget=120)
  with fake_ll2_projection(100):  # simulate LL2 compressed within budget
    decision = gate_choose_protocol(jsonl, freeform, cfg)
    assert decision["route"] == "freeform_llmlingua"
    assert decision["est_tokens"]["ll2"] == 100


def test_gate_falls_back_to_tersetalk_when_projection_unavailable():
  # Long JSONL, LL2 unavailable -> tersetalk
  jsonl = '["g","' + ("x" * 2000) + '"]'
  freeform = "alpha beta gamma " * 200
  cfg = GateCfg(token_budget=120)
  with fake_ll2_projection(None):  # ensure no fake projection
    decision = gate_choose_protocol(jsonl, freeform, cfg)
    assert decision["route"] == "tersetalk"
    # ll2 estimate may be None in this case
    assert decision["est_tokens"]["jsonl"] > cfg.token_budget

