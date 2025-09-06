from __future__ import annotations

import json

from tersetalk.protocol_handler import PHConfig, ProtocolHandler
from tersetalk.memory import MemoryStore


def _mk_long(n: int = 200) -> str:
  return " ".join(["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"] * (max(1, n) // 8))


def test_preoverflow_ll2_avoids_overflow_when_fake_enabled(monkeypatch):
  monkeypatch.setenv("TERSETALK_FAKE_LL2_TEXT", "1")

  caps = {"f": 10, "q": 8, "p": 20, "g": 30, "u": 20, "t": 50}
  long_fact = _mk_long(120)
  mgr = "\n".join(
    [
      "[\"r\",\"M\"]",
      json.dumps(["f", long_fact]),
      json.dumps(["q", "W", _mk_long(80)]),
    ]
  )
  cfg = PHConfig(caps=caps, preoverflow_ll2=True, overflow_ll2=False, deref_ll2=False, deref_policy="never")
  out = ProtocolHandler(cfg).process(mgr, memory=MemoryStore())

  # No overflow lines expected because preoverflow compressed to within caps
  assert all(not ln.startswith("[\"o\"") for ln in out.validated_jsonl.splitlines())
  assert out.counters["preoverflow"]["attempted"] >= 1
  assert out.counters["preoverflow"]["succeeded"] >= 1
  assert out.counters["preoverflow"]["ll2_used"] >= 1


def test_overflow_summary_method_switches_to_llmlingua_without_pkg():
  caps = {"f": 5, "q": 5, "p": 20, "g": 30, "u": 20, "t": 50}
  mgr = "\n".join(
    [
      "[\"r\",\"M\"]",
      json.dumps(["f", _mk_long(120)]),
      json.dumps(["q", "W", _mk_long(100)]),
    ]
  )
  cfg = PHConfig(caps=caps, preoverflow_ll2=False, overflow_ll2=True, deref_ll2=False, deref_policy="never")
  out = ProtocolHandler(cfg).process(mgr, memory=MemoryStore())

  # Expect overflow lines with method == "llmlingua"
  o_lines = [json.loads(ln) for ln in out.validated_jsonl.splitlines() if ln.startswith('["o"')]
  assert len(o_lines) >= 1
  assert all(arr[3] == "llmlingua" for arr in o_lines)
  assert out.counters["overflow"]["count"] >= 1
  assert out.counters["overflow"]["method_llmlingua"] >= 1


def test_deref_injection_and_optional_ll2_compression(monkeypatch):
  monkeypatch.setenv("TERSETALK_FAKE_LL2_TEXT", "1")

  mem = MemoryStore()
  mid = mem.put(_mk_long(160))
  caps = {"f": 12, "q": 20, "p": 20, "g": 30, "u": 20, "t": 50}
  mgr = "\n".join(
    [
      "[\"r\",\"M\"]",
      json.dumps(["d", mid]),
    ]
  )
  cfg = PHConfig(caps=caps, preoverflow_ll2=False, overflow_ll2=False, deref_ll2=True, deref_policy="always")
  out = ProtocolHandler(cfg).process(mgr, memory=mem)

  assert out.post_deref_jsonl is not None
  assert any(ln.startswith('["f"') for ln in out.post_deref_jsonl.splitlines())
  assert all(not ln.startswith('["d"') for ln in out.post_deref_jsonl.splitlines())
  assert out.counters["deref"]["attempted"] >= 1
  assert out.counters["deref"]["injected"] >= 1
  assert out.counters["deref"]["ll2_compressed"] >= 1

