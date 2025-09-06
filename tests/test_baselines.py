from __future__ import annotations

import os

from tersetalk.baselines import build_freeform_prompt, run_freeform_once, run_llmlingua_once
from tersetalk.datasets import load_hotpotqa
from tersetalk.model_io import EchoModel


def _shape_ok(res: dict, expect_ll2: bool | None = None):
  keys = {
    "answer",
    "prompt",
    "response",
    "prompt_tokens",
    "response_tokens",
    "tokens_total",
    "used_llmlingua",
    "origin_tokens",
    "compressed_tokens",
    "compression_ratio",
  }
  assert keys.issubset(set(res.keys()))
  assert isinstance(res["answer"], str)
  assert isinstance(res["prompt_tokens"], int)
  assert isinstance(res["response_tokens"], int)
  assert isinstance(res["tokens_total"], int)
  if expect_ll2 is not None:
    assert res["used_llmlingua"] is expect_ll2
  if not expect_ll2:
    assert res["origin_tokens"] is None
    assert res["compressed_tokens"] is None
    assert res["compression_ratio"] is None


def test_build_freeform_prompt_contains_question_and_fact(monkeypatch):
  monkeypatch.setenv("TERSETALK_OFFLINE_DATA", "1")
  ex = load_hotpotqa(n=1, seed=3)[0]
  prompt = build_freeform_prompt(ex)
  assert "Question:" in prompt
  assert ex["question"][:10] in prompt
  if ex.get("facts"):
    assert str(ex["facts"][0])[:5] in prompt


def test_run_freeform_once_echo_is_deterministic(monkeypatch):
  monkeypatch.setenv("TERSETALK_OFFLINE_DATA", "1")
  ex = load_hotpotqa(n=1, seed=5)[0]
  client = EchoModel()
  r1 = run_freeform_once(ex, client)
  r2 = run_freeform_once(ex, client)
  _shape_ok(r1)
  assert r1["answer"] == r2["answer"]
  assert r1["tokens_total"] == r2["tokens_total"]


def test_run_llmlingua_once_fallback_when_disabled(monkeypatch):
  # Force disable LLMLingua through env to ensure deterministic CI
  monkeypatch.setenv("TERSETALK_OFFLINE_DATA", "1")
  monkeypatch.setenv("TERSETALK_DISABLE_LL2", "1")

  ex = load_hotpotqa(n=1, seed=7)[0]
  client = EchoModel()
  r = run_llmlingua_once(ex, client, target_token=200)
  _shape_ok(r, expect_ll2=False)

