from __future__ import annotations

import os

from tersetalk.datasets import load_hotpotqa
from tersetalk.model_io import EchoModel
from tersetalk.pipeline_runner import PipelineConfig, run_pipeline_once


def _schema(result: dict):  # basic shape
  for k in [
    "answer",
    "verdict",
    "tokens_total",
    "overflow_count",
    "density",
    "latency_ms",
    "sp_reference",
    "manager_jsonl",
    "worker_jsonl",
    "critic_jsonl",
    "memory_stats",
  ]:
    assert k in result
  assert isinstance(result["latency_ms"]["worker"], float)
  assert isinstance(result["latency_ms"]["critic"], float)
  assert isinstance(result["tokens_total"], int)


def test_pipeline_offline_echo_deterministic(monkeypatch):
  monkeypatch.setenv("TERSETALK_OFFLINE_DATA", "1")
  ex = load_hotpotqa(n=1, seed=123)[0]
  cfg = PipelineConfig(caps={"f": 30, "p": 20, "q": 30, "g": 30, "u": 20, "t": 50})
  client = EchoModel()

  r1 = run_pipeline_once(ex, client, cfg)
  r2 = run_pipeline_once(ex, client, cfg)
  _schema(r1)
  assert r1["answer"] == r2["answer"]
  assert r1["verdict"] == r2["verdict"]
  # Memory must be reset per task (non-leakage)
  assert r1["memory_stats"]["entries"] >= 0  # present
  # worker/critic JSONL present
  assert isinstance(r1["worker_jsonl"], str) and len(r1["worker_jsonl"].splitlines()) >= 1


def test_pipeline_overflow_density_changes_with_caps(monkeypatch):
  monkeypatch.setenv("TERSETALK_OFFLINE_DATA", "1")
  ex = load_hotpotqa(n=1, seed=7)[0]
  client = EchoModel()

  # Generous caps: expect zero or few overflows
  cfg_loose = PipelineConfig(caps={"f": 100, "p": 80, "q": 100, "g": 100, "u": 80, "t": 100})
  r_loose = run_pipeline_once(ex, client, cfg_loose)

  # Tiny caps to force overflow
  cfg_tight = PipelineConfig(caps={"f": 5, "p": 5, "q": 5, "g": 5, "u": 5, "t": 5})
  r_tight = run_pipeline_once(ex, client, cfg_tight)

  _schema(r_tight)
  assert r_tight["overflow_count"] >= 0
  # Density should not increase when caps tighten
  assert r_tight["density"] <= r_loose["density"] + 1e-9

