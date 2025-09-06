from __future__ import annotations

from typing import List

from tersetalk.pipeline_runner import PipelineConfig, run_pipeline_once
from tersetalk.datasets import load_hotpotqa
from tersetalk.structured import TerseTalkLine


class FailingWorkerClient:
  def call_jsonl_strict(self, system: str, user_prompt: str, max_tokens: int = 256):
    raise RuntimeError("simulated worker failure")


class FailingCriticClient:
  def __init__(self):
    self._first = True

  def call_jsonl_strict(self, system: str, user_prompt: str, max_tokens: int = 256) -> List[TerseTalkLine]:
    if self._first:
      self._first = False
      # Return a minimal worker line
      return [TerseTalkLine(tag="g", payload=["ok worker"]) ]
    raise RuntimeError("simulated critic failure")


def test_pipeline_handles_worker_failure(monkeypatch):
  monkeypatch.setenv("TERSETALK_OFFLINE_DATA", "1")
  ex = load_hotpotqa(n=1, seed=1)[0]
  cfg = PipelineConfig(caps={})
  client = FailingWorkerClient()
  res = run_pipeline_once(ex, client, cfg)
  assert res["status"] == "error"
  assert "worker_error" in res
  # Density and overflow computed from manager_jsonl still exist
  assert isinstance(res["density"], float)
  assert "manager_jsonl" in res


def test_pipeline_handles_critic_failure(monkeypatch):
  monkeypatch.setenv("TERSETALK_OFFLINE_DATA", "1")
  ex = load_hotpotqa(n=1, seed=2)[0]
  cfg = PipelineConfig(caps={})
  client = FailingCriticClient()
  res = run_pipeline_once(ex, client, cfg)
  assert res["status"] == "error"
  assert "critic_error" in res
  assert isinstance(res["density"], float)
  assert "manager_jsonl" in res

