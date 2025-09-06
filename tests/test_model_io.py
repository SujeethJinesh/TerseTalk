from __future__ import annotations

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
  lines = [
    TerseTalkLine(tag="r", payload=["M"]),
    TerseTalkLine(tag="g", payload=["Compare two dates."]),
  ]
  s = dump_jsonl(lines)
  # Should contain both tags in order
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
  lines = client.call_jsonl_strict(
    "Output a single TerseTalk line with tag 'g' and a very short goal.",
    "One short goal only.",
    max_tokens=64,
  )
  assert isinstance(lines, list) and len(lines) >= 1
  assert isinstance(lines[0], TerseTalkLine)

