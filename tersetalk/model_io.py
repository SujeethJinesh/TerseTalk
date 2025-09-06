from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel  # only for typing clarity in signatures

from tersetalk.structured import TerseTalkLine


# ---------------------------
# Configuration & utilities
# ---------------------------


@dataclass
class ModelCfg:
  """
  Minimal model client configuration.
  - base_url: OpenAI-compatible endpoint (Ollama recommended)
  - api_key: required by OpenAI client but ignored by local Ollama; default 'ollama'
  - model: model name available on the server (e.g., 'mistral' or 'mistral:instruct')
  """

  base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
  api_key: str = os.environ.get("OLLAMA_API_KEY", "ollama")
  model: str = os.environ.get("OLLAMA_MODEL", "mistral")


def _build_instructor_client(cfg: ModelCfg):
  """
  Patch the OpenAI client with Instructor so response_model returns pydantic objects.
  """
  try:
    import instructor  # type: ignore
    from openai import OpenAI  # type: ignore
  except Exception as e:  # pragma: no cover - import guarded for offline tests
    raise RuntimeError(
      "Instructor/OpenAI client not available; install runtime deps or use EchoModel."
    ) from e
  raw = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
  return instructor.patch(raw)


def dump_jsonl(lines: List[TerseTalkLine]) -> str:
  """
  Convert a list of TerseTalkLine into canonical JSONL strings:
  each line is ["<tag>", ...payload...]
  """
  arrs = [[ln.tag, *ln.payload] for ln in lines]
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
    try:
      return (resp.choices[0].message.content or "").strip()
    except Exception:
      return ""


class EchoModel(ModelClient):
  """
  Offline deterministic client for CI.
  - call_jsonl_strict returns a fixed valid TerseTalk line.
  - call_text returns a simple echoed sentence.
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
