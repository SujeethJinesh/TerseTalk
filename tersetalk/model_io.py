from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
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
    # Prefer a robust path on Ollama endpoints where Instructor response_model may be unreliable
    self._is_ollama = isinstance(self.cfg.base_url, str) and (
      "11434" in self.cfg.base_url or "ollama" in self.cfg.base_url.lower()
    )
    self._force_text_jsonl = os.environ.get("TERSETALK_OLLAMA_TEXT_JSONL", "").lower() in {"1","true","yes"}
    # Try to initialize Instructor client, but tolerate absence when we plan to use text-JSONL fallback
    self.client = None
    try:
      if not (self._is_ollama or self._force_text_jsonl):
        self.client = _build_instructor_client(self.cfg)
      else:
        # Build a vanilla OpenAI client for text completions on Ollama
        from openai import OpenAI  # type: ignore
        self.client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
    except Exception:
      # Final fallback: attempt vanilla client; raise only if that fails too
      try:
        from openai import OpenAI  # type: ignore
        self.client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
      except Exception as e:
        raise RuntimeError(
          "OpenAI-compatible client not available; install runtime deps."
        ) from e
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
    Structured generation returning a list of TerseTalkLine.
    Preference order:
      1) If not on Ollama (or forced), use Instructor response_model path.
      2) Else: strict JSONL-by-text path with local parsing and minimal retries.
    """
    # If we have an Instructor-patched client and we're not preferring the text fallback, try it first
    if self.client is not None and not (self._is_ollama or self._force_text_jsonl):
      try:
        result: List[TerseTalkLine] = self.client.chat.completions.create(
          model=self.model,
          messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
          ],
          response_model=List[TerseTalkLine],  # type: ignore[arg-type]
          max_retries=retries,
        )
        return result
      except Exception:
        # Fall through to text JSONL path
        pass

    # Text JSONL fallback (robust on Ollama)
    return self._call_jsonl_via_text(system, user_prompt, max_tokens=max_tokens, retries=max(0, int(retries)))

  # ---- Internal helpers ----
  def _parse_jsonl_text(self, text: str) -> List[TerseTalkLine]:
    allowed = {"r", "g", "f", "u", "p", "q", "d", "v", "o", "t", "x"}
    out: List[TerseTalkLine] = []
    if not isinstance(text, str):
      return out
    s = text.strip()
    if not s:
      return out
    # Strip code fences if present
    if s.startswith("```"):
      s = s.strip("`\n ")
    for raw in s.splitlines():
      ln = raw.strip()
      if not ln or not ln.startswith("["):
        continue
      try:
        arr = json.loads(ln)
      except Exception:
        continue
      if not isinstance(arr, list) or not arr:
        continue
      tag = str(arr[0])
      if tag not in allowed:
        continue
      payload = [str(x) for x in list(arr[1:])]
      try:
        out.append(TerseTalkLine(tag=tag, payload=payload))
      except Exception:
        # Skip malformed line silently
        continue
    return out

  def _call_jsonl_via_text(
    self,
    system: str,
    user_prompt: str,
    max_tokens: int = 256,
    retries: int = 1,
  ) -> List[TerseTalkLine]:
    prefer_json = os.environ.get("TERSETALK_PREFER_JSON", "").lower() in {"1", "true", "yes"}
    strict_system = (
      "You are a TerseTalk generator. Output ONLY JSON Lines (one per line). "
      "Each line MUST be a JSON array like ['r','M'] or ['f','text']. "
      "Allowed tags: r,g,f,u,p,q,d,v,o,t,x. No prose, no markdown, no explanations. "
      "END WITH EXACTLY ONE final line: ['f','<final answer>'] containing ONLY the final answer."
    )
    full_sys = f"{strict_system}\n\n{system or ''}"
    # Single attempt + optional retry if parse yields empty
    for attempt in range(max(1, retries + 1)):
      content = None
      if prefer_json:
        try:
          # Ask for a single JSON value (array-of-arrays)
          resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
              {"role": "system", "content": full_sys + " Return ONE JSON array-of-arrays only."},
              {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
          )
          content = (resp.choices[0].message.content or "").strip()
        except Exception:
          content = None

      if not content:
        content = self.call_text(system=full_sys, user_prompt=user_prompt, max_tokens=max_tokens)

      # Try JSON array-of-arrays first
      lines: List[TerseTalkLine] = []
      try:
        obj = json.loads(content)
        if isinstance(obj, list) and obj and all(isinstance(x, list) for x in obj):
          for arr in obj:
            if not arr:
              continue
            tag = str(arr[0])
            payload = [str(x) for x in list(arr[1:])]
            try:
              lines.append(TerseTalkLine(tag=tag, payload=payload))
            except Exception:
              continue
      except Exception:
        pass

      if not lines:
        lines = self._parse_jsonl_text(content)
      if lines:
        return lines
      # Tighten instruction on retry
      strict_system = (
        "Return ONLY newline-delimited JSON arrays starting with '['. "
        "Allowed tags: r,g,f,u,p,q,d,v,o,t,x. END with exactly one final ['f','<answer>'] line."
      )
      full_sys = f"{strict_system}\n\n{system or ''}"
    # As a last resort, wrap the whole response as a free-text line
    fallback = (content or "").strip()
    if fallback:
      return [TerseTalkLine(tag="t", payload=[fallback])]
    return []

  # Free-form text (baseline support)
  def call_text(
    self,
    system: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float | None = None,
  ) -> str:
    """
    Returns raw assistant text. Keep as a simple baseline utility.
    """
    kwargs = {
      "model": self.model,
      "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
      ],
      "max_tokens": max_tokens,
    }
    if temperature is not None:
      kwargs["temperature"] = float(temperature)
    resp = self.client.chat.completions.create(**kwargs)
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
    temperature: float | None = None,
  ) -> str:
    return "ECHO: hello from EchoModel"
