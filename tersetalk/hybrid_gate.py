from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class GateCfg:
  """
  Configuration for the per-turn hybrid gate.

      token_budget: maximum allowed tokens for the outgoing message.
      use_ll2_tags: reserved for future fine-grain control of which tags to compress;
                    kept here to match the spec, not used in PR-H1 logic.
  """

  token_budget: int = 600
  use_ll2_tags: tuple[str, ...] = ("f", "p", "q")


def estimate_tokens(text: str) -> int:
  """
  Cheap token estimator used repo-wide (~4 chars ≈ 1 token).
  Non-negative and monotonic w.r.t len(text).
  """
  if not text:
    return 0
  return max(0, (len(text) + 3) // 4)


def _fake_ll2_env_projection() -> Optional[int]:
  """
  Testing escape hatch: if TERSETALK_FAKE_LL2_COMPRESS is set,
  return that integer (simulates projected token count).
  """
  val = os.environ.get("TERSETALK_FAKE_LL2_COMPRESS")
  if val is None:
    return None
  try:
    return int(val)
  except Exception:
    return None


def project_ll2_tokens(prompt: str, budget: int) -> Optional[int]:
  """
  Try to project the compressed token length using llmlingua (if available).
  Returns an integer token count or None if unavailable/error.

      Honors TERSETALK_FAKE_LL2_COMPRESS to enable offline tests without the package.
  """
  fake = _fake_ll2_env_projection()
  if fake is not None:
    return fake

  try:
    from llmlingua import PromptCompressor  # type: ignore
  except Exception:
    return None

  try:
    comp = PromptCompressor()
    res = comp.compress(prompt, target_token=budget)
    for key in ("compressed_tokens", "target_token", "projected_tokens", "tokens"):
      v = res.get(key)
      if isinstance(v, int) and v >= 0:
        return v
    for k in ("compressed_prompt", "compressed_text"):
      s = res.get(k)
      if isinstance(s, str):
        return estimate_tokens(s)
    return None
  except Exception:
    return None


def gate_choose_protocol(manager_jsonl: str, freeform_prompt: str, cfg: GateCfg) -> Dict:
  """
  Decide which path to use for the current turn.

      Strategy:
        1) If JSONL estimate <= budget → "tersetalk".
        2) Else try llmlingua projection on freeform; if <= budget → "freeform_llmlingua".
        3) Else → "tersetalk" (with overflow).

      Returns a dict:
      {
        "route": "tersetalk" | "freeform_llmlingua",
        "est_tokens": {"jsonl": int, "ll2": Optional[int]},
        "notes": str
      }
  """
  t_jsonl = estimate_tokens(manager_jsonl)
  if t_jsonl <= cfg.token_budget:
    return {
      "route": "tersetalk",
      "est_tokens": {"jsonl": t_jsonl, "ll2": None},
      "notes": "JSONL within budget; chose tersetalk.",
    }

  t_ll2 = project_ll2_tokens(freeform_prompt, cfg.token_budget)
  if t_ll2 is not None and t_ll2 <= cfg.token_budget:
    return {
      "route": "freeform_llmlingua",
      "est_tokens": {"jsonl": t_jsonl, "ll2": t_ll2},
      "notes": "JSONL over budget; LLMLingua projection fits -> freeform_llmlingua.",
    }

  return {
    "route": "tersetalk",
    "est_tokens": {"jsonl": t_jsonl, "ll2": t_ll2},
    "notes": "JSONL over budget; LLMLingua unavailable or still over -> tersetalk (overflow as needed).",
  }

