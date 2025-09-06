from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Literal

from tersetalk.protocol_jsonl import JSONLValidator
from tersetalk.summarization import Summarizer
from tersetalk.memory import MemoryStore

# -------------------------
# Config & result types
# -------------------------

DerefPolicy = Literal["never", "conditional", "always"]


@dataclass
class PHConfig:
  caps: Dict[str, int]
  summarizer_method: Literal["extractive", "llmlingua"] = "extractive"
  preoverflow_ll2: bool = False
  overflow_ll2: bool = False
  deref_ll2: bool = False
  deref_policy: DerefPolicy = "never"


@dataclass
class PHOutcome:
  validated_jsonl: str
  stats_before: Dict
  post_deref_jsonl: Optional[str]
  stats_after: Optional[Dict]
  counters: Dict[str, Dict[str, int]]

  def to_dict(self) -> Dict:
    d = asdict(self)
    return d


# -------------------------
# Internal helpers
# -------------------------


def _tok_est(text: str) -> int:
  return max(0, (len(text) + 3) // 4)


def _hard_trim_to_tokens(text: str, target_tokens: int) -> str:
  max_chars = max(1, target_tokens * 4)
  if len(text) <= max_chars:
    return text
  # Reserve space for ellipsis
  budget = max(1, max_chars - 3)
  cut = text[:budget]
  sp = cut.rfind(" ")
  if sp >= 16:
    cut = cut[:sp]
  return cut.rstrip() + "..."


def _ll2_compress_text(text: str, target_tokens: int) -> Optional[str]:
  """
  Try LLMLingua compression. If unavailable and TERSETALK_FAKE_LL2_TEXT=1,
  simulate by hard trimming to target token budget. Else return None.
  """
  if os.environ.get("TERSETALK_FAKE_LL2_TEXT", ""):
    return _hard_trim_to_tokens(text, target_tokens)

  try:
    from llmlingua import PromptCompressor  # type: ignore
  except Exception:
    return None

  try:
    comp = PromptCompressor()
    res = comp.compress(text, target_token=int(target_tokens))
    cand = (
      res.get("compressed_prompt")
      or res.get("compressed_text")
      or res.get("text")
      or res.get("prompt")
    )
    if not isinstance(cand, str) or not cand.strip():
      return None
    return _hard_trim_to_tokens(cand.strip(), target_tokens)
  except Exception:
    return None


def _is_textual_tag(tag: str) -> bool:
  return tag in ("f", "p", "g", "u", "t", "q")


# -------------------------
# Protocol Handler
# -------------------------


class ProtocolHandler:
  """
  Coordinates the three LLMLingua touchpoints around the JSONLValidator:
  1) Pre-overflow compression on long payloads
  2) Overflow summarization method selection
  3) Dereference (d:M#) injection and optional compression
  """

  def __init__(self, cfg: PHConfig) -> None:
    self.cfg = cfg

  def _counters_init(self) -> Dict[str, Dict[str, int]]:
    return {
      "preoverflow": {"attempted": 0, "succeeded": 0, "ll2_used": 0, "ll2_unavailable": 0},
      "overflow": {"count": 0, "method_extractive": 0, "method_llmlingua": 0},
      "deref": {"attempted": 0, "injected": 0, "ll2_compressed": 0},
    }

  def _normalize_lines(self, jsonl: str, validator: JSONLValidator) -> List[List]:
    out: List[List] = []
    for raw in [ln for ln in jsonl.splitlines() if ln.strip()]:
      arr = validator.normalize_line(raw)
      if not isinstance(arr, list) or not arr:
        arr = ["t", raw.strip()]
      out.append(arr)
    return out

  def _preoverflow_pass(self, lines: List[List], counters: Dict[str, Dict[str, int]]) -> List[List]:
    """
    Attempt to compress over-cap payloads in-place, before validator overflow logic runs.
    For 'q', payload is at index 2; for others, payload at index 1.
    """
    if not self.cfg.preoverflow_ll2:
      return lines

    caps = self.cfg.caps
    new_lines: List[List] = []
    for arr in lines:
      tag = arr[0] if arr else "t"
      if _is_textual_tag(tag):
        if tag == "q":
          role = arr[1] if len(arr) > 1 else "W"
          text = arr[2] if len(arr) > 2 else ""
          cap = caps.get("q")
          if isinstance(text, str) and cap is not None and _tok_est(text) > cap:
            counters["preoverflow"]["attempted"] += 1
            comp = _ll2_compress_text(text, cap)
            if comp is not None:
              counters["preoverflow"]["ll2_used"] += 1
              if _tok_est(comp) <= cap:
                counters["preoverflow"]["succeeded"] += 1
                new_lines.append(["q", role, comp])
                continue
            else:
              counters["preoverflow"]["ll2_unavailable"] += 1
          new_lines.append(["q", role, text])
        else:
          text = arr[1] if len(arr) > 1 else ""
          cap = caps.get(tag)
          if isinstance(text, str) and cap is not None and _tok_est(text) > cap:
            counters["preoverflow"]["attempted"] += 1
            comp = _ll2_compress_text(text, cap)
            if comp is not None:
              counters["preoverflow"]["ll2_used"] += 1
              if _tok_est(comp) <= cap:
                counters["preoverflow"]["succeeded"] += 1
                new_lines.append([tag, comp] + (arr[2:] if len(arr) > 2 else []))
                continue
            else:
              counters["preoverflow"]["ll2_unavailable"] += 1
          new_lines.append(arr)
      else:
        new_lines.append(arr)
    return new_lines

  def _overflow_counters(self, stats: Dict, counters: Dict[str, Dict[str, int]], method: str) -> None:
    cnt = int(stats.get("overflow", {}).get("count", 0))
    counters["overflow"]["count"] += cnt
    if method == "llmlingua":
      counters["overflow"]["method_llmlingua"] += cnt
    else:
      counters["overflow"]["method_extractive"] += cnt

  def _deref_inject(self, jsonl: str, memory: MemoryStore, counters: Dict[str, Dict[str, int]]) -> str:
    """
    Replace each ["d","M#id"] with an inline ["f", <(maybe compressed) text>].
    Then return a new JSONL string (array form).
    """
    out_lines: List[str] = []
    for raw in [ln for ln in jsonl.splitlines() if ln.strip()]:
      arr = None
      try:
        arr = json.loads(raw)
      except Exception:
        out_lines.append(raw)
        continue
      if not isinstance(arr, list) or not arr:
        out_lines.append(raw)
        continue
      tag = arr[0]
      if tag != "d" or len(arr) < 2:
        out_lines.append(json.dumps(arr))
        continue

      counters["deref"]["attempted"] += 1
      mid = arr[1]
      text = memory.get(mid)
      if not isinstance(text, str):
        out_lines.append(json.dumps(arr))
        continue

      if self.cfg.deref_ll2:
        cap = int(self.cfg.caps.get("f", 30))
        comp = _ll2_compress_text(text, cap)
        if comp is not None:
          counters["deref"]["ll2_compressed"] += 1
          text = comp

      counters["deref"]["injected"] += 1
      out_lines.append(json.dumps(["f", text]))
    return "\n".join(out_lines)

  # --------- Public API ---------

  def process(self, manager_jsonl: str, memory: Optional[MemoryStore] = None) -> PHOutcome:
    """
    Run pre-overflow compression, validate (with chosen summarizer method),
    then optional dereference injection + revalidation.
    """
    memory = memory or MemoryStore()
    counters = self._counters_init()

    # Normalization pass
    tmp_validator = JSONLValidator(caps=self.cfg.caps, memory=memory, summarizer=Summarizer(method="extractive"))
    norm_lines = self._normalize_lines(manager_jsonl, tmp_validator)

    # Pre-overflow compression (in-place), if enabled
    pre_lines = self._preoverflow_pass(norm_lines, counters)

    # Overflow summarization method selection
    ov_method = "llmlingua" if self.cfg.overflow_ll2 else self.cfg.summarizer_method
    validator = JSONLValidator(caps=self.cfg.caps, memory=memory, summarizer=Summarizer(method=ov_method))

    # Validate + overflow
    pre_jsonl = "\n".join(json.dumps(l) for l in pre_lines)
    validated_jsonl, stats_before = validator.validate_and_overflow(pre_jsonl)
    self._overflow_counters(stats_before, counters, method=ov_method)

    # Dereference policy
    post_jsonl: Optional[str] = None
    stats_after: Optional[Dict] = None
    if self.cfg.deref_policy in ("conditional", "always"):
      post_jsonl = self._deref_inject(validated_jsonl, memory, counters)
      validated2, stats2 = validator.validate_and_overflow(post_jsonl)
      post_jsonl, stats_after = validated2, stats2
      self._overflow_counters(stats2, counters, method=ov_method)

    return PHOutcome(
      validated_jsonl=validated_jsonl,
      stats_before=stats_before,
      post_deref_jsonl=post_jsonl,
      stats_after=stats_after,
      counters=counters,
    )
