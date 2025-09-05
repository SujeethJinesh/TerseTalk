from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple
from tersetalk.summarization import Summarizer

# --- Protocol Tags (single letters) -----------------------------

TAGS: Dict[str, str] = {
  "r": "role",  # ["r","M"|"W"|"C"]
  "g": "subgoal",  # ["g","<short text>"]
  "f": "fact",  # ["f","<text>","M#id"?]
  "u": "assumption",  # ["u","<short text>"]
  "p": "plan_step",  # ["p","<short text>"]
  "q": "question",  # ["q","M|W|C","<question>"]
  "d": "delta_ref",  # ["d","M#id"]
  "v": "verdict",  # ["v","A"|"R"|"E"]
  "o": "overflow",  # ["o","<summary>","M#ref","extractive"]
  "t": "free_text",  # ["t","<very short>"]
  "x": "context",  # ["x","<key>","<val>"]
}


# --- Minimal memory interface for PR-02 only --------------------


class SupportsPut(Protocol):
  def put(self, text: str) -> str: ...


class _LocalMemory:
  """
  PR-02-only, lightweight in-module memory to mint M# ids.
  PR-03 will provide a full MemoryStore; this is intentionally minimal.
  """

  def __init__(self) -> None:
    self._store: Dict[str, str] = {}
    self._ctr: int = 0

  def put(self, text: str) -> str:  # pragma: no cover - trivial
    self._ctr += 1
    mid = f"M#{self._ctr}"
    self._store[mid] = text
    return mid


# --- Helper ------------------------------------------------------


def _token_estimate(text: str) -> int:
  """Cheap token estimator used throughout this repo."""
  return max(0, (len(text) + 3) // 4)


# --- JSONL Validator --------------------------------------------


@dataclass
class JSONLValidator:
  """
  Validate + normalize JSONL, enforce soft caps, and create overflow lines.

  caps: per-tag target token caps. Missing keys fall back to defaults.
  memory: object with .put(text)->"M#k" used to store overflow text.
  """

  caps: Dict[str, int] = field(default_factory=dict)
  memory: Optional[SupportsPut] = None
  summarizer: Optional[Summarizer] = None

  def __post_init__(self) -> None:
    default_caps = {"f": 30, "p": 20, "q": 30, "g": 30, "u": 20, "t": 50}
    merged = dict(default_caps)
    merged.update(self.caps or {})
    self.caps = merged
    # Do not rely on truthiness; MemoryStore defines __len__ and may be empty.
    if self.memory is None:
      self.memory = _LocalMemory()
    # Default summarizer: extractive
    self.summarizer = self.summarizer or Summarizer(method="extractive")
    self.overflow_freq: Dict[str, int] = {}

  # ---------- Detection ----------
  def detect_format_break(self, output: str) -> Tuple[bool, int]:
    """
    Returns (is_mixed, line_number_of_break).
    Mixed if any non-empty line does not start with '[' or '{'.
    """
    lines = [ln for ln in output.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
      s = line.lstrip()
      if not (s.startswith("[") or s.startswith("{")):
        return True, i
    return False, -1

  # ---------- Normalization ----------
  def normalize_line(self, raw: str) -> List[Any]:
    """
    Convert a lenient object form to the canonical array form.
    Unknown structures become ["t", "<stringified>"].
    """
    s = raw.strip()
    if not s:
      return ["t", ""]
    if s[0] == "[":
      arr = json.loads(s)
      if isinstance(arr, list):
        return arr
      return ["t", json.dumps(arr, separators=(",", ":"))]
    if s[0] == "{":
      obj = json.loads(s)
      # If explicit 'tag'
      if "tag" in obj:
        tag = obj.get("tag")
        if tag == "f":
          text = obj.get("text") or obj.get("value") or obj.get("f") or ""
          ref = obj.get("ref") or obj.get("id")
          return ["f", str(text)] + ([str(ref)] if ref else [])
        if tag == "q":
          who = obj.get("role") or obj.get("to") or obj.get("who") or "W"
          text = obj.get("text") or obj.get("question") or ""
          return ["q", str(who), str(text)]
        if tag == "r":
          role = obj.get("role") or obj.get("r") or "M"
          return ["r", str(role)]
        if tag in TAGS:
          # generic object: prefer 'text' or tag-named key
          v = obj.get("text", obj.get(tag, ""))
          if isinstance(v, list):
            return [tag] + v
          return [tag, str(v)]
        # unknown tag → t
        return ["t", json.dumps(obj, separators=(",", ":"))]

      # No explicit 'tag': look for single-letter key
      for k in TAGS.keys():
        if k in obj:
          v = obj[k]
          if isinstance(v, list):
            return [k] + v
          return [k, str(v)]
      # Fall back
      return ["t", json.dumps(obj, separators=(",", ":"))]

    # Fallback for raw text lines (not valid JSON) → 't'
    return ["t", s]

  # ---------- (De)serialization helpers ----------
  def jsonl_to_prose(self, lines: str) -> str:
    """Convert canonical array-lines to simple prose (for SP reference)."""
    prose: List[str] = []
    for ln in [ln for ln in lines.splitlines() if ln.strip()]:
      try:
        arr = json.loads(ln)
      except Exception:
        continue
      if not isinstance(arr, list) or not arr:
        continue
      tag = arr[0]
      if tag == "g":
        prose.append(f"Goal: {arr[1] if len(arr)>1 else ''}")
      elif tag == "f":
        prose.append(f"Fact: {arr[1] if len(arr)>1 else ''}")
      elif tag == "u":
        prose.append(f"Assumption: {arr[1] if len(arr)>1 else ''}")
      elif tag == "p":
        prose.append(f"Plan: {arr[1] if len(arr)>1 else ''}")
      elif tag == "q":
        who = arr[1] if len(arr) > 1 else ""
        txt = arr[2] if len(arr) > 2 else ""
        prose.append(f"Question ({who}): {txt}")
      elif tag == "v":
        prose.append(f"Verdict: {arr[1] if len(arr)>1 else ''}")
      elif tag == "o":
        mid = arr[2] if len(arr) > 2 else ""
        prose.append(f"Overflow: {arr[1] if len(arr)>1 else ''} [{mid}]")
      elif tag == "d":
        prose.append(f"Ref: {arr[1] if len(arr)>1 else ''}")
      elif tag == "r":
        prose.append(f"Role: {arr[1] if len(arr)>1 else ''}")
      elif tag == "t":
        prose.append(f"Note: {arr[1] if len(arr)>1 else ''}")
      elif tag == "x":
        key = arr[1] if len(arr) > 1 else ""
        val = arr[2] if len(arr) > 2 else ""
        prose.append(f"Meta: {key}={val}")
    return "\n".join(prose)

  # ---------- Internal: summarization shim ----------
  def _summarize(self, text: str, target_tokens: int) -> str:  # pragma: no cover
    # Back-compat: route to summarizer (tag unknown → treat as free text)
    return self.summarizer.summarize(text, "t", target_tokens)

  # ---------- Core: validation + overflow ----------
  def validate_and_overflow(self, jsonl: str) -> Tuple[str, Dict[str, Any]]:
    """
    Validate JSONL, normalize to arrays, enforce caps, and emit overflow lines.
    Returns (validated_jsonl_str, stats_dict).
    """
    out_lines: List[str] = []
    self.overflow_freq.clear()

    raw_lines = [ln for ln in jsonl.splitlines() if ln.strip()]
    for raw in raw_lines:
      arr = self.normalize_line(raw)
      if not isinstance(arr, list) or not arr:
        arr = ["t", json.dumps(raw)]

      tag = arr[0]
      if tag not in TAGS:
        # Unknown tag → coerce to free_text
        arr = ["t", json.dumps(arr, separators=(",", ":"))]
        tag = "t"

      # Enforce caps on textual payloads; q has (role, text)
      cap = self.caps.get(tag)
      if tag == "q":
        role = arr[1] if len(arr) > 1 else "W"
        text = arr[2] if len(arr) > 2 else ""
        if isinstance(text, str) and cap is not None and _token_estimate(text) > cap:
          summary = self.summarizer.summarize(text, tag, cap)
          method = getattr(self.summarizer, "method", "extractive")
          mid = self.memory.put(text)  # type: ignore[union-attr]
          self.overflow_freq[tag] = self.overflow_freq.get(tag, 0) + 1
          out_lines.append(json.dumps(["q", role, summary]))
          out_lines.append(json.dumps(["o", summary, mid, method]))
        else:
          out_lines.append(json.dumps(["q", role, text]))
        continue

      if tag in ("f", "p", "g", "u", "t"):
        text = arr[1] if len(arr) > 1 else ""
        if isinstance(text, str) and cap is not None and _token_estimate(text) > cap:
          summary = self.summarizer.summarize(text, tag, cap)
          method = getattr(self.summarizer, "method", "extractive")
          mid = self.memory.put(text)  # type: ignore[union-attr]
          self.overflow_freq[tag] = self.overflow_freq.get(tag, 0) + 1
          # For 'f', attach inline M# pointer as third element; for others, only summary text.
          new_line = ["f", summary, mid] if tag == "f" else [tag, summary]
          out_lines.append(json.dumps(new_line))
          out_lines.append(json.dumps(["o", summary, mid, method]))
        else:
          out_lines.append(json.dumps(arr))
        continue

      # Pass-through for non-textual or control tags ('r','d','v','x','o')
      out_lines.append(json.dumps(arr))

    total_lines = len(out_lines)
    overflow_count = sum(self.overflow_freq.values())
    rate = overflow_count / total_lines if total_lines else 0.0
    density = 1.0 - rate

    stats = {
      "lines_total": total_lines,
      "overflow": {
        "count": overflow_count,
        "per_tag": {k: v for k, v in self.overflow_freq.items() if v},
        "rate": rate,
      },
      "density": density,
    }
    return "\n".join(out_lines), stats

  # ---------- Public utility ----------
  def estimate_tokens(self, text: str) -> int:  # pragma: no cover - trivial
    return _token_estimate(text)
