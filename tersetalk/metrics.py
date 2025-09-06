from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Iterable, Optional, Union

_TIKTOKEN = None
_BERT_SCORE = None


def _lazy_import_tiktoken(encoder_name: str = "cl100k_base"):
  global _TIKTOKEN
  if _TIKTOKEN is not None:
    return _TIKTOKEN
  try:
    import tiktoken  # type: ignore

    enc = tiktoken.get_encoding(encoder_name)
    _TIKTOKEN = enc
  except Exception:
    _TIKTOKEN = None
  return _TIKTOKEN


def _lazy_import_bertscore():
  global _BERT_SCORE
  if _BERT_SCORE is not None:
    return _BERT_SCORE
  try:
    from bert_score import score  # type: ignore

    _BERT_SCORE = score
  except Exception:
    _BERT_SCORE = None
  return _BERT_SCORE


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

_FRACTION_RE = re.compile(
  r"""
(?P<whole>-?\d+)? # optional whole number (may be negative)
(?:\s+)? # optional space
(?P<num>\d+)\s*/\s*(?P<den>\d+) # numerator/denominator
""",
  re.VERBOSE,
)

_NUM_RE = re.compile(
  r"""
(?P<sign>[-+]?)
(
  (?P<intg>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))? # 1,234 or 1234 or 1,234.56
  |
  \.(?P<leadfrac>\d+) # .5
)
""",
  re.VERBOSE,
)

_BOXED_RE = re.compile(r"\\boxed\s*\{([^}]*)\}")


def _normalize(text: str) -> str:
  s = (text or "").lower()
  s = _PUNCT_RE.sub(" ", s)
  s = _ARTICLE_RE.sub(" ", s)
  s = _WS_RE.sub(" ", s).strip()
  return s


_YES_EQUIV = {"yes", "true", "y", "1", "correct"}
_NO_EQUIV = {"no", "false", "n", "0", "incorrect"}


def _yn_canonical(s: str) -> Optional[str]:
  ss = _normalize(s)
  if ss in _YES_EQUIV:
    return "yes"
  if ss in _NO_EQUIV:
    return "no"
  return None


def _last_fraction_as_fraction(s: str) -> Optional[Fraction]:
  matches = list(_FRACTION_RE.finditer(s))
  if not matches:
    return None
  m = matches[-1]
  num = int(m.group("num"))
  den = int(m.group("den"))
  whole_str = m.group("whole")
  frac = Fraction(num, den)
  if whole_str is not None:
    whole = int(whole_str)
    sign = -1 if whole < 0 else 1
    frac = Fraction(abs(whole), 1) + frac
    frac *= sign
  return frac


def _last_number_as_fraction(s: str) -> Optional[Fraction]:
  frac = _last_fraction_as_fraction(s)
  if frac is not None:
    return frac
  matches = list(_NUM_RE.finditer(s))
  if not matches:
    return None
  m = matches[-1]
  sign = -1 if m.group("sign") == "-" else 1
  if m.group("leadfrac"):
    dec_str = "0." + m.group("leadfrac")
  else:
    intg = m.group("intg").replace(",", "")
    frac_part = m.group("frac")
    dec_str = intg if frac_part is None else f"{intg}.{frac_part}"
  try:
    dec = Decimal(dec_str)
    f = Fraction(dec)
    return -f if sign == -1 else f
  except (InvalidOperation, ValueError):
    return None


def _strip_latex_wrappers(s: str) -> str:
  out = _BOXED_RE.sub(lambda m: m.group(1), s)
  out = out.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
  return out


@dataclass
class MetricsComputer:
  use_tiktoken: bool = False
  encoder_name: str = "cl100k_base"

  def __post_init__(self):
    self._encoder = _lazy_import_tiktoken(self.encoder_name) if self.use_tiktoken else None

  def count_tokens(self, text: str) -> int:
    s = "" if text is None else str(text)
    if self._encoder is not None:
      try:
        return len(self._encoder.encode(s))
      except Exception:
        pass
    n = len(s)
    return math.ceil(n / 4) if n else 0

  def exact_match(self, pred: str, gold: str) -> bool:
    p_can = _yn_canonical(pred)
    g_can = _yn_canonical(gold)
    if p_can is not None and g_can is not None:
      return p_can == g_can
    return _normalize(pred) == _normalize(gold)

  def gsm8k_correct(self, pred: str, gold: str) -> bool:
    pred_s = _strip_latex_wrappers(pred or "")
    gold_s = _strip_latex_wrappers(gold or "")
    p_val = _last_number_as_fraction(pred_s)
    g_val = _last_number_as_fraction(gold_s)
    if p_val is None or g_val is None:
      return False
    return p_val == g_val

  def jaccard_sp(self, reference: str, candidate: str) -> float:
    ref = set((_normalize(reference) or "").split())
    cand = set((_normalize(candidate) or "").split())
    if not ref and not cand:
      return 1.0
    if not ref or not cand:
      return 0.0
    return len(ref & cand) / len(ref | cand)

  def bertscore_sp(self, reference: str, candidate: str, force_fallback: bool = False) -> float:
    if force_fallback:
      return self.jaccard_sp(reference, candidate)
    scorer = _lazy_import_bertscore()
    if scorer is None:
      return self.jaccard_sp(reference, candidate)
    try:
      P, R, F1 = scorer([candidate], [reference], lang="en")
      return float(F1[0].item())
    except Exception:
      return self.jaccard_sp(reference, candidate)

  def measure_generation_timing(self, start: float, ttft: float, end: float, tokens: int) -> dict:
    total = max(0.0, end - start)
    ttft_ms = max(0.0, (ttft - start) * 1000.0)
    if tokens and tokens > 1 and end > ttft:
      itl_ms = ((end - ttft) / (tokens - 1)) * 1000.0
    else:
      itl_ms = 0.0
    tps = (tokens / total) if (tokens > 0 and total > 0) else 0.0
    return {"tps": tps, "ttft_ms": ttft_ms, "itl_ms": itl_ms, "total_time_s": total}

  def bytes_on_wire(self, payload: Union[str, Iterable[str]]) -> int:
    if isinstance(payload, str):
      return len(payload.encode("utf-8"))
    total = 0
    for p in payload:
      total += len(str(p).encode("utf-8"))
    return total

