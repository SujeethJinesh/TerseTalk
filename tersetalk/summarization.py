from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List


def _token_estimate(text: str) -> int:
  """~4 chars ≈ 1 token heuristic used across the repo."""
  return max(0, (len(text) + 3) // 4)


_STOPWORDS = {
  "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at",
  "by","with","from","as","that","this","these","those","be","is","are","was","were",
  "it","its","into","over","under","about","after","before","so","we","you","they",
  "he","she","him","her","them","i","my","your","our","their","not","no","yes","do",
  "does","did","done","can","could","may","might","must","should","would","will",
}


def _norm_ws(s: str) -> str:
  return re.sub(r"\s+", " ", s).strip()


def _split_sentences(text: str) -> List[str]:
  text = _norm_ws(text)
  if not text:
    return []
  parts = re.split(r"(?<=[.!?])\s+", text)
  out = [p.strip() for p in parts if p.strip()]
  if not out:
    out = [text]
  return out


def _words(text: str) -> List[str]:
  return re.findall(r"[A-Za-z0-9']+", text.lower())


@dataclass
class Summarizer:
  """
  Lightweight summarizer with two modes:
  - method="extractive" (default): stdlib-only scoring
  - method="llmlingua": optional; falls back to extractive if not available
  """

  method: str = "extractive"

  def summarize(self, text: str, tag: str, target_tokens: int = 20) -> str:
    text = _norm_ws(text)
    if not text:
      return ""
    if target_tokens <= 0:
      return text[:1]

    if self.method == "llmlingua":
      out = self._llmlingua_summary(text, target_tokens)
      if out is not None:
        return out

    # Default / fallback
    return self._extractive_summary(text, target_tokens)

  # --------- Methods ---------
  def _extractive_summary(self, text: str, target_tokens: int) -> str:
    sents = _split_sentences(text)
    if len(sents) == 1:
      return self._hard_trim(sents[0], target_tokens)

    # Build TF and DF
    sent_tokens: List[List[str]] = []
    df: Dict[str, int] = {}
    for s in sents:
      toks = [w for w in _words(s) if w not in _STOPWORDS]
      if not toks:
        toks = _words(s)
      sent_tokens.append(toks)
      for w in set(toks):
        df[w] = df.get(w, 0) + 1

    N = len(sents)
    idf: Dict[str, float] = {}
    for w, d in df.items():
      idf[w] = math.log((N + 1) / (1 + d)) + 1.0  # smoothed

    # Score sentences: sum(TF * IDF)/sqrt(len) + position bonus
    scores: List[float] = []
    for i, toks in enumerate(sent_tokens):
      if toks:
        tf: Dict[str, int] = {}
        for w in toks:
          tf[w] = tf.get(w, 0) + 1
        max_tf = max(tf.values())
        core = sum((tf[w] / max_tf) * idf.get(w, 0.0) for w in tf)
        core /= math.sqrt(len(toks) + 1e-6)
      else:
        core = 0.0
      pos_bonus = 0.1 * (1.0 - (i / max(1, N - 1))) if N > 1 else 0.0
      scores.append(core + pos_bonus)

    order = sorted(range(N), key=lambda i: scores[i], reverse=True)

    # Greedy selection under token budget
    chosen: List[int] = []
    acc_text = ""
    for idx in order:
      cand = (acc_text + " " + sents[idx]).strip() if acc_text else sents[idx]
      if _token_estimate(cand) <= target_tokens:
        chosen.append(idx)
        acc_text = cand

    if not chosen:
      best = sents[order[0]]
      return self._hard_trim(best, target_tokens)

    chosen.sort()
    summary = " ".join(sents[i] for i in chosen)
    return self._hard_trim(summary, target_tokens)

  def _hard_trim(self, text: str, target_tokens: int) -> str:
    if _token_estimate(text) <= target_tokens:
      return text
    max_chars = max(1, target_tokens * 4)
    if len(text) <= max_chars:
      return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    if space >= 20:
      cut = cut[:space]
    return cut.rstrip() + "..."

  def _llmlingua_summary(self, text: str, target_tokens: int) -> str | None:
    """Attempt LLMLingua compression; fallback to extractive on any error/import failure."""
    try:
      from llmlingua import PromptCompressor  # type: ignore
    except Exception:
      return None
    try:
      comp = PromptCompressor()
      result = comp.compress(text, target_token=target_tokens)
      cand = (
        result.get("compressed_prompt")
        or result.get("compressed_text")
        or result.get("prompt")
        or result.get("text")
      )
      if not isinstance(cand, str) or not cand.strip():
        return None
      return self._hard_trim(cand.strip(), target_tokens)
    except Exception:
      return None

