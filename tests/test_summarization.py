from __future__ import annotations

from tersetalk.summarization import Summarizer


def test_extractive_produces_shorter_or_equal_and_keeps_keywords():
  text = (
    "Alpha beta gamma delta. Gamma delta epsilon zeta eta. "
    "This sentence is filler. Alpha appears again with beta and gamma."
  )
  s = Summarizer(method="extractive")
  out = s.summarize(text, tag="t", target_tokens=10)
  assert isinstance(out, str) and len(out) > 0

  def est(t: str) -> int:
    return (len(t) + 3) // 4

  assert est(out) <= 10 or est(text) <= 10
  lowered = out.lower()
  assert ("alpha" in lowered) or ("gamma" in lowered) or ("delta" in lowered)


def test_llmlingua_path_falls_back_if_unavailable():
  text = " ".join(["alpha beta gamma delta epsilon zeta"] * 5)
  s = Summarizer(method="llmlingua")
  out = s.summarize(text, tag="f", target_tokens=8)
  assert isinstance(out, str) and len(out) > 0

  def est(t: str) -> int:
    return (len(t) + 3) // 4

  assert est(out) <= est(text)

