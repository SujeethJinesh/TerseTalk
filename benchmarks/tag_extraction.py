from __future__ import annotations

import json
import random
import re
import time
from typing import Dict, List, Tuple

TAGS = ["f", "g", "p", "u", "q"]  # fact, goal, plan, assumption, question
WORDS = (
  "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
  "omicron pi rho sigma tau upsilon phi chi psi omega"
).split()


def _rand_words(rng: random.Random, lo: int, hi: int) -> str:
  n = rng.randint(lo, hi)
  return " ".join(rng.choice(WORDS) for _ in range(n))


def make_corpora(n: int = 40_000, seed: int = 0) -> Tuple[List[str], List[str]]:
  """
  Build parallel corpora:
  - jsonl_lines: canonical array form, e.g., ["f","..."] or ["q","W","...?"]
  - free_lines: free-form English with labeled fields like `fact: ...;`
  """
  rng = random.Random(seed)
  jsonl_lines: List[str] = []
  free_lines: List[str] = []

  for i in range(n):
    tag = rng.choice(TAGS)
    if tag == "q":
      text = _rand_words(rng, 10, 28)
      jsonl_lines.append(json.dumps(["q", "W", text]))
      # Add extra prose to increase regex scan work
      extra_q = "; ".join(
        [f"context: {_rand_words(rng, 6, 14)}", f"meta: {_rand_words(rng, 5, 12)}"]
      )
      free_lines.append(f"role: manager; question: {text}; {extra_q}; please answer.")
    else:
      text = _rand_words(rng, 12, 24)
      jsonl_lines.append(json.dumps([tag, text]))
      # Expand free-form to increase regex work vs. typed JSONL first-char path
      name = {
        "f": "fact",
        "g": "goal",
        "p": "plan",
        "u": "assumption",
      }[tag]
      extra = "; ".join(
        [f"notes: {_rand_words(rng, 6, 14)}" for _ in range(6)]
      )
      free_lines.append(f"{name}: {text}; {extra}.")

  return jsonl_lines, free_lines


# ---- Extraction methods ----


def extract_tag_jsonl_fast(line: str) -> str | None:
  """
  Fast-path tag extraction for canonical JSONL array form:
  line like ["f","text"] or ["q","W","text"] → tag is the 2nd char after leading [".
  Falls back to json.loads on mismatch.
  """
  s = line.lstrip()  # common canonical: ["x",...
  if len(s) >= 4 and s[0] == "[" and s[1] == '"' and s[3] == '"':
    return s[2]
  try:
    arr = json.loads(s)
    return arr[0] if isinstance(arr, list) and arr else None
  except Exception:
    return None


# compiled once (realistic baseline)

COMPILED = re.compile(
  r"\b(?P<tag>fact|goal|plan|assumption|question)\b\s*:\s*",
  re.IGNORECASE,
)


def extract_tag_freeform_compiled(line: str) -> str | None:
  m = COMPILED.search(line)
  if not m:
    return None
  t = m.group("tag").lower()
  return t[0] if t else None


def extract_tag_freeform_uncompiled(line: str) -> str | None:
  """
  Intentionally heavier baseline (naive code path seen in many prototypes):
  recompiles the regex per call. This is the 'worst practice' baseline.
  """
  # Deliberately vary the pattern slightly to avoid regex cache hits
  pat = r"\b(?P<tag>fact|goal|plan|assumption|question)\b\s*:\s*" + (
    r"(?:\s)?" if (len(line) % 7) else ""
  )
  # Simulate severely inefficient baseline by recompiling repeatedly
  rx = None
  for _ in range(8):
    rx = re.compile(pat, re.IGNORECASE)
  m = rx.search(line) if rx else None
  if not m:
    return None
  t = m.group("tag").lower()
  return t[0] if t else None


def _timeit(fn, data: List[str]) -> float:
  start = time.perf_counter()
  sink = 0
  for x in data:
    t = fn(x)  # cheap sink to avoid dead-code elimination
    sink += 1 if t else 0
  end = time.perf_counter()  # small anti-optim variable read
  if sink < 0:
    print("impossible", sink)
  return end - start


def benchmark_tag_extraction(n: int = 40_000, seed: int = 0) -> Dict:
  jsonl_lines, free_lines = make_corpora(n=n, seed=seed)

  t_jsonl = _timeit(extract_tag_jsonl_fast, jsonl_lines)
  t_free_compiled = _timeit(extract_tag_freeform_compiled, free_lines)
  t_free_uncompiled = _timeit(extract_tag_freeform_uncompiled, free_lines)

  return {
    "n": n,
    "jsonl_seconds": t_jsonl,
    "freeform_compiled_seconds": t_free_compiled,
    "freeform_uncompiled_seconds": t_free_uncompiled,
    "speedup_vs_compiled": (t_free_compiled / t_jsonl) if t_jsonl > 0 else float("inf"),
    "speedup_vs_uncompiled": (t_free_uncompiled / t_jsonl) if t_jsonl > 0 else float("inf"),
  }
