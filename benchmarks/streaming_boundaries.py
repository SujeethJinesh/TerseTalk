from __future__ import annotations

import random
import re
import time
from typing import Dict, Tuple, List

from .tag_extraction import WORDS, TAGS, _rand_words


def make_streams(n_msgs: int = 40_000, seed: int = 0) -> Tuple[str, str]:
  """
  Build two streams of concatenated messages:
  - jsonl_stream: one JSON value per line (O(1) boundary detection)
  - free_stream: multi-sentence prose with punctuation, abbreviations
  """
  rng = random.Random(seed)
  # JSONL stream: a variety of short lines (reuse from tag gen)
  jsonl_lines: List[str] = []
  for i in range(n_msgs):
    tag = rng.choice(TAGS)
    if tag == "q":
      text = _rand_words(rng, 8, 18)
      jsonl_lines.append(f'["q","W","{text}"]')
    else:
      text = _rand_words(rng, 8, 18)
      jsonl_lines.append(f'["{tag}","{text}"]')
  jsonl_stream = "\n".join(jsonl_lines)

  # Free-form stream: approximate sentence boundaries with tricky cases
  abbrs = ["e.g.", "i.e.", "Dr.", "Mr.", "Ms.", "vs."]
  sentences: List[str] = []
  for i in range(n_msgs):
    s1 = f"{_rand_words(rng, 5, 12)}."
    s2 = f"{rng.choice(abbrs)} {_rand_words(rng, 4, 9)}."
    s3 = f"{_rand_words(rng, 4, 9)}?"
    s4 = f"{_rand_words(rng, 4, 9)}!"
    sentences.append(" ".join([s1, s2, s3, s4]))
  free_stream = " ".join(sentences)
  return jsonl_stream, free_stream


# JSONL O(1) boundary detector: scan for '\n'


def count_jsonl_boundaries(stream: str) -> int:
  if not stream:
    return 0
  cnt = 0
  pos = 0
  while True:
    i = stream.find("\n", pos)
    if i == -1:
      break
    cnt += 1
    pos = i + 1
  # last line (if not ending with newline)
  if stream[-1] != "\n":
    cnt += 1
  return cnt


def count_freeform_sentences(stream: str) -> int:
  """
  Heuristic sentence counting without regex lookbehinds (Py's fixed-width limitation):
  Count .!? followed by whitespace, except when part of common abbreviations like
  e.g., i.e., Dr., Mr., Ms., vs., Mrs.
  """
  if not stream:
    return 0
  abbr3 = {"e.g", "i.e"}
  abbr2 = {"dr", "mr", "ms", "vs", "mrs"}
  n = len(stream)
  count = 0
  i = 0
  while i < n:
    ch = stream[i]
    if ch in ".!?" and (i + 1 < n and stream[i + 1].isspace()):
      # Check preceding abbreviation tokens (case-insensitive)
      prev3 = stream[i - 3 : i].lower() if i >= 3 else ""
      prev2 = stream[i - 2 : i].lower() if i >= 2 else ""
      if prev3 in abbr3 or prev2 in abbr2:
        i += 1
        continue
      count += 1
    i += 1
  # Add last fragment if stream does not end with a boundary+space
  # Ensure at least one sentence
  return max(1, count + 1)


def _timeit(fn, arg: str) -> float:
  start = time.perf_counter()
  val = fn(arg)
  end = time.perf_counter()  # lightweight sink
  if val < 0:
    print("impossible", val)
  return end - start


def benchmark_streaming(n_msgs: int = 40_000, seed: int = 0) -> Dict:
  j_stream, f_stream = make_streams(n_msgs=n_msgs, seed=seed)
  t_jsonl = _timeit(count_jsonl_boundaries, j_stream)
  t_free = _timeit(count_freeform_sentences, f_stream)
  return {
    "n_msgs": n_msgs,
    "jsonl_seconds": t_jsonl,
    "freeform_seconds": t_free,
    "speedup": (t_free / t_jsonl) if t_jsonl > 0 else float("inf"),
  }
