from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, List, Tuple

from .tag_extraction import WORDS, _rand_words


def make_messages(n: int = 5_000, seed: int = 0) -> List[Dict[str, Any]]:
  rng = random.Random(seed)
  msgs = []
  for i in range(n):
    goal = f"Compare values; case {i}."
    facts = [_rand_words(rng, 8, 16), _rand_words(rng, 6, 12)]
    if i % 4 == 0:
      facts.append(_rand_words(rng, 6, 12))
    q = "Which is earlier?"
    msgs.append({"role": "M", "goal": goal, "facts": facts, "q_to": "W", "question": q})
  return msgs


def serialize_jsonl(msgs: List[Dict[str, Any]]) -> str:
  lines: List[str] = []
  for m in msgs:
    lines.append(json.dumps(["r", m["role"]]))
    lines.append(json.dumps(["g", m["goal"]]))
    for f in m["facts"]:
      lines.append(json.dumps(["f", f]))
    lines.append(json.dumps(["q", m["q_to"], m["question"]]))
  return "\n".join(lines)


def serialize_freeform_lean(msgs: List[Dict[str, Any]]) -> str:
  parts: List[str] = []
  for m in msgs:
    parts.append(f"Role: {m['role']}\n")
    parts.append(f"Goal: {m['goal']}\n")
    for f in m["facts"]:
      parts.append(f"Fact: {f}\n")
    parts.append(f"Question({m['q_to']}): {m['question']}\n\n")
  return "".join(parts)


def serialize_freeform_verbose(msgs: List[Dict[str, Any]]) -> str:
  parts: List[str] = []
  for m in msgs:
    parts.append(f"Role: {m['role']}\n")
    parts.append(f"Goal: {m['goal']}\n")
    for f in m["facts"]:
      parts.append(f"Fact: {f}\n")
    parts.append(f"Question({m['q_to']}): {m['question']}\n")
    # Add verbose context to resemble prose payloads
    parts.append(f"Context: {_rand_words(random.Random(len(m['goal'])), 12, 24)}\n")
    parts.append(f"Metadata: {_rand_words(random.Random(len(m['facts'])), 8, 16)}\n\n")
  return "".join(parts)


def maybe_msgpack(msgs: List[Dict[str, Any]]) -> Tuple[bool, bytes, float]:
  try:
    import msgpack  # type: ignore
  except Exception:
    return False, b"", 0.0
  start = time.perf_counter()
  blob = msgpack.packb(msgs, use_bin_type=True)
  t = time.perf_counter() - start
  return True, blob, t


def maybe_protobuf(msgs: List[Dict[str, Any]]) -> Tuple[bool, bytes, float]:
  """
  Guarded placeholder: skip real schema to keep PR dependency-free.
  """
  return False, b"", 0.0


def benchmark_serde_bytes(n: int = 5_000, seed: int = 0) -> Dict:
  msgs = make_messages(n=n, seed=seed)

  t0 = time.perf_counter()
  jsonl = serialize_jsonl(msgs)
  t_jsonl = time.perf_counter() - t0

  t1 = time.perf_counter()
  free_lean = serialize_freeform_lean(msgs)
  t_free_lean = time.perf_counter() - t1

  t2 = time.perf_counter()
  free_verbose = serialize_freeform_verbose(msgs)
  t_free_verbose = time.perf_counter() - t2

  b_jsonl = len(jsonl.encode("utf-8"))
  b_free_lean = len(free_lean.encode("utf-8"))
  b_free_verbose = len(free_verbose.encode("utf-8"))

  has_msgpack, blob_mp, t_mp = maybe_msgpack(msgs)
  res = {
    "n_msgs": n,
    "jsonl_seconds": t_jsonl,
    "freeform_lean_seconds": t_free_lean,
    "freeform_verbose_seconds": t_free_verbose,
    "jsonl_bytes": b_jsonl,
    "freeform_lean_bytes": b_free_lean,
    "freeform_verbose_bytes": b_free_verbose,
    "bytes_ratio_jsonl_over_freeform_lean": (b_jsonl / b_free_lean) if b_free_lean > 0 else 0.0,
    "bytes_ratio_jsonl_over_freeform_verbose": (b_jsonl / b_free_verbose) if b_free_verbose > 0 else 0.0,
  }
  if has_msgpack:
    res.update(
      {
        "msgpack_bytes": len(blob_mp),
        "msgpack_seconds": t_mp,
        "bytes_ratio_msgpack_over_jsonl": (len(blob_mp) / b_jsonl) if b_jsonl > 0 else 0.0,
      }
    )
  return res
