from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure package import when running directly
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from tersetalk.protocol_jsonl import JSONLValidator
from tersetalk.summarization import Summarizer


def main() -> None:
  ap = argparse.ArgumentParser(description="TerseTalk JSONL guard (PR-02+PR-04)")
  ap.add_argument(
    "--caps",
    type=str,
    default='{"f":30,"p":20,"q":30}',
    help='Per-tag caps JSON, e.g. "{"f":20,"q":25}"',
  )
  ap.add_argument(
    "--fail-on-mixed",
    action="store_true",
    help="Exit nonzero if mixed format detected.",
  )
  ap.add_argument(
    "--summarizer",
    choices=["extractive", "llmlingua"],
    default="extractive",
    help="Summarizer method for overflow summaries.",
  )
  ap.add_argument(
    "--input",
    type=str,
    default="-",
    help="Path to JSONL file or '-' for stdin.",
  )
  args = ap.parse_args()

  try:
    caps = json.loads(args.caps)
    if not isinstance(caps, dict):
      raise ValueError
  except Exception:
    print("Error: --caps must be a JSON object.", file=sys.stderr)
    sys.exit(2)

  if args.input == "-":
    data = sys.stdin.read()
  else:
    with open(args.input, "r", encoding="utf-8") as f:
      data = f.read()

  validator = JSONLValidator(caps=caps, summarizer=Summarizer(method=args.summarizer))
  mixed, idx = validator.detect_format_break(data)
  if mixed and args.fail_on_mixed:
    print(json.dumps({"mixed_format": True, "break_line": idx}), file=sys.stderr)
    sys.exit(3)

  out, stats = validator.validate_and_overflow(data)
  res = {"mixed_format": mixed, "break_line": idx, "stats": stats, "out": out}
  print(json.dumps(res, indent=2))


if __name__ == "__main__":
  main()
