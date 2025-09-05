from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure package import when running directly
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from tersetalk.memory import MemoryStore
from tersetalk.protocol_jsonl import JSONLValidator


def do_store(args):
  mem = MemoryStore()
  ids = [mem.put(s) for s in args.items]
  fetched = {mid: mem.get(mid) for mid in ids}
  out = {
    "mode": "store",
    "minted_ids": ids,
    "fetched": fetched,
    "stats": mem.stats(),
  }
  print(json.dumps(out, indent=2))


def _check_o_refs(jsonl_out: str, mem: MemoryStore) -> bool:
  """Verify that every M# in overflow lines can be dereferenced."""
  ok = True
  for line in [ln for ln in jsonl_out.splitlines() if ln.strip()]:
    if line.startswith('["o"'):
      try:
        arr = json.loads(line)
      except Exception:
        continue
      if len(arr) >= 3:
        mid = arr[2]
        if not isinstance(mem.get(mid), str):
          ok = False
  return ok


def do_validate(args):
  try:
    caps = json.loads(args.caps)
    if not isinstance(caps, dict):
      raise ValueError
  except Exception:
    print("Error: --caps must be a JSON object", file=sys.stderr)
    sys.exit(2)

  data = sys.stdin.read() if args.input == "-" else open(args.input, "r", encoding="utf-8").read()
  mem = MemoryStore()
  validator = JSONLValidator(caps=caps, memory=mem)
  mixed, idx = validator.detect_format_break(data)
  out, stats = validator.validate_and_overflow(data)
  res = {
    "mode": "validate",
    "mixed_format": mixed,
    "break_line": idx,
    "validated_jsonl": out,
    "validator_stats": stats,
    "memory_stats": mem.stats(),
    "o_refs_retrievable": _check_o_refs(out, mem),
  }
  print(json.dumps(res, indent=2))


def main():
  ap = argparse.ArgumentParser(description="PR-03 MemoryStore smoke utility")
  sub = ap.add_subparsers(dest="cmd", required=True)

  p_store = sub.add_parser("store", help="Mint M# ids for provided items and fetch them back.")
  p_store.add_argument("items", nargs="+", help="Strings to store")
  p_store.set_defaults(func=do_store)

  p_val = sub.add_parser("validate", help="Validate JSONL with real MemoryStore and show stats.")
  p_val.add_argument("--caps", default='{"f":30,"p":20,"q":30}', help='Per-tag caps JSON, e.g. "{"f":20,"q":25}"')
  p_val.add_argument("--input", default="-", help="Path to JSONL file or '-' for stdin.")
  p_val.set_defaults(func=do_validate)

  args = ap.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()

