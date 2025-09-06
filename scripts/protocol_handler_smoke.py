from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure package import for direct execution
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from tersetalk.protocol_handler import PHConfig, ProtocolHandler
from tersetalk.memory import MemoryStore
from tersetalk.calibration import synth_shard


def main() -> None:
  ap = argparse.ArgumentParser(description="PR-H4: Protocol Handler smoke tool")
  ap.add_argument("--input", default="-", help="JSONL input or '-' for synthetic")
  ap.add_argument("--seed", type=int, default=0, help="Seed for synthetic example")
  ap.add_argument("--caps", default='{"f":30,"p":20,"q":30,"g":30,"u":20,"t":50}')
  ap.add_argument("--summarizer", choices=["extractive", "llmlingua"], default="extractive")
  ap.add_argument("--preoverflow-ll2", action="store_true")
  ap.add_argument("--overflow-ll2", action="store_true")
  ap.add_argument("--deref-ll2", action="store_true")
  ap.add_argument("--deref-policy", choices=["never", "conditional", "always"], default="never")
  args = ap.parse_args()

  if args.input == "-":
    data = sys.stdin.read()
    if data.strip():
      mgr = data
    else:
      ex = synth_shard(1, args.seed)[0]
      mgr = ex["jsonl"]
  else:
    with open(args.input, "r", encoding="utf-8") as f:
      mgr = f.read()

  caps = json.loads(args.caps)
  cfg = PHConfig(
    caps=caps,
    summarizer_method=args.summarizer,
    preoverflow_ll2=args.preoverflow_ll2,
    overflow_ll2=args.overflow_ll2,
    deref_ll2=args.deref_ll2,
    deref_policy=args.deref_policy,
  )
  handler = ProtocolHandler(cfg)
  out = handler.process(mgr, memory=MemoryStore())
  print(json.dumps(out.to_dict(), indent=2))


if __name__ == "__main__":
  main()
