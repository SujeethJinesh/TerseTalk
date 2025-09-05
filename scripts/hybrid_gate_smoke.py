from __future__ import annotations

import argparse
import json

import os
import sys

# Ensure package import when running directly
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from tersetalk.hybrid_gate import GateCfg, gate_choose_protocol

EXAMPLE_JSONL = '["r","M"]\n["g","Compare dates"]\n["f","Event A: 2001-07-16"]\n["f","Event B: 1999-05-02"]\n["q","W","Which is earlier?"]'
EXAMPLE_FREEFORM = (
  "Role: Manager\n"
  "Goal: Compare dates\n"
  "Facts: Event A: 2001-07-16; Event B: 1999-05-02\n"
  "Question: Which is earlier?"
)


def main() -> None:
  ap = argparse.ArgumentParser(description="PR-H1 hybrid gate smoke tool")
  ap.add_argument("--jsonl", default=EXAMPLE_JSONL, help="JSONL probe text")
  ap.add_argument("--freeform", default=EXAMPLE_FREEFORM, help="Free-form probe text")
  ap.add_argument("--token-budget", type=int, default=600, help="Token budget for gate")
  args = ap.parse_args()

  cfg = GateCfg(token_budget=args.token_budget)
  decision = gate_choose_protocol(args.jsonl, args.freeform, cfg)
  print(json.dumps({"cfg": cfg.__dict__, "decision": decision}, indent=2))


if __name__ == "__main__":
  main()
