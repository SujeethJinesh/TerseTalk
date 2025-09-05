from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure package import when running directly
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from tersetalk.structured import (
  EchoGenerator,
  InstructorGenerator,
  lines_to_jsonl,
  GoalLine,
  FactLine,
  QuestionLine,
  RoleLine,
  AnyTypedLine,
)
from tersetalk.protocol_jsonl import JSONLValidator


def main() -> None:
  ap = argparse.ArgumentParser(description="PR-02S Structured generation demo")
  ap.add_argument(
    "--mode",
    choices=["echo", "instructor"],
    default="echo",
    help="Generation mode (default: echo)",
  )
  ap.add_argument("--model", default="mistral", help="Model name for Instructor mode")
  ap.add_argument(
    "--goal",
    default="Compare dates of two events; return the earlier.",
    help="Goal text",
  )
  ap.add_argument(
    "--facts",
    nargs="*",
    default=["Event A: 2001-07-16", "Event B: 1999-05-02"],
    help="Facts list",
  )
  ap.add_argument("--question", default="Which is earlier?", help="Question text")
  ap.add_argument("--caps", default='{"f":30,"p":20,"q":30}', help="Caps JSON for validator")
  args = ap.parse_args()

  # Produce lines
  if args.mode == "echo":
    gen = EchoGenerator()
    lines = gen.generate(args.goal, args.facts, args.question)
    sys_prompt = "You are a Worker."
    usr_prompt = "Echo generator used (offline)."
  else:
    try:
      gen = InstructorGenerator(model=args.model)
    except Exception as e:
      print(json.dumps({"error": str(e)}), file=sys.stderr)
      sys.exit(2)
    sys_prompt = "You are a Manager. Produce concise TerseTalk lines."
    usr_prompt = (
      f"Goal: {args.goal}\nFacts: {args.facts}\n"
      f"Question: {args.question}\nRespond as a list of TerseTalkLine items."
    )
    lines = gen.generate(sys_prompt, usr_prompt)

  # JSONL + validation
  jsonl = lines_to_jsonl(lines)
  try:
    caps = json.loads(args.caps)
    if not isinstance(caps, dict):
      raise ValueError
  except Exception:
    print("Error: --caps must be JSON object", file=sys.stderr)
    sys.exit(2)

  validator = JSONLValidator(caps=caps)
  mixed, idx = validator.detect_format_break(jsonl)
  out, stats = validator.validate_and_overflow(jsonl)

  res = {
    "mode": args.mode,
    "system_prompt": sys_prompt,
    "user_prompt": usr_prompt,
    "lines_count": len(lines),
    "compliance_ok": (not mixed),
    "break_line": idx,
    "lines_jsonl": jsonl,
    "validated_jsonl": out,
    "validator_stats": stats,
  }
  print(json.dumps(res, indent=2))


if __name__ == "__main__":
  main()

