from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure package import when running directly
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from tersetalk.summarization import Summarizer


def main() -> None:
  ap = argparse.ArgumentParser(description="PR-04 Summarizer smoke tool")
  ap.add_argument("--method", choices=["extractive", "llmlingua"], default="extractive")
  ap.add_argument("--target-tokens", type=int, default=20)
  ap.add_argument("--input", default="-", help="Path or '-' for stdin")
  args = ap.parse_args()

  text = sys.stdin.read() if args.input == "-" else open(args.input, "r", encoding="utf-8").read()
  s = Summarizer(method=args.method)
  summary = s.summarize(text, tag="t", target_tokens=args.target_tokens)

  def est(t: str) -> int:
    return max(0, (len(t) + 3) // 4)

  print(
    json.dumps(
      {
        "method": args.method,
        "target_tokens": args.target_tokens,
        "orig_tokens_est": est(text),
        "summary_tokens_est": est(summary),
        "summary": summary,
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()

