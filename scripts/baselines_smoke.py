from __future__ import annotations

import argparse
import json
import os

from tersetalk.datasets import load_hotpotqa, load_gsm8k
from tersetalk.model_io import EchoModel
from tersetalk.baselines import run_freeform_once, run_llmlingua_once


def main():
  ap = argparse.ArgumentParser(description="PR-08: Baselines smoke (free-form & LLMLingua)")
  ap.add_argument("--task", choices=["hotpotqa", "gsm8k"], default="hotpotqa")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--offline", action="store_true")
  ap.add_argument("--model", choices=["echo", "real"], default="echo")
  ap.add_argument("--disable-ll2", action="store_true", help="Force disable LLMLingua via env")
  ap.add_argument("--max-tokens", type=int, default=256, help="Max tokens for baseline generation")
  args = ap.parse_args()

  if args.disable_ll2:
    os.environ["TERSETALK_DISABLE_LL2"] = "1"

  offline = args.offline or (os.environ.get("TERSETALK_OFFLINE_DATA") == "1")

  if args.task == "hotpotqa":
    ex = load_hotpotqa(n=1, seed=args.seed, offline=offline)[0]
  else:
    ex = load_gsm8k(n=1, seed=args.seed, offline=offline)[0]

  if args.model == "real":
    if os.environ.get("RUN_REAL_MODEL") != "1":
      print("SKIP: real model not enabled (set RUN_REAL_MODEL=1)")
      return
    from tersetalk.model_io import ModelClient
    client = ModelClient()
  else:
    client = EchoModel()

  print("=== Free-form baseline ===")
  res_free = run_freeform_once(ex, client, max_tokens=args.max_tokens)
  print(json.dumps(res_free, indent=2))

  print("\n=== Free-form + LLMLingua baseline ===")
  res_ll2 = run_llmlingua_once(ex, client, target_token=200, max_tokens=args.max_tokens)
  print(json.dumps(res_ll2, indent=2))


if __name__ == "__main__":
  main()
