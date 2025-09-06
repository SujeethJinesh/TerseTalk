from __future__ import annotations

import argparse
import json
import os

from tersetalk.datasets import load_hotpotqa, load_gsm8k
from tersetalk.model_io import EchoModel
from tersetalk.pipeline_runner import PipelineConfig, run_pipeline_once


def main():
  ap = argparse.ArgumentParser(description="PR-07: Manager→Worker→Critic pipeline smoke")
  ap.add_argument("--task", choices=["hotpotqa", "gsm8k"], default="hotpotqa")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--offline", action="store_true")
  ap.add_argument("--model", choices=["echo", "real"], default="echo")
  ap.add_argument("--caps", default='{"f":30,"p":20,"q":30,"g":30,"u":20,"t":50}')
  ap.add_argument("--use-handler", action="store_true", help="Use protocol handler if available")
  ap.add_argument("--preoverflow-ll2", action="store_true")
  ap.add_argument("--overflow-ll2", action="store_true")
  ap.add_argument("--deref-ll2", action="store_true")
  ap.add_argument("--deref-policy", choices=["never","conditional","always"], default="never")
  args = ap.parse_args()

  offline = args.offline or (os.environ.get("TERSETALK_OFFLINE_DATA") == "1")
  caps = json.loads(args.caps)

  if args.task == "hotpotqa":
    ex = load_hotpotqa(n=1, seed=args.seed, offline=offline)[0]
  else:
    ex = load_gsm8k(n=1, seed=args.seed, offline=offline)[0]

  cfg = PipelineConfig(
    caps=caps,
    use_protocol_handler=args.use_handler,
    preoverflow_ll2=args.preoverflow_ll2,
    overflow_ll2=args.overflow_ll2,
    deref_ll2=args.deref_ll2,
    deref_policy=args.deref_policy,
  )
  if args.model == "real":
    if os.environ.get("RUN_REAL_MODEL") != "1":
      print("SKIP: real model not enabled (set RUN_REAL_MODEL=1)")
      return
    from tersetalk.model_io import ModelClient
    client = ModelClient()
  else:
    client = EchoModel()
  res = run_pipeline_once(ex, client, cfg)
  print(json.dumps(res, indent=2))


if __name__ == "__main__":
  main()
