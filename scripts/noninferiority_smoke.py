from __future__ import annotations

import argparse
import json
import os
import sys
import random
from typing import List

# Ensure package import when running directly
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from tersetalk.noninferiority import noninferiority_test


def _gen_bernoulli(n: int, p: float, seed: int) -> List[int]:
  rng = random.Random(seed)
  return [1 if rng.random() < p else 0 for _ in range(n)]


def main() -> None:
  ap = argparse.ArgumentParser(description="PR-H3: One-sided non-inferiority smoke")
  ap.add_argument("--n", type=int, default=500, help="Number of examples")
  ap.add_argument("--p-hybrid", type=float, default=0.79, help="Hybrid accuracy (synthetic)")
  ap.add_argument("--p-ll", type=float, default=0.78, help="LLMLingua accuracy (synthetic)")
  ap.add_argument("--delta", type=float, default=0.02, help="Non-inferiority margin (absolute)")
  ap.add_argument("--alpha", type=float, default=0.05, help="One-sided alpha")
  ap.add_argument("--seed", type=int, default=0, help="Random seed")
  ap.add_argument("--n-boot", type=int, default=800, help="Bootstrap replicates")
  ap.add_argument(
    "--from-json",
    type=str,
    default="",
    help="Path to JSON file with {'hybrid':[0/1...],'ll':[0/1...]}",
  )
  args = ap.parse_args()

  if args.from_json:
    with open(args.from_json, "r", encoding="utf-8") as f:
      payload = json.load(f)
    hybrid = payload["hybrid"]
    ll = payload["ll"]
  else:
    hybrid = _gen_bernoulli(args.n, args.p_hybrid, args.seed)
    ll = _gen_bernoulli(args.n, args.p_ll, args.seed + 1)

  report = noninferiority_test(
    hybrid, ll, delta=args.delta, alpha=args.alpha, n_boot=args.n_boot, seed=args.seed
  )
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()

