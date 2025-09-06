from __future__ import annotations

import argparse
import json
import os

from tersetalk.datasets import load_gsm8k, load_hotpotqa


def main():
    ap = argparse.ArgumentParser(description="PR-06: Dataset adapters smoke tool")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offline", action="store_true", help="Force synthetic shards")
    args = ap.parse_args()

    offline = args.offline or (os.environ.get("TERSETALK_OFFLINE_DATA") == "1")

    hotpot = load_hotpotqa(n=args.n, seed=args.seed, offline=offline)
    gsm = load_gsm8k(n=args.n, seed=args.seed, offline=offline)

    out = {
        "config": {"n": args.n, "seed": args.seed, "offline": offline},
        "hotpotqa_samples": hotpot[:2],
        "gsm8k_samples": gsm[:2],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

