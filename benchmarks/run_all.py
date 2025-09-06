from __future__ import annotations

import argparse
import json

from .tag_extraction import benchmark_tag_extraction
from .streaming_boundaries import benchmark_streaming
from .serde_bytes import benchmark_serde_bytes


def main() -> None:
  ap = argparse.ArgumentParser(description="TerseTalk Microbenchmark Suite")
  ap.add_argument("--fast", action="store_true", help="Use smaller sizes (CI-friendly).")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  if args.fast:
    n_tag = 20_000
    n_stream = 20_000
    n_serde = 2_000
  else:
    n_tag = 40_000
    n_stream = 40_000
    n_serde = 5_000

  mb1 = benchmark_tag_extraction(n=n_tag, seed=args.seed)
  mb2 = benchmark_streaming(n_msgs=n_stream, seed=args.seed)
  mb3 = benchmark_serde_bytes(n=n_serde, seed=args.seed)

  report = {
    "MB1_tag_extraction": mb1,
    "MB2_streaming_boundaries": mb2,
    "MB3_serde_bytes": mb3,
    "notes": "Times are wall-clock seconds (time.perf_counter). JSON is valid YAML.",
  }
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()

