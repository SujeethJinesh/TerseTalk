from __future__ import annotations

import json
from tersetalk.reproducibility import set_global_seed, fingerprint_snapshot


def run_once(seed: int, n: int = 5) -> dict:
  cfg = set_global_seed(seed)
  fp = fingerprint_snapshot(n=n)
  return {"seed": seed, "defaults": cfg, "fingerprint": fp}


def main():
  same1 = run_once(1234)
  same2 = run_once(1234)
  diff = run_once(1235)

  out = {
    "same_seed_equal": same1["fingerprint"] == same2["fingerprint"],
    "diff_seed_unequal": same1["fingerprint"] != diff["fingerprint"],
    "runs": [same1, same2, diff],
  }
  print(json.dumps(out, indent=2))


if __name__ == "__main__":
  main()

