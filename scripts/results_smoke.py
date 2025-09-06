from __future__ import annotations

import json
from tersetalk.results_manager import ResultsManager


def main() -> None:
  rm = ResultsManager(base_dir="results")
  run_dir = rm.get_run_dir("hotpotqa_tersetalk", timestamp=True)

  cfg = {"task": "hotpotqa", "system": "tersetalk", "seed": 42, "caps": {"f": 30, "p": 20, "q": 30}}
  rm.save_config(run_dir, cfg)

  rm.append_jsonl(run_dir, "raw_outputs.jsonl", {"id": 1, "answer": "Paris", "em": True})
  rm.append_jsonl(run_dir, "raw_outputs.jsonl", {"id": 2, "answer": "Tokyo", "em": False})

  rm.append_csv_row(run_dir, "metrics.csv", {"id": 1, "em": 1, "tokens": 128})
  rm.append_csv_row(run_dir, "metrics.csv", {"id": 2, "em": 0, "tokens": 142})

  summary = {"examples": 2, "em": 0.5, "avg_tokens": 135}
  rm.save_summary(run_dir, summary)

  latest = run_dir.parent / "latest"
  print(json.dumps({
    "run_dir": str(run_dir),
    "latest": str(latest),
    "has_config": (latest / "config.json").exists(),
    "has_jsonl": (latest / "raw_outputs.jsonl").exists(),
    "has_csv": (latest / "metrics.csv").exists(),
    "has_summary": (latest / "summary.json").exists(),
  }, indent=2))


if __name__ == "__main__":
  main()

