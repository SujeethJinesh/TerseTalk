from __future__ import annotations

import json
from tersetalk.metrics import MetricsComputer


def main() -> None:
  mc = MetricsComputer(use_tiktoken=False)

  pred = "The Eiffel Tower!"
  gold = "eiffel tower"
  em = mc.exact_match(pred, gold)

  p = "Therefore, the answer is \\boxed{1 3/4}."
  g = "7/4"
  gsm_ok = mc.gsm8k_correct(p, g)

  sp = mc.bertscore_sp("a quick brown fox", "a fast brown fox", force_fallback=True)

  toks = mc.count_tokens('["f","Paris"]\n["q","W","Capital?"]')
  timing = mc.measure_generation_timing(start=0.00, ttft=0.02, end=0.12, tokens=8)

  bow = mc.bytes_on_wire('["f","Paris"]\n')

  out = {
    "em_demo": em,
    "gsm8k_demo": gsm_ok,
    "sp_demo": sp,
    "token_count_demo": toks,
    "timing_demo": timing,
    "bytes_on_wire_demo": bow,
  }
  print(json.dumps(out, indent=2))


if __name__ == "__main__":
  main()

