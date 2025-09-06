from __future__ import annotations

import math
from tersetalk.metrics import MetricsComputer


def test_em_normalization_and_yes_no():
  mc = MetricsComputer()
  assert mc.exact_match("The Eiffel; Tower", "eiffel tower")
  assert mc.exact_match("YES", "true")
  assert mc.exact_match("No", "FALSE")
  assert not mc.exact_match("paris", "london")


def test_gsm8k_integer_decimal_fraction_mixed_and_boxed():
  mc = MetricsComputer()
  assert mc.gsm8k_correct("Result: \\boxed{42}", "42")
  assert mc.gsm8k_correct("Therefore 1,234.50", "1234.5")
  assert mc.gsm8k_correct("the value is -12", "-12")
  assert mc.gsm8k_correct("ratio = 3/4", "0.75")
  assert mc.gsm8k_correct("time = 1 3/4 hours", "7/4")


def test_jaccard_and_bertscore_fallback():
  mc = MetricsComputer()
  j = mc.jaccard_sp("a b c", "a c d")
  # 'a' is an article and removed by normalization, so sets are {b,c} and {c,d} → 1/3
  assert abs(j - (1.0/3.0)) < 1e-9
  bf = mc.bertscore_sp("a b c", "a c d", force_fallback=True)
  assert abs(bf - j) < 1e-9


def test_token_count_heuristic_is_reasonable():
  mc = MetricsComputer(use_tiktoken=False)
  assert mc.count_tokens("") == 0
  assert mc.count_tokens("abcd") == 1
  assert mc.count_tokens("hello") == 2


def test_timing_metrics_shapes_and_values():
  mc = MetricsComputer()
  res = mc.measure_generation_timing(start=0.0, ttft=0.02, end=0.12, tokens=6)
  assert set(res.keys()) == {"tps", "ttft_ms", "itl_ms", "total_time_s"}
  assert res["total_time_s"] > 0
  assert res["ttft_ms"] > 0
  assert res["tps"] > 0
  assert res["itl_ms"] > 0


def test_bytes_on_wire_str_and_list():
  mc = MetricsComputer()
  s = "€"
  assert mc.bytes_on_wire(s) == len(s.encode("utf-8"))
  lines = ["a", "€"]
  assert mc.bytes_on_wire(lines) == sum(len(x.encode("utf-8")) for x in lines)
