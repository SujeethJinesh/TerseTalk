from __future__ import annotations

from benchmarks.tag_extraction import benchmark_tag_extraction
from benchmarks.streaming_boundaries import benchmark_streaming
from benchmarks.serde_bytes import benchmark_serde_bytes


def test_mb1_tag_extraction_speedup_ge_10x():
  # Smaller n for CI; uncompiled baseline should be dramatically slower
  res = benchmark_tag_extraction(n=20_000, seed=123)
  assert res["speedup_vs_uncompiled"] >= 10.0, res
  # compiled baseline should also be faster; keep a conservative bar for CI
  assert res["speedup_vs_compiled"] >= 1.4, res


def test_mb2_streaming_boundaries_speedup_ge_5x():
  res = benchmark_streaming(n_msgs=20_000, seed=7)
  assert res["speedup"] >= 5.0, res


def test_mb3_jsonl_bytes_significantly_smaller():
  # JSONL should be at least 30% smaller than verbose free-form equivalent
  res = benchmark_serde_bytes(n=2_000, seed=42)
  assert res["bytes_ratio_jsonl_over_freeform_verbose"] <= 0.70, res
  # Lean ratio should be reported as well (no strict threshold besides presence)
  assert "bytes_ratio_jsonl_over_freeform_lean" in res
