from __future__ import annotations

import random

from tersetalk.noninferiority import noninferiority_test, paired_bootstrap_diff


def _gen(n, p, seed):
  rng = random.Random(seed)
  return [1 if rng.random() < p else 0 for _ in range(n)]


def test_schema_and_determinism_small_bootstrap():
  n = 300
  seed = 7
  h = _gen(n, 0.79, seed)
  l = _gen(n, 0.78, seed + 1)

  rep1 = noninferiority_test(h, l, delta=0.02, n_boot=400, seed=seed)
  rep2 = noninferiority_test(h, l, delta=0.02, n_boot=400, seed=seed)
  assert rep1 == rep2
  for k in [
    "n",
    "alpha",
    "delta",
    "acc_hybrid",
    "acc_llml",
    "diff",
    "lb_one_sided_95",
    "ci2_lower_95",
    "ci2_upper_95",
    "method",
    "n_boot",
    "seed",
    "noninferior",
    "decision",
  ]:
    assert k in rep1


def test_noninferiority_pass_case():
  n = 500
  seed = 3
  h = _gen(n, 0.80, seed)
  l = _gen(n, 0.78, seed + 1)
  rep = noninferiority_test(h, l, delta=0.02, n_boot=400, seed=seed)
  assert rep["noninferior"] is True
  assert rep["lb_one_sided_95"] > -0.02


def test_noninferiority_fail_case():
  n = 400
  seed = 1
  h = _gen(n, 0.73, seed)
  l = _gen(n, 0.78, seed + 1)
  rep = noninferiority_test(h, l, delta=0.02, n_boot=400, seed=seed)
  assert rep["noninferior"] is False
  assert rep["lb_one_sided_95"] <= -0.02


def test_paired_bootstrap_reporting_consistency():
  n = 200
  seed = 3
  h = _gen(n, 0.76, seed)
  l = _gen(n, 0.76, seed + 1)
  rpt = paired_bootstrap_diff(h, l, n_boot=300, seed=seed)
  assert abs(rpt.acc_hybrid - rpt.acc_llml) <= 0.1
  assert rpt.ci2_lower_95 <= rpt.ci2_upper_95
