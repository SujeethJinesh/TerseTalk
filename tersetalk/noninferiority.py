from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple


def _to01(x) -> int:
  if isinstance(x, bool):
    return 1 if x else 0
  if isinstance(x, int):
    if x in (0, 1):
      return int(x)
  raise ValueError("Outcomes must be 0/1 or bool.")


def _validate_inputs(hybrid: Iterable, lingua: Iterable) -> Tuple[List[int], List[int]]:
  h = [_to01(v) for v in list(hybrid)]
  l = [_to01(v) for v in list(lingua)]
  if len(h) == 0 or len(l) == 0 or len(h) != len(l):
    raise ValueError("Hybrid and LLMLingua arrays must be non-empty and equal-length.")
  return h, l


def _acc(arr: List[int]) -> float:
  return sum(arr) / len(arr) if arr else 0.0


def _acc_subset(arr: List[int], idxs: List[int]) -> float:
  if not idxs:
    return 0.0
  s = 0
  for i in idxs:
    s += arr[i]
  return s / len(idxs)


def _percentile(sorted_vals: List[float], q: float) -> float:
  """Simple percentile on a sorted list; q in [0,1]. Linear interpolation."""
  if not sorted_vals:
    return 0.0
  n = len(sorted_vals)
  if n == 1:
    return sorted_vals[0]
  pos = q * (n - 1)
  lo = int(pos)
  hi = min(lo + 1, n - 1)
  frac = pos - lo
  return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@dataclass
class NIReport:
  n: int
  alpha: float
  delta: float
  acc_hybrid: float
  acc_llml: float
  diff: float
  lb_one_sided_95: float
  ci2_lower_95: float
  ci2_upper_95: float
  method: str
  n_boot: int
  seed: int
  noninferior: bool

  def to_dict(self) -> Dict:
    d = asdict(self)
    d["decision"] = "non-inferior" if self.noninferior else "fail-to-demonstrate"
    return d


def paired_bootstrap_diff(
  hybrid: Iterable,
  lingua: Iterable,
  n_boot: int = 1000,
  alpha: float = 0.05,
  seed: int = 0,
) -> NIReport:
  """
  Paired bootstrap of the accuracy difference d = acc(H) - acc(L).
  Returns an NIReport with two-sided 95% CI and the one-sided 95% lower bound.
  """
  h, l = _validate_inputs(hybrid, lingua)
  n = len(h)
  rng = random.Random(seed)

  base_h = _acc(h)
  base_l = _acc(l)
  base_d = base_h - base_l

  diffs: List[float] = []
  idxs = list(range(n))
  for _ in range(max(1, n_boot)):
    sample = [idxs[rng.randrange(n)] for _ in range(n)]  # resample paired indices with replacement
    dh = _acc_subset(h, sample)
    dl = _acc_subset(l, sample)
    diffs.append(dh - dl)

  diffs.sort()
  ci2_lo = _percentile(diffs, 0.025)
  ci2_hi = _percentile(diffs, 0.975)
  lb = _percentile(diffs, alpha)  # one-sided lower 95% bound at alpha=0.05

  return NIReport(
    n=n,
    alpha=alpha,
    delta=0.02,  # default margin; caller may override in decision step
    acc_hybrid=base_h,
    acc_llml=base_l,
    diff=base_d,
    lb_one_sided_95=lb,
    ci2_lower_95=ci2_lo,
    ci2_upper_95=ci2_hi,
    method="paired-bootstrap",
    n_boot=n_boot,
    seed=seed,
    noninferior=False,
  )


def noninferiority_test(
  hybrid: Iterable,
  lingua: Iterable,
  delta: float = 0.02,
  alpha: float = 0.05,
  n_boot: int = 1000,
  seed: int = 0,
) -> Dict:
  """
  One-sided non-inferiority: H0: d <= -delta; HA: d > -delta.
  We declare non-inferiority if lb_one_sided_95(d) > -delta.
  Returns a JSON-serializable dict (NIReport + decision).
  """
  rep = paired_bootstrap_diff(hybrid, lingua, n_boot=n_boot, alpha=alpha, seed=seed)
  decision = rep.lb_one_sided_95 > (-delta)
  rep.delta = float(delta)
  rep.noninferior = bool(decision)
  return rep.to_dict()

