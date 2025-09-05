from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from tersetalk.reproducibility import set_global_seed
from tersetalk.protocol_jsonl import JSONLValidator
from tersetalk.summarization import Summarizer
from tersetalk.memory import MemoryStore
from tersetalk.hybrid_gate import GateCfg, gate_choose_protocol, estimate_tokens


# --------------------------
# Synthetic shard generator
# --------------------------

_LOREM = (
  "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
  "omicron pi rho sigma tau upsilon phi chi psi omega"
).split()


def _rand_words(rng: random.Random, lo: int, hi: int) -> str:
  n = rng.randint(lo, hi)
  return " ".join(rng.choice(_LOREM) for _ in range(n)).strip()


def _synth_example(rng: random.Random, idx: int) -> Dict:
  """
  Produce a single synthetic Manager task with variability to trigger overflow.
  Deterministic for a fixed RNG state.
  """
  goal = f"Compare entities and return the earlier or smaller value (case {idx})."
  # Alternate between date-like and description-like facts
  if idx % 3 == 0:
    f1 = f"Item A: 200{rng.randint(0,9)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
    f2 = f"Item B: 199{rng.randint(0,9)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
  else:  # Longish facts to force caps
    f1 = _rand_words(rng, 12, 28)
    f2 = _rand_words(rng, 6, 18)

  # Optional third fact to vary density
  facts = [f1, f2]
  if idx % 4 == 0:
    facts.append(_rand_words(rng, 10, 20))

  question = "Which is earlier or smaller? Provide only the answer."
  # Compose JSONL (lenient mix to exercise normalizer)
  lines: List[str] = [
    '["r","M"]',
    json.dumps(["g", goal]),
  ]
  for f in facts:
    # Mix array/object forms
    if rng.random() < 0.5:
      lines.append(json.dumps(["f", f]))
    else:
      lines.append(json.dumps({"f": f}))
  # occasional assumptions/plans to change tag mix
  if rng.random() < 0.5:
    lines.append(json.dumps(["u", "Use ISO dates if dates are present."]))
  if rng.random() < 0.4:
    lines.append(json.dumps(["p", "Compare A and B; output one token."]))
  lines.append(json.dumps(["q", "W", question]))
  return {"jsonl": "\n".join(lines)}


def synth_shard(n: int, seed: int) -> List[Dict]:
  rng = random.Random(seed)
  return [_synth_example(rng, i) for i in range(n)]


# --------------------------
# Grid + scoring
# --------------------------

Caps = Dict[str, int]
SummMethod = Literal["extractive", "llmlingua"]
DerefPolicy = Literal["never", "conditional", "always"]  # placeholder for PR-H4


def default_caps_grid() -> List[Caps]:
  return [
    {"f": 20, "p": 15, "q": 20, "g": 30, "u": 20, "t": 50},  # aggressive
    {"f": 30, "p": 20, "q": 30, "g": 30, "u": 20, "t": 50},  # baseline
    {"f": 50, "p": 40, "q": 50, "g": 40, "u": 25, "t": 60},  # relaxed
    {"f": 100, "p": 80, "q": 100, "g": 60, "u": 30, "t": 80},  # very relaxed
  ]


@dataclass(frozen=True)
class CalibSpec:
  caps: Caps
  summarizer: SummMethod
  deref_policy: DerefPolicy
  gate_enabled: bool
  token_budget: int


@dataclass
class CalibMetrics:
  avg_est_tokens: float
  avg_density: float
  avg_overflow_rate: float
  routed_freeform_frac: float
  n: int


@dataclass
class CalibEval:
  spec: CalibSpec
  metrics: CalibMetrics


def evaluate_spec_on_shard(shard: List[Dict], spec: CalibSpec, seed: int) -> CalibEval:
  """
  Evaluate one calibration spec on a synthetic shard.
  Deterministic for fixed (shard, spec, seed).
  """
  # Ensure deterministic behavior of any randomness
  set_global_seed(seed)

  summarizer = Summarizer(method=spec.summarizer)

  densities: List[float] = []
  token_estimates: List[int] = []
  overflow_rates: List[float] = []
  routed_freeform = 0
  gate_cfg = GateCfg(token_budget=spec.token_budget)

  for ex in shard:
    memory = MemoryStore()
    validator = JSONLValidator(caps=spec.caps, memory=memory, summarizer=summarizer)

    # Normalize/overflow with the selected caps/summarizer
    validated_jsonl, stats = validator.validate_and_overflow(ex["jsonl"])

    # Free-form probe
    freeform = validator.jsonl_to_prose(validated_jsonl)

    # Gate decision
    if spec.gate_enabled:
      decision = gate_choose_protocol(validated_jsonl, freeform, gate_cfg)
      route = decision["route"]
      if route == "freeform_llmlingua":
        routed_freeform += 1
        ll2 = decision["est_tokens"].get("ll2")
        token_estimates.append(int(ll2))
      else:
        token_estimates.append(estimate_tokens(validated_jsonl))
    else:
      # No gate: always TerseTalk tokens post-validation
      token_estimates.append(estimate_tokens(validated_jsonl))

    densities.append(float(stats["density"]))
    of_count = stats["overflow"]["count"]
    total_lines = max(1, stats["lines_total"])
    overflow_rates.append(of_count / total_lines)

    memory.reset()

  n = len(shard)
  metrics = CalibMetrics(
    avg_est_tokens=sum(token_estimates) / n if n else 0.0,
    avg_density=sum(densities) / n if n else 0.0,
    avg_overflow_rate=sum(overflow_rates) / n if n else 0.0,
    routed_freeform_frac=(routed_freeform / n) if n else 0.0,
    n=n,
  )
  return CalibEval(spec=spec, metrics=metrics)


def sweep_grid(
  n: int,
  seed: int,
  caps_grid: List[Caps],
  summarizers: List[SummMethod],
  deref_policies: List[DerefPolicy],
  gate_modes: List[bool],
  token_budgets: List[int],
  density_min: float = 0.75,
) -> Dict:
  """
  Run a full sweep over the grid and return a deterministic report dict:
  {
  "n": int, "seed": int, "density_min": float,
  "grid_evaluations": [ { "spec": {...}, "metrics": {...} }, ... ],
  "best": { "spec": {...}, "metrics": {...} }
  }
  """
  shard = synth_shard(n=n, seed=seed)
  evals: List[CalibEval] = []

  # Deterministic iteration order
  for caps in caps_grid:
    for sm in summarizers:
      for dp in deref_policies:
        for gate_on in gate_modes:
          for budget in token_budgets:
            spec = CalibSpec(
              caps=caps,
              summarizer=sm,
              deref_policy=dp,
              gate_enabled=gate_on,
              token_budget=int(budget),
            )
            ev = evaluate_spec_on_shard(shard, spec, seed=seed)
            evals.append(ev)

  # Selection: filter by density_min; pick lowest avg_est_tokens
  def _rank_key(ev: CalibEval) -> Tuple[float, float, float]:
    # Lower tokens better; higher density better; lower freeform frac better (tie-breaker)
    return (ev.metrics.avg_est_tokens, -ev.metrics.avg_density, ev.metrics.routed_freeform_frac)

  feasible = [ev for ev in evals if ev.metrics.avg_density >= density_min]
  if feasible:
    feasible.sort(key=_rank_key)
    chosen = feasible[0]
  else:
    # No candidate meets density; pick by highest density then lowest tokens
    evals.sort(key=lambda ev: (-ev.metrics.avg_density, ev.metrics.avg_est_tokens))
    chosen = evals[0]

  # Deterministic JSON-like report (JSON is valid YAML)
  def _ev_to_dict(ev: CalibEval) -> Dict:
    return {"spec": asdict(ev.spec), "metrics": asdict(ev.metrics)}

  report = {
    "n": n,
    "seed": seed,
    "density_min": density_min,
    "grid_evaluations": [_ev_to_dict(ev) for ev in evals],
    "best": _ev_to_dict(chosen),
  }
  return report


def save_calibration_yaml(report: Dict, out_path: str | Path) -> Path:
  """
  Write the report as JSON (which is valid YAML 1.2) to out_path.
  Returns the Path. Deterministic content (no timestamps).
  """
  p = Path(out_path)
  p.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(report, indent=2, sort_keys=True)
  p.write_text(text, encoding="utf-8")
  return p

