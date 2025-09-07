from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from tersetalk.model_io import ModelClient, EchoModel, dump_jsonl
from tersetalk.structured import TerseTalkLine
from tersetalk.memory import MemoryStore
from tersetalk.protocol_jsonl import JSONLValidator

# Optional protocol handler (PR-H4); import-guarded
try:  # pragma: no cover
  from tersetalk.protocol_handler import PHConfig, ProtocolHandler  # type: ignore
except Exception:  # pragma: no cover
  PHConfig = None  # type: ignore[assignment]
  ProtocolHandler = None  # type: ignore[assignment]


# ---------------------------
# Utilities
# ---------------------------


def _approx_tokens(text: str) -> int:  # Rule-of-thumb: 4 chars ≈ 1 token
  return max(0, (len(text) + 3) // 4)


def _jsonl_from_lines(lines: List[TerseTalkLine]) -> str:
  """Canonical JSONL emitter for typed lines (reuse dump_jsonl)."""
  return dump_jsonl(lines)


def _overflow_count(stats: Dict) -> int:
  if not isinstance(stats, dict):
    return 0
  if "overflow" in stats and isinstance(stats["overflow"], dict):
    c = stats["overflow"].get("count")
    return int(c) if isinstance(c, (int, float)) else 0
  for k in ("overflow_count", "count"):
    if k in stats and isinstance(stats[k], (int, float)):
      return int(stats[k])
  return 0


def _nonempty_jsonl_lines(s: str) -> int:
  return sum(1 for ln in s.splitlines() if ln.strip())


# ---------------------------
# Pipeline configuration
# ---------------------------


@dataclass
class PipelineConfig:
  """Soft caps per tag; handler toggles."""

  caps: Optional[Dict[str, int]] = None  # e.g., {"f":30, ...}
  # Whether to use ProtocolHandler instead of raw JSONLValidator
  use_protocol_handler: bool = True
  # Optional handler flags (mirrors PR-H4)
  preoverflow_ll2: bool = False
  overflow_ll2: bool = False
  deref_ll2: bool = False
  deref_policy: str = "never"  # "never" | "conditional" | "always"

  def ensure_defaults(self) -> None:
    if self.caps is None:
      self.caps = {"f": 30, "p": 20, "q": 30, "g": 30, "u": 20, "t": 50}
    if self.deref_policy not in ("never", "conditional", "always"):
      self.deref_policy = "never"


# ---------------------------
# Message builders & extractors
# ---------------------------


def build_manager_message(example: Dict, caps: Dict[str, int]) -> str:
  """
  Build a TerseTalk JSONL message for the Manager from a normalized example:
  {question, answer, facts, subgoal, assumptions}
  """
  lines: List[List] = []
  lines.append(["r", "M"])
  if example.get("subgoal"):
    lines.append(["g", str(example["subgoal"])])
  for f in example.get("facts", [])[:10]:
    lines.append(["f", str(f)])
  for a in example.get("assumptions", [])[:5]:
    lines.append(["u", str(a)])
  q = example.get("question", "")
  lines.append(["q", "W", str(q)])
  return "\n".join(json.dumps(l, ensure_ascii=False) for l in lines)


def prepare_critic_input(worker_lines: List[TerseTalkLine], question: Optional[str] = None) -> str:
  """
  Prepare a concise critic prompt in JSONL: set role 'C', brief goal, then copy worker lines.
  The critic is expected to emit a ['v','A'|'R'|'E'] verdict among its lines.
  """
  pre: List[List[str]] = [
    ["r", "C"],
    ["g", "Verify worker output; emit verdict tag 'v' = A|R|E."],
  ]
  if question:
    pre.append(["f", f"Task: {question}"])
  wl = [[ln.tag, *ln.payload] for ln in worker_lines]
  return "\n".join(json.dumps(l, ensure_ascii=False) for l in (pre + wl))


def extract_answer(worker_lines: List[TerseTalkLine]) -> str:
  """
  Heuristics: prefer the last 'f' payload, else last 't', else last 'g', else empty.
  """
  ans = ""
  for ln in reversed(worker_lines):
    if ln.tag == "f" and ln.payload:
      return str(ln.payload[0])
    if ln.tag == "t" and ln.payload:
      ans = ans or str(ln.payload[0])
    if ln.tag == "g" and ln.payload:
      ans = ans or str(ln.payload[0])
  return ans


def extract_verdict(critic_lines: List[TerseTalkLine]) -> str:
  """
  Return 'A'|'R'|'E'; default to 'A' if no explicit verdict.
  """
  for ln in reversed(critic_lines):
    if ln.tag == "v" and ln.payload:
      v = str(ln.payload[0]).strip().upper()
      if v in ("A", "R", "E"):
        return v
  return "A"


# ---------------------------
# Core pipeline
# ---------------------------


def run_pipeline_once(
  example: Dict,
  client: ModelClient,
  cfg: PipelineConfig,
  client_worker: Optional[ModelClient] = None,
  client_critic: Optional[ModelClient] = None,
) -> Dict:
  """
  Manager → Worker → Critic, single pass (manager-coordinated).
  - Validates Manager JSONL (caps/overflow)
  - Calls Worker (structured)
  - Prepares Critic input from Worker output and calls Critic (structured)
  - Aggregates tokens, latency, overflow counts, and memory stats
  """
  cfg.ensure_defaults()
  memory = MemoryStore()

  # Build Manager message
  manager_jsonl = build_manager_message(example, cfg.caps or {})

  # Convert manager JSONL to a "semantic preservation" prose reference
  validator_probe = JSONLValidator(caps=cfg.caps or {}, memory=memory)
  try:
    sp_reference = validator_probe.jsonl_to_prose(manager_jsonl)
  except Exception:
    sp_reference = ""

  # Apply protocol handler if enabled and available; else do a raw validate/overflow
  if cfg.use_protocol_handler and ProtocolHandler is not None and PHConfig is not None:  # pragma: no cover
    phcfg = PHConfig(
      caps=cfg.caps or {},
      summarizer_method="extractive",
      preoverflow_ll2=cfg.preoverflow_ll2,
      overflow_ll2=cfg.overflow_ll2,
      deref_ll2=cfg.deref_ll2,
      deref_policy=cfg.deref_policy,  # type: ignore[arg-type]
    )
    handler = ProtocolHandler(phcfg)
    outcome = handler.process(manager_jsonl, memory=memory)
    validated_jsonl = outcome.validated_jsonl
    stats_before = outcome.stats_before or {}
    overflow_cnt = _overflow_count(stats_before)
  else:
    validator = JSONLValidator(caps=cfg.caps or {}, memory=memory)
    validated_jsonl, stats = validator.validate_and_overflow(manager_jsonl)
    overflow_cnt = _overflow_count(stats)

  # Manager → Worker (robust to schema/tool failures)
  t0 = time.perf_counter()
  worker_lines: List[TerseTalkLine] = []
  worker_error: Optional[str] = None
  try:
    c_w = client_worker or client
    worker_lines = c_w.call_jsonl_strict(
      system="You are a Worker. Read TerseTalk-JSONL and respond with valid TerseTalk lines.",
      user_prompt=validated_jsonl,
      max_tokens=256,
    )
  except Exception as e:  # pragma: no cover - exercised via tests with fakes
    # Fallback: request a concise final answer via text, then wrap as a simple TerseTalk line
    try:
      text = (client_worker or client).call_text(
        system="You are a Worker. Read the JSONL and return only the final answer with no extra words.",
        user_prompt=validated_jsonl,
        max_tokens=128,
      )
      if text:
        worker_lines = [TerseTalkLine(tag="f", payload=[text.strip()])]
        worker_error = None  # consider fallback success
      else:
        worker_error = f"{type(e).__name__}: {e}"
    except Exception as e2:
      worker_error = f"{type(e).__name__}: {e}; FallbackError: {type(e2).__name__}: {e2}"
  t1 = time.perf_counter()

  # Worker → Critic
  critic_lines: List[TerseTalkLine] = []
  critic_error: Optional[str] = None
  if worker_error is None and worker_lines:
    critic_input = prepare_critic_input(worker_lines, example.get("question"))
    try:
      c_c = client_critic or client
      critic_lines = c_c.call_jsonl_strict(
        system="You are a Critic. Verify and emit ['v','A'|'R'|'E'] among your lines.",
        user_prompt=critic_input,
        max_tokens=128,
      )
    except Exception as e:  # pragma: no cover - exercised via tests with fakes
      # Critic fallback: request a one-letter verdict; default to 'A' if undecidable
      try:
        txt = (client_critic or client).call_text(
          system="You are a Critic. Read the JSONL and return only one letter: A (accept), R (revise), or E (error). No extra words.",
          user_prompt=critic_input,
          max_tokens=16,
        )
        v = (txt or "").strip().upper()
        v = "A" if not v else ("A" if v.startswith("A") else ("R" if v.startswith("R") else ("E" if v.startswith("E") else "A")))
        critic_lines = [TerseTalkLine(tag="v", payload=[v])]
        critic_error = None
      except Exception as e2:
        critic_error = f"{type(e).__name__}: {e}; FallbackError: {type(e2).__name__}: {e2}"
  t2 = time.perf_counter()

  # Aggregate outputs
  worker_jsonl = _jsonl_from_lines(worker_lines) if worker_lines else ""
  critic_jsonl = _jsonl_from_lines(critic_lines) if critic_lines else ""

  tokens_total = (
    _approx_tokens(validated_jsonl) + _approx_tokens(worker_jsonl) + _approx_tokens(critic_jsonl)
  )
  total_lines = _nonempty_jsonl_lines(validated_jsonl)
  density = 1.0 - (float(overflow_cnt) / max(1, total_lines))

  # Extract answer & verdict
  answer = extract_answer(worker_lines) if worker_lines else ""
  verdict = extract_verdict(critic_lines) if critic_lines else "A"

  # Grab memory stats and reset between tasks (non-leakage)
  mem_stats_before_reset = memory.stats()
  memory.reset()

  status = "ok" if worker_error is None and critic_error is None else "error"
  result = {
    "answer": answer,
    "verdict": verdict,
    "tokens_total": int(tokens_total),
    "overflow_count": int(overflow_cnt),
    "density": float(density),
    "latency_ms": {
      "worker": (t1 - t0) * 1000.0,
      "critic": (t2 - t1) * 1000.0,
      "total": (t2 - t0) * 1000.0,
    },
    "sp_reference": sp_reference,
    "manager_jsonl": validated_jsonl,
    "worker_jsonl": worker_jsonl,
    "critic_jsonl": critic_jsonl,
    "memory_stats": mem_stats_before_reset,
    "status": status,
  }
  if worker_error is not None:
    result["worker_error"] = worker_error
  if critic_error is not None:
    result["critic_error"] = critic_error
  return result


def run_pipeline_batch(examples: List[Dict], client: ModelClient, cfg: PipelineConfig) -> List[Dict]:
  out: List[Dict] = []
  for ex in examples:
    out.append(run_pipeline_once(ex, client, cfg))
  return out
