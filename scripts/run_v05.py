from __future__ import annotations

import csv
import json
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Ensure repository root is importable when invoked as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

import click
from tqdm import tqdm

from tersetalk.reproducibility import set_global_seed
from tersetalk.results_manager import ResultsManager
from tersetalk.metrics import MetricsComputer
from tersetalk.model_io import ModelClient, EchoModel, ModelCfg
from tersetalk.baselines import run_freeform_once, run_llmlingua_once, build_freeform_prompt
from tersetalk.pipeline_runner import run_pipeline_once, PipelineConfig
from tersetalk.datasets import load_hotpotqa, load_gsm8k

from tersetalk.protocol_jsonl import JSONLValidator
from tersetalk.memory import MemoryStore


def _load_examples(task: str, n: int, seed: int) -> List[Dict[str, Any]]:
  if task == "hotpotqa":
    return load_hotpotqa(split="validation", n=n, seed=seed)
  if task == "gsm8k":
    return load_gsm8k(split="test", n=n, seed=seed)
  if task == "synth":
    data: List[Dict[str, Any]] = []
    base = [
      {
        "question": "What city is the Eiffel Tower in?",
        "answer": "Paris",
        "facts": ["Eiffel Tower is a landmark in Paris, France."],
        "subgoal": "Answer the question concisely.",
        "assumptions": ["Use common knowledge", "Return one word"],
      },
      {
        "question": "Which number is larger: 7 or 3?",
        "answer": "7",
        "facts": ["7 > 3"],
        "subgoal": "Compare numbers and answer.",
        "assumptions": ["Use integers", "Be concise"],
      },
      {
        "question": "2 + 2 = ?",
        "answer": "4",
        "facts": ["Basic arithmetic"],
        "subgoal": "Compute a simple sum.",
        "assumptions": ["Return a numeral"],
      },
    ]
    for i in range(min(n, len(base))):
      data.append(base[i])
    return data
  raise click.ClickException(f"Unknown task: {task}")


def _manager_jsonl_from_example(example: Dict[str, Any]) -> str:
  lines: List[List[Any]] = []
  lines.append(["r", "M"])
  if example.get("subgoal"):
    lines.append(["g", str(example["subgoal"])[:512]])
  for f in (example.get("facts") or [])[:10]:
    lines.append(["f", str(f)[:2048]])
  for a in (example.get("assumptions") or [])[:5]:
    lines.append(["u", str(a)[:256]])
  if example.get("question"):
    lines.append(["q", "W", str(example["question"])[:2048]])
  return "\n".join(json.dumps(x, ensure_ascii=False) for x in lines)


def _jsonl_prose_pair(manager_jsonl: str, caps: Dict[str, int]) -> Tuple[str, str, Dict[str, Any], str]:
  memory = MemoryStore()
  validator = JSONLValidator(caps=caps, memory=memory)
  ref_prose = validator.jsonl_to_prose(manager_jsonl)
  validated_jsonl, of_stats = validator.validate_and_overflow(manager_jsonl)
  cand_prose = validator.jsonl_to_prose(validated_jsonl)
  memory.reset()
  return ref_prose, cand_prose, of_stats, validated_jsonl


def _compute_quality(task: str, mc: MetricsComputer, pred: str, gold: str) -> bool:
  if task == "gsm8k":
    return mc.gsm8k_correct(pred, gold)
  return mc.exact_match(pred, gold)


def _save_incremental(raw_path: Path, rows: List[Dict[str, Any]]) -> None:
  if not rows:
    return
  with raw_path.open("a", encoding="utf-8") as f:
    for r in rows:
      f.write(json.dumps(r, ensure_ascii=False) + "\n")
  rows.clear()


def _write_metrics_csv(csv_path: Path, rows: Iterable[Dict[str, Any]]) -> None:
  fieldnames = [
    "idx",
    "task",
    "system",
    "seed",
    "correct",
    "tokens_total",
    "bytes_on_wire",
    "sp_score",
    "overflow_count",
  ]
  with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
      w.writerow({k: r.get(k) for k in fieldnames})


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
  import statistics as stats
  if not rows:
    return {}
  n = len(rows)
  bools = [bool(r.get("correct", False)) for r in rows]
  toks = [int(r.get("tokens_total", 0)) for r in rows]
  bows = [int(r.get("bytes_on_wire", 0)) for r in rows]
  sps = [float(r.get("sp_score")) for r in rows if r.get("sp_score") is not None]
  ofs = [int(r.get("overflow_count", 0)) for r in rows if r.get("overflow_count") is not None]

  def safe_mean(xs):
    return float(stats.mean(xs)) if xs else 0.0

  def safe_median(xs):
    return float(stats.median(xs)) if xs else 0.0

  return {
    "num_examples": n,
    "accuracy": sum(bools) / n,
    "tokens_avg": safe_mean(toks),
    "tokens_median": safe_median(toks),
    "bytes_on_wire_avg": safe_mean(bows),
    "sp_avg": safe_mean(sps),
    "overflow_avg": safe_mean(ofs),
  }


def _load_completed_indices(raw_path: Path) -> set[int]:
  done: set[int] = set()
  if not raw_path.exists():
    return done
  with raw_path.open("r", encoding="utf-8") as f:
    for line in f:
      try:
        obj = json.loads(line)
        idx = int(obj.get("idx"))
        done.add(idx)
      except Exception:
        continue
  return done


from tersetalk import __version__ as _TT_VERSION


@click.command()
@click.version_option(version=_TT_VERSION, message="TerseTalk v0.5 runner %(version)s")
@click.option('--task', type=click.Choice(['hotpotqa', 'gsm8k', 'synth']), required=True)
@click.option('--system', type=click.Choice(['tersetalk', 'freeform', 'llmlingua']), required=True)
@click.option('--n', default=100, show_default=True)
@click.option('--seed', default=0, show_default=True)
@click.option('--caps', default='{"f":30,"p":20,"q":30}', show_default=True)
@click.option('--model', default='mistral', show_default=True)
@click.option('--out', default='results', show_default=True)
@click.option('--hybrid', is_flag=True, default=False)
@click.option('--token-budget', default=600, show_default=True)
@click.option('--deref-policy', type=click.Choice(['always','conditional','never']), default='conditional', show_default=True)
@click.option('--summarizer', type=click.Choice(['extractive','llmlingua','truncate']), default='extractive', show_default=True)
@click.option('--preoverflow-ll2', is_flag=True, default=False)
@click.option('--overflow-ll2', is_flag=True, default=False)
@click.option('--deref-ll2', is_flag=True, default=False)
@click.option('--use-tiktoken', is_flag=True, default=False)
@click.option('--sp', type=click.Choice(['auto','jaccard']), default='auto', show_default=True)
@click.option('--save-every', default=10, show_default=True)
@click.option('--resume', is_flag=True, default=False)
@click.option('--verbose', is_flag=True, default=False)
@click.option('--dry-run', is_flag=True, default=False, help='Print config JSON and exit')
def main(task, system, n, seed, caps, model, out,
         hybrid, token_budget, deref_policy, summarizer,
         preoverflow_ll2, overflow_ll2, deref_ll2,
         use_tiktoken, sp, save_every, resume, verbose, dry_run):

  model_cfg = set_global_seed(seed)
  results_mgr = ResultsManager(out)
  experiment_id = f"{task}_{system}"
  run_dir = results_mgr.get_run_dir(experiment_id, timestamp=True)

  try:
    caps_dict = json.loads(caps)
    assert isinstance(caps_dict, dict)
  except Exception:
    raise click.ClickException("--caps must be valid JSON mapping")

  config = {
    "task": task,
    "system": system,
    "n": n,
    "seed": seed,
    "caps": caps_dict,
    "model": model,
    "hybrid": hybrid,
    "token_budget": int(token_budget),
    "deref_policy": deref_policy,
    "summarizer": summarizer,
    "preoverflow_ll2": preoverflow_ll2,
    "overflow_ll2": overflow_ll2,
    "deref_ll2": deref_ll2,
    "use_tiktoken": use_tiktoken,
    "sp_mode": sp,
    **model_cfg,
  }
  if dry_run:
    # For CLI scaffold tests: echo minimal config and defaults
    defaults = set_global_seed(seed)
    print(json.dumps({
      "task": task,
      "system": system,
      "n": n,
      "seed": seed,
      "caps": json.loads(caps),
      "defaults": defaults,
    }, indent=2))
    return

  results_mgr.save_config(run_dir, config)

  examples = _load_examples(task, n, seed)

  if model.strip().lower() == "echo":
    client = EchoModel()
  else:
    cfg = ModelCfg(model=model)
    client = ModelClient(cfg)

  mc = MetricsComputer(use_tiktoken=use_tiktoken)

  raw_path = run_dir / "raw_outputs.jsonl"
  csv_path = run_dir / "metrics.csv"
  sum_path = run_dir / "summary.json"

  completed = _load_completed_indices(raw_path) if resume else set()

  pending_rows: List[Dict[str, Any]] = []
  all_rows: List[Dict[str, Any]] = []

  def flush():
    _save_incremental(raw_path, pending_rows)
    _write_metrics_csv(csv_path, all_rows)
    if verbose:
      click.echo(f"[flush] saved {raw_path.name}, {csv_path.name}")

  def handle_sigint(signum, frame):
    click.echo("\n[run_v05] Caught SIGINT, flushing buffers...")
    flush()
    sys.exit(130)

  signal.signal(signal.SIGINT, handle_sigint)

  for idx, ex in enumerate(tqdm(examples, desc=f"Running {system} on {task}", unit="ex")):
    if idx in completed:
      continue

    question = str(ex.get("question", ""))
    gold = str(ex.get("answer", ""))

    route_str = "na"
    overflow_count = None
    bytes_wire = 0
    sp_score: Optional[float] = None
    pred_answer = ""
    tokens_total = 0
    aux: Dict[str, Any] = {}

    if system == "tersetalk":
      pipe_cfg = PipelineConfig(
        caps=caps_dict,
        use_protocol_handler=False,
        preoverflow_ll2=preoverflow_ll2,
        overflow_ll2=overflow_ll2,
        deref_ll2=deref_ll2,
        deref_policy=deref_policy,
      )
      result = run_pipeline_once(ex, client, pipe_cfg)
      pred_answer = str(result.get("answer", ""))
      tokens_total = int(result.get("tokens_total", 0))
      overflow_count = int(result.get("overflow_count", 0))
      route_str = "tersetalk"

      manager_jsonl = _manager_jsonl_from_example(ex)
      ref_prose, cand_prose, of_stats, validated_jsonl = _jsonl_prose_pair(manager_jsonl, caps_dict)
      bytes_wire = mc.bytes_on_wire(validated_jsonl)
      sp_score = mc.jaccard_sp(ref_prose, cand_prose) if sp == "jaccard" else mc.bertscore_sp(ref_prose, cand_prose)
      aux.update({
        "overflow_rate_est": of_stats.get("rate", None),
        "latency_ms": result.get("latency_ms", None),
        "status": result.get("status", "ok"),
      })
      if result.get("worker_error"):
        aux["worker_error"] = result.get("worker_error")
      if result.get("critic_error"):
        aux["critic_error"] = result.get("critic_error")

    elif system == "freeform":
      b = run_freeform_once(ex, client)
      pred_answer = str(b.get("answer", ""))
      tokens_total = int(b.get("tokens_total", b.get("prompt_tokens", 0) + b.get("response_tokens", 0)))
      route_str = "freeform"
      origin_prompt = b.get("origin_prompt") or b.get("prompt") or ""
      bytes_wire = mc.bytes_on_wire(origin_prompt)
      manager_jsonl = _manager_jsonl_from_example(ex)
      ref_prose, _, _, _ = _jsonl_prose_pair(manager_jsonl, caps_dict)
      cand_prose = origin_prompt
      sp_score = mc.jaccard_sp(ref_prose, cand_prose) if sp == "jaccard" else mc.bertscore_sp(ref_prose, cand_prose)

    else:  # llmlingua
      b = run_llmlingua_once(ex, client)
      pred_answer = str(b.get("answer", ""))
      tokens_total = int(b.get("tokens_total", b.get("prompt_tokens", 0) + b.get("response_tokens", 0)))
      route_str = "freeform_llmlingua"
      origin_prompt = b.get("origin_prompt", "")
      compressed_prompt = b.get("compressed_prompt", "")
      bytes_wire = mc.bytes_on_wire(compressed_prompt or origin_prompt)
      sp_score = mc.jaccard_sp(origin_prompt, compressed_prompt or origin_prompt) if sp == "jaccard" else mc.bertscore_sp(origin_prompt, compressed_prompt or origin_prompt)
      aux.update({"compression_ratio": b.get("compression_ratio")})

    correct = _compute_quality(task, mc, pred_answer, gold)
    row = {
      "idx": idx,
      "task": task,
      "system": system,
      "seed": seed,
      "question": question,
      "gold_answer": gold,
      "pred_answer": pred_answer,
      "correct": bool(correct),
      "tokens_total": tokens_total,
      "bytes_on_wire": bytes_wire,
      "sp_score": sp_score,
      "overflow_count": overflow_count,
      "latency_ms": aux.get("latency_ms"),
      "route": route_str,
      "aux": {k: v for k, v in aux.items() if k not in ("latency_ms",)},
    }
    pending_rows.append(row)
    all_rows.append(row)

    if (idx + 1) % int(save_every) == 0:
      flush()

  flush()
  summary = _summarize(all_rows)
  (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
  click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
