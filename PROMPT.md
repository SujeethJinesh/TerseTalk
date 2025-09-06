Read through your AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the AGENTS.md. To ask claude code for a review, you can do something like `echo "Please ensure you understand @RESEARCH_PROPOSAL.md. I am implementing PR-X. Please review the following files I created: @README.md, @xxx. Ensure that my implementation is properly aligned with the goals of the project. Also here is some additional data on the runs I gathered, please critique it as if you're a senior data scientist and ensure that I'm not cheating on the results, lying, or misrepresenting things" | claude -p --dangerously-skip-permissions --model opus`. Please ensure you are indeed calling it like `claude -p --dangerously-skip-permissions --model opus` and ensure you get both the code review, and the data review, and an additional PI review about the state of the project with Claude and yourself. You must be truthful and honest, and address all the points Claude makes (though keep away from making fakes or mocks). If you need to create fakes or mocks for debugging, delete them afterwards so as to not confuse fakes and mocks for actual results. You must only present results from real model runs as much as possible for our project. It's imperative. Ensure that you are going back and incorporating feedback from claude and yourself as necessary. You must continue this loop until both you and claude agree with each other and give yourselfs both approval without nits. After that you should push your changes up for review. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update AGENTS.md and ask CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly. Before you merge the PR, you must wait for my approval. Also at the end I want you to outline any risks we are seeing in this project, are our expectations aligned with how the progress is going? Will our project succeed and achieve the baselines we expect using your best judgement? Absolutely make sure you're reporting results truthfully and honestly. Avoid fake, mocked, or other non genuine results. You should also analyze the results we get for each run and determine if they meet our figures of merit, when you report back, it's crucial to include that analysis (e.g. compression amount, failure rate, latency, etc.). We should be aiming to properly fix things and run proper evaluations. If we do have any expected goals or outcomes (e.g. >= 10x on xyz) and they aren't achieved, then explain why, but do not lie or cheat and use drastically contrived inefficient metrics. It's important to generally be comparing to a standard implementation. Please also report the results on real data runs, it's important that we start gathering as much data on real runs as possible now so it can be reviewed by Claude AND YOU. If you see there are problems or optimizations, don't hesitate to suggest them as we collect more data. Here are more detailed instructions for PR implementation.

### PR Summary

You will implement 2 PRs in one.

PR‑08B — Baselines knobs & minor nits

- Parameterize max_tokens in run_freeform_once/run_llmlingua_once; surface via smoke CLI.
- Add brief comment in build_freeform_prompt explaining lone "Question:" line retention.
- Tests: determinism unchanged with default params; schema intact.

PR‑11 — Experiment Driver

Role: You are a senior engineer implementing the Experiment Driver for TerseTalk v0.5. Your task is to create a robust CLI script and a smoke test that align with the proposal’s PR‑11 plan, integrating previously merged modules.

Objectives

Add a single entrypoint script scripts/run_v05.py that:

Loads a task split (hotpotqa, gsm8k, and a lightweight synth for CI).

Runs one of {tersetalk | freeform | llmlingua} systems.

Computes per‑example and aggregate metrics (accuracy, tokens, SP, bytes‑on‑wire, overflow counts).

Saves results in a versioned run directory via ResultsManager (PR‑09):

config.json (exact run configuration)

raw_outputs.jsonl (one JSON object per example)

metrics.csv (per‑example)

summary.json (aggregates)

Supports resume, incremental writes, and is SIGINT‑safe (flushes buffers on Ctrl‑C).

Slightly extend baselines to expose the free‑form prompt(s) so we can compute SP against compression:

run_llmlingua_once should include origin_prompt and compressed_prompt in its return payload (keep existing keys too).

If this change is not yet present from PR‑08, add it now.

Add a smoke test: tests/test_run_v05_smoke.py

Runs scripts/run_v05.py with --task synth --system tersetalk --n 3 --model echo into a temp output dir.

Asserts that config.json, raw_outputs.jsonl, and summary.json exist and are non‑empty.

Does not require internet or GPUs.

Requirements & Interfaces (must use)

Repro: from tersetalk.reproducibility import set_global_seed

Datasets: from tersetalk.datasets import load_hotpotqa, load_gsm8k

Pipeline: from tersetalk.pipeline_runner import run_pipeline_once

Baselines: from tersetalk.baselines import run_freeform_once, run_llmlingua_once

Results: from tersetalk.results_manager import ResultsManager

Metrics: from tersetalk.metrics import MetricsComputer

(Optional for SP on manager messages) from tersetalk.protocol_jsonl import JSONLValidator and from tersetalk.memory import MemoryStore

Model clients: from tersetalk.model_io import ModelClient, EchoModel

CLI surface (match proposal + add synth + a few practical flags)
python scripts/run_v05.py \
  --task {hotpotqa,gsm8k,synth} \
  --system {tersetalk,freeform,llmlingua} \
  --n 100 --seed 0 \
  --caps '{"f":30,"p":20,"q":30}' \
  --model mistral \
  --out results \
  [--hybrid] [--token-budget 600] \
  [--deref-policy {always,conditional,never}] \
  [--summarizer {extractive,llmlingua,truncate}] \
  [--preoverflow-ll2] [--overflow-ll2] [--deref-ll2] \
  [--use-tiktoken] [--sp {auto,jaccard}] \
  [--save-every 10] [--resume] [--verbose]


Notes:

--model echo must route to EchoModel() for GPU‑free CI.

--sp auto tries BERTScore then falls back; --sp jaccard forces fallback.

--resume reuses existing raw_outputs.jsonl to skip already completed indices.

Design details

Per‑example record (JSON line in raw_outputs.jsonl), minimally:

{
  "idx": 17,
  "task": "hotpotqa",
  "system": "tersetalk",
  "seed": 0,
  "question": "...",
  "gold_answer": "...",
  "pred_answer": "...",
  "correct": true,
  "tokens_total": 413,
  "bytes_on_wire": 912,
  "sp_score": 0.87,
  "overflow_count": 2,
  "latency_ms": {"manager":0,"worker":...,"critic":...},
  "route": "tersetalk|freeform_llmlingua|na",
  "aux": { "compression_ratio": 0.63, "notes": "optional small bag" }
}


Accuracy:

hotpotqa → MetricsComputer.exact_match

gsm8k → MetricsComputer.gsm8k_correct

synth → treat like EM

SP (semantic preservation): score the Manager→Worker transmission vs a reference “golden prose” of the manager’s intent.

Build reference prose via JSONLValidator.jsonl_to_prose(manager_jsonl) using the un‑overflowed content (i.e., start from the unvalidated manager JSONL string you assemble from the example; see helper below).

Candidate for systems:

tersetalk: jsonl_to_prose(validated_jsonl) (after caps/overflow)

freeform: the free‑form prompt itself (SP ≈ 1 vs reference prose built from the same content)

llmlingua: compressed_prompt vs origin_prompt prose (use the origin prompt as the reference)

Bytes‑on‑wire: use MetricsComputer.bytes_on_wire(<transmitted string>) for the exact string given to the next role (validated JSONL, free‑form prompt, or compressed prompt).

Hybrid flags: pass through into the config dict you hand to run_pipeline_once so future PR‑H4 wiring can use them (no special handling needed in the driver beyond config plumbing).

Files to add / update
A) scripts/run_v05.py (new)

Create the file with the following full content:

from __future__ import annotations

import csv
import json
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click
from tqdm import tqdm

from tersetalk.reproducibility import set_global_seed
from tersetalk.results_manager import ResultsManager
from tersetalk.metrics import MetricsComputer
from tersetalk.model_io import ModelClient, EchoModel
from tersetalk.baselines import run_freeform_once, run_llmlingua_once
from tersetalk.pipeline_runner import run_pipeline_once
from tersetalk.datasets import load_hotpotqa, load_gsm8k

# Optional (for SP ref/candidate prose of JSONL)
from tersetalk.protocol_jsonl import JSONLValidator
from tersetalk.memory import MemoryStore

# ----------------------------
# Helpers for dataset handling
# ----------------------------

def _load_examples(task: str, n: int, seed: int) -> List[Dict[str, Any]]:
    if task == "hotpotqa":
        return load_hotpotqa(split="validation", n=n, seed=seed)
    if task == "gsm8k":
        return load_gsm8k(split="test", n=n, seed=seed)
    if task == "synth":
        # Offline, tiny CI-friendly set
        # Fields mimick our normalized example shape
        data = []
        base = [
            {
                "question": "What city is the Eiffel Tower in?",
                "answer": "Paris",
                "facts": ["Eiffel Tower is a landmark in Paris, France."],
                "subgoal": "Answer the question concisely.",
                "assumptions": ["Use common knowledge", "Return one word"]
            },
            {
                "question": "Which number is larger: 7 or 3?",
                "answer": "7",
                "facts": ["7 > 3"],
                "subgoal": "Compare numbers and answer.",
                "assumptions": ["Use integers", "Be concise"]
            },
            {
                "question": "2 + 2 = ?",
                "answer": "4",
                "facts": ["Basic arithmetic"],
                "subgoal": "Compute a simple sum.",
                "assumptions": ["Return a numeral"]
            },
        ]
        # Ensure deterministic slice of N
        for i in range(min(n, len(base))):
            data.append(base[i])
        return data
    raise click.ClickException(f"Unknown task: {task}")


def _manager_jsonl_from_example(example: Dict[str, Any]) -> str:
    """
    Construct Manager→Worker message in TerseTalk JSONL style for SP reference.
    Keep simple & consistent with earlier PRs (role 'M', subgoal, facts, question).
    """
    import json as _json

    lines = []
    lines.append(["r", "M"])
    if example.get("subgoal"):
        lines.append(["g", str(example["subgoal"])[:512]])
    for f in example.get("facts", [])[:10]:
        lines.append(["f", str(f)[:2048]])
    if example.get("assumptions"):
        for a in example["assumptions"][:5]:
            lines.append(["u", str(a)[:256]])
    if example.get("question"):
        lines.append(["q", "W", str(example["question"])[:2048]])

    return "\n".join(_json.dumps(x, ensure_ascii=False) for x in lines)


def _jsonl_prose_pair(manager_jsonl: str, caps: Dict[str, int]) -> Tuple[str, str, Dict[str, Any], str]:
    """
    Returns (reference_prose, candidate_prose, overflow_stats, validated_jsonl)
    - reference_prose: prose from original (pre-validation) JSONL
    - candidate_prose: prose after caps/overflow
    """
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
    # hotpotqa and synth: EM normalization
    return mc.exact_match(pred, gold)


def _save_incremental(raw_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with raw_path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows.clear()


def _write_metrics_csv(csv_path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    # Write/overwrite a small csv for convenience; minimal columns
    fieldnames = [
        "idx", "task", "system", "seed",
        "correct", "tokens_total", "bytes_on_wire",
        "sp_score", "overflow_count"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    import math
    import statistics as stats

    if not rows:
        return {}
    n = len(rows)
    bools = [bool(r.get("correct", False)) for r in rows]
    toks = [int(r.get("tokens_total", 0)) for r in rows]
    bows = [int(r.get("bytes_on_wire", 0)) for r in rows]
    sps  = [float(r.get("sp_score")) for r in rows if r.get("sp_score") is not None]
    ofs  = [int(r.get("overflow_count", 0)) for r in rows if r.get("overflow_count") is not None]

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
    done = set()
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


# ----------------------------
# Main CLI
# ----------------------------

@click.command()
@click.option('--task', type=click.Choice(['hotpotqa', 'gsm8k', 'synth']), required=True)
@click.option('--system', type=click.Choice(['tersetalk', 'freeform', 'llmlingua']), required=True)
@click.option('--n', default=100, show_default=True, help='Number of examples')
@click.option('--seed', default=0, show_default=True, help='Random seed')
@click.option('--caps', default='{"f":30,"p":20,"q":30}', show_default=True, help='Soft caps JSON for TerseTalk')
@click.option('--model', default='mistral', show_default=True, help='Model name (use "echo" for CI)')
@click.option('--out', default='results', show_default=True, help='Output base directory')

# Hybrid / LLMLingua toggles (plumbed into config)
@click.option('--hybrid', is_flag=True, default=False, help='Enable per-turn hybrid gate')
@click.option('--token-budget', default=600, show_default=True, help='Token budget for hybrid/LL2')
@click.option('--deref-policy', type=click.Choice(['always','conditional','never']), default='conditional', show_default=True)
@click.option('--summarizer', type=click.Choice(['extractive','llmlingua','truncate']), default='extractive', show_default=True)
@click.option('--preoverflow-ll2', is_flag=True, default=False, help='Use LLMLingua before overflow')
@click.option('--overflow-ll2', is_flag=True, default=False, help='Use LLMLingua to produce overflow summary')
@click.option('--deref-ll2', is_flag=True, default=False, help='Use LLMLingua on dereference contents')

# Metrics/runtime options
@click.option('--use-tiktoken', is_flag=True, default=False, help='Use exact token counting when available')
@click.option('--sp', type=click.Choice(['auto','jaccard']), default='auto', show_default=True, help='Semantic preservation scoring')
@click.option('--save-every', default=10, show_default=True, help='Flush raw outputs every K examples')
@click.option('--resume', is_flag=True, default=False, help='Resume based on existing raw_outputs.jsonl')
@click.option('--verbose', is_flag=True, default=False, help='Verbose logging to stdout')
def main(task, system, n, seed, caps, model, out,
         hybrid, token_budget, deref_policy, summarizer,
         preoverflow_ll2, overflow_ll2, deref_ll2,
         use_tiktoken, sp, save_every, resume, verbose):

    # Reproducibility config
    model_cfg = set_global_seed(seed)

    # Resolve run directory
    results_mgr = ResultsManager(out)
    experiment_id = f"{task}_{system}"
    run_dir = results_mgr.get_run_dir(experiment_id, timestamp=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Compose config
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
    results_mgr.save_config(run_dir, config)

    # Load dataset
    examples = _load_examples(task, n, seed)

    # Create/choose model client
    if model.strip().lower() == "echo":
        client = EchoModel()
    else:
        client = ModelClient()
        client.init(model_name=model)

    # Initialize metrics computer
    mc = MetricsComputer(use_tiktoken=use_tiktoken)

    # Output file paths
    raw_path = run_dir / "raw_outputs.jsonl"
    csv_path = run_dir / "metrics.csv"
    sum_path = run_dir / "summary.json"

    # Resume support
    completed = _load_completed_indices(raw_path) if resume else set()

    # SIGINT-safe flushing
    pending_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    def flush():
        _save_incremental(raw_path, pending_rows)
        # Refresh aggregate CSV on every flush (lightweight)
        _write_metrics_csv(csv_path, all_rows)
        if verbose:
            click.echo(f"[flush] saved {raw_path.name}, {csv_path.name}")

    def handle_sigint(signum, frame):
        click.echo("\n[run_v05] Caught SIGINT, flushing buffers...")
        flush()
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    # Main loop
    for idx, ex in enumerate(tqdm(examples, desc=f"Running {system} on {task}", unit="ex")):
        if idx in completed:
            continue

        question = str(ex.get("question", ""))
        gold = str(ex.get("answer", ""))

        # Per-system execution + content for SP/bytes accounting
        route_str = "na"
        overflow_count = None
        bytes_wire = 0
        sp_score: Optional[float] = None
        pred_answer = ""
        tokens_total = 0
        aux: Dict[str, Any] = {}

        if system == "tersetalk":
            # Run the coordinated pipeline
            result = run_pipeline_once(ex, client, config)
            pred_answer = str(result.get("answer", ""))
            tokens_total = int(result.get("tokens_total", 0))
            overflow_count = int(result.get("overflow_count", 0))
            route_str = "tersetalk"

            # For SP/bytes: reconstruct manager JSONL and validated JSONL
            manager_jsonl = _manager_jsonl_from_example(ex)
            ref_prose, cand_prose, of_stats, validated_jsonl = _jsonl_prose_pair(manager_jsonl, caps_dict)
            bytes_wire = mc.bytes_on_wire(validated_jsonl)

            # Semantic preservation
            if sp == "jaccard":
                sp_score = mc.jaccard_sp(ref_prose, cand_prose)
            else:
                sp_score = mc.bertscore_sp(ref_prose, cand_prose)

            aux.update({
                "overflow_rate_est": of_stats.get("rate", None),
                "memory_stats_end": result.get("memory_stats", None),
                "verdict": result.get("verdict", None),
            })

        elif system == "freeform":
            b = run_freeform_once(ex, client)
            pred_answer = str(b.get("answer", ""))
            tokens_total = int(b.get("tokens", 0))
            route_str = "freeform"

            origin_prompt = b.get("origin_prompt") or b.get("prompt") or ""
            bytes_wire = mc.bytes_on_wire(origin_prompt)

            # SP: freeform prompt vs reference prose ≈ 1 since it's the uncompressed prose itself
            manager_jsonl = _manager_jsonl_from_example(ex)
            ref_prose, _, _, _ = _jsonl_prose_pair(manager_jsonl, caps_dict)
            cand_prose = origin_prompt
            if sp == "jaccard":
                sp_score = mc.jaccard_sp(ref_prose, cand_prose)
            else:
                sp_score = mc.bertscore_sp(ref_prose, cand_prose)

        else:  # llmlingua
            b = run_llmlingua_once(ex, client)  # must return origin/compressed prompts
            pred_answer = str(b.get("answer", ""))
            tokens_total = int(b.get("tokens", 0))
            route_str = "freeform_llmlingua"

            origin_prompt = b.get("origin_prompt", "")
            compressed_prompt = b.get("compressed_prompt", "")
            bytes_wire = mc.bytes_on_wire(compressed_prompt or origin_prompt)

            # SP: compressed vs origin prompt
            if sp == "jaccard":
                sp_score = mc.jaccard_sp(origin_prompt, compressed_prompt or origin_prompt)
            else:
                sp_score = mc.bertscore_sp(origin_prompt, compressed_prompt or origin_prompt)

            aux.update({
                "compression_ratio": b.get("compression_ratio"),
            })

        # Quality
        correct = _compute_quality(task, mc, pred_answer, gold)

        # Assemble record
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
            "latency_ms": None,   # available in pipeline result if needed
            "route": route_str,
            "aux": aux or None,
        }
        pending_rows.append(row)
        all_rows.append(row)

        # periodic flush
        if (idx + 1) % int(save_every) == 0:
            flush()

    # Final flush and summary
    flush()
    summary = _summarize(all_rows)
    sum_path.write_text(json.dumps(summary, indent=2))
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

B) Patch tersetalk/baselines.py (extend LLMLingua return payload)

If these keys are not already returned from PR‑08, add them now (non‑breaking):

# In tersetalk/baselines.py, inside run_llmlingua_once(...)
def run_llmlingua_once(example: dict, client) -> dict:
    from llmlingua import PromptCompressor
    compressor = PromptCompressor()

    origin_prompt = build_freeform_prompt(example)  # ensure this helper exists; if not, add it
    compressed = compressor.compress(origin_prompt, target_token=100)

    compressed_prompt = compressed.get('compressed_prompt', '')
    response = client.call(compressed_prompt)

    return {
        "answer": response,
        "tokens": compressed.get('origin_tokens', 0),   # keep original fields
        "compression_ratio": compressed.get('ratio', None),
        # new fields for PR-11 SP/bytes:
        "origin_prompt": origin_prompt,
        "compressed_prompt": compressed_prompt,
    }


If build_freeform_prompt(example) is missing, also add a simple implementation in the same module:

def build_freeform_prompt(example: dict) -> str:
    parts = [
        "Role: Manager",
        f"Goal: {example.get('subgoal','')}",
        f"Facts: {'; '.join(map(str, example.get('facts', [])))}",
        f"Question: {example.get('question','')}",
    ]
    return "\n".join(parts)

C) Test: tests/test_run_v05_smoke.py (new)
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

def test_run_v05_synth_echo(tmp_path: Path):
    script = Path("scripts") / "run_v05.py"
    assert script.exists(), "run_v05.py not found"

    outdir = tmp_path / "results"
    cmd = [
        sys.executable, str(script),
        "--task", "synth",
        "--system", "tersetalk",
        "--n", "3",
        "--seed", "0",
        "--model", "echo",
        "--out", str(outdir),
        "--save-every", "1",
        "--sp", "jaccard",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr

    # Find the single timestamped run directory
    runs = list((outdir / "synth_tersetalk").glob("*"))
    assert runs, "No run directory created"
    # 'latest' symlink plus timestamp dir(s)
    rundir = [p for p in runs if p.is_dir() and p.name != "latest"][0]

    raw = rundir / "raw_outputs.jsonl"
    cfg = rundir / "config.json"
    summ = rundir / "summary.json"

    assert cfg.exists() and cfg.read_text().strip(), "config missing/empty"
    assert raw.exists() and raw.read_text().strip(), "raw_outputs.jsonl missing/empty"
    assert summ.exists() and summ.read_text().strip(), "summary.json missing/empty"

    # Summary should be valid JSON and contain num_examples
    s = json.loads(summ.read_text())
    assert "num_examples" in s and s["num_examples"] == 3

Commit message
PR-11: Experiment Driver (scripts/run_v05.py) + LLMLingua payloads + smoke test

- Add scripts/run_v05.py:
  * CLI to run {tersetalk|freeform|llmlingua} on {hotpotqa|gsm8k|synth}
  * Reproducibility via set_global_seed
  * ResultsManager integration: config.json, raw_outputs.jsonl, metrics.csv, summary.json
  * Per-example metrics: accuracy (EM/GSM8K), tokens, bytes-on-wire, SP (BERTScore→Jaccard), overflow counts
  * Incremental saves, resume support, SIGINT-safe flush
  * Hybrid/LL2 flags plumbed into config for future PR-H4
- Extend baselines.run_llmlingua_once to return origin/compressed prompts (for SP + bytes)
- Add tests/test_run_v05_smoke.py: EchoModel + synth task offline smoke

Notes & rationale

Synth task keeps CI offline and fast while exercising the same code paths.

SP scoring follows the proposal’s spirit: measure semantic drift introduced by protocol caps/overflow vs a reference description. For LLMLingua, we explicitly compare compressed vs origin prompt.

Bytes‑on‑wire and token counts are both reported to support systems metrics and Pareto plots later (PR‑12).

Hybrid flags are collected but not used here (they will be consumed by PR‑H4 wiring).
