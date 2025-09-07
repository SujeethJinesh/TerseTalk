from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

import click
from tqdm import tqdm

# Ensure repository root on path when invoked as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tersetalk.datasets import load_hotpotqa, load_gsm8k
from tersetalk.pipeline_runner import (
    run_pipeline_once,
    PipelineConfig,
    build_manager_message,
)
from tersetalk.baselines import run_freeform_once, run_llmlingua_once, build_freeform_prompt
from tersetalk.model_io import ModelClient, EchoModel
from tersetalk.metrics import MetricsComputer
from tersetalk.reproducibility import set_global_seed
from tersetalk.results_manager import ResultsManager
from tersetalk.hybrid_gate import GateCfg, gate_choose_protocol


CAPS_GRID: List[Dict[str, int]] = [
    {"f": 20, "p": 15, "q": 20},
    {"f": 30, "p": 20, "q": 30},
    {"f": 50, "p": 40, "q": 50},
    {"f": 100, "p": 80, "q": 100},
]
HYBRID_BUDGETS = [400, 600, 800]


def _summarize(rows: List[Dict]) -> Dict:
    ok = [r for r in rows if str(r.get("status", "success")) != "error"]
    n_all = len(rows)
    if not ok:
        return {"n_total": n_all, "n_successful": 0, "avg_tokens": 0.0, "accuracy": 0.0, "compliance_rate": 0.0}
    avg_tok = sum(float(r.get("tokens") or r.get("tokens_total") or 0) for r in ok) / len(ok)
    acc = sum(1 for r in ok if bool(r.get("correct"))) / len(ok)
    return {
        "n_total": n_all,
        "n_successful": len(ok),
        "avg_tokens": float(avg_tok),
        "accuracy": float(acc),
        "compliance_rate": float(len(ok) / n_all),
    }


def _save_jsonl(run_dir: Path, name: str, rows: List[Dict]) -> None:
    p = run_dir / f"{name}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@click.command()
@click.option('--task', type=click.Choice(['hotpotqa', 'gsm8k']), required=True)
@click.option('--systems', multiple=True, type=click.Choice(['tersetalk','freeform','llmlingua','hybrid']), default=['tersetalk','freeform','llmlingua'])
@click.option('--n', default=100, show_default=True)
@click.option('--seed', default=42, show_default=True)
@click.option('--caps-grid/--no-caps-grid', default=True, show_default=True)
@click.option('--model', default='echo', show_default=True)
@click.option('--out', default='results/evaluation', show_default=True)
@click.option('--worker-model', type=str, default=None, help='Override model for Worker role')
@click.option('--critic-model', type=str, default=None, help='Override model for Critic role')
@click.option('--temperature', type=float, default=0.2, show_default=True)
@click.option('--dry-run', is_flag=True, help='Use n=10 and echo model')
def main(task, systems, n, seed, caps_grid, model, out, worker_model, critic_model, temperature, dry_run):
    """Run v0.5 evaluation across systems; save JSONL and summary.json (offline-safe)."""
    set_global_seed(seed)
    if dry_run:
        n, model = 10, 'echo'

    rm = ResultsManager(out)
    run_dir = rm.get_run_dir(task, timestamp=True)
    rm.save_config(run_dir, {"task": task, "systems": systems, "n": n, "seed": seed, "caps_grid": bool(caps_grid), "model": model})

    # Data
    ds = load_hotpotqa(n=n, seed=seed) if task == 'hotpotqa' else load_gsm8k(n=n, seed=seed)

    # Model
    client = EchoModel() if (model or '').lower() == 'echo' else ModelClient()
    # Optional per-role clients for Worker/Critic
    worker_client = None
    critic_client = None
    if worker_model and (worker_model.strip().lower() != 'echo'):
        from tersetalk.model_io import ModelCfg
        worker_client = ModelClient(ModelCfg(model=worker_model))
    if critic_model and (critic_model.strip().lower() != 'echo'):
        from tersetalk.model_io import ModelCfg
        critic_client = ModelClient(ModelCfg(model=critic_model))
    mc = MetricsComputer()

    def evaluate_answer(ex: Dict, ans: str) -> bool:
        gold = str(ex.get('answer', ''))
        try:
            return mc.exact_match(ans, gold) if task == 'hotpotqa' else mc.gsm8k_correct(ans, gold)
        except Exception:
            return False

    def run_tersetalk(examples: List[Dict], caps: Dict[str, int]) -> List[Dict]:
        cfg = PipelineConfig(caps=caps, use_protocol_handler=False, deref_policy='never')
        rows: List[Dict] = []
        for ex in tqdm(examples, desc=f"tersetalk caps={caps}"):
            r = run_pipeline_once(ex, client, cfg, client_worker=worker_client, client_critic=critic_client)
            ans = str(r.get('answer', ''))
            r['correct'] = bool(evaluate_answer(ex, ans))
            r['tokens'] = int(r.get('tokens_total', 0))
            r['status'] = r.get('status', 'ok')
            rows.append(r)
        return rows

    def run_hybrid(examples: List[Dict], caps: Dict[str, int], budget: int) -> List[Dict]:
        rows: List[Dict] = []
        gate = GateCfg(token_budget=int(budget))
        for ex in tqdm(examples, desc=f"hybrid budget={budget}"):
            manager = build_manager_message(ex, caps)
            prompt = build_freeform_prompt(ex)
            route = gate_choose_protocol(manager, prompt, gate)
            if route.get('route') == 'freeform_llmlingua':
                r = run_llmlingua_once(ex, client)
                tokens = int(r.get('tokens_total') or r.get('tokens') or 0)
            else:
                r = run_pipeline_once(ex, client, PipelineConfig(caps=caps, use_protocol_handler=False), client_worker=worker_client, client_critic=critic_client)
                tokens = int(r.get('tokens_total', 0))
            ans = str(r.get('answer', ''))
            r['correct'] = bool(evaluate_answer(ex, ans))
            r['tokens'] = tokens
            r['status'] = r.get('status', 'success')
            rows.append(r)
        return rows

    def run_baseline(tag: str, examples: List[Dict]) -> List[Dict]:
        rows: List[Dict] = []
        for ex in tqdm(examples, desc=tag):
            if tag == 'freeform':
                r = run_freeform_once(ex, client, temperature=temperature)
                tokens = int(r.get('tokens_total') or r.get('tokens') or 0)
            else:
                r = run_llmlingua_once(ex, client, temperature=temperature)
                tokens = int(r.get('tokens_total') or r.get('tokens') or 0)
            ans = str(r.get('answer', ''))
            r['correct'] = bool(evaluate_answer(ex, ans))
            r['tokens'] = tokens
            r['status'] = r.get('status', 'success')
            rows.append(r)
        return rows

    summary: Dict[str, Dict] = {}

    # TerseTalk grid
    if 'tersetalk' in systems:
        grid = CAPS_GRID if caps_grid else [{"f": 30, "p": 20, "q": 30}]
        for caps in grid:
            name = f"tersetalk_f{caps['f']}_p{caps['p']}_q{caps['q']}"
            rows = run_tersetalk(ds, caps)
            out_path = run_dir / f"{name}.jsonl"
            _save_jsonl(run_dir, name, rows)
            summary[name] = _summarize(rows)

        # Create tersetalk_baseline.jsonl symlink to f30_p20_q30 for significance convenience
        try:
            base = run_dir / "tersetalk_f30_p20_q30.jsonl"
            if base.exists():
                link = run_dir / "tersetalk_baseline.jsonl"
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(base.name)
        except Exception:
            # On filesystems without symlink support, skip silently
            pass

    # Hybrid budgets
    if 'hybrid' in systems:
        base_caps = {"f": 30, "p": 20, "q": 30}
        for b in HYBRID_BUDGETS:
            name = f"hybrid_budget_{b}"
            rows = run_hybrid(ds, base_caps, b)
            _save_jsonl(run_dir, name, rows)
            summary[name] = _summarize(rows)

    # Baselines
    for base in ('freeform', 'llmlingua'):
        if base in systems:
            rows = run_baseline(base, ds)
            _save_jsonl(run_dir, base, rows)
            summary[base] = _summarize(rows)

    # Persist & print summary
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 62)
    print(f"SUMMARY → {run_dir}")
    print("-" * 62)
    print(f"{'system':<28} {'tokens':>10} {'accuracy':>10} {'compl.':>8}")
    for name, s in sorted(summary.items()):
        print(f"{name:<28} {s['avg_tokens']:>10.1f} {s['accuracy']:>10.2%} {s['compliance_rate']:>8.2%}")


if __name__ == '__main__':
    main()
