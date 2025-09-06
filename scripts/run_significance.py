from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure repository root is importable when invoked as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import click

from tersetalk.statistics import (
    percent_reduction,
    bootstrap_mean_ci,
    bootstrap_ci_diff,
    one_sided_p_gt_zero,
    noninferiority,
)


def _load_tokens_acc(path: Path) -> Tuple[List[float], List[float]]:
    toks: List[float] = []
    accs: List[float] = []
    if not path.exists():
        return toks, accs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                r = json.loads(s)
            except Exception:
                continue
            if str(r.get("status", "success")) == "error":
                continue
            tok = r.get("tokens_total", r.get("tokens", None))
            if tok is not None:
                try:
                    toks.append(float(tok))
                except Exception:
                    toks.append(0.0)
            accs.append(1.0 if r.get("correct") else 0.0)
    return toks, accs


@click.command()
@click.option("--results-dir", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--terse-file", default="tersetalk_baseline.jsonl", show_default=True)
@click.option("--free-file", default="freeform.jsonl", show_default=True)
@click.option("--ll2-file", default="llmlingua.jsonl", show_default=True)
@click.option("--hybrid-file", default="hybrid_budget_600.jsonl", show_default=True)
@click.option("--confidence", default=0.95, show_default=True)
@click.option("--boots", default=5000, show_default=True)
@click.option("--out", default="significance_tests.json", show_default=True)
def main(
    results_dir: Path,
    terse_file: str,
    free_file: str,
    ll2_file: str,
    hybrid_file: str,
    confidence: float,
    boots: int,
    out: str,
):
    """Run bootstrap significance tests for tokens + quality + non-inferiority (numpy-only)."""
    terse_t, terse_a = _load_tokens_acc(results_dir / terse_file)
    free_t, free_a = _load_tokens_acc(results_dir / free_file)
    ll2_t, ll2_a = _load_tokens_acc(results_dir / ll2_file)
    hyb_t, hyb_a = _load_tokens_acc(results_dir / hybrid_file)

    report = {}

    # 1) Token reduction: freeform -> tersetalk (%), paired
    pct = percent_reduction(free_t, terse_t)
    mean_pct, lo_pct, hi_pct = bootstrap_mean_ci(pct, n_boot=boots, conf=confidence)
    p_one = one_sided_p_gt_zero(pct, n_boot=max(boots, 5000))
    report["token_reduction"] = {
        "mean_reduction_pct": float(mean_pct),
        "ci_lower": float(lo_pct),
        "ci_upper": float(hi_pct),
        "p_one_sided_gt0": float(p_one),
        "n": int(len(pct)),
    }

    # 2) Quality preservation: terse - free accuracy (paired)
    mean_q, lo_q, hi_q = bootstrap_ci_diff(terse_a, free_a, n_boot=boots, conf=confidence, paired=True)
    report["quality_preservation"] = {
        "mean_diff": float(mean_q),
        "ci_lower": float(lo_q),
        "ci_upper": float(hi_q),
        "n": int(min(len(terse_a), len(free_a))),
    }

    # 3) Hybrid non-inferiority vs LLMLingua (δ=0.02)
    report["hybrid_noninferiority"] = noninferiority(hyb_a, ll2_a, delta=0.02, conf=confidence)

    # Print concise summary
    print("\n" + "=" * 60)
    print("SIGNIFICANCE RESULTS")
    print("=" * 60)
    tr = report["token_reduction"]
    print(
        f"Token Reduction (free→terse): {tr['mean_reduction_pct']:.1%} "
        f"[{tr['ci_lower']:.1%}, {tr['ci_upper']:.1%}]  "
        f"p(one‑sided>0)={tr['p_one_sided_gt0']:.4f}  n={tr['n']}"
    )
    qp = report["quality_preservation"]
    print(
        f"Quality Δ (terse‑free): {qp['mean_diff']:.3f} "
        f"[{qp['ci_lower']:.3f}, {qp['ci_upper']:.3f}]  n={qp['n']}"
    )
    hi = report["hybrid_noninferiority"]
    print(
        f"Hybrid vs LLMLingua Non‑Inferiority (δ=0.02): "
        f"{'PASS' if hi['is_noninferior'] else 'FAIL'}; "
        f"Δ={hi['mean_difference']:.3f}, CI=[{hi['ci_lower']:.3f},{hi['ci_upper']:.3f}]"
    )

    # Save full JSON
    out_path = results_dir / out
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

