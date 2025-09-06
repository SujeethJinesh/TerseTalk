from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class RunRecord:
    run_path: Path
    run_id: str
    task: str
    system: str
    model: str
    seed: int
    n: int
    caps: Optional[Dict[str, int]]
    summarizer: Optional[str]
    deref_policy: Optional[str]
    hybrid: bool
    token_budget: Optional[int]
    accuracy: float
    tokens_avg: float
    tokens_median: float
    bytes_on_wire_avg: float
    sp_avg: Optional[float]
    overflow_avg: Optional[float]


# ----------------------------
# I/O helpers
# ----------------------------

def _safe_read_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _discover_runs(indir: Path) -> List[Path]:
    """
    A run directory is any directory that contains a config.json file.
    We search recursively under indir/*/* to be tolerant of organize patterns.
    """
    return [cfg.parent for cfg in indir.rglob("config.json")]


def _extract_run_record(run_dir: Path) -> Optional[RunRecord]:
    cfg = _safe_read_json(run_dir / "config.json")
    summ = _safe_read_json(run_dir / "summary.json")

    if not cfg or not summ:
        # Incomplete run; skip
        return None

    # Required config fields with defaults
    task = str(cfg.get("task", "unknown"))
    system = str(cfg.get("system", "unknown"))
    model = str(cfg.get("model", "unknown"))
    seed = int(cfg.get("seed", 0))
    n = int(cfg.get("n", 0))
    caps = cfg.get("caps", None)
    summarizer = cfg.get("summarizer", None)
    deref_policy = cfg.get("deref_policy", None)
    hybrid = bool(cfg.get("hybrid", False))
    token_budget = cfg.get("token_budget", None)
    try:
        token_budget = int(token_budget) if token_budget is not None else None
    except Exception:
        token_budget = None

    # Summary metrics (see PR-11 _summarize)
    accuracy = float(summ.get("accuracy", 0.0))
    tokens_avg = float(summ.get("tokens_avg", 0.0))
    tokens_median = float(summ.get("tokens_median", 0.0))
    bytes_on_wire_avg = float(summ.get("bytes_on_wire_avg", 0.0))
    sp_avg = summ.get("sp_avg", None)
    sp_avg = float(sp_avg) if sp_avg is not None else None
    overflow_avg = summ.get("overflow_avg", None)
    overflow_avg = float(overflow_avg) if overflow_avg is not None else None

    return RunRecord(
        run_path=run_dir,
        run_id=run_dir.name,
        task=task,
        system=system,
        model=model,
        seed=seed,
        n=n,
        caps=caps if isinstance(caps, dict) else None,
        summarizer=summarizer,
        deref_policy=deref_policy,
        hybrid=hybrid,
        token_budget=token_budget,
        accuracy=accuracy,
        tokens_avg=tokens_avg,
        tokens_median=tokens_median,
        bytes_on_wire_avg=bytes_on_wire_avg,
        sp_avg=sp_avg,
        overflow_avg=overflow_avg,
    )


def _rows_from_records(recs: List[RunRecord]) -> List[Dict[str, Any]]:
    rows = []
    for r in recs:
        rows.append({
            "run_path": str(r.run_path),
            "run_id": r.run_id,
            "task": r.task,
            "system": r.system,
            "model": r.model,
            "seed": r.seed,
            "n": r.n,
            "caps": json.dumps(r.caps) if r.caps else "",
            "summarizer": r.summarizer or "",
            "deref_policy": r.deref_policy or "",
            "hybrid": int(r.hybrid),
            "token_budget": r.token_budget if r.token_budget is not None else "",
            "accuracy": r.accuracy,
            "tokens_avg": r.tokens_avg,
            "tokens_median": r.tokens_median,
            "bytes_on_wire_avg": r.bytes_on_wire_avg,
            "sp_avg": r.sp_avg if r.sp_avg is not None else "",
            "overflow_avg": r.overflow_avg if r.overflow_avg is not None else "",
        })
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        # Write empty CSV with header to keep tooling happy
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("run_path,run_id,task,system,model,seed,n,caps,summarizer,deref_policy,hybrid,token_budget,accuracy,tokens_avg,tokens_median,bytes_on_wire_avg,sp_avg,overflow_avg\n")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


# ----------------------------
# Pareto frontier
# ----------------------------

def pareto_frontier(points: List[Tuple[float, float]]) -> List[int]:
    """
    Compute indices of points that form the Pareto frontier for:
        minimize x (tokens), maximize y (accuracy).
    Args:
        points: list of (x=tokens_avg, y=accuracy)
    Returns:
        indices (list) into 'points' that lie on frontier, sorted by x asc.
    """
    if not points:
        return []
    # Sort by x asc, y desc so we can single-scan
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    best_y = -1.0
    frontier: List[int] = []
    for i in order:
        x, y = points[i]
        if y > best_y:
            frontier.append(i)
            best_y = y
    # Return frontier indices in x-ascending order
    return sorted(frontier, key=lambda i: points[i][0])


# ----------------------------
# Plotting
# ----------------------------

def plot_pareto_for_task(task: str, rows: List[Dict[str, Any]], outdir: Path, fmt: str = "pdf", annotate: bool = True) -> Optional[Path]:
    """
    Build Accuracy vs Tokens scatter by system for 'task'; overlay global Pareto frontier.
    """
    task_rows = [r for r in rows if r["task"] == task]
    if not task_rows:
        return None

    # Prepare points by system
    systems = sorted({r["system"] for r in task_rows})
    points: List[Tuple[float,float]] = []
    labels: List[str] = []
    for r in task_rows:
        try:
            x = float(r["tokens_avg"])
            y = float(r["accuracy"])
        except Exception:
            continue
        points.append((x, y))
        labels.append(f'{r["system"]}')

    # Compute frontier on all points (across systems)
    frontier_idx = pareto_frontier(points)

    # Plot
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, marker="o", s=36, alpha=0.8)

    if annotate:
        for (x, y), lab in zip(points, labels):
            ax.annotate(lab, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)

    # Overlay frontier as line
    fx = [points[i][0] for i in frontier_idx]
    fy = [points[i][1] for i in frontier_idx]
    if len(fx) >= 2:
        ax.plot(fx, fy, linewidth=1.5)

    ax.set_xlabel("Avg tokens per example")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Pareto — {task}")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"pareto_{task}.{fmt}"
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    # Save frontier points CSV
    frontier_rows = []
    for i in frontier_idx:
        frontier_rows.append({"tokens_avg": points[i][0], "accuracy": points[i][1]})
    _write_csv(outdir / f"pareto_points_{task}.csv", frontier_rows)

    return outpath


def plot_caps_ablation_for_task(task: str, rows: List[Dict[str, Any]], outdir: Path, fmt: str = "pdf") -> Optional[Path]:
    """
    For TerseTalk runs only, aggregate by 'caps' dictionary and produce:
      - Avg cap size vs Tokens (line with markers)
      - Overflow rate vs Quality (line with markers)
    Skip gracefully if fewer than 2 distinct cap settings.
    """
    trs = [r for r in rows if r["task"] == task and r["system"] == "tersetalk" and r.get("caps")]
    if not trs:
        return None

    # Aggregate by caps (stringified as canonical JSON with sorted keys)
    buckets: Dict[str, Dict[str, Any]] = {}
    for r in trs:
        try:
            caps = json.loads(r["caps"]) if isinstance(r["caps"], str) else r["caps"]
        except Exception:
            caps = None
        if not isinstance(caps, dict):
            continue

        avg_cap = float(np.mean([caps.get("f", 0), caps.get("p", 0), caps.get("q", 0)]))
        key = json.dumps(caps, sort_keys=True)

        b = buckets.setdefault(key, {"avg_cap": avg_cap, "tokens": [], "quality": [], "overflow": []})
        b["tokens"].append(float(r["tokens_avg"]))
        b["quality"].append(float(r["accuracy"]))
        ov = r.get("overflow_avg")
        if ov != "" and ov is not None:
            b["overflow"].append(float(ov))

    # Reduce
    entries = []
    for k, b in buckets.items():
        if not b["tokens"] or not b["quality"]:
            continue
        entries.append({
            "caps_json": k,
            "avg_cap": b["avg_cap"],
            "tokens": float(np.mean(b["tokens"])),
            "quality": float(np.mean(b["quality"])),
            "overflow_rate": float(np.mean(b["overflow"])) if b["overflow"] else float("nan"),
        })

    if len(entries) < 2:
        # Not enough points to plot lines; write CSV and return None
        _write_csv(outdir / f"ablation_caps_{task}.csv", entries)
        return None

    # Sort by avg_cap
    entries.sort(key=lambda e: e["avg_cap"])

    # Plot
    fig = plt.figure(figsize=(12, 5))
    # left: cap size vs tokens
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot([e["avg_cap"] for e in entries], [e["tokens"] for e in entries], marker="o")
    ax1.set_xlabel("Average cap size")
    ax1.set_ylabel("Avg tokens per example")
    ax1.set_title(f"Caps vs Tokens — {task}")
    ax1.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    # right: overflow vs quality
    ax2 = fig.add_subplot(1, 2, 2)
    xs = [e["overflow_rate"] for e in entries if not math.isnan(e["overflow_rate"])]
    ys = [e["quality"] for e in entries if not math.isnan(e["overflow_rate"])]
    if xs and ys:
        ax2.plot(xs, ys, marker="o")
    ax2.set_xlabel("Overflow rate")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Overflow vs Quality — {task}")
    ax2.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    outpath = outdir / f"ablation_caps_{task}.{fmt}"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    # CSV
    _write_csv(outdir / f"ablation_caps_{task}.csv", entries)
    return outpath


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyze TerseTalk v0.5 results and generate figures.")
    ap.add_argument("--indir", type=str, default="results", help="Root directory containing run outputs")
    ap.add_argument("--outdir", type=str, default=None, help="Directory to write figures/CSVs (default: <indir>/figures)")
    ap.add_argument("--task", type=str, action="append", default=None, help="Filter by task (repeatable)")
    ap.add_argument("--system", type=str, action="append", default=None, help="Filter by system (repeatable)")
    ap.add_argument("--format", dest="fmt", choices=["pdf", "png", "svg"], default="pdf", help="Figure format")
    ap.add_argument("--no-annotate", action="store_true", help="Disable point labels on Pareto plots")
    args = ap.parse_args()

    indir = Path(args.indir).resolve()
    if not indir.exists():
        print(f"[analyze_v05] input directory does not exist: {indir}", file=sys.stderr)
        sys.exit(2)

    outdir = Path(args.outdir).resolve() if args.outdir else (indir / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_runs(indir)
    if not run_dirs:
        print(f"[analyze_v05] no run directories found under {indir}", file=sys.stderr)
        sys.exit(0)

    # Build records
    records: List[RunRecord] = []
    for rd in run_dirs:
        rec = _extract_run_record(rd)
        if rec is None:
            continue
        if args.task and rec.task not in set(args.task):
            continue
        if args.system and rec.system not in set(args.system):
            continue
        records.append(rec)

    if not records:
        print(f"[analyze_v05] no records after filtering", file=sys.stderr)
        sys.exit(0)

    # Write by_run.csv
    by_run_rows = _rows_from_records(records)
    _write_csv(outdir / "by_run.csv", by_run_rows)

    # Unique tasks
    tasks = sorted({r.task for r in records})

    # Pareto plots per task
    for task in tasks:
        try:
            plot_pareto_for_task(task, by_run_rows, outdir, fmt=args.fmt, annotate=not args.no_annotate)
        except Exception as e:
            print(f"[analyze_v05] WARN: pareto plot failed for task={task}: {e}", file=sys.stderr)

    # Caps ablation per task (TerseTalk only)
    for task in tasks:
        try:
            plot_caps_ablation_for_task(task, by_run_rows, outdir, fmt=args.fmt)
        except Exception as e:
            print(f"[analyze_v05] WARN: caps ablation failed for task={task}: {e}", file=sys.stderr)

    print(f"[analyze_v05] Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
