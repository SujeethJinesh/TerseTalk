from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import subprocess
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
# PR-15 additions: tree aggregation + provenance
# ----------------------------

def _discover_summary_runs(root: Path) -> List[Path]:
    return sorted({p.parent for p in root.rglob("summary.json")})


def _list_system_jsonls(run_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in run_dir.glob("*.jsonl"):
        out[p.stem] = p
    return out


def _load_jsonl_rows(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def _summarize_system_file(jsonl_path: Path) -> Optional[Dict[str, float]]:
    toks: List[float] = []
    accs: List[float] = []
    ofl: List[float] = []
    for r in _load_jsonl_rows(jsonl_path):
        if str(r.get("status", "success")) == "error":
            continue
        t = r.get("tokens_total", r.get("tokens", 0))
        toks.append(float(t or 0))
        accs.append(1.0 if r.get("correct") else 0.0)
        if "overflow_count" in r:
            try:
                ofl.append(float(r.get("overflow_count") or 0))
            except Exception:
                pass
    if not toks:
        return None
    avg_tokens = float(np.mean(toks))
    accuracy = float(np.mean(accs))
    overflow_rate = float(np.mean(ofl)) if ofl else float("nan")
    return {"avg_tokens": avg_tokens, "accuracy": accuracy, "overflow_rate": overflow_rate}


def _save_pareto_points(points: List[Dict[str, float]], out_csv: Path, out_pdf: Path) -> None:
    # Build frontier flags using existing pareto_frontier helper
    pairs = [(p["tokens"], p["accuracy"]) for p in points]
    idx = set(pareto_frontier(pairs))
    rows = []
    for i, p in enumerate(points):
        rows.append({
            "system": p.get("system", ""),
            "tokens": float(p.get("tokens", 0.0)),
            "accuracy": float(p.get("accuracy", 0.0)),
            "is_pareto": 1 if i in idx else 0,
        })
    # CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["system", "tokens", "accuracy", "is_pareto"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for i, p in enumerate(points):
        is_front = i in idx
        ax.scatter(p["tokens"], p["accuracy"], marker=("o" if is_front else "x"))
        if is_front:
            ax.annotate(str(p.get("system", "")), (p["tokens"], p["accuracy"]), xytext=(3, 3), textcoords="offset points", fontsize=8)
    fx = [points[i]["tokens"] for i in sorted(idx, key=lambda j: points[j]["tokens"])]
    fy = [points[i]["accuracy"] for i in sorted(idx, key=lambda j: points[j]["tokens"])]
    if len(fx) >= 2:
        ax.plot(fx, fy, linestyle="--", linewidth=1)
    ax.set_xlabel("Total Tokens (lower is better)")
    ax.set_ylabel("Task Accuracy (higher is better)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _collect_pareto_points(runs: List[Path]) -> List[Dict[str, float]]:
    pts: List[Dict[str, float]] = []
    skipped = 0
    for run in runs:
        for name, path in _list_system_jsonls(run).items():
            s = _summarize_system_file(path)
            if not s:
                skipped += 1
                continue
            pts.append({"system": name, "tokens": s["avg_tokens"], "accuracy": s["accuracy"]})
    print(f"[analyze_v05] pareto: {len(pts)} points; skipped {skipped} files", file=sys.stderr)
    # Deduplicate by (system, tokens, accuracy)
    uniq: Dict[Tuple[str, float, float], Dict[str, float]] = {}
    for p in pts:
        key = (str(p["system"]), round(float(p["tokens"]), 4), round(float(p["accuracy"]), 4))
        uniq[key] = p
    return list(uniq.values())


def _parse_avg_cap_from_filename(p: Path) -> Optional[float]:
    name = p.stem
    # Expect names like tersetalk_f30_p20_q30 or tersetalk_aggressive
    try:
        if "_f" in name and "_p" in name and "_q" in name:
            import re
            m_f = re.search(r"_f(\d+)", name)
            m_p = re.search(r"_p(\d+)", name)
            m_q = re.search(r"_q(\d+)", name)
            if m_f and m_p and m_q:
                f = float(m_f.group(1)); pval = float(m_p.group(1)); q = float(m_q.group(1))
                return float(np.mean([f, pval, q]))
        mapping = {"aggressive": 20.0, "baseline": 30.0, "relaxed": 50.0, "very_relaxed": 100.0}
        for k, v in mapping.items():
            if k in name:
                return v
    except Exception:
        return None
    return None


def _collect_ablation_rows(latest_run: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for name, path in _list_system_jsonls(latest_run).items():
        if not name.startswith("tersetalk"):
            continue
        s = _summarize_system_file(path)
        if not s:
            continue
        avg_cap = _parse_avg_cap_from_filename(path)
        if avg_cap is None:
            # heuristically use tokens as proxy ordering if caps unknown
            avg_cap = float(s["avg_tokens"]) if s["avg_tokens"] else 0.0
        rows.append({
            "name": name,
            "avg_cap": float(avg_cap),
            "tokens": float(s["avg_tokens"]),
            "accuracy": float(s["accuracy"]),
            "overflow_rate": float(s["overflow_rate"]),
        })
    rows.sort(key=lambda r: r["avg_cap"])  # deterministic
    return rows


def _save_ablation(rows: List[Dict[str, float]], out_csv: Path, out_pdf: Path) -> None:
    # CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "avg_cap", "tokens", "accuracy", "overflow_rate"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot (1x2)
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot([r["avg_cap"] for r in rows], [r["tokens"] for r in rows], marker="o")
    ax1.set_xlabel("Average Cap Size")
    ax1.set_ylabel("Total Tokens Used")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    xs = [r["overflow_rate"] for r in rows if not math.isnan(r["overflow_rate"]) ]
    ys = [r["accuracy"] for r in rows if not math.isnan(r["overflow_rate"]) ]
    if xs and ys:
        ax2.plot(xs, ys, marker="o")
    ax2.set_xlabel("Overflow rate")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _git_short() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _enrich_summary_with_provenance(summary_path: Path) -> None:
    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        print(f"[analyze_v05] WARN: failed to load summary.json at {summary_path}", file=sys.stderr)
        return
    import datetime as _dt
    for name, stats in obj.items():
        if not isinstance(stats, dict):
            continue
        stats["tokens_method"] = "tiktoken" if _tiktoken_available() else "heuristic"
        stats["sp_method"] = "bertscore" if _bertscore_available() else "jaccard"
        stats["timestamp"] = _dt.datetime.now().isoformat()
        stats["version"] = _git_short()
    summary_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"[analyze_v05] enriched provenance in {summary_path}", file=sys.stderr)


def _tiktoken_available() -> bool:
    try:
        import tiktoken  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _bertscore_available() -> bool:
    try:
        import bert_score  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


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
    # PR-15 toggles (default on); allow disabling via --no-*
    ap.add_argument("--enrich-provenance", dest="enrich_prov", action="store_true", default=True)
    ap.add_argument("--no-enrich-provenance", dest="enrich_prov", action="store_false")
    ap.add_argument("--pareto", dest="do_pareto", action="store_true", default=True)
    ap.add_argument("--no-pareto", dest="do_pareto", action="store_false")
    ap.add_argument("--ablation", dest="do_ablation", action="store_true", default=True)
    ap.add_argument("--no-ablation", dest="do_ablation", action="store_false")
    args = ap.parse_args()

    indir = Path(args.indir).resolve()
    if not indir.exists():
        print(f"[analyze_v05] input directory does not exist: {indir}", file=sys.stderr)
        sys.exit(2)

    outdir = Path(args.outdir).resolve() if args.outdir else (indir / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_runs(indir)
    if not run_dirs:
        print(f"[analyze_v05] no run directories (config.json) found under {indir}", file=sys.stderr)

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
        print(f"[analyze_v05] no records after filtering (by_run.csv skipped)", file=sys.stderr)
        by_run_rows = []
    else:
        # Write by_run.csv
        by_run_rows = _rows_from_records(records)
        _write_csv(outdir / "by_run.csv", by_run_rows)

    # Unique tasks
    tasks = sorted({r.task for r in records}) if records else []

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

    # PR-15: Tree-wide Pareto and latest-run ablation + provenance
    try:
        sum_runs = _discover_summary_runs(indir)
        print(f"[analyze_v05] discovered {len(sum_runs)} run(s) under {indir}", file=sys.stderr)
        if sum_runs:
            latest = max(sum_runs, key=lambda p: p.stat().st_mtime)
            print(f"[analyze_v05] using latest run for ablation: {latest}", file=sys.stderr)
            if args.do_pareto:
                points = _collect_pareto_points(sum_runs)
                if points:
                    _save_pareto_points(points, outdir / "pareto_points.csv", outdir / "pareto_frontier.pdf")
            if args.do_ablation:
                rows = _collect_ablation_rows(latest)
                if rows:
                    _save_ablation(rows, outdir / "ablation_caps.csv", outdir / "ablation_caps.pdf")
                else:
                    print("[analyze_v05] no tersetalk* jsonl files found for ablation", file=sys.stderr)
            if args.enrich_prov:
                sp = latest / "summary.json"
                if sp.exists():
                    _enrich_summary_with_provenance(sp)
                else:
                    print(f"[analyze_v05] summary.json missing at {sp}", file=sys.stderr)
    except Exception as e:
        print(f"[analyze_v05] WARN: PR-15 extras failed: {e}", file=sys.stderr)

    print(f"[analyze_v05] Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
