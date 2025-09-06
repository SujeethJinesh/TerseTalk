from __future__ import annotations

import csv
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")


@dataclass
class _LatestInfo:
  """Tracks how we maintain the 'latest' pointer per experiment."""

  path: Path
  is_symlink: bool
  is_mirror_dir: bool


class ResultsManager:
  """
  Minimal, dependency-free results manager with:
  - Versioned run dirs: results/<experiment>/<YYYY-MM-DD-HH-MM-SS>/
  - 'latest' pointer (symlink when possible; else a real dir we keep mirrored)
  - Atomic JSON writes, JSONL append, CSV append
  - Simple cleanup keeping the newest N runs
  """

  def __init__(self, base_dir: str | os.PathLike = "results") -> None:
    self.base_dir = Path(base_dir)
    self.base_dir.mkdir(parents=True, exist_ok=True)
    self._latest_cache: dict[str, _LatestInfo] = {}

  # ---------- public API ----------
  def get_run_dir(
    self,
    experiment_id: str,
    timestamp: bool = True,
    run_id: Optional[str] = None,
  ) -> Path:
    """
    Create and return a new run directory for an experiment.
    - If run_id is provided, use it (must be filesystem-safe).
    - Else if timestamp is True, generate a timestamp-based run_id.
    - Also set/update 'latest' pointer (symlink preferred; mirror dir fallback).
    """
    exp_dir = self._exp_dir(experiment_id)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if run_id is None:
      run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") if timestamp else "run"

    run_id = self._dedupe_run_id(exp_dir, run_id)
    run_dir = exp_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    latest_info = self._set_latest_pointer(exp_dir, run_dir)
    self._latest_cache[experiment_id] = latest_info
    return run_dir

  def save_config(self, run_dir: Path, config: dict, name: str = "config.json") -> Path:
    """Atomically write config JSON in run_dir (and mirror to latest if needed)."""
    path = run_dir / name
    self._atomic_write_json(path, config)
    self._mirror_to_latest(run_dir, path)
    return path

  def append_jsonl(self, run_dir: Path, filename: str, record: dict | str) -> Path:
    """
    Append a JSONL record. If 'record' is a dict, it is json.dumps()-ed.
    Also append to latest mirror when symlinks are unavailable.
    """
    path = run_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    line = record if isinstance(record, str) else json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8", newline="") as f:
      f.write(line)
      if not line.endswith("\n"):
        f.write("\n")
    self._mirror_jsonl_append(run_dir, filename, line)
    return path

  def append_csv_row(self, run_dir: Path, filename: str, row: dict) -> Path:
    """
    Append a row to a CSV file. If file doesn't exist, write header first.
    Header is the sorted set of current row keys when creating file.
    (Subsequent rows should be consistent with the same header.)
    """
    path = run_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    fieldnames = sorted(row.keys())

    with path.open("a", encoding="utf-8", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      if is_new:
        writer.writeheader()
      writer.writerow(row)

    self._mirror_csv_row(run_dir, filename, row, fieldnames, write_header=is_new)
    return path

  def save_summary(self, run_dir: Path, summary: dict, name: str = "summary.json") -> Path:
    """Atomically write a small summary JSON (and mirror to latest if needed)."""
    path = run_dir / name
    self._atomic_write_json(path, summary)
    self._mirror_to_latest(run_dir, path)
    return path

  def cleanup_old_runs(self, experiment_id: str, keep_last_n: int = 5) -> list[Path]:
    """
    Remove older run directories, keeping the newest N.
    Ignores the 'latest' pointer entry (symlink or mirror dir).
    Returns a list of removed run dirs.
    """
    exp_dir = self._exp_dir(experiment_id)
    if not exp_dir.exists():
      return []
    run_dirs = [p for p in exp_dir.iterdir() if p.is_dir() and p.name != "latest"]
    run_dirs.sort(key=lambda p: (self._sort_key(p.name), p.stat().st_mtime), reverse=True)

    to_keep = set(run_dirs[: max(keep_last_n, 0)])
    removed: list[Path] = []
    for p in run_dirs:
      if p not in to_keep:
        shutil.rmtree(p, ignore_errors=True)
        removed.append(p)
    return removed

  # ---------- helpers ----------
  def _exp_dir(self, experiment_id: str) -> Path:
    safe = str(experiment_id).strip().replace(" ", "_")
    return self.base_dir / safe

  def _dedupe_run_id(self, exp_dir: Path, run_id: str) -> str:
    if not (exp_dir / run_id).exists():
      return run_id
    i = 1
    while True:
      cand = f"{run_id}-{i:02d}"
      if not (exp_dir / cand).exists():
        return cand
      i += 1

  def _set_latest_pointer(self, exp_dir: Path, run_dir: Path) -> _LatestInfo:
    latest = exp_dir / "latest"
    if latest.exists() or latest.is_symlink():
      try:
        if latest.is_symlink() or latest.is_file():
          latest.unlink()
        else:
          shutil.rmtree(latest, ignore_errors=True)
      except Exception:
        pass
    try:
      target = Path(os.path.relpath(run_dir, exp_dir))
      latest.symlink_to(target, target_is_directory=True)
      return _LatestInfo(path=latest, is_symlink=True, is_mirror_dir=False)
    except Exception:
      latest.mkdir(parents=True, exist_ok=True)
      return _LatestInfo(path=latest, is_symlink=False, is_mirror_dir=True)

  def _atomic_write_json(self, path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
      json.dump(obj, f, ensure_ascii=False, indent=2)
      f.write("\n")
    os.replace(tmp, path)

  def _mirror_to_latest(self, run_dir: Path, written_file: Path) -> None:
    exp_dir = run_dir.parent
    info = self._latest_cache.get(exp_dir.name)
    if not info:
      return
    if info.is_mirror_dir and not info.is_symlink:
      dst = info.path / written_file.name
      dst.parent.mkdir(parents=True, exist_ok=True)
      tmp = dst.with_suffix(dst.suffix + ".tmp")
      shutil.copyfile(written_file, tmp)
      os.replace(tmp, dst)

  def _mirror_jsonl_append(self, run_dir: Path, filename: str, line: str) -> None:
    exp_dir = run_dir.parent
    info = self._latest_cache.get(exp_dir.name)
    if not info or not info.is_mirror_dir or info.is_symlink:
      return
    dst = info.path / filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a", encoding="utf-8", newline="") as f:
      f.write(line)
      if not line.endswith("\n"):
        f.write("\n")

  def _mirror_csv_row(
    self,
    run_dir: Path,
    filename: str,
    row: dict,
    fieldnames: list[str],
    write_header: bool,
  ) -> None:
    exp_dir = run_dir.parent
    info = self._latest_cache.get(exp_dir.name)
    if not info or not info.is_mirror_dir or info.is_symlink:
      return
    dst = info.path / filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    is_new = not dst.exists()
    with dst.open("a", encoding="utf-8", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      if write_header and is_new:
        writer.writeheader()
      writer.writerow(row)

  def _sort_key(self, name: str) -> tuple[int, str]:
    return (1 if _TIMESTAMP_RE.match(name) else 0, name)

