from __future__ import annotations

import time
from typing import Dict, Optional


class MemoryStore:
  """
  Bounded key-value store for overflowed text, referenced by ids like "M#23".

  Eviction policy: oldest-by-last-access (access time min).
  Ids are minted monotonically per process/session and reset by reset().
  """

  MAX_ENTRIES: int = 10_000

  def __init__(self) -> None:
    self.store: Dict[str, str] = {}
    self.access_times: Dict[str, float] = {}
    self.counter: int = 0

  # -------- Core API --------
  def put(self, text: str) -> str:
    """
    Store text and return an M# reference.
    Evicts the oldest-by-last-access entry if at capacity.
    """
    if len(self.store) >= self.MAX_ENTRIES:
      self._evict_oldest()

    self.counter += 1
    mid = f"M#{self.counter}"
    text = text if isinstance(text, str) else str(text)
    self.store[mid] = text
    self.access_times[mid] = time.time()
    return mid

  def get(self, mid: str) -> Optional[str]:
    """
    Retrieve text by M# reference. Updates last-access time on hit.
    Returns None if the id is not present.
    """
    if mid in self.store:
      self.access_times[mid] = time.time()
      return self.store[mid]
    return None

  def reset(self) -> None:
    """Clear all entries and reset the id counter."""
    self.store.clear()
    self.access_times.clear()
    self.counter = 0

  def stats(self) -> Dict[str, Optional[float]]:
    """
    Memory usage statistics:
    - entries: number of keys
    - bytes: sum of len(text) across entries (ASCII/UTF-8 char count)
    - oldest: earliest last-access timestamp (epoch seconds) or None
    """
    oldest = None
    if self.access_times:
      oldest = min(self.access_times.values())
    return {
      "entries": len(self.store),
      "bytes": sum(len(v) for v in self.store.values()),
      "oldest": oldest,
    }

  # -------- Helpers --------
  def _evict_oldest(self) -> None:
    """Remove the id with the smallest last-access time."""
    if not self.access_times:
      return
    oldest_id = min(self.access_times.items(), key=lambda kv: kv[1])[0]
    self.access_times.pop(oldest_id, None)
    self.store.pop(oldest_id, None)

  # Convenience for tests / introspection
  def __len__(self) -> int:  # pragma: no cover
    return len(self.store)

