from __future__ import annotations

import json
import re
import time

from tersetalk.memory import MemoryStore
from tersetalk.protocol_jsonl import JSONLValidator


def test_put_get_and_id_pattern():
  mem = MemoryStore()
  m1 = mem.put("foo")
  m2 = mem.put("bar")
  assert m1 != m2
  assert re.match(r"^M#\d+$", m1)
  assert mem.get(m1) == "foo"
  assert mem.get(m2) == "bar"
  st = mem.stats()
  assert st["entries"] == 2
  assert st["bytes"] >= len("foo") + len("bar")


def test_oldest_by_last_access_eviction():
  mem = MemoryStore()
  mem.MAX_ENTRIES = 2  # shrink capacity for test
  a = mem.put("A")  # oldest access initially
  time.sleep(0.002)
  b = mem.put("B")
  _ = mem.get(a)  # touch A so B becomes oldest by last-access
  time.sleep(0.002)
  c = mem.put("C")  # should evict B (oldest by last access)
  assert mem.get(b) is None
  assert mem.get(a) == "A"
  assert mem.get(c) == "C"
  assert mem.stats()["entries"] == 2


def test_reset_clears_and_resets_counter():
  mem = MemoryStore()
  m1 = mem.put("X")
  mem.reset()
  assert mem.get(m1) is None
  assert mem.stats()["entries"] == 0
  m2 = mem.put("Y")
  assert m2 == "M#1"  # counter restarted


def test_validator_integration_with_real_memory():
  mem = MemoryStore()
  caps = {"f": 5, "q": 5}
  v = JSONLValidator(caps=caps, memory=mem)
  long_fact = "alpha beta gamma delta epsilon zeta eta theta"
  long_q = "please compare the following two things and return the earlier now"
  raw = "\n".join(
    [
      '["r","M"]',
      json.dumps(["f", long_fact]),
      json.dumps(["q", "W", long_q]),
      '["g","short goal"]',
    ]
  )
  out, stats = v.validate_and_overflow(raw)
  # Must have at least two overflow lines
  o_lines = [ln for ln in out.splitlines() if ln.startswith('["o"')]
  assert len(o_lines) >= 2
  # Every overflow M# must be retrievable from memory
  for ln in o_lines:
    arr = json.loads(ln)
    mid = arr[2]
    assert isinstance(mem.get(mid), str)

  # Memory should have stored the long items
  mstats = mem.stats()
  assert mstats["entries"] >= 2
  assert "density" in stats and 0.0 <= stats["density"] <= 1.0

