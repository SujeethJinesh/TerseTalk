from __future__ import annotations

import os
from tersetalk.reproducibility import (
  set_global_seed,
  snapshot_prng_state,
  fingerprint_snapshot,
)
import pytest


def test_api_returns_defaults():
  cfg = set_global_seed(42)
  assert cfg["temperature"] == 0.0
  assert cfg["seed"] == 42
  assert cfg["top_p"] == 1.0
  assert cfg["top_k"] == 1


def test_env_vars_set():
  set_global_seed(7)
  assert os.environ.get("PYTHONHASHSEED") == "7"
  assert os.environ.get("LLAMA_CPP_SEED") == "7"
  # CUBLAS_WORKSPACE_CONFIG should be set when CUDA is available; otherwise may be absent.
  # We don't assert presence to avoid false negatives on CPU-only CI.


def test_same_seed_produces_identical_snapshot():
  set_global_seed(123)
  fp1 = fingerprint_snapshot(n=7)
  set_global_seed(123)
  fp2 = fingerprint_snapshot(n=7)
  assert fp1 == fp2


def test_different_seed_changes_snapshot():
  set_global_seed(123)
  fp1 = fingerprint_snapshot(n=7)
  set_global_seed(124)
  fp2 = fingerprint_snapshot(n=7)
  assert fp1 != fp2


def test_snapshot_keys_present():
  set_global_seed(0)
  snap = snapshot_prng_state(n=3)
  # Libraries may be None if not installed; keys must exist
  assert "python" in snap and snap["python"] is not None and len(snap["python"]) == 3
  assert "numpy" in snap  # value may be None
  assert "torch" in snap  # value may be None


def test_invalid_seed_raises():
  with pytest.raises(ValueError):
    set_global_seed(-1)
  with pytest.raises(ValueError):
    set_global_seed("abc")  # type: ignore[arg-type]
