from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional

# Optional deps guarded at import time
try:  # NumPy (optional)
  import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
  _np = None  # type: ignore

try:  # PyTorch (optional)
  import torch as _torch  # type: ignore
except Exception:  # pragma: no cover
  _torch = None  # type: ignore


def set_global_seed(seed: int) -> Dict[str, Any]:
  """
  Set deterministic seeds across common libraries and environment.
  Returns model defaults dict suitable for generation calls.

  - Python: random + PYTHONHASHSEED
  - NumPy: np.random.seed (if installed)
  - PyTorch: manual_seed, cuda manual_seed_all, deterministic flags (if installed)
  - Llama.cpp-compatible: LLAMA_CPP_SEED env
  """
  if not isinstance(seed, int) or seed < 0:
    raise ValueError("seed must be a non-negative integer")

  # Python + hashing determinism
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)

  # NumPy determinism (optional)
  if _np is not None:
    try:
      _np.random.seed(seed)  # type: ignore
    except Exception:
      pass

  # PyTorch determinism (optional)
  if _torch is not None:
    try:
      _torch.manual_seed(seed)
      # CUDA seeding + cuBLAS workspace (if CUDA available)
      if getattr(_torch, "cuda", None) is not None and _torch.cuda.is_available():  # type: ignore[attr-defined]
        _torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
      # Prefer new deterministic API; ignore if unavailable
      try:
        _torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
      except Exception:
        # May not exist or may be unsupported on some PyTorch versions
        pass
      # cuDNN flags (safe to set on CPU-only too)
      try:
        _torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        _torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
      except Exception:
        # Some backends may be unavailable; skip without failing
        pass
    except Exception:
      # Do not fail the whole setup if torch misbehaves on the host
      pass

  # For llama.cpp or other engines that read this
  os.environ["LLAMA_CPP_SEED"] = str(seed)

  # Model defaults for deterministic decoding
  return {
    "temperature": 0.0,
    "seed": seed,
    "top_p": 1.0,
    "top_k": 1,
  }


def _stable_json_hash(obj: Any) -> str:
  """
  Stable SHA-256 of an arbitrary object by JSON encoding with sort_keys.
  Tensors/ndarrays are converted to lists to avoid dtype/device nondeterminism.
  """

  def _default(o: Any):
    # NumPy arrays
    if _np is not None and hasattr(o, "tolist"):
      try:
        return o.tolist()
      except Exception:
        pass
    # Torch tensors
    if _torch is not None and getattr(o.__class__, "__name__", "").endswith("Tensor"):
      try:
        return o.detach().cpu().tolist()
      except Exception:
        return str(o)
    return str(o)

  data = json.dumps(obj, sort_keys=True, default=_default).encode("utf-8")
  return hashlib.sha256(data).hexdigest()


def snapshot_prng_state(n: int = 5) -> Dict[str, Optional[List[float]]]:
  """
  Produce a small snapshot of PRNG outputs for determinism checks.
  Uses CPU-only operations to avoid GPU nondeterminism in tests.

  Returns a dict with keys: 'python', 'numpy', 'torch'
  - Keys may be None if the library is not available.
  """
  snap: Dict[str, Optional[List[float]]] = {}

  # Python random
  snap["python"] = [random.random() for _ in range(n)]

  # NumPy
  if _np is not None:
    try:
      snap["numpy"] = _np.random.random(n).tolist()  # type: ignore
    except Exception:
      snap["numpy"] = None
  else:
    snap["numpy"] = None

  # Torch (CPU)
  if _torch is not None:
    try:
      cpu_vals = _torch.randn(n).tolist()
      snap["torch"] = [float(x) for x in cpu_vals]
    except Exception:
      snap["torch"] = None
  else:
    snap["torch"] = None

  return snap


def fingerprint_snapshot(n: int = 5) -> str:
  """One-line helper: hash of the current PRNG snapshot."""
  return _stable_json_hash(snapshot_prng_state(n=n))
