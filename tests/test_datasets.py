from __future__ import annotations

import os

from tersetalk.datasets import load_gsm8k, load_hotpotqa


def _schema_ok(example):
    assert isinstance(example["question"], str) and example["question"]
    assert isinstance(example["answer"], str) and example["answer"]
    assert isinstance(example["facts"], list)
    assert isinstance(example["subgoal"], str) and example["subgoal"]
    assert isinstance(example["assumptions"], list)


def test_offline_hotpotqa_determinism(monkeypatch):
    monkeypatch.setenv("TERSETALK_OFFLINE_DATA", "1")
    a = load_hotpotqa(n=5, seed=123)
    b = load_hotpotqa(n=5, seed=123)
    c = load_hotpotqa(n=5, seed=124)
    assert a == b
    assert a != c
    for ex in a:
        _schema_ok(ex)
        assert len(ex["facts"]) >= 2  # our synthetic has 2 facts


def test_offline_gsm8k_determinism(monkeypatch):
    monkeypatch.setenv("TERSETALK_OFFLINE_DATA", "1")
    a = load_gsm8k(n=6, seed=77)
    b = load_gsm8k(n=6, seed=77)
    c = load_gsm8k(n=6, seed=78)
    assert a == b
    assert a != c
    for ex in a:
        _schema_ok(ex)


def test_real_optional_smoke():
    """
    Optional: If RUN_REAL_DATASETS=1, call real loaders (may use local HF cache).
    This is skipped by default in CI.
    """
    if os.environ.get("RUN_REAL_DATASETS") != "1":
        return
    os.environ.pop("TERSETALK_OFFLINE_DATA", None)  # Small n to keep fast
    hotpot = load_hotpotqa(n=3, seed=0, offline=False)
    gsm = load_gsm8k(n=3, seed=0, offline=False)
    assert len(hotpot) == 3 and len(gsm) == 3
    for ex in hotpot + gsm:
        _schema_ok(ex)

