from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional


# --------------------------
# Env & utilities
# --------------------------


def _truthy(env: str, default: bool = False) -> bool:
    v = os.environ.get(env)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _deterministic_indices(total: int, n: int, seed: int) -> List[int]:
    n = max(0, min(n, total))
    rng = random.Random(int(seed))  # Use sample without replacement for stable subset
    return rng.sample(range(total), n)


def _shorten(text: str, max_chars: int = 64) -> str:
    if len(text) <= max_chars:
        return text
    base = text[: max_chars]
    sp = base.rfind(" ")
    if sp >= 16:
        cut = base[:sp]
    else:
        # Ensure room for ellipsis when no reasonable space to cut on
        cut = text[: max(0, max_chars - 3)]
    return cut.rstrip() + "..."


# --------------------------
# Synthetic shards (offline)
# --------------------------


def _synth_hotpotqa(n: int, seed: int = 0) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))

    def _date() -> str:
        y = rng.randint(1980, 2021)
        m = rng.randint(1, 12)
        d = rng.randint(1, 28)
        return f"{y:04d}-{m:02d}-{d:02d}"

    out: List[Dict[str, Any]] = []
    for i in range(n):
        a_title = f"EventA{i}"
        b_title = f"EventB{i}"
        a_date = _date()
        b_date = _date()
        earlier = min(a_date, b_date)
        q = (
            f"Between {a_title} on {a_date} and {b_title} on {b_date}, which happened first?"
        )
        facts = [f"{a_title}: occurred on {a_date}", f"{b_title}: occurred on {b_date}"]
        out.append(
            {
                "question": q,
                "answer": earlier,
                "facts": facts,
                "subgoal": f"Identify earlier date for pair {i}.",
                "assumptions": ["Use ISO dates", "Return earlier only"],
            }
        )
    return out


def _synth_gsm8k(n: int, seed: int = 0) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        a = rng.randint(2, 9)
        b = rng.randint(2, 9)
        c = rng.randint(10, 50)
        q = (
            f"Tom has {a} boxes with {b} apples each and buys {c} more apples. "
            f"How many apples does he have in total?"
        )
        ans = a * b + c
        out.append(
            {
                "question": q,
                "answer": str(ans),  # keep as string for uniformity
                "facts": [],  # GSM8K facts generally emerge in reasoning; keep empty
                "subgoal": f"Compute {a}×{b}+{c}.",
                "assumptions": ["Do integer arithmetic", "Final answer only"],
            }
        )
    return out


# --------------------------
# Real loaders (guarded)
# --------------------------


def _maybe_import_datasets():
    try:
        import datasets  # type: ignore

        return datasets
    except Exception:
        return None


def _hotpot_supporting_sentences(item: Dict[str, Any]) -> List[str]:
    """
    Map Hotpot 'supporting_facts' (title, sent_id) pairs into textual sentences using 'context'.
    Fallback: first sentence of up to two contexts if supports are missing.
    """
    facts: List[str] = []
    ctx_raw = item.get("context") or []
    # Normalize contexts into iterable of (title, sents)
    if isinstance(ctx_raw, dict):
        ctx_iter = list(ctx_raw.items())
        title_to_sents = {}
        for title, sents in ctx_iter:
            if isinstance(title, str) and isinstance(sents, (list, tuple)):
                title_to_sents[title] = list(sents)
    elif isinstance(ctx_raw, list) and ctx_raw and isinstance(ctx_raw[0], dict):
        ctx_iter = list(ctx_raw)
        title_to_sents = {}
        for doc in ctx_iter:
            title = doc.get("title")
            sents = doc.get("sentences")
            if isinstance(title, str) and isinstance(sents, (list, tuple)):
                title_to_sents[title] = list(sents)
    else:
        ctx_iter = list(ctx_raw)
        # Be robust to shapes like [title, [sents]] or longer tuples
        title_to_sents = {}
        for pair in ctx_iter:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                title = pair[0]
                sents = pair[1]
            else:
                continue
            if isinstance(title, str) and isinstance(sents, (list, tuple)):
                title_to_sents[title] = list(sents)
    for pair in item.get("supporting_facts") or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        title, idx = pair
        if title not in title_to_sents:
            continue
        sents = title_to_sents.get(title) or []
        if isinstance(idx, int) and 0 <= idx < len(sents):
            sent = sents[idx]
            if isinstance(sent, str):
                facts.append(f"{title}: {sent}")
    if not facts:  # safe fallback
        tmp: List[str] = []
        if isinstance(ctx_iter, list) and ctx_iter and isinstance(ctx_iter[0], dict):
            for doc in ctx_iter[:2]:
                title = doc.get("title")
                sents = doc.get("sentences")
                if isinstance(title, str) and isinstance(sents, (list, tuple)) and sents:
                    tmp.append(f"{title}: {sents[0]}")
        else:
            for pair in ctx_iter[:2]:  # type: ignore[index]
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    title = pair[0]
                    sents = pair[1]
                    if isinstance(title, str) and isinstance(sents, (list, tuple)) and sents:
                        tmp.append(f"{title}: {sents[0]}")
        facts = tmp
    # de-dup and cap
    seen = set()
    uniq: List[str] = []
    for f in facts:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq[:10]


def load_hotpotqa(
    split: str = "validation", n: int = 100, seed: int = 0, offline: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    Load HotpotQA and normalize to the TerseTalk schema.
    If offline (env TERSETALK_OFFLINE_DATA=1 or offline=True), return synthetic data.
    """
    if offline is None:
        offline = _truthy("TERSETALK_OFFLINE_DATA", default=False)
    if offline:
        return _synth_hotpotqa(n, seed)

    ds_lib = _maybe_import_datasets()
    if ds_lib is None:
        return _synth_hotpotqa(n, seed)

    # Try common configs robustly: 'distractor' (validation/train), fallback to 'fullwiki'
    ds = None
    for config in ("distractor", "fullwiki", None):
        try:
            if config is None:
                ds = ds_lib.load_dataset("hotpot_qa", split=split)
            else:
                ds = ds_lib.load_dataset("hotpot_qa", config, split=split)
            break
        except Exception:
            ds = None
    if ds is None:
        return _synth_hotpotqa(n, seed)

    total = len(ds)
    idxs = _deterministic_indices(total, n, seed)
    out: List[Dict[str, Any]] = []
    for i in idxs:
        item = ds[int(i)]
        q = item.get("question", "")
        a = item.get("answer", "")
        facts = _hotpot_supporting_sentences(item)
        out.append(
            {
                "question": q,
                "answer": str(a),
                "facts": facts,
                "subgoal": _shorten(f"Answer: {q}", 64),
                "assumptions": ["Use provided facts", "Be concise"],
            }
        )
    return out


def _gsm8k_extract_answer(ans: str) -> str:
    """
    GSM8K 'answer' often ends with '#### <number>'. Keep full string for gold,
    but this helper can extract the numeric if needed elsewhere.
    """
    if not isinstance(ans, str):
        return str(ans)
    s = ans.strip()
    if "####" in s:
        tail = s.split("####", 1)[-1].strip()  # keep as string but strip spaces
        return tail
    return s


def load_gsm8k(
    split: str = "test", n: int = 100, seed: int = 0, offline: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    Load GSM8K ('main' config) and normalize to the TerseTalk schema.
    If offline (env TERSETALK_OFFLINE_DATA=1 or offline=True), return synthetic data.
    """
    if offline is None:
        offline = _truthy("TERSETALK_OFFLINE_DATA", default=False)
    if offline:
        return _synth_gsm8k(n, seed)

    ds_lib = _maybe_import_datasets()
    if ds_lib is None:
        return _synth_gsm8k(n, seed)

    # GSM8K has config 'main' with splits 'train'/'test'
    try:
        ds = ds_lib.load_dataset("gsm8k", "main", split=split)
    except Exception:
        # Fallback to 'test' if unknown split
        try:
            ds = ds_lib.load_dataset("gsm8k", "main", split="test")
        except Exception:
            return _synth_gsm8k(n, seed)

    total = len(ds)
    idxs = _deterministic_indices(total, n, seed)
    out: List[Dict[str, Any]] = []
    for i in idxs:
        item = ds[int(i)]
        q = item.get("question", "")
        a_raw = item.get("answer", "")
        a = _gsm8k_extract_answer(a_raw)
        out.append(
            {
                "question": q,
                "answer": str(a),
                "facts": [],  # reasoning steps are model-generated; keep empty
                "subgoal": _shorten(f"Solve: {q}", 64),
                "assumptions": ["Show concise reasoning if needed", "Return final number"],
            }
        )
    return out
