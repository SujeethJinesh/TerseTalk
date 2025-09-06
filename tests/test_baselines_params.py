from __future__ import annotations

from tersetalk.baselines import run_freeform_once
from tersetalk.hybrid_gate import estimate_tokens


class FailingClient:
    def call_text(self, system: str, user_prompt: str, max_tokens: int = 256) -> str:  # type: ignore[override]
        raise RuntimeError("boom")


class EchoClient:
    def call_text(self, system: str, user_prompt: str, max_tokens: int = 256) -> str:  # type: ignore[override]
        return "OK"


def test_freeform_success_echo():
    ex = {"question": "test?", "facts": [], "assumptions": [], "subgoal": "Answer."}
    res = run_freeform_once(ex, EchoClient(), max_tokens=123)
    assert res["status"] == "success"
    assert res["tokens"] >= estimate_tokens("")
    assert isinstance(res["tokens_total"], int)


def test_freeform_error_handling():
    ex = {"question": "test?", "facts": [], "assumptions": [], "subgoal": "Answer."}
    res = run_freeform_once(ex, FailingClient(), max_tokens=64)
    assert res["status"] == "error"
    assert res["tokens"] >= 0
    assert res["tokens_total"] >= 0

