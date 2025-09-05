from __future__ import annotations

import json
import os
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError


# -----------------------------
# Generic line for Instructor
# -----------------------------


class TerseTalkLine(BaseModel):
  """
  Generic typed line that is easy for Instructor to produce.
  Use convert_to_specific(...) to map into strongly-typed models.
  """

  tag: Literal["r", "g", "f", "u", "p", "q", "d", "v", "o", "t", "x"]
  payload: List[str]

  def to_array(self) -> List[str]:
    return [self.tag] + self.payload


# -----------------------------
# Strongly-typed line models
# -----------------------------


RefStr = Annotated[str, Field(pattern=r"^M#\d+$")]


class _BaseLine(BaseModel):
  def to_array(self) -> List[str]:  # pragma: no cover - overridden
    raise NotImplementedError


class RoleLine(_BaseLine):
  tag: Literal["r"] = "r"
  role: Literal["M", "W", "C"]

  def to_array(self) -> List[str]:
    return ["r", self.role]


class GoalLine(_BaseLine):
  tag: Literal["g"] = "g"
  text: str

  def to_array(self) -> List[str]:
    return ["g", self.text]


class FactLine(_BaseLine):
  tag: Literal["f"] = "f"
  text: str
  ref: Optional[RefStr] = None  # Optional inline M# pointer

  def to_array(self) -> List[str]:
    arr = ["f", self.text]
    if self.ref:
      arr.append(self.ref)
    return arr


class AssumptionLine(_BaseLine):
  tag: Literal["u"] = "u"
  text: str

  def to_array(self) -> List[str]:
    return ["u", self.text]


class PlanLine(_BaseLine):
  tag: Literal["p"] = "p"
  text: str

  def to_array(self) -> List[str]:
    return ["p", self.text]


class QuestionLine(_BaseLine):
  tag: Literal["q"] = "q"
  to: Literal["M", "W", "C"]
  text: str

  def to_array(self) -> List[str]:
    return ["q", self.to, self.text]


class DeltaRefLine(_BaseLine):
  tag: Literal["d"] = "d"
  ref: RefStr

  def to_array(self) -> List[str]:
    return ["d", self.ref]


class VerdictLine(_BaseLine):
  tag: Literal["v"] = "v"
  verdict: Literal["A", "R", "E"]  # Accept / Revise / Escalate

  def to_array(self) -> List[str]:
    return ["v", self.verdict]


class OverflowLine(_BaseLine):
  tag: Literal["o"] = "o"
  summary: str
  ref: RefStr
  method: Literal["extractive", "llmlingua"] = "extractive"

  def to_array(self) -> List[str]:
    return ["o", self.summary, self.ref, self.method]


class FreeTextLine(_BaseLine):
  tag: Literal["t"] = "t"
  text: str

  def to_array(self) -> List[str]:
    return ["t", self.text]


class ContextLine(_BaseLine):
  tag: Literal["x"] = "x"
  key: str
  value: str

  def to_array(self) -> List[str]:
    return ["x", self.key, self.value]


AnyTypedLine = Union[
  RoleLine,
  GoalLine,
  FactLine,
  AssumptionLine,
  PlanLine,
  QuestionLine,
  DeltaRefLine,
  VerdictLine,
  OverflowLine,
  FreeTextLine,
  ContextLine,
]


# -----------------------------
# Converters
# -----------------------------


def terse_to_specific(lines: List[TerseTalkLine]) -> List[AnyTypedLine]:
  """
  Convert generic TerseTalkLine objects into specific typed models.
  Unknown shapes are mapped to FreeTextLine.
  """
  out: List[AnyTypedLine] = []
  for ln in lines:
    t = ln.tag
    pl = ln.payload or []
    try:
      if t == "r":
        out.append(RoleLine(role=(pl[0] if pl else "M")))
      elif t == "g":
        out.append(GoalLine(text=(pl[0] if pl else "")))
      elif t == "f":
        text = pl[0] if pl else ""
        ref = pl[1] if len(pl) > 1 else None
        out.append(FactLine(text=text, ref=ref))
      elif t == "u":
        out.append(AssumptionLine(text=(pl[0] if pl else "")))
      elif t == "p":
        out.append(PlanLine(text=(pl[0] if pl else "")))
      elif t == "q":
        to = pl[0] if pl else "W"
        text = pl[1] if len(pl) > 1 else ""
        out.append(QuestionLine(to=to, text=text))
      elif t == "d":
        out.append(DeltaRefLine(ref=(pl[0] if pl else "M#1")))
      elif t == "v":
        out.append(VerdictLine(verdict=(pl[0] if pl else "A")))
      elif t == "o":
        summary = pl[0] if pl else ""
        ref = pl[1] if len(pl) > 1 else "M#1"
        method = pl[2] if len(pl) > 2 else "extractive"
        out.append(OverflowLine(summary=summary, ref=ref, method=method))  # type: ignore[arg-type]
      elif t == "t":
        out.append(FreeTextLine(text=(pl[0] if pl else "")))
      elif t == "x":
        key = pl[0] if pl else ""
        val = pl[1] if len(pl) > 1 else ""
        out.append(ContextLine(key=key, value=val))
      else:
        out.append(FreeTextLine(text=json.dumps(ln.to_array())))
    except ValidationError:
      out.append(FreeTextLine(text=json.dumps(ln.to_array())))
  return out


def lines_to_jsonl(lines: List[AnyTypedLine]) -> str:
  """Emit newline-delimited JSON (canonical array form)."""
  return "\n".join(json.dumps(l.to_array(), ensure_ascii=False) for l in lines)


# -----------------------------
# Structured generation harness
# -----------------------------


class EchoGenerator:
  """
  Offline-safe, deterministic stub for CI and tests.
  """

  def generate(self, goal: str, facts: List[str], question: str) -> List[AnyTypedLine]:
    lines: List[AnyTypedLine] = [
      RoleLine(role="M"),
      GoalLine(text=goal),
    ]
    for f in facts:
      lines.append(FactLine(text=f))
    lines.append(QuestionLine(to="W", text=question))
    return lines


class InstructorGenerator:
  """
  Optional: requires 'instructor' and 'openai' installed and a server
  speaking the OpenAI API (e.g., Ollama at http://localhost:11434/v1).
  """

  def __init__(
    self,
    model: str = "mistral",
    base_url: str = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    api_key: str = os.environ.get("OPENAI_API_KEY", "ollama"),
    max_retries: int = 2,
  ) -> None:
    try:
      import instructor  # type: ignore
      from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover - import guarded
      raise RuntimeError(
        "InstructorGenerator requires 'instructor' and 'openai' to be installed."
      ) from e

    self._instructor = instructor
    self._OpenAI = OpenAI
    self.model = model
    self.base_url = base_url
    self.api_key = api_key
    self.max_retries = max_retries
    self._client = self._instructor.patch(
      self._OpenAI(base_url=self.base_url, api_key=self.api_key)
    )

  def generate(self, system_prompt: str, user_prompt: str) -> List[AnyTypedLine]:
    """
    Ask the model to produce a list of generic TerseTalkLine objects,
    then convert to specific typed lines.
    """
    result: List[TerseTalkLine] = self._client.chat.completions.create(
      model=self.model,
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
      ],
      response_model=List[TerseTalkLine],  # Typed magic by Instructor
      max_retries=self.max_retries,
    )
    return terse_to_specific(result)

