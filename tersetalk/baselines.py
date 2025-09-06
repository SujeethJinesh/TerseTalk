from __future__ import annotations

import os
from typing import Dict, Optional

from tersetalk.model_io import ModelClient


# ---- Utilities ----


def approx_token_count(text: str) -> int:
  """4 chars ≈ 1 token fallback. Non-negative."""
  if not isinstance(text, str):
    return 0
  return max(0, (len(text) + 3) // 4)


def build_freeform_prompt(example: Dict) -> str:
  """
  Construct a concise, readable free-form prompt from a normalized example:
  {question, answer, facts, subgoal, assumptions}
  Note: We retain a lone 'Question:' line even if the following line is empty
  to keep the overall prompt shape consistent across examples.
  """
  question = str(example.get("question", "")).strip()
  subgoal = str(example.get("subgoal", "")).strip()
  facts = [str(f) for f in (example.get("facts", []) or [])][:10]
  assumptions = [str(a) for a in (example.get("assumptions", []) or [])][:5]

  facts_block = ""
  if facts:
    facts_block = "Facts:\n" + "\n".join(f"- {f}" for f in facts)

  assumptions_block = ""
  if assumptions:
    assumptions_block = "Assumptions:\n" + "\n".join(f"- {a}" for a in assumptions)

  # Keep compact and deterministic; baseline expects plain text.
  prompt = f"""Role: Manager

Goal: {subgoal or "Answer the user's question correctly and concisely."}
{facts_block}

{assumptions_block}

Question:
{question}

Instructions:
Provide only the final answer as one or two short sentences.
"""
  # Remove empty lines except allow lone 'Question:' if needed
  return "\n".join(ln for ln in prompt.splitlines() if ln.strip() or ln == "Question:")


# ---- Baseline runners ----


def run_freeform_once(example: Dict, client: ModelClient, max_tokens: int = 256) -> Dict:
  """
  Free-form (no compression) baseline.
  Uses ModelClient.call_text to obtain a plain-text response.
  """
  prompt = build_freeform_prompt(example)
  system = "You are a helpful, concise assistant."
  response = client.call_text(system=system, user_prompt=prompt, max_tokens=max_tokens)

  prompt_tokens = approx_token_count(prompt)
  response_tokens = approx_token_count(response)
  return {
    "answer": response.strip(),
    "prompt": prompt,
    "response": response,
    "prompt_tokens": int(prompt_tokens),
    "response_tokens": int(response_tokens),
    "tokens_total": int(prompt_tokens + response_tokens),
    "used_llmlingua": False,
    "origin_tokens": None,
    "compressed_tokens": None,
    "compression_ratio": None,
    "origin_prompt": prompt,
    "compressed_prompt": None,
  }


def run_llmlingua_once(example: Dict, client: ModelClient, target_token: int = 400, max_tokens: int = 256) -> Dict:
  """
  Free-form + LLMLingua baseline.
  - If LLMLingua is unavailable or disabled (TERSETALK_DISABLE_LL2=1), falls back to
    uncompressed prompt but still returns the schema with used_llmlingua=False and
    None for compression fields.
  """
  prompt = build_freeform_prompt(example)
  system = "You are a helpful, concise assistant."

  # Honor env switch to make CI deterministic
  if os.environ.get("TERSETALK_DISABLE_LL2", "0") == "1":
    response = client.call_text(system=system, user_prompt=prompt, max_tokens=max_tokens)
    pt = approx_token_count(prompt)
    rt = approx_token_count(response)
    return {
      "answer": response.strip(),
      "prompt": prompt,
      "response": response,
      "prompt_tokens": int(pt),
      "response_tokens": int(rt),
      "tokens_total": int(pt + rt),
      "used_llmlingua": False,
      "origin_tokens": None,
      "compressed_tokens": None,
      "compression_ratio": None,
      "origin_prompt": prompt,
      "compressed_prompt": None,
    }

  try:
    from llmlingua import PromptCompressor  # type: ignore

    compressor = PromptCompressor()
    comp = compressor.compress(prompt, target_token=int(target_token))

    compressed_prompt = comp.get("compressed_prompt") or comp.get("compressed_text") or prompt
    origin_tokens = int(comp.get("origin_tokens") or approx_token_count(prompt))
    compressed_tokens = int(comp.get("compressed_tokens") or approx_token_count(compressed_prompt))
    ratio = float(comp.get("ratio") or (compressed_tokens / max(1, origin_tokens)))

    response = client.call_text(system=system, user_prompt=compressed_prompt, max_tokens=max_tokens)
    resp_tokens = approx_token_count(response)
    return {
      "answer": response.strip(),
      "prompt": compressed_prompt,
      "response": response,
      "prompt_tokens": int(compressed_tokens),
      "response_tokens": int(resp_tokens),
      "tokens_total": int(compressed_tokens + resp_tokens),
      "used_llmlingua": True,
      "origin_tokens": origin_tokens,
      "compressed_tokens": compressed_tokens,
      "compression_ratio": ratio,
      "origin_prompt": prompt,
      "compressed_prompt": compressed_prompt,
    }
  except Exception:
    # Graceful fallback if library missing or runtime error
    response = client.call_text(system=system, user_prompt=prompt, max_tokens=max_tokens)
    pt = approx_token_count(prompt)
    rt = approx_token_count(response)
    return {
      "answer": response.strip(),
      "prompt": prompt,
      "response": response,
      "prompt_tokens": int(pt),
      "response_tokens": int(rt),
      "tokens_total": int(pt + rt),
      "used_llmlingua": False,
      "origin_tokens": None,
      "compressed_tokens": None,
      "compression_ratio": None,
      "origin_prompt": prompt,
      "compressed_prompt": None,
    }
