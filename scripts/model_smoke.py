from __future__ import annotations

import argparse
import json
import os

from tersetalk.model_io import ModelClient, EchoModel, ModelCfg, dump_jsonl

SYSTEM_JSONL = (
  "You output compact, typed JSONL lines for the TerseTalk protocol. "
  "Each line is an array: ['r'|'g'|'f'|'u'|'p'|'q'|'d'|'v'|'o'|'t'|'x', ...]. "
  "Return 2-4 lines that set a role and a short goal."
)
USER_JSONL = "Create a manager role and a concise subgoal about comparing two dates."

SYSTEM_TEXT = "You are a concise assistant."
USER_TEXT = "Say one short sentence about efficiency."


def main() -> None:
  ap = argparse.ArgumentParser(description="PR-05: Model I/O smoke tool")
  ap.add_argument("--mode", choices=["echo", "real"], default=os.environ.get("MODEL_CLIENT", "echo"))
  ap.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
  ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "mistral"))
  ap.add_argument("--api-key", default=os.environ.get("OLLAMA_API_KEY", "ollama"))
  args = ap.parse_args()

  if args.mode == "real":
    client = ModelClient(ModelCfg(base_url=args.base_url, api_key=args.api_key, model=args.model))
  else:
    client = EchoModel()

  # Structured call
  try:
    lines = client.call_jsonl_strict(SYSTEM_JSONL, USER_JSONL, max_tokens=200)
    jsonl = dump_jsonl(lines)
  except Exception as e:
    lines = []
    jsonl = f"<error: {e}>"

  # Free-form call
  try:
    text = client.call_text(SYSTEM_TEXT, USER_TEXT, max_tokens=50)
  except Exception as e:
    text = f"<error: {e}>"

  out = {
    "mode": args.mode,
    "structured_lines": [{"tag": ln.tag, "payload": ln.payload} for ln in lines],
    "structured_jsonl": jsonl,
    "freeform_text": text,
  }
  print(json.dumps(out, indent=2))


if __name__ == "__main__":
  main()

