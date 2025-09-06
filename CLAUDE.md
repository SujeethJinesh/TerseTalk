# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TerseTalk is a Python-first repository with Make-based tooling for dev tasks. Our work aligns with RESEARCH_PROPOSAL.md, focusing on a compact, typed JSONL inter-agent protocol (v0.5) and a scoped topology extension (v1.0).

## Development Setup

### Python Environment

- Create a virtual environment: `python -m venv .venv`
- Install dev deps via Make: `make setup`
- Run tests: `make test`
- Lint/format: `make lint` / `make fmt`
  - Target Python version: 3.12 (Makefile auto-detects `python3.12` or falls back to `python3`).

### Dependencies

- Tools are defined via `pyproject.toml` and installed by `make setup` (pytest, ruff, black). Prefer suggesting changes to these over adding new tools.
- Research (v0.5) minimal deps that may be added per PR: `datasets`, `llmlingua==0.2.1`, `pytest-timeout`.
- Optional deps (guarded, auto-disabled if missing): `bert-score` + `torch`, `tiktoken`.

## Common Commands

### Testing

- Primary runner is `pytest` via `make test`.

### Linting and Formatting

- Use `ruff check .` and `black .` (or `make lint` / `make fmt`). Keep style advice practical and minimal.

## Project Structure

- Code under `src/`; tests under `tests/`; assets under `assets/`; scripts under `scripts/`; docs under `docs/`.
 - For research modules, place code under `src/tersetalk/` mirroring proposal layout (protocol_jsonl.py, memory.py, model_io.py, star_runner.py, baselines.py, datasets.py, metrics.py, logging_utils.py). Mirror in `tests/`.

## Review Protocol (for Claude)

- Objectives: review for correctness, clarity, minimalism, and alignment with the field guide, repo conventions, and RESEARCH_PROPOSAL.md goals.
- Output:
  - Brief “Review” noting strengths per file.
  - “Minor suggestions” list (max ~5 bullets) of small, actionable improvements.
  - If fully acceptable, end with exactly: `Approved: no nits.` (case-sensitive)
- Scope:
  - Do not recommend heavy tooling, broad refactors, or speculative defenses unless requested.
  - Favor Makefile/config tweaks, tiny doc fixes, and focused code nits.
- Research-specific checks:
  - Confirm PR scope aligns with `RESEARCH_PROPOSAL.md` (≤250 LOC where feasible; includes context, before→after, public APIs, tests, DoD). Explicitly call out alignment in your verdict.
  - Confirm minimal deps usage; optional deps are guarded; no unnecessary tool sprawl.
  - Check metrics hooks or placeholders align with the evaluation plan (Quality vs Tokens, SP optional, latency, overflow/memory stats).
  - Always read `@RESEARCH_PROPOSAL.md` relevant section first and review in its spirit.
  - Evaluate any real-run snippets included: assess honesty, compare against baselines, and provide a short PI status note on whether results seem on track with figures of merit.
- Referencing code:
  - The agent sends `File: <path>\n---\n<content>` blocks only for files changed.
  - Avoid echoing full files back; refer to filenames and line numbers when helpful.
- Re-review:
  - When prompted “Re-review after applying your suggestions,” check changes, then either add concise follow-ups or reply exactly `Approved: no nits.`

### For PR-00 — Reproducibility Infrastructure

- Validate that `tersetalk/reproducibility.py` seeds Python, optional NumPy/Torch, and env vars as specified; returns the defaults dict.
- Ensure tests pass without NumPy/Torch installed and cover same/different seed behavior and env var setting.
- Require evidence in PR body: pytest summary and `scripts/repro_smoke.py` JSON output.

### Environment usage

- Always assume a local `.venv` is used (`source .venv/bin/activate`). Prefer commands that respect the active venv.

### For PR-01 — Repository Scaffold & CLI Skeleton

- Validate that `make install` installs runtime + dev deps (requirements.txt + requirements-dev.txt).
- Confirm `scripts/run_v05.py` prints `--help`, `--version`, and that `--dry-run` returns a JSON config including `defaults` from `set_global_seed`.
- Ensure PR‑00 files and tests remain intact and passing.
- Confirm runtime deps minimality (click, tqdm); no heavy tooling added.

## Notes

- The .gitignore is configured for Python projects with comprehensive exclusions for common Python artifacts
- VSCode settings include ChatGPT extension configuration

THE MAKE IT WORK FIRST FIELD GUIDE

CORE TRUTH
Defensive code before functionality is theater.
Prove it works. Then protect it.

THE RULES

1. Build the Happy Path First – Code that DOES the thing
2. No Theoretical Defenses – Naked first version
3. Learn from Real Failures – Fix reality, not ghosts
4. Guard Only What Breaks – Add checks only for facts
5. Keep the Engine Visible – Action, not paranoia

ANTI-PATTERNS TO BURN
❌ Fortress Validation
❌ Defensive Exit Theater
❌ Connection State Paranoia

PATTERNS TO LIVE BY
✅ Direct Execution
✅ Natural Failure
✅ Continuous Progress

THE TEST
Can someone grok your code in 10 seconds?
YES → You lived the manifesto
NO → Delete defenses

THE PROMISE
Readable. Debuggable. Maintainable. Honest.

THE METAPHOR
Don’t bolt on airbags before the engine runs.
First: make it move.
Then: guard against real crashes.

MAKE IT WORK FIRST.
MAKE IT WORK ALWAYS.
GUARDS EARN THEIR KEEP.

# CHANGES BETWEEN ORIGINAL MANIFESTO AND FIELD GUIDE

## Core Truth

- Original: "Every line of defensive code you write before proving your feature works is a lie..."

* Field Guide: "Defensive code before functionality is theater. Prove it works. Then protect it."
  (Phrased shorter, sharper, no metaphor drift.)

## Philosophy / Rules

- Original had 5 sections with long explanations (e.g. “Write code that does the thing. Not checks...”)

* Field Guide reduced to 5 short rules, one line each.
  (Compression: removed repetition, slogans instead of prose.)

## Anti-Patterns

- Original: Full code samples showing Fortress Validation, Defensive Exit Theater, Connection State Paranoia.

* Field Guide: Only names listed with ❌ icons.
  (Removed examples for poster readability.)

## Patterns We Embrace

- Original: Full code samples for Direct Execution, Natural Failure, Continuous Progress.

* Field Guide: Only names listed with ✅ icons.
  (Same compression—patterns as mantras.)

## Mindset Shift

- Original: "From: X → To: Y" contrasts across multiple lines.

* Field Guide: Removed section entirely.
  (The core shift is implied by the rules; stripped for brevity.)

## The Path

- Original: 5 steps (Write It, Run It, Break It, Guard It, Ship It).

* Field Guide: Removed entirely.
  (Field guide favors slogans, not process.)

## The Test

- Original: "Can someone read your code and understand what it does in 10 seconds?"

* Field Guide: "Can someone grok your code in 10 seconds?"
  (Simplified, kept essence.)

## The Promise

- Original: Bullet list: Readable, Debuggable, Maintainable, Honest (with explanations).

* Field Guide: "Readable. Debuggable. Maintainable. Honest."
  (Compressed into a chant-like line.)

## The Metaphor

- Original: "Don’t add airbags to a car that doesn’t have an engine yet..."

* Field Guide: "Don’t bolt on airbags before the engine runs. First: make it move. Then: guard against real crashes."
  (Metaphor shortened, same spirit.)

## Call to Action

- Original: Long section: "Stop writing code that apologizes... Stop defending... Stop hiding..."

* Field Guide: 3 bold lines: "MAKE IT WORK FIRST. MAKE IT WORK ALWAYS. GUARDS EARN THEIR KEEP."
  (Stripped to rallying cry.)

# Reviewing code

You must ensure that code you review follows the above principles

### For PR-06 — Dataset Adapters

- Validate `@tersetalk/datasets.py` provides `load_hotpotqa()` and `load_gsm8k()` returning examples with keys: question, answer (string), facts (list[str]), subgoal (<= ~64 chars), assumptions (list[str]).
- Confirm deterministic sampling by seed and offline‑first synthetic fallback when `TERSETALK_OFFLINE_DATA=1` or datasets missing.
- Check `@tests/test_datasets.py` covers determinism, schema, and offline behavior; optional real smoke gated by `RUN_REAL_DATASETS=1`.
- Smoke: `@scripts/datasets_smoke.py` prints two samples per task with `offline` flag; include JSON in PR description.
- If fully acceptable, end with exactly: `Approved: no nits.`
### For PR-07 — Pipeline Runner (Manager→Worker→Critic)

- Validate `@tersetalk/pipeline_runner.py` provides:
  - `PipelineConfig`, `build_manager_message`, `prepare_critic_input`, `extract_answer`, `extract_verdict`, `run_pipeline_once`, `run_pipeline_batch`.
  - Uses `JSONLValidator` + `MemoryStore`; optional `ProtocolHandler` path guarded.
  - Returns rich result dict with answer, verdict, tokens_total, overflow_count, density, latency_ms, sp_reference, manager/worker/critic JSONL, and memory_stats.
- Check `@tests/test_pipeline_runner.py` covers schema and density behavior (loose vs tight caps) and determinism with Echo model + synthetic data.
- Smoke: `@scripts/pipeline_smoke.py` prints one result JSON for Echo + synthetic tasks; include loose vs tight caps snippets in PR body.
- Ensure reference to `@RESEARCH_PROPOSAL.md` (pipeline section) guides the review for alignment and scope.
- If fully acceptable, end with exactly: `Approved: no nits.`

### For PR-08 — Baselines (Free‑form & LLMLingua)

- Validate `@tersetalk/baselines.py` provides:
  - `build_freeform_prompt`, `approx_token_count`, `run_freeform_once`, `run_llmlingua_once`.
  - LLMLingua path import‑guarded; honors `TERSETALK_DISABLE_LL2=1` to force fallback in CI.
  - Returns dict with: answer, prompt, response, prompt_tokens, response_tokens, tokens_total, used_llmlingua, origin_tokens, compressed_tokens, compression_ratio.
- Check `@tests/test_baselines.py` covers prompt content, schema, deterministic Echo, and LL2 disabled fallback.
- Smoke: `@scripts/baselines_smoke.py` prints both baselines; include JSON snippets in PR body. Ensure review is aligned with @RESEARCH_PROPOSAL.md.
- If fully acceptable, end with exactly: `Approved: no nits.`
