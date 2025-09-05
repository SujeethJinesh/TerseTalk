# Repository Guidelines

## Project Structure & Module Organization

- Current repo is minimal: `README.md`, `LICENSE`, and contributor docs. Place new code under `src/`, tests under `tests/`, assets under `assets/`, scripts under `scripts/`, and additional docs under `docs/`.
- Keep modules small and cohesive. Mirror `src/` structure in `tests/` (e.g., `src/core/parser.ts` → `tests/core/parser.spec.ts`).

## Build, Test, and Development Commands

- No build system is defined yet. When introducing tooling, prefer Make targets for a consistent DX:
  - `make setup`: install dependencies
  - `make lint`: run formatters/linters
  - `make test`: run the test suite
  - `make dev`: run local app/CLI
- Example: `make test` should run your project’s primary test runner.

## Coding Style & Naming Conventions

- Indentation: 2 spaces. Line length: 100 chars target.
- Files/dirs: `kebab-case/` for folders and scripts; code files follow language norms (`snake_case.py`, `kebab-case.ts`).
- Naming: Classes `CamelCase`; functions/vars `snake_case` (Python) or `camelCase` (JS/TS).
- Formatting/Linting: adopt language standards (Python: Black + ruff; JS/TS: Prettier + ESLint). Include config in the repo.

## Testing Guidelines

- Tests live in `tests/`, mirroring `src/`. Name tests `test_*.py` (Python) or `*.spec.ts` (TS/JS).
- Aim for ≥80% line coverage; include edge cases and error paths. Fixtures go in `tests/fixtures/`.
- Provide a single entry command (e.g., `pytest` or `npm test`) wired to `make test` if using Make.

## Commit & Pull Request Guidelines

- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`.
- PRs: clear description, linked issues, reproduction/verification steps, and screenshots for UI changes. Update docs when behavior changes.
- Keep PRs focused and reasonably small; add tests and pass CI before requesting review.

## Security & Configuration Tips

- Never commit secrets. Use `.env` and provide `.env.example` with safe defaults.
- Pin dependencies and document required versions in `README.md`.

## Agent-Specific Notes

- Prefer minimal, focused diffs; follow existing patterns. If you add tooling, include Make targets and update `README.md` and this guide as needed.

## Research Alignment (with RESEARCH_PROPOSAL.md)

- North Star: Implement and evaluate the TerseTalk-JSONL protocol (v0.5) and a scoped topology extension (v1.0). Keep every implementation unit ≤250 LOC with clear before→after, public APIs, tests, and a Definition of Done (DoD).
- Code placement: implement new modules under `src/tersetalk/` mirroring the proposal’s layout (protocol_jsonl.py, memory.py, model_io.py, star_runner.py, baselines.py, datasets.py, metrics.py, logging_utils.py). Mirror paths in `tests/`.
- PR discipline:
  - ≤250 LOC per PR where feasible; tightly scoped.
  - Include: context, before→after, public APIs, tests, DoD checklist.
  - Avoid heavy frameworks; stick to proposal’s minimal deps.
- Dependencies:
  - Minimal now: `pytest`, `pytest-timeout`, `ruff`, `black`. Research tasks may also need `datasets` and `llmlingua==0.2.1` per v0.5.
  - Optional (auto-disabled if missing): `bert-score` + `torch`, `tiktoken`. Add guards so absence downgrades to fallback metrics (e.g., Jaccard, len/4 tokens).
- Evaluation targets (v0.5):
  - Datasets: HotpotQA and GSM8K subsets.
  - Comparisons: TerseTalk-JSONL, Free-form, Free-form + LLMLingua/‑2.
  - Metrics: Quality vs Tokens (Pareto), semantic preservation (BERTScore optional), (de)serialization latency, overflow/memory stats.
- Process: prefer a single CLI runner and CSV outputs; keep UX simple and reproducible; make each unit runnable independently.

## Claude Code Review Protocol

- Preconditions:
  - Ensure local checks pass: `make fmt && make lint && make test`.
  - For PR-00 specifically, run `make install && make test && make smoke` and include smoke JSON + pytest summary in the PR.
  - Claude CLI must work with Node LTS (>=18). On macOS with Homebrew:
    - `brew install node@20` and add `export PATH="/usr/local/opt/node@20/bin:$PATH"`.
  - Export `ANTHROPIC_API_KEY` in your shell.
  - Quick sanity check: `echo "pong" | claude -p --dangerously-skip-permissions --model opus` ⇒ outputs `pong`.

- Prompt template (reference-only; do not paste file contents):
  - Start with a 1–2 sentence goal and explicit ask for a verdict.
  - Explicitly instruct Claude to read `@RESEARCH_PROPOSAL.md` (relevant PR section) first.
  - List edited files using `@` references only (e.g., `@tersetalk/reproducibility.py`, `@tests/test_reproducibility.py`).
  - End with: `If fully acceptable with no nits, explicitly reply: 'Approved: no nits.'`

- Recommended command shape:
  - `( printf "<context + ask with @file refs only>\n" ) | claude -p --dangerously-skip-permissions --model opus`

- Iteration loop (multi-round, required):
  - Self-review first: ensure changes compile, tests pass, and scope aligns with the proposal. Keep diffs minimal.
  - Request Claude review using @file references only; instruct Claude to read `@RESEARCH_PROPOSAL.md` first.
  - If Claude has comments: evaluate necessity. Implement minimal changes or respond with rationale for declining.
  - After any changes: re-run `make test` (and `make smoke` when applicable).
  - Re-run Claude review with a brief “Re-review after applying suggestions” note.
  - Repeat until Claude replies exactly: `Approved: no nits.` and confirms proposal alignment.
  - Perform a final self-review: confirm all tests pass and code quality meets your standard before requesting human review.
  - For research features, ask Claude to explicitly confirm alignment with `RESEARCH_PROPOSAL.md` (PR scope, metrics, minimal deps, and DoD). Always reference the relevant proposal section in your prompt.

### Always reference RESEARCH_PROPOSAL.md

- When requesting Claude review, start by citing the applicable section(s) in `RESEARCH_PROPOSAL.md` and ask Claude to judge alignment with that scope and spirit.
- Keep diffs small (≤250 LOC/unit) and include DoD in the PR body.

### Claude Prompt Formatting

- In the prompt preface, reference files using `@<path>` style (e.g., `@RESEARCH_PROPOSAL.md`, `@tersetalk/reproducibility.py`) to aid context.
- Explicitly instruct Claude to read `@RESEARCH_PROPOSAL.md` first (focus on the relevant PR section) before reviewing the changed files.
- Still send content blocks only for the files you edited using the `File: <path>\n---\n<content>` format.

- Scope and pushback:
  - It’s fine to push back on large or out-of-scope requests; explain constraints and propose a minimal alternative, then ask Claude to confirm.
  - Do not introduce heavy tools or broad refactors unless explicitly requested by the user.

- Hygiene:
  - Never pipe secrets or `.env` contents to Claude.
  - Reference only the specific files you edited with `@path` (no file bodies).
  - It's expected that Claude fetches referenced files; avoid pasting long excerpts.

## Troubleshooting Claude CLI

- CLI prints a Node stack trace on start:
  - Install Node LTS and ensure it’s first in `PATH` (see above), then retry.
- Auth errors from Claude:
  - Ensure `ANTHROPIC_API_KEY` is exported and valid.
- Permission prompts in sandbox:
  - Use `--dangerously-skip-permissions` as this environment is sandboxed.

## Environment (Python 3.12 + venv)

- This project targets Python 3.12. Ensure it’s installed (macOS/Homebrew):
  - `brew install python@3.12`
  - Optionally add: `export PATH="/usr/local/opt/python@3.12/bin:$PATH"`
- Create venv and install dev tools:
  - `make setup` (auto-detects Python 3.12; falls back to `python3` if unavailable)
- Validate:
  - `make fmt && make lint && make test`

Always use the local `.venv` when running any commands. Activate with `source .venv/bin/activate` before `make` if the Makefile does not manage activation for you.

## PR Summaries (Running Log)

Record a brief summary after each PR is merged to accelerate context-loading in new sessions. Include acceptance evidence and next steps.

- PR-00 — Reproducibility Infrastructure:
  - Summary: Adds `tersetalk/reproducibility.py` with `set_global_seed`, fingerprint helpers, tests, and a smoke script. Optional NumPy/Torch guarded.
  - Evidence: pytest passed locally (7 passed); smoke JSON: `same_seed_equal: true`, `diff_seed_unequal: true`. Full JSON and pytest summary included in PR body.
  - Next: Proceed to PR-01 per `RESEARCH_PROPOSAL.md`.

- PR-01 — Repository Scaffold & CLI Skeleton:
  - Summary: Adds `.gitignore`, `requirements.txt`, package versioning (`tersetalk/_version.py`), updates `pyproject.toml` to 0.0.2, introduces Click-based CLI at `scripts/run_v05.py`, and adds CLI tests. Preserves PR‑00.
  - Evidence: pytest passed locally (all tests green); `--help` and `--version` print; `--dry-run` emits JSON with defaults from `set_global_seed`. See PR body for copies of outputs.
  - Next: Proceed to PR-02 per `RESEARCH_PROPOSAL.md`.

- PR-02 — JSONL Protocol & Validator:
  - Summary: Adds `tersetalk/protocol_jsonl.py` with `JSONLValidator` for detection, normalization, caps + overflow (M# pointers, ["o", ...] lines), density metric, and a `jsonl_to_prose` helper; includes `scripts/jsonl_guard.py` and tests.
  - Evidence: pytest all green; `jsonl_guard.py` outputs JSON showing mixed_format, stats {overflow: {count, per_tag, rate}, density}, and normalized+overflowed `out`. See PR body for snippets.
  - Next: Proceed to PR-02S/PR-03 per `RESEARCH_PROPOSAL.md`.

- PR-02S — Structured Output with Instructor:
  - Summary: Adds `tersetalk/structured.py` with typed Pydantic models for all tags, generic `TerseTalkLine`, converters, `EchoGenerator` (offline), `InstructorGenerator` (guarded). Adds `scripts/structured_demo.py` and tests.
  - Evidence: pytest all green; Echo demo JSON shows `compliance_ok: true`, canonical `lines_jsonl`, and validator stats. Runtime deps pinned to: pydantic==2.11.7, instructor==1.11.2, openai==1.106.1.
  - Next: Proceed to PR-03 MemoryStore per `RESEARCH_PROPOSAL.md`.

- PR-03 — Memory Store:
  - Summary: Adds `tersetalk/memory.py` with bounded `MemoryStore` (MAX_ENTRIES=10_000), oldest-by-last-access eviction, deterministic M# id minting, and `put/get/reset/stats`. Includes `scripts/memory_smoke.py` and tests validating eviction, id pattern, reset, stats, and validator integration.
  - Evidence: pytest all green; smoke shows minted_ids, fetched echo mapping, and validator run with `o_refs_retrievable: true` plus non-zero overflow stats and density ∈ [0,1].
  - Next: Proceed to PR-04 Summarizer per `RESEARCH_PROPOSAL.md`.

- PR-04 — Summarization Module:
  - Summary: Adds `tersetalk/summarization.py` with Summarizer (extractive default; optional llmlingua), integrates into `JSONLValidator` so overflow summaries use the summarizer and record the method. Adds `scripts/summarize_smoke.py`, updates `scripts/jsonl_guard.py` with `--summarizer`, and adds tests.
  - Evidence: pytest all green; summarizer smoke shows token reduction; guard outputs `o` lines with `method` field; validator interop verified.
  - Next: Proceed to PR‑H1 Hybrid Gate per `RESEARCH_PROPOSAL.md`.

- PR‑H1 — Hybrid Gate:
  - Summary: Adds `tersetalk/hybrid_gate.py` with `GateCfg`, token estimator, optional llmlingua projection (import-guarded), and `gate_choose_protocol`. Extends `scripts/run_v05.py` dry‑run with optional gate preview and adds `scripts/hybrid_gate_smoke.py`. Tests cover routing and estimator.
  - Evidence: pytest all green; smoke shows `route`, `est_tokens`, and `notes`; env var `TERSETALK_FAKE_LL2_COMPRESS` demo routes to `freeform_llmlingua`; dry‑run output includes `gate` object when probes provided.
  - Next: Proceed to pipeline/evaluation PRs per `RESEARCH_PROPOSAL.md`.

- PR‑H2 — Calibration Sweep:
  - Summary: Adds `tersetalk/calibration.py` and `scripts/calibrate_caps.py` to sweep {caps, summarizer, deref_policy (placeholder), gate on/off, token_budget} on a synthetic shard; selects best per policy and writes `configs/calibration.yaml` (JSON-as-YAML). Deterministic, stdlib-only; Hybrid Gate consulted when enabled.
  - Evidence: pytest all green; CLI prints compact summary (best_spec + best_metrics) and writes `configs/calibration.yaml`. Gate effect verified via `TERSETALK_FAKE_LL2_COMPRESS` (routed_freeform_frac > 0 when on).
  - Next: Proceed to pipeline/evaluation PRs per `RESEARCH_PROPOSAL.md`.

- PR‑H3 — Non-Inferiority (paired bootstrap):
  - Summary: Adds `tersetalk/noninferiority.py` with a paired-bootstrap one‑sided non‑inferiority test for d=acc(H)−acc(L) at δ=0.02, plus `scripts/noninferiority_smoke.py` and tests. Stdlib-only, deterministic via seed.
  - Evidence: pytest all green; smoke PASS example shows `noninferior: true` and LB > −δ; smoke FAIL shows `noninferior: false` and LB ≤ −δ. Report includes fields: n, alpha, delta, acc_hybrid, acc_llml, diff, lb_one_sided_95, ci2_lower_95, ci2_upper_95, method, n_boot, seed, noninferior, decision.
  - Next: Proceed to pipeline/evaluation PRs per `RESEARCH_PROPOSAL.md`.

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

# Use Claude Code for Review

You must always do a self-review first, then a Claude Code review, then a final self-review:

1) Self-review: ensure changes compile, lint, test; confirm minimal diffs and alignment with the task.
2) Claude review: use the protocol above and the exact verdict phrase `Approved: no nits.`
3) Final self-review: confirm that Claude’s suggestions were implemented or consciously declined with rationale, and all checks pass.

Always point Claude only to the specific files you are editing (not the whole repo).

You may not necessarily agree with everything that Claude says, you should feel ok to push back on those changes if you think they're invalid. But you have to make sure you address all feedback with Claude and also bring justification to it about why you may not agree.

After that round of review, I want you to finally review with yourself until you LGTM your own code.

All in all, you will have one self review, one review with claude code, and finally a self-review with yourself before ending.

# Alignment

You must always ensure you're aligned with me and check with me on next steps. Write small pieces of code for me and claude to review. You must always ensure you test on real benchmarks or our actual experiments and do not produce your own benchmarks or such.

# You are a researcher

You are writing code to test the minimal viable research proposal.
