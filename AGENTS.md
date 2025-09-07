# Repository Guidelines

## Project Structure & Module Organization

- Current repo is minimal: `README.md`, `LICENSE`, and contributor docs. Place new code under `src/`, tests under `tests/`, assets under `assets/`, scripts under `scripts/`, and additional docs under `docs/`.
- Keep modules small and cohesive. Mirror `src/` structure in `tests/` (e.g., `src/core/parser.ts` → `tests/core/parser.spec.ts`).

### Handoff Quick Start (Structure & Entrypoints)

- Code layout:
  - `tersetalk/`: core modules (protocol_jsonl.py, memory.py, model_io.py, baselines.py, datasets.py, metrics.py, pipeline_runner.py, hybrid_gate.py, protocol_handler.py, calibration.py, noninferiority.py, statistics.py, etc.)
  - `scripts/`: CLIs (run_v05.py, run_evaluation.py, analyze_v05.py, run_significance.py, plus smoke utilities)
  - `tests/`: pytest suite mirroring modules; includes smokes for CLIs and offline determinism
  - `benchmarks/`: microbenchmarks (optional)
  - `results/`: run outputs (ResultsManager writes date-stamped runs and a `latest` pointer)

- Main CLIs:
  - `scripts/run_v05.py`: v0.5 experiment runner (tersetalk/freeform/llmlingua)
  - `scripts/run_evaluation.py`: paper-grade evaluation driver (caps grid + hybrid budgets)
  - `scripts/analyze_v05.py`: analysis (by_run.csv, tree-wide Pareto CSV/PDF, deterministic caps ablation CSV/PDF, provenance enrichment)
  - `scripts/run_significance.py`: bootstrap significance + non-inferiority (numpy-only)

- Real runs (local-first):
  - Default real backend is Ollama (OpenAI-compatible): base_url `http://localhost:11434/v1`, api_key `ollama`, model from `OLLAMA_MODEL` (e.g., `phi:latest`, `llama3.1:8b`).
  - EchoModel is used only for offline smoke/tests; avoid mocks in reported results.

- Quick routine:
  1) `. .venv/bin/activate && make install && make test`
  2) Small local eval (e.g., 3 samples):
     - `export OLLAMA_MODEL="phi:latest"`
     - `python scripts/run_evaluation.py --task hotpotqa --systems tersetalk freeform llmlingua hybrid --n 3 --seed 0 --model $OLLAMA_MODEL --out results/eval_local`
  3) Analyze: `python scripts/analyze_v05.py --indir results/eval_local --outdir results/eval_local/figures`
  4) Significance: `python scripts/run_significance.py --results-dir results/eval_local/<task>/<timestamp>`
  5) Attach figures/JSON to PR as evidence.

### PR Workflow Mandate (copy/paste for each PR)

```
Read through your @AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated @RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the @RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the @AGENTS.md. To ask claude code for a review, you can do something like `echo "Please ensure you understand @RESEARCH_PROPOSAL.md. I am implementing PR-X. Please review the following files I created: @README.md, @xxx. Ensure that my implementation is properly aligned with the goals of the project. Also here is some additional data on the runs I gathered, please critique it as if you're a senior data scientist and ensure that I'm not cheating on the results, lying, or misrepresenting things" | claude -p --dangerously-skip-permissions --model opus`. Please ensure you are indeed calling it like `claude -p --dangerously-skip-permissions --model opus` and ensure you get both the code review, and the data review, and an additional PI review about the state of the project with Claude and yourself. You must be truthful and honest, and address all the points Claude makes (though keep away from making fakes or mocks). If you need to create fakes or mocks for debugging, delete them afterwards so as to not confuse fakes and mocks for actual results. You must only present results from real model runs as much as possible for our project. It's imperative. Ensure that you are going back and incorporating feedback from claude and yourself as necessary. You must continue this loop until both you and claude agree with each other and give yourselfs both approval without nits. After that you should push your changes up for review. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your @AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the @RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the @RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update @AGENTS.md and ask @CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly. Before you merge the PR, you must wait for my approval. Also at the end I want you to outline any risks we are seeing in this project, are our expectations aligned with how the progress is going? Will our project succeed and achieve the baselines we expect using your best judgement? Absolutely make sure you're reporting results truthfully and honestly. Avoid fake, mocked, or other non genuine results. You should also analyze the results we get for each run and determine if they meet our figures of merit, when you report back, it's crucial to include that analysis (e.g. compression amount, failure rate, latency, etc.). We should be aiming to properly fix things and run proper evaluations. If we do have any expected goals or outcomes (e.g. >= 10x on xyz) and they aren't achieved, then explain why, but do not lie or cheat and use drastically contrived inefficient metrics. It's important to generally be comparing to a standard implementation. Please also report the results on real data runs, it's important that we start gathering as much data on real runs as possible now so it can be reviewed by Claude AND YOU. If you see there are problems or optimizations, don't hesitate to suggest them as we collect more data. Here are more detailed instructions for PR implementation.
```

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

### Recent PRs (running log)


- PR‑08B — Baselines knobs & minor nits:
  - Parameterize max_tokens in baselines and smoke; include origin/compressed prompts. Echo determinism unchanged; schema intact.

- PR‑11 — Experiment Driver (scripts/run_v05.py):
  - CLI runs {tersetalk|freeform|llmlingua} on {hotpotqa|gsm8k|synth}; saves config.json, raw_outputs.jsonl, metrics.csv, summary.json via ResultsManager; supports resume and SIGINT-safe flush. Compatible with --help/--version/--dry-run.

- PR‑07 — Pipeline Runner (Manager→Worker→Critic):
  - Adds `tersetalk/pipeline_runner.py` with `PipelineConfig`, message builders, extractors, and `run_pipeline_once/batch` using `MemoryStore` + `JSONLValidator` (optional `ProtocolHandler`).
  - Adds `scripts/pipeline_smoke.py` and `tests/test_pipeline_runner.py` (Echo model + synthetic data; determinism and density/overflow behavior).
  - Evidence: pytest all green; smoke shows loose caps (density≈1.0, low overflow) vs tight caps (lower density, higher overflow).
  - Summary: Echo model yields deterministic outputs; manager message density drops as caps tighten; tokens_total and latency_ms reported.

- PR‑08 — Baselines (Free‑form & Free‑form+LLMLingua):
  - Adds `tersetalk/baselines.py` with `build_freeform_prompt`, `approx_token_count`, `run_freeform_once`, and `run_llmlingua_once` (import‑guarded, env `TERSETALK_DISABLE_LL2=1` to force fallback in CI).
  - Adds `scripts/baselines_smoke.py` and `tests/test_baselines.py` (prompt content, schema checks, deterministic Echo, and LLMLingua disabled fallback).
  - Evidence: pytest all green; offline smoke prints both baselines. With LL2 disabled: `used_llmlingua=false` and compression fields are null.
  - Summary: Baselines return comparable token metrics and are offline‑safe; ready for head‑to‑head with pipeline in future PRs.

## Real Runs & Analysis (ongoing discipline)

- For every PR, include a small real‑run snippet (opt‑in via `RUN_REAL_MODEL=1`) for baselines and, when robust, the pipeline. Report: tokens, latency, and brief qualitative accuracy (e.g., EM snippet or numeric correctness on GSM8K item).
- Default “real model” runs are local via Ollama (OpenAI‑compatible):
  - base_url: `http://localhost:11434/v1`, api_key: `ollama`, model from `OLLAMA_MODEL`.
  - Avoid mocks/fakes; EchoModel is only for offline smoke and unit tests.
  - Preferred quick models locally: `phi:latest` or similar lightweight variants.
- Provide a sober analysis vs figures of merit (e.g., token reduction targets, failure rate, density), call out limitations (heuristic tokens, randomness), and avoid overstated claims.
- If real runs are blocked (e.g., model schema/tooling), document the root cause and the follow‑up PR fixing it.

- PR‑06 — Dataset Adapters (HotpotQA & GSM8K):
  - Adds `tersetalk/datasets.py` with `load_hotpotqa()` and `load_gsm8k()` returning normalized examples: {question, answer, facts, subgoal, assumptions}. Deterministic sampling by seed; offline‑first synthetic shards controlled by `TERSETALK_OFFLINE_DATA=1`.
  - Adds `scripts/datasets_smoke.py` and `tests/test_datasets.py` (determinism, schema, offline behavior; optional real smoke via `RUN_REAL_DATASETS=1`).
  - Evidence: pytest all green; smoke JSON with `"offline": true` showing two samples per task with all five keys.
  - Next: Proceed per RESEARCH_PROPOSAL.md to dataset‑driven evaluation wiring and metrics.

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
  - Explicitly instruct Claude to read `@RESEARCH_PROPOSAL.md` (relevant PR section) first and judge alignment with its spirit.
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

- PR‑H4 — Protocol Handler (LLMLingua touchpoints):
  - Summary: Adds `tersetalk/protocol_handler.py` with `PHConfig` and `ProtocolHandler` implementing three toggleable touchpoints: pre‑overflow LL2 compression, overflow summarization method (llmlingua), and dereference injection with optional LL2 compression; includes counters. Adds `scripts/protocol_handler_smoke.py` and extends `scripts/run_v05.py` with flags and a `--protocol-demo` dry‑run preview.
  - Evidence: pytest all green; smoke shows preoverflow success (no `o` lines, counters.succeeded>0), overflow `o` lines labeled `llmlingua`, and deref injection replacing `d` with `f` plus counters.deref.ll2_compressed>0; run_v05 dry‑run includes protocol_demo.
  - Next: Proceed to pipeline/evaluation PRs per `RESEARCH_PROPOSAL.md`.

- PR‑MB — Microbenchmark Suite:
  - Summary: Adds `benchmarks/` package (MB‑1 tag extraction, MB‑2 streaming boundaries, MB‑3 SerDe/bytes) with a `run_all` runner and tests. Optional msgpack is guard‑checked.
  - Evidence: pytest all green; run_all --fast JSON shows MB‑1 ≥10× vs uncompiled and >1.4× vs compiled, MB‑2 ≥5×, and MB‑3 JSONL ≤70% of verbose free‑form bytes (lean ratio also reported).
  - Next: Proceed to pipeline/evaluation PRs per `RESEARCH_PROPOSAL.md`.

- PR‑14 — Evaluation Driver:
  - Summary: Adds `scripts/run_evaluation.py` to run tersetalk/freeform/llmlingua/hybrid over a caps grid and hybrid budgets; saves per‑system JSONL + summary.json; offline‑safe.
  - Evidence: pytest green; local real runs (Ollama `phi:latest`) with n=3:
    - hotpotqa/freeform → avg_tokens≈455.0, accuracy=0.00, compliance=1.00
    - gsm8k/freeform → avg_tokens≈221.0, accuracy=0.00, compliance=1.00
  - Next: Expand to tersetalk/hybrid on small n; feed outputs to PR‑12 analysis for Pareto/ablation.

- PR-12 — Analysis Scripts:
  - Summary: Adds `scripts/analyze_v05.py` (by_run.csv, Pareto plots + points CSVs, caps ablation figure/CSV) with headless Matplotlib; stdlib + numpy + matplotlib only; smoke test runs Echo+synth and asserts artifacts.
  - Evidence: pytest all green; smoke created `by_run.csv` and `pareto_synth.pdf`; CLI works headless.
  - Next: Populate with HotpotQA/GSM8K runs; consider SP vs bytes-on-wire plot.

- PR-13 — Baselines Parameters & Robustness:
  - Summary: Parameterizes baselines (`max_tokens`, `target_compression`), adds robust error handling with `status` and `error` fields; extends smoke CLI with `--target-compression`; defaults/back-compat preserved.
  - Evidence: pytest all green; Echo smoke prints status=success; injected failing client returns status=error with non-negative tokens.
  - Next: Use knobs in sweeps; surface error counts in summaries/analysis.

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

### PR-12 — Analysis Scripts

- Summary: Adds `scripts/analyze_v05.py` to aggregate PR‑11 run outputs into `by_run.csv`, per‑task Pareto figures and points CSVs, plus an optional caps ablation plot. Headless Matplotlib; stdlib + numpy + matplotlib only.
- Evidence: local pytest smoke passes (`tests/test_analyze_v05_smoke.py` creates synth results via Echo and generates `pareto_synth.pdf` and `by_run.csv`).
- Next: Expand usage on HotpotQA/GSM8K shards and integrate figures into reports; consider adding SP vs bytes plot in a later PR.


- PR-15 — Analysis Polish & Provenance:
  - Summary: Upgrades `scripts/analyze_v05.py` with tree-wide Pareto CSV+PDF, deterministic cap ablations from filename-parsed caps, and provenance enrichment (tokens_method, sp_method, timestamp, git hash). Headless; stdlib+numpy+matplotlib only; keeps prior outputs for back-compat.
  - Evidence: pytest green; new smoke test `tests/test_analyze_min.py` creates a tiny faux run and the tool emits `pareto_points.csv` + `pareto_frontier.pdf`. Compatible with PR‑14 outputs (e.g., `tersetalk_f*_p*_q*.jsonl`).
  - Next: Run on latest Ollama runs (phi, llama3.1:8b) and attach figures to PRs.

- PR-16 — Statistical Significance & Non‑Inferiority:
  - Summary: Adds `tersetalk/statistics.py` (numpy-only bootstrap: paired CI, percent reduction, mean CI, one-sided p, noninferiority) and `scripts/run_significance.py` CLI. Produces concise console lines and `significance_tests.json` from PR‑14 JSONL outputs.
  - Evidence: pytest green; `tests/test_statistics_smoke.py` creates minimal aligned rows and verifies JSON output; Claude review Approved: no nits.
  - Next: Run on Ollama evaluation outputs (phi, llama3.1:8b) to quantify token reductions and quality deltas; attach reports to PRs.


- PR-19 — Worker Fallback (Ollama-safe):
  - Summary: Adds Worker text fallback when Instructor fails; wraps final answer as TerseTalk ["f", ...]; status OK on success.
  - Evidence: tests green; reduced error rows in pipeline on Ollama.
  - Next: Add Critic fallback to avoid dropping rows when Critic schema fails.

- PR-20 — Temperature + Instruct-run Plan (n=2):
  - Summary: Adds temperature controls to ModelClient/baselines; updates PROMPT with instruct-run plan (n=2).
  - Evidence: pytest green; codellama:13b-instruct freeform runs (n=2) completed; figures generated.
  - Next: Execute joint runs with tersetalk+freeform; produce significance JSON.

- PR-21 — Critic Fallback (text):
  - Summary: Adds Critic text fallback (A/R/E) when structured validation fails; complements Worker fallback to preserve rows.
  - Evidence: tests green; ready to re-run instruct evaluations and analysis.
  - Next: Run paired instruct evaluations (n=2), analyze (Pareto), run significance, attach artifacts.
