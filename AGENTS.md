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

## Claude Code Review Protocol

- Preconditions:
  - Ensure local checks pass: `make fmt && make lint && make test`.
  - Claude CLI must work with Node LTS (>=18). On macOS with Homebrew:
    - `brew install node@20` and add `export PATH="/usr/local/opt/node@20/bin:$PATH"`.
  - Export `ANTHROPIC_API_KEY` in your shell.
  - Quick sanity check: `echo "pong" | claude -p --dangerously-skip-permissions --model opus` ⇒ outputs `pong`.

- Prompt template (pipe only the files you touched):
  - Start with a 1–2 sentence goal and explicit ask for a verdict.
  - Include focus areas relevant to the change.
  - For each file, send as:
    - `File: path/to/file\n---\n<file contents>`
  - End with: `If fully acceptable with no nits, explicitly reply: 'Approved: no nits.'`

- Recommended command shape:
  - `( printf "<context + ask>\n\n"; printf "File: X\n---\n"; sed -n '1,200p' X; ... ) | claude -p --dangerously-skip-permissions --model opus` (the `sed -n '1,200p'` prints the first 200 lines)

- Iteration loop:
  - Capture Claude’s feedback verbatim in your task response.
  - Apply only changes that meet the intent and keep diffs minimal.
  - Re-run `make fmt && make lint && make test`.
  - Re-run the review with a short “Re-review after applying your suggestions” preface.
  - Stop when Claude replies exactly: `Approved: no nits.` (case-sensitive, must match exactly)

- Scope and pushback:
  - It’s fine to push back on large or out-of-scope requests; explain constraints and propose a minimal alternative, then ask Claude to confirm.
  - Do not introduce heavy tools or broad refactors unless explicitly requested by the user.

- Hygiene:
  - Never pipe secrets or `.env` contents to Claude.
  - Send only the specific files you edited; avoid dumping unrelated large files.
  - Keep each file excerpt <=200 lines where possible; trim to relevant hunks if larger.

## Troubleshooting Claude CLI

- CLI prints a Node stack trace on start:
  - Install Node LTS and ensure it’s first in `PATH` (see above), then retry.
- Auth errors from Claude:
  - Ensure `ANTHROPIC_API_KEY` is exported and valid.
- Permission prompts in sandbox:
  - Use `--dangerously-skip-permissions` as this environment is sandboxed.

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
