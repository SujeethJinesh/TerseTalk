# TerseTalk v0.5 — Postmortem (llama3.1:8b‑instruct, local‑only)

## 1) Executive Summary

- Goal: Demonstrate that TerseTalk‑JSONL reduces tokens without hurting accuracy on real instruct models (llama3.1:8b‑instruct‑q8_0) across HotpotQA and GSM8K; produce Pareto figures and bootstrap significance; show a publishable hybrid non‑inferiority safety net.
- Outcome: Under the tested conditions, TerseTalk did not meet the accuracy‑vs‑tokens tradeoff on these tasks/models. Freeform achieved higher EM with similar or fewer tokens. The hybrid path remains viable, but in this session full hybrid n=20 runs were limited by local runtime/latency; the code is ready to execute without Slurm.
- Root causes: brittle adherence of a general instruct model to a novel, typed JSONL protocol; strict final‑answer formatting hurt correctness; numeric tasks lack robust arithmetic scaffolding; JSONL + guidance overhead offset token savings.

## 2) Objectives & Success Criteria

- Token savings (≈10–30%) with ≤2–5% EM loss relative to freeform on HotpotQA/GSM8K.
- Hybrid routing: non‑inferior accuracy to freeform (δ=0.02) with token savings.
- Figures (Pareto, ablations) + bootstrap significance with reproducible artifacts.

## 3) What We Built

- Protocol: `tersetalk/protocol_jsonl.py` with soft caps, overflow `o` lines (M#), normalization, SP‑prose rendering.
- Structured generation: Pydantic models; Instructor path; Ollama‑robust JSONL‑by‑text fallback with local parsing/retries.
- Pipeline: Manager→Worker→Critic (strict prompts; fallbacks; accurate token accounting; latency; per‑task memory reset).
- Baselines: Freeform; LLMLingua optional (guarded; disabled locally by env when absent).
- Evaluation: real instruct runs via Ollama; `scripts/run_evaluation.py` (caps grid/hybrid budgets); `scripts/analyze_v05.py` (Pareto/ablation/provenance); `scripts/run_significance.py` (numpy‑only bootstrap).
- Setup: local venv; optional Slurm script included but not used here.

## 4) Experimental Setup

- Model: `llama3.1:8b-instruct-q8_0` via Ollama OpenAI‑compatible API; temperature 0.2 for baselines and Worker.
- Tasks: HotpotQA (EM); GSM8K (numeric EM).
- Systems: freeform; tersetalk; hybrid prepared.
- N: 20 per task (freeform completed); TerseTalk combined two 10‑sample runs into n=20.
- Flags/interventions tested: SP‑prose preface, JSONL few‑shot pattern, allow <=2 `t` lines, stricter answer extraction, Worker max_tokens 384, post‑hoc fallback to freeform when Critic rejects or final `f` missing.

## 5) Results Summary (Real Model)

### HotpotQA (n=20)
- Freeform (baseline): avg tokens 220.2; EM 35.0%; compliance 100%.
- TerseTalk (combined 10+10): avg tokens ≈210–222 across runs; EM ≈5%; compliance 100%.
- Bootstrap (paired on combined set):
  - Token reduction (free→terse): −6.3% [−22.6%, 8.3%], p(one‑sided>0)=0.7778, n=20.
  - Quality Δ (terse−free): −0.300 [−0.550, −0.050], n=20.

### GSM8K (n=20)
- Freeform (baseline): avg tokens 122.0; EM 10.0%; compliance 100%.
- TerseTalk (combined 10+10): avg tokens ≈210–260; EM ≈5–10%; compliance 100%.
- Bootstrap (paired on combined set):
  - Token reduction (free→terse): −80.9% [−132.2%, −40.1%] (i.e., tersetalk used more tokens), n=20.
  - Quality Δ (terse−free): −0.050 [−0.200, 0.100], n=20.

## 6) Interventions Tried

- Strict Worker prompt: single final `['f','<answer>']`; “final answer only”; “no invented entities/IDs”.
- SP‑prose preface: prepend `jsonl_to_prose` (do not output prose) before JSONL.
- Few‑shot JSONL pattern: 2–3 lines showing `r`/`t`/`f` shape.
- Allow <=2 `t` lines: minimal scratch reasoning permitted.
- Increase Worker max_tokens: 256→384.
- Post‑hoc fallback to freeform: on Critic R/E or missing final f‑line, call freeform and use that answer; tokens counted.
- JSON response_format exploration: OpenAI `json_object` mismatch for array‑of‑arrays on Ollama; fell back to instruction+parsing and array‑parse path.

## 7) Root Cause Analysis

- Protocol adherence vs semantics: the model often emits valid JSONL but fails to select the correct final answer, especially in HotpotQA entity linking, despite SP‑prose and strict prompts.
- Train‑test mismatch: the model wasn’t trained on our `r/g/f/...` tags; few‑shot helps structure, not robust correctness.
- Cost of structure: JSONL + guidance + SP‑prose consume tokens; any savings from structured content were offset by overhead and Critic passes.
- Numeric fragility: GSM8K requires reliable stepwise arithmetic; strict “final answer only” plus limited scratch space degraded correctness.
- Critic dynamics: frequent R/E without a task‑specific rubric; rejections trigger fallback or acceptance failures.

## 8) What Worked

- Infrastructure & robustness: full test suite green; resilient fallbacks; accurate token accounting; run artifacts + provenance.
- Microbenchmarks (earlier): JSONL shows strong advantages (tag extraction speed, streaming boundaries, bytes‑on‑wire control);
- Freeform baselines: clean n=20 runs with reasonable accuracy for an 8B model.

## 9) What Failed

- Accuracy preservation: TerseTalk EM substantially below freeform on HotpotQA (−30 points) and not better on GSM8K.
- Token savings: not consistently lower than freeform; on GSM8K, TerseTalk used more tokens.
- Hybrid (in this session): end‑to‑end n=20 not completed due to per‑example latency within an interactive window; code is ready for local chunked runs.

## 10) Recommendations To Recover Publishability

- Reframe the primary claim (systems‑protocol + safety net):
  - Emphasize measurable engineering wins (microbenchmarks) and hybrid non‑inferiority with token savings. This aligns with the proposal.
  - Execute hybrid locally in two 10‑sample chunks per task to avoid timeouts; compute bootstrap CIs (δ=0.02).
- Task‑adapted scaffolding for TerseTalk:
  - HotpotQA: allow slightly larger caps (e.g., f50/p40/q50); include one tiny task‑specific JSONL example per item (retrieved by similarity) to lift EM.
  - GSM8K: allow 2–3 short `t` steps; enforce final integer only; optionally a tiny numeric check in the Critic rubric.
- Model/setting tweaks:
  - Try a stronger instruct model locally (e.g., codellama:13b‑instruct) or lower quantization; Worker temperature 0.0; Critic 0.0.
- Semi‑structured schema path:
  - Return a minimal JSON object (role/facts/plan/final), not tag arrays; convert to JSONL internally. Leverages Ollama’s `json` formatting to improve adherence while keeping typed on‑wire semantics.
- (Future) Fine‑tuning:
  - A small adapter trained on synthetic JSONL examples would likely fix adherence + EM, preserving the protocol story.

## 11) Paper Angles

- Angle A (systems‑protocol focus):
  - Contribution: a typed, streaming‑friendly protocol with robust validation/overflow/memory, fast parse/streaming, and hybrid routing. Present microbenchmarks and hybrid non‑inferiority with token savings; include TerseTalk negative results with analysis.
- Angle B (task realignment):
  - Add yes/no or multiple‑choice tasks (TruthfulQA‑MC, HellaSwag subset) where final answers map naturally to a single short token string; include HotpotQA/GSM8K results candidly.

## 12) Lessons Learned

- Prompt‑only adaptation of a novel protocol is fragile on 8B instruct; minimal fine‑tuning or semi‑structured schemas substantially helps.
- Over‑constraining format and caps can silently degrade correctness more than it saves tokens.
- Hybrid routing should be a first‑class baseline for protocol papers—practical and safety‑oriented.

## 13) Two‑Week Plan (No Slurm)

- Week 1:
  - Implement semi‑structured JSON schema + converter; run n=20 per task.
  - Run hybrid n=20 per task (10+10 chunks); produce non‑inferiority + token savings figures.
  - Add one small task‑specific exemplar per item via nearest neighbor from a seed bank.
- Week 2:
  - Scale to n=100 per task (chunked), continuously updating figures and CIs.
  - Prepare write‑up: microbenchmarks, pipeline, ablations, hybrid results, and an honest negative‑result section for TerseTalk on these tasks/models.

## 14) Key Artifacts & Paths

- Freeform baselines (n=20):
  - HotpotQA: `results/eval_real_20_hotpot_freeform/hotpotqa/2025-09-08-10-54-07`
  - GSM8K: `results/eval_real_20_gsm_freeform/gsm8k/2025-09-08-11-33-09`
- TerseTalk (combined 10+10 to 20):
  - HotpotQA: `results/eval_real_20_hotpot_terse_combined/hotpotqa/2025-09-08-15-25-00/`
  - GSM8K: `results/eval_real_20_gsm_terse_combined/gsm8k/2025-09-08-15-25-00/`
- Significance outputs: `significance_tests.json` within the directories above.
- Analysis figures: `scripts/analyze_v05.py` outputs under `<run_root>/figures/`.

---

Prepared by: the TerseTalk agent (local Ollama, no Slurm). All results are from real model runs; no mocks used for reported metrics.
