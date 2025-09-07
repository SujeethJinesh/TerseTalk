Read through your @AGENTS.md and ensure you follow that precisely. Make sure you fully understand and have read through the updated @RESEARCH_PROPOSAL.md. First verify, which branch we're on and return to the main branch if we're not there. Ensure you pull the latest main branch before branching off for the rest of the PR. I want you to implement a PR of the @RESEARCH_PROPOSAL.md on a new branch and create a PR for it with a very good review using yourself, claude code, and yourself again of course as mentioned in the @AGENTS.md. To ask claude code for a review, you can do something like `echo "Please ensure you understand @RESEARCH_PROPOSAL.md. I am implementing PR-X. Please review the following files I created: @README.md, @xxx. Ensure that my implementation is properly aligned with the goals of the project. Also here is some additional data on the runs I gathered, please critique it as if you're a senior data scientist and ensure that I'm not cheating on the results, lying, or misrepresenting things" | claude -p --dangerously-skip-permissions --model opus`. Please ensure you are indeed calling it like `claude -p --dangerously-skip-permissions --model opus` and ensure you get both the code review, and the data review, and an additional PI review about the state of the project with Claude and yourself. You must be truthful and honest, and address all the points Claude makes (though keep away from making fakes or mocks). If you need to create fakes or mocks for debugging, delete them afterwards so as to not confuse fakes and mocks for actual results. You must only present results from real model runs as much as possible for our project. It's imperative. Ensure that you are going back and incorporating feedback from claude and yourself as necessary. You must continue this loop until both you and claude agree with each other and give yourselfs both approval without nits. After that you should push your changes up for review. Ensure you're aligned with the spirit of the proposal. Then send a PR out so I can review it after your implementation is pushed up. If you run into any major roadblocks, let me know and be detailed. Also update your @AGENTS.md and CLAUDE.md to ensure that when you are asking for a review from CLAUDE, to refer to the @RESEARCH_PROPOSAL.md and ensure you review it in the spirit of that as well. It's imperative that we always keep the @RESEARCH_PROPOSAL.md up to date. You must provide a very small summary of the results after the PR was checked in (update @AGENTS.md and ask @CLAUDE.md to review that as well every PR). This way when a new session starts, it is easy to get back up to speed on the latest work that needs to be done. Please also ensure you use the .venv created at all times. Ensure that if you are debugging failures or such and need to create additional scripts, that you clean them up afterwards. It's incredibly important that you keep the code clean and minimal. We want it to do the job correctly. Before you merge the PR, you must wait for my approval. Also at the end I want you to outline any risks we are seeing in this project, are our expectations aligned with how the progress is going? Will our project succeed and achieve the baselines we expect using your best judgement? Absolutely make sure you're reporting results truthfully and honestly. Avoid fake, mocked, or other non genuine results. You should also analyze the results we get for each run and determine if they meet our figures of merit, when you report back, it's crucial to include that analysis (e.g. compression amount, failure rate, latency, etc.). We should be aiming to properly fix things and run proper evaluations. If we do have any expected goals or outcomes (e.g. >= 10x on xyz) and they aren't achieved, then explain why, but do not lie or cheat and use drastically contrived inefficient metrics. It's important to generally be comparing to a standard implementation. Please also report the results on real data runs, it's important that we start gathering as much data on real runs as possible now so it can be reviewed by Claude AND YOU. If you see there are problems or optimizations, don't hesitate to suggest them as we collect more data. Here are more detailed instructions for PR implementation.

### PR Summary

PR‑20 — Instruct Model Runs (n=2) + Strict Generation Settings

Scope (≤250 LOC where needed):
- Use Ollama instruct model `llama3.1:8b-instruct-q8_0` where available
- Run paired, real‑data evaluations with strict generation settings
- Produce initial figures and significance (n=2) to validate plumbing

Settings
- Temperature: 0.2; max_tokens: 256–384 where applicable
- Systems: tersetalk (with Worker fallback), freeform (final answer only), hybrid (if time permits), llmlingua (guarded fallback)
- Tasks: hotpotqa and gsm8k; Seeds: [0]
- Per‑role models: set Worker/Critic to instruct model when available

Commands (example)
- export OLLAMA_MODEL="llama3.1:8b-instruct-q8_0"
- python scripts/run_evaluation.py --task hotpotqa --systems tersetalk freeform --n 2 --seed 0 --model "$OLLAMA_MODEL"   --worker-model "$OLLAMA_MODEL" --critic-model "$OLLAMA_MODEL" --temperature 0.2 --out results/eval_instruct
- Repeat for gsm8k

Analysis
- python scripts/analyze_v05.py --indir results/eval_instruct --outdir results/eval_instruct/figures
- Artifacts: pareto_points.csv, pareto_frontier.pdf, ablation_caps.{csv,pdf}

Significance (paired)
- python scripts/run_significance.py --results-dir results/eval_instruct/<task>/<timestamp> --boots 2000
- Uses tersetalk_baseline.jsonl symlink and freeform.jsonl in the same run dir

Acceptance
- Both tasks produce non‑NaN paired stats for token reduction and quality Δ
- Pareto/ablation artifacts saved; console summaries present
- Short analysis note in PR body: token savings, EM deltas, any anomalies (guarded paths)


Focus (≤250 LOC per PR; minimal diffs):
- Run real local models via Ollama on real tasks; debug issues in the pipeline path.
- Generate consistent, audit‑ready figures and stats for TerseTalk vs baselines.
- Tighten comparison correctness: same examples, same seeds, paired analyses.

Plan (phased)
- Refactors/Polish (small, high‑impact):
  - Per‑role models/timeouts: allow `--worker-model`/`--critic-model` in eval; set per‑call timeouts; lower temp (≈0.2) and raise max_tokens modestly.
  - Instructor fallback: if structured parse fails on Ollama, downgrade Worker/Critic to text path with a typed shim (still recorded in status).
  - Naming consistency: ensure tersetalk outputs include cap tuple and a `tersetalk_baseline.jsonl` symlink for significance.
  - Determinism: ensure seeds and n are consistent across all systems; pair by index; log skipped/error rows.

- Real‑Run Matrix (initial):
  - Tasks: hotpotqa, gsm8k
  - Systems: tersetalk, freeform, llmlingua, hybrid
  - Models: llama3.1:8b (primary), phi:latest (fallback for smoke)
  - N×seeds: {n=50, seeds [0,42]} initial; {n=200+} later if stable
  - Caps grid (terse): {(20,15,20), (30,20,30), (50,40,50)}; Hybrid budgets: {400,600,800}

- Execution (commands):
  - Evaluate:
    - `export OLLAMA_MODEL="llama3.1:8b"`
    - `python scripts/run_evaluation.py --task hotpotqa --systems tersetalk freeform llmlingua hybrid --n 50 --seed 0 --model $OLLAMA_MODEL --out results/eval_llama31`
    - Repeat for gsm8k and seed 42
  - Analyze:
    - `python scripts/analyze_v05.py --indir results/eval_llama31 --outdir results/eval_llama31/figures`
    - Artifacts: `by_run.csv`, `pareto_points.csv`, `pareto_frontier.pdf`, `ablation_caps.{csv,pdf}`
  - Significance:
    - `python scripts/run_significance.py --results-dir results/eval_llama31/<task>/<timestamp> --boots 5000`
    - Artifacts: `significance_tests.json` + console lines

- Comparison correctness:
  - Pairing by index across systems (truncate to common length) for tokens & accuracy.
  - Use the same examples (seeded loaders), same model, same n for all systems.
  - Report compliance rate and status/error counts to ensure fair “n”.
  - SP method explicitly recorded (jaccard vs bertscore) and consistent per run.

- Risks & mitigations:
  - Slow inference on llama3.1:8b → start with n=10–50; extend timeouts; batch serially.
  - Structured schema failures → Instructor fallback; record status; don’t crash.
  - LLMLingua unavailable → guarded fallback already present; label runs properly.
  - Low accuracy for small models → document; emphasize token savings and non‑inferiority when applicable.

- DoD (acceptance):
  - Real runs (no Echo) complete for both tasks at n≥50 without crashes.
  - Figures: Pareto & ablation generated and committed as artifacts.
  - Significance JSON produced with meaningful (non‑NaN) results for all three tests.
  - PRs include concise evidence (paths + screenshots) and pass Claude review with “Approved: no nits.”
