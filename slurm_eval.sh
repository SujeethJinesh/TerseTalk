#!/usr/bin/env bash
#SBATCH --job-name=tt-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# Slurm evaluation script for TerseTalk (paired instruct runs + analysis + significance)
# Assumes you already ran ./setup.sh in this repo and pulled models.
# You can modify TASKS, N, SEED, MODEL, and OUT_PREFIX below or via env.

set -euo pipefail

# ---------- Config (override via env) ----------
: "${TASKS:=hotpotqa gsm8k}"
: "${SYSTEMS:=tersetalk freeform}"
: "${N:=10}"
: "${SEED:=0}"
: "${MODEL:=llama3.1:8b-instruct-q8_0}"
: "${TEMP:=0.2}"
: "${OUT_PREFIX:=results/eval_instruct_joint}"

# User-global npm (HPC-safe) for CLI tools if needed
export NPM_CONFIG_PREFIX="$HOME/.npm-global"
export PATH="$HOME/.npm-global/bin:$PATH"

# Use the repo's venv
cd "$SLURM_SUBMIT_DIR"
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "ERROR: .venv not found. Please run ./setup.sh first." >&2
  exit 1
fi

# Ollama env
export OLLAMA_MODEL="$MODEL"
export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"

# GPU visibility
nvidia-smi || true

# Start Ollama locally if not already running
if ! curl -sS localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting ollama serve..."
  nohup ollama serve > /tmp/ollama.$SLURM_JOB_ID.log 2>&1 &
  echo -n "Waiting for Ollama"
  for i in $(seq 1 60); do
    if curl -sS localhost:11434/api/tags >/dev/null 2>&1; then
      echo " OK"; break
    fi
    echo -n "."; sleep 2
  done
fi

# Pull model (no-op if present)
ollama pull "$MODEL" || true

# Run paired evaluations per task
for T in $TASKS; do
  OUT_DIR="${OUT_PREFIX}_${T}_l31i"
  echo "Running task=$T N=$N seed=$SEED model=$MODEL -> $OUT_DIR"
  python scripts/run_evaluation.py     --task "$T"     --systems $SYSTEMS     --n "$N" --seed "$SEED"     --model "$MODEL"     --worker-model "$MODEL" --critic-model "$MODEL"     --temperature "$TEMP"     --out "$OUT_DIR"     --no-caps-grid

  # Analysis + significance
  python scripts/analyze_v05.py --indir "$OUT_DIR" --outdir "$OUT_DIR/figures"
  python scripts/run_significance.py --results-dir "$OUT_DIR/$T/latest" --boots 5000

done

# Summary
echo "
Finished. Artifacts under:"
for T in $TASKS; do
  OUT_DIR="${OUT_PREFIX}_${T}_l31i"
  echo "  $OUT_DIR/$T/latest"
  echo "  $OUT_DIR/figures"
  ls -1 "$OUT_DIR/figures" | sed 's/^/    /'
  echo "  Significance: $OUT_DIR/$T/latest/significance_tests.json"
  echo
  done
