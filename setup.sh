#!/usr/bin/env bash
set -euo pipefail

# TerseTalk setup script for Ubuntu 22.04
# - Installs system deps, Python 3.12 + venv, pip deps
# - Installs and starts Ollama
# - Pulls default models (configurable)
# - Writes a .env with sane defaults

# Usage:
#   bash setup.sh                 # full setup with default models
#   OLLAMA_MODELS="llama3.1:8b-instruct-q8_0" bash setup.sh
#   INSTALL_LLMLINGUA=1 bash setup.sh   # optionally install llmlingua==0.2.1

: "${PY_VER:=3.12}"
: "${VENV_DIR:=.venv}"
: "${INSTALL_LLMLINGUA:=0}"
: "${OLLAMA_MODELS:=llama3.1:8b-instruct-q8_0}"
: "${OLLAMA_BASE_URL:=http://localhost:11434/v1}"
: "${OLLAMA_API_KEY:=ollama}"
: "${OLLAMA_MODEL:=llama3.1:8b-instruct-q8_0}"

ensure_apt() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "This script expects apt-get (Ubuntu/Debian)." >&2
    exit 1
  fi
}

install_system_deps() {
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
    curl ca-certificates git jq \
    software-properties-common build-essential \
    python3-pip python3-venv

  # Install Python ${PY_VER}
  if ! command -v python${PY_VER} >/dev/null 2>&1; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python${PY_VER} python${PY_VER}-venv python${PY_VER}-dev
  fi

  # Install Node.js (LTS 20.x) for CLI tools (Claude/Codex)
  if ! command -v node >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
  fi
}

create_venv() {
  if [ ! -d "${VENV_DIR}" ]; then
    python${PY_VER} -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  python -m pip install -U pip
  if [ -f requirements.txt ]; then
    python -m pip install -r requirements.txt || true
  fi
  if [ -f requirements-dev.txt ]; then
    python -m pip install -r requirements-dev.txt || true
  fi
  # Editable install of the package (if pyproject present)
  if [ -f pyproject.toml ]; then
    python -m pip install -e .
  fi
  if [ "${INSTALL_LLMLINGUA}" = "1" ]; then
    python -m pip install 'llmlingua==0.2.1' || true
  fi
}

install_ollama() {
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
  fi

  # Try to start/enable service; fallback to foreground server in background
  if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl enable --now ollama || true
  fi

  if ! curl -sS localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting ollama serve in the background..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
  fi

  echo -n "Waiting for Ollama to become ready"
  for i in $(seq 1 60); do
    if curl -sS localhost:11434/api/tags >/dev/null 2>&1; then
      echo " OK"
      break
    fi
    echo -n "."
    sleep 2
  done
}

pull_models() {
  echo "Pulling models: ${OLLAMA_MODELS}"
  for m in ${OLLAMA_MODELS}; do
    echo "ollama pull ${m}"
    ollama pull "${m}" || true
  done
}

write_env() {
  cat > .env <<EOF
OLLAMA_BASE_URL=${OLLAMA_BASE_URL}
OLLAMA_API_KEY=${OLLAMA_API_KEY}
OLLAMA_MODEL=${OLLAMA_MODEL}
EOF
  echo "Wrote .env with OLLAMA_* defaults"
}

post_checks() {
  echo "Python: $(python -V) in ${VENV_DIR}"
  echo "Pip:    $(pip -V)"
  echo "Ollama: $(ollama --version || true)"
  echo "Available models:"
  curl -sS localhost:11434/api/tags | jq -r '.models[].name' || true
  echo "Node:   $(node -v || true)"
  echo "NPM:    $(npm -v || true)"
}

main() {
  ensure_apt
  install_system_deps
  create_venv
  install_ollama
  pull_models
  write_env
  # Configure user-level global npm prefix (HPC/Slurm friendly)
  mkdir -p "$HOME/.npm-global"
  npm config set prefix "$HOME/.npm-global" || true
  export PATH="$HOME/.npm-global/bin:$PATH"
  if ! grep -q 'NPM_CONFIG_PREFIX' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export NPM_CONFIG_PREFIX="$HOME/.npm-global"' >> "$HOME/.bashrc"
  fi
  if ! grep -q '.npm-global/bin' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> "$HOME/.bashrc"
  fi
  # Install CLI tools (user-global)
  npm install -g @openai/codex || true
  npm install -g @anthropic-ai/claude-code || true
  post_checks

  cat <<'MSG'

Setup complete.

Next steps (examples):
  # Activate the venv in this shell
  source .venv/bin/activate

  # Quick echo dry-run (no network)
  python scripts/run_v05.py --task synth --system tersetalk --n 2 --seed 0 --model echo --dry-run

  # Small real instruct run (HotpotQA, tersetalk+freeform, n=2)
  export OLLAMA_MODEL="llama3.1:8b-instruct-q8_0"
  python scripts/run_evaluation.py --task hotpotqa --systems tersetalk --systems freeform \
    --n 2 --seed 0 --model "$OLLAMA_MODEL" --worker-model "$OLLAMA_MODEL" --critic-model "$OLLAMA_MODEL" \
    --temperature 0.2 --out results/eval_instruct --no-caps-grid

  # Analysis & significance
  python scripts/analyze_v05.py --indir results/eval_instruct --outdir results/eval_instruct/figures
  python scripts/run_significance.py --results-dir results/eval_instruct/hotpotqa/latest --boots 2000

Notes:
  - To enable real LLMLingua compression, rerun setup with INSTALL_LLMLINGUA=1 or pip install llmlingua==0.2.1.
  - You can change OLLAMA_MODELS before running setup to pull different models.

MSG
}

main "$@"
