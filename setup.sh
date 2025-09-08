#!/usr/bin/env bash
set -euo pipefail

# TerseTalk setup for HPC/Marlowe (no sudo required)

: "${VENV_DIR:=.venv}"
: "${INSTALL_LLMLINGUA:=0}"
: "${OLLAMA_MODELS:=llama3.1:8b-instruct-q8_0}"
: "${OLLAMA_BASE_URL:=http://localhost:11434/v1}"
: "${OLLAMA_API_KEY:=ollama}"
: "${OLLAMA_MODEL:=llama3.1:8b-instruct-q8_0}"
: "${OLLAMA_DIR:=$HOME/.local/ollama}"

check_prerequisites() {
  # Check if we're on a compute node (recommended) or login node
  if [[ "${SLURM_JOB_ID:-}" == "" ]]; then
    echo "WARNING: Not running on a compute node. Consider using:"
    echo "  srun --partition=preempt --time=01:00:00 --mem=8GB --pty bash"
    echo "  Then run this script"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      exit 1
    fi
  fi
}

load_modules() {
  # Load HPC modules instead of installing packages
  echo "Loading modules..."
  module load python/3.12 || module load python/3.11 || module load python/3.10
  
  # Check if node/npm module exists
  module load node || module load nodejs || true
}

create_venv() {
  if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
  fi
  source "${VENV_DIR}/bin/activate"
  python -m pip install -U pip
  
  if [ -f requirements.txt ]; then
    python -m pip install -r requirements.txt
  fi
  if [ -f requirements-dev.txt ]; then
    python -m pip install -r requirements-dev.txt
  fi
  if [ -f pyproject.toml ]; then
    python -m pip install -e .
  fi
  if [ "${INSTALL_LLMLINGUA}" = "1" ]; then
    python -m pip install 'llmlingua==0.2.1'
  fi
}

install_ollama_user() {
  # Install Ollama in user directory (no sudo needed)
  mkdir -p "${OLLAMA_DIR}"
  
  if [ ! -f "${OLLAMA_DIR}/ollama" ]; then
    echo "Installing Ollama to ${OLLAMA_DIR}..."
    curl -L https://github.com/ollama/ollama/releases/download/v0.1.48/ollama-linux-amd64 \
         -o "${OLLAMA_DIR}/ollama"
    chmod +x "${OLLAMA_DIR}/ollama"
  fi
  
  export PATH="${OLLAMA_DIR}:$PATH"
  
  # Set Ollama data directory to user space
  export OLLAMA_MODELS="${HOME}/.ollama/models"
  export OLLAMA_HOST="127.0.0.1:11434"
}

start_ollama_background() {
  # Check if already running
  if curl -sS localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama already running"
    return
  fi
  
  echo "Starting Ollama in background..."
  mkdir -p "${HOME}/.ollama/logs"
  nohup "${OLLAMA_DIR}/ollama" serve > "${HOME}/.ollama/logs/ollama.log" 2>&1 &
  OLLAMA_PID=$!
  
  echo "Waiting for Ollama to start (PID: $OLLAMA_PID)..."
  for i in $(seq 1 30); do
    if curl -sS localhost:11434/api/tags >/dev/null 2>&1; then
      echo "Ollama ready!"
      echo $OLLAMA_PID > "${HOME}/.ollama/ollama.pid"
      break
    fi
    sleep 2
  done
}

pull_models() {
  echo "Pulling models: ${OLLAMA_MODELS}"
  for m in ${OLLAMA_MODELS}; do
    echo "Pulling ${m}..."
    "${OLLAMA_DIR}/ollama" pull "${m}"
  done
}

setup_npm_user() {
  # Configure npm for user-level installs
  mkdir -p "$HOME/.npm-global"
  npm config set prefix "$HOME/.npm-global" 2>/dev/null || true
  export PATH="$HOME/.npm-global/bin:$PATH"
  
  # Add to bashrc if not already there
  if ! grep -q 'NPM_CONFIG_PREFIX' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export NPM_CONFIG_PREFIX="$HOME/.npm-global"' >> "$HOME/.bashrc"
    echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> "$HOME/.bashrc"
  fi
}

write_env() {
  cat > .env <<EOF
OLLAMA_BASE_URL=${OLLAMA_BASE_URL}
OLLAMA_API_KEY=${OLLAMA_API_KEY}
OLLAMA_MODEL=${OLLAMA_MODEL}
OLLAMA_DIR=${OLLAMA_DIR}
OLLAMA_MODELS=${HOME}/.ollama/models
EOF
  echo "Wrote .env file"
}

main() {
  check_prerequisites
  load_modules
  create_venv
  install_ollama_user
  start_ollama_background
  pull_models
  write_env
  
  # Only setup npm if available
  if command -v npm >/dev/null 2>&1; then
    setup_npm_user
  else
    echo "npm not available - skipping npm setup"
  fi
  
  echo ""
  echo "Setup complete!"
  echo "Ollama running with PID: $(cat ${HOME}/.ollama/ollama.pid 2>/dev/null || echo 'unknown')"
  echo ""
  echo "For Slurm jobs, add to your script:"
  echo "  export PATH=\"${OLLAMA_DIR}:\$PATH\""
  echo "  export OLLAMA_MODELS=\"${HOME}/.ollama/models\""
  echo ""
}

main "$@"