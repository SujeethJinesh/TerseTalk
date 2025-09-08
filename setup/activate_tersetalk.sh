#!/usr/bin/env bash
set -euo pipefail

# TerseTalk setup for Marlowe login node

: "${VENV_DIR:=.venv}"
: "${PROJECT_DIR:=/projects/m000066}"  # Your project directory with more space

echo "==============================================="
echo "TerseTalk Login Node Setup"
echo "==============================================="

check_login_node() {
  if [[ "${SLURM_JOB_ID:-}" != "" ]]; then
    echo "WARNING: You're on a compute node. This script is meant for login nodes."
  fi
  echo "Current node: $(hostname)"
  echo "Python version: $(python3 --version)"
  echo ""
}

create_venv() {
  echo "Setting up Python virtual environment..."
  
  # Only prompt if venv exists and script is run interactively
  if [ -d "${VENV_DIR}" ]; then
    if [ -t 0 ]; then  # Check if running interactively
      echo "Virtual environment already exists at ${VENV_DIR}"
      read -p "Remove and recreate? (y/n) " -n 1 -r
      echo
      if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${VENV_DIR}"
      else
        echo "Keeping existing venv"
        source "${VENV_DIR}/bin/activate"
        return
      fi
    else
      echo "Venv exists, activating..."
      source "${VENV_DIR}/bin/activate"
      return
    fi
  fi
  
  if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "Created virtual environment at ${VENV_DIR}"
  fi
  
  source "${VENV_DIR}/bin/activate"
  
  echo "Upgrading pip..."
  python -m pip install --quiet --upgrade pip
  
  if [ -f requirements.txt ]; then
    echo "Installing requirements.txt..."
    python -m pip install -r requirements.txt
  fi
  
  if [ -f pyproject.toml ]; then
    echo "Installing package in editable mode..."
    python -m pip install -e .
  fi
  
  echo "Installing LLMLingua (required)"
  python -m pip install 'llmlingua==0.2.1' || true
  echo ""
}

setup_directories() {
  echo "Setting up directory structure..."
  
  mkdir -p "${HOME}/.local/bin"
  mkdir -p "${HOME}/.local/ollama"
  mkdir -p "${HOME}/.ollama/models"
  mkdir -p "${HOME}/.ollama/logs"
  mkdir -p "${HOME}/TerseTalk/logs"
  
  # Consider using project directory for models if low on home space
  if [ -d "${PROJECT_DIR}" ]; then
    echo "Project directory available at ${PROJECT_DIR}"
    mkdir -p "${PROJECT_DIR}/ollama-models"
    ln -sfn "${PROJECT_DIR}/ollama-models" "${HOME}/.ollama/models-project"
    echo "Created link to project directory for additional model storage"
  fi
  
  echo ""
}

check_disk_space() {
  echo "Checking disk space..."
  echo "Home directory:"
  df -h "$HOME" | head -1
  df -h "$HOME" | tail -1
  
  if [ -d "${PROJECT_DIR}" ]; then
    echo ""
    echo "Project directory (${PROJECT_DIR}):"
    df -h "${PROJECT_DIR}" | tail -1
  fi
  echo ""
}

write_env() {
  if [ -f .env ]; then
    echo ".env file already exists, skipping..."
    return
  fi
  
  echo "Writing .env configuration..."
  cat > .env <<EOF
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama
OLLAMA_MODEL=llama3.1:8b-instruct-q8_0
OLLAMA_DIR=$HOME/.local/ollama
OLLAMA_MODELS=$HOME/.ollama/models
EOF
  echo ""
}

create_activation_script() {
  echo "Creating activation script..."
  
  cat > activate_tersetalk.sh <<'EOF'
#!/bin/bash
# Quick activation script for TerseTalk environment

cd ~/TerseTalk

# Activate venv
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
  echo "✓ Activated Python venv"
fi

# Set Ollama paths
export PATH="$HOME/.local/ollama:$PATH"
export OLLAMA_MODELS="$HOME/.ollama/models"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"

echo "✓ TerseTalk environment ready"
echo "  Python: $(which python)"
echo "  Ollama: $HOME/.local/ollama/ollama"
EOF
  
  chmod +x activate_tersetalk.sh
  echo "Created activate_tersetalk.sh"
  echo ""
}

main() {
  check_login_node
  create_venv
  setup_directories
  check_disk_space
  write_env
  create_activation_script
  
  echo "==============================================="
  echo "Login node setup complete!"
  echo ""
  echo "To activate environment:"
  echo "  source ~/TerseTalk/activate_tersetalk.sh"
  echo ""
  echo "Or add to ~/.bashrc for auto-activation:"
  echo "  echo 'source ~/TerseTalk/activate_tersetalk.sh' >> ~/.bashrc"
  echo ""
  echo "Next: Get compute node and run setup_compute_node.sh"
  echo "==============================================="
}

main "$@"
