#!/usr/bin/env bash
set -euo pipefail

# TerseTalk setup for Marlowe compute node
# Run this on a compute node to install Ollama and pull models

: "${OLLAMA_DIR:=$HOME/.local/ollama}"
: "${OLLAMA_MODELS:=llama3.1:8b-instruct-q8_0}"
: "${OLLAMA_HOST:=127.0.0.1:11434}"
: "${VENV_DIR:=.venv}"

echo "==============================================="
echo "TerseTalk Compute Node Setup"
echo "==============================================="

check_compute_node() {
  if [[ "${SLURM_JOB_ID:-}" == "" ]]; then
    echo "ERROR: This script must run on a compute node"
    echo ""
    echo "First, allocate a compute node:"
    echo "  srun --partition=preempt --time=01:00:00 --mem=16GB --pty bash"
    echo ""
    echo "Then run this script again."
    exit 1
  fi
  
  echo "Running on compute node: $(hostname)"
  echo "SLURM Job ID: ${SLURM_JOB_ID}"
  echo "Allocated memory: ${SLURM_MEM_PER_NODE:-unknown}MB"
  echo "Allocated CPUs: ${SLURM_CPUS_ON_NODE:-unknown}"
  echo ""
}

activate_environment() {
  echo "Activating Python environment..."
  
  # Load modules if available
  if [ -f load_modules.sh ]; then
    source load_modules.sh
  fi
  
  # Activate venv
  if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    echo "Python: $(which python)"
    echo "Version: $(python --version)"
  else
    echo "WARNING: No virtual environment found. Run setup_login_node.sh first."
  fi
  echo ""
}

install_ollama() {
  echo "Installing Ollama..."
  
  mkdir -p "${OLLAMA_DIR}"
  
  if [ -f "${OLLAMA_DIR}/ollama" ]; then
    echo "Ollama already installed at ${OLLAMA_DIR}/ollama"
    CURRENT_VERSION=$("${OLLAMA_DIR}/ollama" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
    echo "Current version: ${CURRENT_VERSION}"
    read -p "Reinstall/update? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Keeping existing installation"
      return
    fi
  fi
  
  echo "Downloading Ollama..."
  
  # Try to get latest release
  OLLAMA_URL=$(curl -s https://api.github.com/repos/ollama/ollama/releases/latest 2>/dev/null | \
               grep "browser_download_url.*linux-amd64\"" | \
               head -1 | \
               cut -d '"' -f 4)
  
  if [ -z "$OLLAMA_URL" ]; then
    echo "Could not fetch latest release, using fallback version"
    OLLAMA_URL="https://github.com/ollama/ollama/releases/download/v0.3.12/ollama-linux-amd64"
  else
    echo "Found latest release: ${OLLAMA_URL}"
  fi
  
  curl -L --progress-bar "$OLLAMA_URL" -o "${OLLAMA_DIR}/ollama"
  chmod +x "${OLLAMA_DIR}/ollama"
  
  # Add to PATH
  export PATH="${OLLAMA_DIR}:$PATH"
  
  echo "Ollama installed successfully"
  echo "Version: $("${OLLAMA_DIR}/ollama" --version)"
  echo ""
}

start_ollama() {
  echo "Starting Ollama service..."
  
  # Set environment
  export OLLAMA_MODELS="${HOME}/.ollama/models"
  export OLLAMA_HOST="${OLLAMA_HOST}"
  export PATH="${OLLAMA_DIR}:$PATH"
  
  # Check if already running
  if curl -sS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "Ollama is already running"
    return
  fi
  
  # Start Ollama
  mkdir -p "${HOME}/.ollama/logs"
  nohup "${OLLAMA_DIR}/ollama" serve > "${HOME}/.ollama/logs/ollama_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
  OLLAMA_PID=$!
  echo $OLLAMA_PID > "${HOME}/.ollama/ollama.pid"
  
  echo "Started Ollama with PID: ${OLLAMA_PID}"
  echo "Waiting for Ollama to be ready..."
  
  # Wait for service to be ready
  MAX_ATTEMPTS=30
  for i in $(seq 1 $MAX_ATTEMPTS); do
    if curl -sS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      echo "Ollama is ready!"
      break
    fi
    if [ $i -eq $MAX_ATTEMPTS ]; then
      echo "ERROR: Ollama failed to start after ${MAX_ATTEMPTS} attempts"
      echo "Check logs at: ${HOME}/.ollama/logs/"
      exit 1
    fi
    echo -n "."
    sleep 2
  done
  echo ""
}

check_disk_space() {
  echo "Checking available disk space..."
  
  AVAILABLE_GB=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | sed 's/G//')
  echo "Available space in home: ${AVAILABLE_GB}GB"
  
  if [ "$AVAILABLE_GB" -lt 15 ]; then
    echo "WARNING: Low disk space!"
    echo "Each model can be 5-10GB. You have ${AVAILABLE_GB}GB available."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Aborting. Free up space and try again."
      exit 1
    fi
  fi
  echo ""
}

pull_models() {
  echo "Pulling Ollama models..."
  echo "Models to pull: ${OLLAMA_MODELS}"
  echo ""
  
  for model in ${OLLAMA_MODELS}; do
    echo "Pulling ${model}..."
    "${OLLAMA_DIR}/ollama" pull "${model}"
    if [ $? -eq 0 ]; then
      echo "Successfully pulled ${model}"
    else
      echo "ERROR: Failed to pull ${model}"
    fi
    echo ""
  done
  
  echo "Installed models:"
  "${OLLAMA_DIR}/ollama" list
  echo ""
}

create_test_script() {
  echo "Creating test script..."
  
  cat > test_ollama.py <<'EOF'
#!/usr/bin/env python3
"""Test Ollama installation"""

import requests
import json
import os

ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
model = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q8_0")

print(f"Testing Ollama at {ollama_host} with model {model}")

try:
    # Test API endpoint
    response = requests.get(f"http://{ollama_host}/api/tags")
    models = response.json()
    print(f"Available models: {[m['name'] for m in models.get('models', [])]}")
    
    # Test generation
    prompt = "What is 2+2? Answer in one word."
    response = requests.post(
        f"http://{ollama_host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Test prompt: {prompt}")
        print(f"Response: {result.get('response', 'No response')}")
        print("✓ Ollama is working correctly!")
    else:
        print(f"✗ Generation failed: {response.status_code}")
        
except Exception as e:
    print(f"✗ Error: {e}")
EOF
  
  chmod +x test_ollama.py
  echo "Created test_ollama.py"
  echo ""
}

cleanup_info() {
  echo "==============================================="
  echo "Compute node setup complete!"
  echo ""
  echo "Ollama is running with PID: $(cat ${HOME}/.ollama/ollama.pid 2>/dev/null || echo 'unknown')"
  echo ""
  echo "To test Ollama:"
  echo "  python test_ollama.py"
  echo ""
  echo "To stop Ollama:"
  echo "  kill \$(cat ${HOME}/.ollama/ollama.pid)"
  echo ""
  echo "For Slurm jobs, add to your script:"
  echo "  export PATH=\"${OLLAMA_DIR}:\$PATH\""
  echo "  export OLLAMA_MODELS=\"${HOME}/.ollama/models\""
  echo "  ${OLLAMA_DIR}/ollama serve &"
  echo "  OLLAMA_PID=\$!"
  echo "  sleep 10"
  echo "  # Your code here"
  echo "  kill \$OLLAMA_PID"
  echo ""
  echo "NOTE: Ollama will stop when this compute session ends"
  echo "==============================================="
}

main() {
  check_compute_node
  activate_environment
  install_ollama
  check_disk_space
  start_ollama
  pull_models
  create_test_script
  
  # Run test
  if [ -f test_ollama.py ]; then
    echo "Running test..."
    python test_ollama.py
    echo ""
  fi
  
  cleanup_info
}

main "$@"