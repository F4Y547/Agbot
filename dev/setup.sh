#!/usr/bin/env bash
# Sets up the local agent environment (Linux/macOS).
#   bash dev/setup.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
ollama_model="${OLLAMA_MODEL:-qwen3:8b}"

echo "==> Creating venv (.venv)"
python3 -m venv .venv
./.venv/bin/python -m pip install -q -U pip
./.venv/bin/python -m pip install -q -r requirements.txt
echo "==> Python deps installed"

if command -v ollama >/dev/null 2>&1; then
  if ollama list | grep -q "${ollama_model%%:*}"; then
    echo "==> Ollama model '$ollama_model' already present"
  else
    echo "==> Pulling Ollama model '$ollama_model'"
    ollama pull "$ollama_model"
  fi
else
  echo "!! Ollama not found - install from https://ollama.com/download for the local fallback"
fi

[ -f .env ] || { cp .env.example .env; echo "==> Created .env from .env.example"; }

echo
echo "Done. Try:"
echo "  ./.venv/bin/python -m agbot --selftest"
echo "  ./.venv/bin/python -m agbot"
