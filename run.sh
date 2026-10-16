#!/usr/bin/env bash
# Run Agbot from the repo root:  ./run.sh [prompt] [options]
set -euo pipefail
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then
  exec .venv/bin/python -m agbot "$@"            # Linux/macOS venv layout
elif [ -x ".venv/Scripts/python.exe" ]; then
  exec .venv/Scripts/python.exe -m agbot "$@"    # Windows venv layout (Git Bash)
else
  echo "No .venv found - set it up first:  bash dev/setup.sh"
  exit 1
fi
