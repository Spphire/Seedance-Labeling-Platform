#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p tmp
PYTHON_BIN="${PYTHON:-python3}"

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

export TMPDIR="$ROOT/tmp"
export PIP_CACHE_DIR="$ROOT/tmp"

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt

if [ -f "vendor/wheels/nmx_msg-2.2.0-py3-none-any.whl" ]; then
  .venv/bin/python -m pip install --no-cache-dir vendor/wheels/nmx_msg-2.2.0-py3-none-any.whl
fi

echo "Setup complete: $ROOT/.venv"
