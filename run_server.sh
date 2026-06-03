#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing venv. Run setup.sh first." >&2
  exit 1
fi

HOST="${SEEDANCE_HOST:-0.0.0.0}"
PORT="${SEEDANCE_PORT:-18080}"
RELOAD_ARGS=()
if [ "${SEEDANCE_RELOAD:-0}" = "1" ]; then
  RELOAD_ARGS=(--reload)
fi

exec .venv/bin/python -m uvicorn app.backend.main:app --host "$HOST" --port "$PORT" "${RELOAD_ARGS[@]}"
