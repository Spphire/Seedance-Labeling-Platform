#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing venv. Run setup.sh first." >&2
  exit 1
fi

.venv/bin/python -m unittest discover -s tests -p "test_*.py"
