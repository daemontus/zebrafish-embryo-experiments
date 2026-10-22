#!/usr/bin/env bash
# Create the project-local Python environment and install pinned dependencies.
# Run from anywhere:  bash scripts/00_setup_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
    echo "[setup] creating .venv with $(python3 --version)"
    python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
if [[ "${USE_LOCK:-1}" == "1" && -f requirements.lock.txt ]]; then
    echo "[setup] installing exact versions from requirements.lock.txt"
    .venv/bin/pip install -r requirements.lock.txt
else
    echo "[setup] installing top-level pins from requirements.txt"
    .venv/bin/pip install -r requirements.txt
fi

echo "[setup] done"
.venv/bin/python - <<'EOF'
import importlib.metadata as md
for pkg in ("scanpy", "anndata", "numpy", "pandas", "scipy", "harmonypy"):
    print(f"  {pkg} {md.version(pkg)}")
EOF
