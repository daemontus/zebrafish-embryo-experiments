#!/usr/bin/env bash
# Run the analysis pipeline in order (stages 01-05).
# Usage:  bash scripts/run_pipeline.sh [--force]
#   --force  rebuild stage 01/03/04 outputs instead of reusing them
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "error: .venv not found; run: bash scripts/00_setup_env.sh" >&2
    exit 1
fi

ARGS=()
if [[ "${1:-}" == "--force" ]]; then
    ARGS+=(--force)
fi

echo "== stage 01: build subset =="
"$PY" scripts/01_build_dropseq_subset.py "${ARGS[@]}"

echo "== stage 02: QC summary =="
"$PY" scripts/02_qc_summary.py

echo "== stage 03: UMAP sanity check =="
"$PY" scripts/03_umap_sanity.py "${ARGS[@]}"

echo "== stage 04: Harmony-corrected diagnostic UMAP =="
"$PY" scripts/04_umap_harmony.py "${ARGS[@]}"

echo "== stage 05: dataset structure (stage x batch x library) =="
"$PY" scripts/05_dataset_structure.py

echo "== pipeline complete =="
