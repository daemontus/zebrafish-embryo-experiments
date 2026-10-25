#!/usr/bin/env bash
# Run the analysis pipeline in order (stages 01-13).
# Usage:  bash scripts/run_pipeline.sh [--force]
#   --force  rebuild stage 01/03/04/06/07/08/09/12 outputs instead of reusing them
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

echo "== stage 06: per-library Leiden clustering =="
"$PY" scripts/06_library_clustering.py "${ARGS[@]}"

echo "== stage 07: merged contexts (DS5 across time, 50% across batches) =="
"$PY" scripts/07_merged_contexts.py "${ARGS[@]}"

echo "== stage 08: 50% epiboly batch correction (Harmony, scVI) =="
"$PY" scripts/08_batch_correction_50pct.py "${ARGS[@]}"

echo "== stage 09: cNMF gene programs per 50% epiboly library =="
"$PY" scripts/09_gene_programs_cnmf.py "${ARGS[@]}"

echo "== stage 10: replicate + annotate programs (needs data/reference) =="
if [[ ! -s data/reference/zfin_aliases.txt ]]; then
    bash scripts/download_references.sh
fi
"$PY" scripts/10_program_annotation.py

echo "== stage 11: replicated programs across K =5..20 =="
"$PY" scripts/11_program_k_scan.py

echo "== stage 12: DS5 gene programs and match to 50% epiboly =="
"$PY" scripts/12_ds5_programs.py "${ARGS[@]}"

echo "== stage 13: signalling-pathway proxies =="
"$PY" scripts/13_pathway_proxies.py

echo "== pipeline complete =="
