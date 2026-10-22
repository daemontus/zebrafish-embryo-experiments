#!/usr/bin/env bash
# Download the complete GEO SuperSeries GSE106587 (Farrell et al., 2018,
# "Single-cell reconstruction of developmental trajectories during zebrafish
# embryogenesis", Science 360:eaar3131).
#
# The SuperSeries has three components:
#   GSE106474  Drop-seq, wild-type, 12 timepoints (the main atlas, ~38k cells)
#   GSE106466  SMART-seq2, wild-type + MZoep, 50% epiboly (416 single cells)
#   GSE106587  10x Genomics, wild-type + MZoep, 6-somite (2 samples)
#
# Usage:  bash scripts/download_gse106587.sh
# Re-running is safe: existing complete files are left in place.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${ROOT}/data/raw/GSE106587"
META="${ROOT}/data/metadata"
BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE106nnn"

fetch() {
    local url="$1" dest="$2"
    mkdir -p "$(dirname "$dest")"
    if [[ -s "$dest" ]]; then
        echo "[skip]  ${dest#${ROOT}/}"
        return 0
    fi
    echo "[fetch] ${url}"
    curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 \
         -o "${dest}.part" "$url"
    mv "${dest}.part" "$dest"
}

# --- processed expression matrices (supplementary files) -------------------
fetch "${BASE}/GSE106474/suppl/GSE106474_UMICounts.txt.gz" \
      "${RAW}/GSE106474/GSE106474_UMICounts.txt.gz"
fetch "${BASE}/GSE106466/suppl/GSE106466_SS2_umi_cnts.txt.gz" \
      "${RAW}/GSE106466/GSE106466_SS2_umi_cnts.txt.gz"
fetch "${BASE}/GSE106466/suppl/GSE106466_SS2_fpkm.txt.gz" \
      "${RAW}/GSE106466/GSE106466_SS2_fpkm.txt.gz"
fetch "${BASE}/GSE106587/suppl/GSE106587_RAW.tar" \
      "${RAW}/GSE106587/GSE106587_RAW.tar"

# --- GEO metadata: SOFT family files + series matrices ---------------------
for gse in GSE106587 GSE106466 GSE106474; do
    fetch "${BASE}/${gse}/soft/${gse}_family.soft.gz" \
          "${META}/soft/${gse}_family.soft.gz"
done
fetch "${BASE}/GSE106587/matrix/GSE106587-GPL18413_series_matrix.txt.gz" \
      "${META}/series_matrix/GSE106587-GPL18413_series_matrix.txt.gz"
fetch "${BASE}/GSE106587/matrix/GSE106587-GPL20828_series_matrix.txt.gz" \
      "${META}/series_matrix/GSE106587-GPL20828_series_matrix.txt.gz"
fetch "${BASE}/GSE106466/matrix/GSE106466_series_matrix.txt.gz" \
      "${META}/series_matrix/GSE106466_series_matrix.txt.gz"
fetch "${BASE}/GSE106474/matrix/GSE106474_series_matrix.txt.gz" \
      "${META}/series_matrix/GSE106474_series_matrix.txt.gz"

# --- provenance / integrity -------------------------------------------------
# Only the SuperSeries has a published filelist.txt; the SubSeries do not.
fetch "${BASE}/GSE106587/suppl/filelist.txt" \
      "${META}/filelist_GSE106587.txt"

echo "[check] gzip integrity"
find "$RAW" "$META" -type f -name '*.gz' -print0 | xargs -0 -r gzip -t

# --- unpack the 10x MatrixMarket archive (idempotent) ----------------------
if [[ -f "${RAW}/GSE106587/GSE106587_RAW.tar" ]]; then
    echo "[untar] GSE106587_RAW.tar"
    tar -xf "${RAW}/GSE106587/GSE106587_RAW.tar" -C "${RAW}/GSE106587"
fi

echo "[hash]  writing SHA256SUMS"
(cd "$RAW" && find . -type f ! -name 'SHA256SUMS' -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS)

echo "[done]  data in ${RAW}"
