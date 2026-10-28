#!/usr/bin/env bash
# Download the ZFIN files used to assign genes to expression programs
# (script 09): wild-type expression records, publications (for the year
# filter) and gene aliases (old symbols used by the Drop-seq matrix).
# ZFIN updates these files continuously; SHA256SUMS records the version used.
#
# Usage:  bash scripts/download_zfin.sh
# Re-running is safe: existing files are left in place.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${ROOT}/data/raw/zfin"
BASE="https://zfin.org/downloads"

mkdir -p "$RAW"
for name in wildtype-expression_fish.txt zfinpubs.txt aliases.txt; do
    if [[ -s "${RAW}/${name}" ]]; then
        echo "[skip]  data/raw/zfin/${name}"
        continue
    fi
    echo "[fetch] ${BASE}/${name}"
    curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 \
         -o "${RAW}/${name}.part" "${BASE}/${name}"
    mv "${RAW}/${name}.part" "${RAW}/${name}"
done
(cd "$RAW" && shasum -a 256 wildtype-expression_fish.txt zfinpubs.txt aliases.txt > SHA256SUMS)
cat "${RAW}/SHA256SUMS"
