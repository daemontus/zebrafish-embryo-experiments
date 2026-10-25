#!/usr/bin/env bash
# Download public annotation references (gene-program interpretation, stages 10-12;
# pathway-proxy gene-set resolution, stage 13).
# Usage:  bash scripts/download_references.sh
#
# These databases are updated continuously, so the retrieval date and SHA-256
# of each file are recorded in data/reference/MANIFEST.tsv. Re-running skips
# files that already exist; delete a file to refresh it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/data/reference"
mkdir -p "$OUT"

FILES=(
  # ZFIN curated wild-type gene expression (gene x anatomy x stage, mostly in situ)
  "zfin_wildtype_expression.txt https://zfin.org/downloads/wildtype-expression_fish.txt"
  # ZFIN developmental stage ontology (stage names, ordering, hours)
  "zfin_stage_ontology.txt https://zfin.org/downloads/stage_ontology.txt"
  # GO Consortium zebrafish gene annotations (GAF 2.2, from ZFIN)
  "zfin.gaf.gz https://current.geneontology.org/annotations/zfin.gaf.gz"
  # GO ontology (is_a / part_of only)
  "go-basic.obo http://purl.obolibrary.org/obo/go/go-basic.obo"
  # Zebrafish anatomy ontology (ZFA), to propagate ZFIN anatomy terms to parents
  "zfa.obo http://purl.obolibrary.org/obo/zfa.obo"
  # ZFIN gene IDs + current symbols, and previous names / aliases (stage 13 gene-set resolution)
  "zfin_gene.txt https://zfin.org/downloads/gene.txt"
  "zfin_aliases.txt https://zfin.org/downloads/aliases.txt"
)

for entry in "${FILES[@]}"; do
  name="${entry%% *}"
  url="${entry#* }"
  if [[ -s "$OUT/$name" ]]; then
    echo "[skip] $name"
    continue
  fi
  echo "[get ] $name <- $url"
  curl -fsSL --retry 3 -o "$OUT/$name.part" "$url"
  mv "$OUT/$name.part" "$OUT/$name"
done

{
  printf "file\tsource\tretrieved\tbytes\tsha256\n"
  for entry in "${FILES[@]}"; do
    name="${entry%% *}"
    url="${entry#* }"
    retrieved=$(date -r "$OUT/$name" +%Y-%m-%d)
    bytes=$(wc -c < "$OUT/$name" | tr -d ' ')
    sha=$(shasum -a 256 "$OUT/$name" | cut -d' ' -f1)
    printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$url" "$retrieved" "$bytes" "$sha"
  done
} > "$OUT/MANIFEST.tsv"
echo "[done] manifest: $OUT/MANIFEST.tsv"
