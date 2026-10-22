#!/usr/bin/env python3
"""Sanity-check the downloaded GSE106587 matrices (standard library only).

Writes data/metadata/validation_report.txt and exits non-zero on failure.
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "GSE106587"
OUT = ROOT / "data" / "metadata" / "validation_report.txt"

problems: list[str] = []
lines: list[str] = []


def check(condition: bool, message: str) -> None:
    lines.append(("OK   " if condition else "FAIL ") + message)
    if not condition:
        problems.append(message)


def gz_lines(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            yield line.rstrip("\n")


def header(path: Path) -> list[str]:
    return next(gz_lines(path)).split("\t")


def first_column(path: Path) -> list[str]:
    return [line.split("\t", 1)[0] for line in gz_lines(path)]


# --- Drop-seq ---------------------------------------------------------------
drop = RAW / "GSE106474" / "GSE106474_UMICounts.txt.gz"
drop_cells = [c for c in header(drop) if c]
drop_genes = first_column(drop)[1:]
check(len(drop_cells) == len(set(drop_cells)),
      f"Drop-seq: {len(drop_cells):,} unique cell IDs")
check(len(drop_genes) == len(set(drop_genes)),
      f"Drop-seq: {len(drop_genes):,} unique gene symbols")
if len(drop_genes) != len(set(drop_genes)):
    from collections import Counter
    counts = Counter(drop_genes)
    dups = sorted(g for g, n in counts.items() if n > 1)
    lines.append(f"     duplicate genes: {', '.join(dups[:20])}")
check(all("_" in c for c in drop_cells),
      "Drop-seq: every cell ID encodes stage/batch/barcode")

# --- SMART-seq2 -------------------------------------------------------------
ss2 = RAW / "GSE106466" / "GSE106466_SS2_umi_cnts.txt.gz"
fpkm = RAW / "GSE106466" / "GSE106466_SS2_fpkm.txt.gz"
ss2_cells = [c for c in header(ss2) if c]
fpkm_cells = [c for c in header(fpkm) if c]
check(len(ss2_cells) == len(set(ss2_cells)),
      f"SMART-seq2: {len(ss2_cells):,} unique UMI cell IDs")
check(len(fpkm_cells) == len(set(fpkm_cells)),
      f"SMART-seq2: {len(fpkm_cells):,} unique FPKM cell IDs")
check(set(fpkm_cells) <= set(ss2_cells),
      "SMART-seq2: FPKM cells are a subset of UMI cells")
check(len(set(ss2_cells) - set(fpkm_cells)) == 88,
      "SMART-seq2: 88 cells (oep p0) have UMI counts but no FPKM")

# --- 10x --------------------------------------------------------------------
for gsm, label in [("GSM3084402", "10xWT6S"), ("GSM3084403", "10xMZoep6S")]:
    base = RAW / "GSE106587" / f"{gsm}_{label}"
    genes = [line.split("\t")[0] for line in gz_lines(Path(f"{base}_genes.tsv.gz"))]
    barcodes = list(gz_lines(Path(f"{base}_barcodes.tsv.gz")))
    with gzip.open(f"{base}_matrix.mtx.gz", "rt") as fh:
        check(next(fh).startswith("%%MatrixMarket"), f"{label}: MatrixMarket header")
        for line in fh:
            if not line.startswith("%"):
                dims = line.split()
                break
        else:
            dims = []
    n_genes, n_barcodes = int(dims[0]), int(dims[1])
    check(n_genes == len(genes), f"{label}: {n_genes:,} genes match genes.tsv")
    check(n_barcodes == len(barcodes),
          f"{label}: {n_barcodes:,} barcodes match barcodes.tsv")
    check(len(barcodes) == len(set(barcodes)),
          f"{label}: barcodes are unique")

text = "\n".join(lines) + "\n\n"
text += "RESULT: PASS\n" if not problems else f"RESULT: {len(problems)} FAILURE(S)\n"
OUT.write_text(text, encoding="utf-8")
print(text, end="")
sys.exit(1 if problems else 0)
