"""Shared paths, constants and plotting helpers for the analysis pipeline."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "GSE106587"
METADATA = ROOT / "data" / "metadata"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

RAW_DROPSEQ = RAW / "GSE106474" / "GSE106474_UMICounts.txt.gz"
CELL_METADATA = METADATA / "dropseq_cell_metadata.csv"

SUBSET_H5AD = PROCESSED / "dropseq_3.3-6.0hpf_raw.h5ad"
QC_H5AD = PROCESSED / "dropseq_3.3-6.0hpf_qc.h5ad"
UMAP_H5AD = PROCESSED / "dropseq_3.3-6.0hpf_umap.h5ad"
HARMONY_H5AD = PROCESSED / "dropseq_3.3-6.0hpf_umap_harmony.h5ad"

# Stages inside the requested 3.3-6.0 hpf window, in developmental order.
STAGE_ORDER = [
    "high",
    "oblong",
    "dome",
    "30% epiboly",
    "50% epiboly",
    "shield",
]
STAGE_HPF = {
    "high": 3.3,
    "oblong": 3.8,
    "dome": 4.3,
    "30% epiboly": 4.7,
    "50% epiboly": 5.3,
    "shield": 6.0,
}
STAGE_COLORS = {
    "high": "#4c72b0",
    "oblong": "#55a868",
    "dome": "#c44e52",
    "30% epiboly": "#8172b2",
    "50% epiboly": "#ccb974",
    "shield": "#64b5cd",
}


def library_of(cell_ids) -> "pd.Index":
    """Library (GEO sample) name from Drop-seq cell IDs, e.g. ZF50_WT_DS4b."""
    import pandas as pd

    return pd.Index(cell_ids).str.rsplit("_", n=1).str[0]


def ensure_dirs() -> None:
    """Create output directories used by all stages."""
    for path in (PROCESSED, FIGURES, TABLES):
        path.mkdir(parents=True, exist_ok=True)


def save_figure(fig, stem: str) -> None:
    """Save a matplotlib figure as PNG and PDF in results/figures."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
