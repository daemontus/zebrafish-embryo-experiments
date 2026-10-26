"""Shared paths, constants and plotting helpers for the analysis scripts."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "GSE106587"
METADATA = ROOT / "data" / "metadata"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
REPORT_FIGURES = ROOT / "report" / "figures"
REPORT_TABLES = ROOT / "report" / "tables"

RAW_DROPSEQ = RAW / "GSE106474" / "GSE106474_UMICounts.txt.gz"
CELL_METADATA = METADATA / "dropseq_cell_metadata.csv"
DROPSEQ_SOFT = METADATA / "soft" / "GSE106474_family.soft.gz"

# All Drop-seq stages in developmental order, with GEO hpf values.
STAGE_HPF = {
    "high": 3.3,
    "oblong": 3.8,
    "dome": 4.3,
    "30% epiboly": 4.7,
    "50% epiboly": 5.3,
    "shield": 6.0,
    "60% epiboly": 7.0,
    "75% epiboly": 8.0,
    "90% epiboly": 9.0,
    "bud": 10.0,
    "3-somite": 11.0,
    "6-somite": 12.0,
}
STAGES = list(STAGE_HPF)

# Sequencing batches, in fixed order with fixed (Okabe-Ito) colours.
BATCHES = ["DS2", "DS3", "DS4", "DS5"]
BATCH_COLORS = {"DS2": "#0072B2", "DS3": "#E69F00", "DS4": "#009E73", "DS5": "#D55E00"}

# Report text width (A4, 2.5cm margins) in inches.
REPORT_WIDTH_IN = 16.0 / 2.54


def library_of(cell_ids) -> "pd.Index":
    """Library (GEO sample) name from Drop-seq cell IDs, e.g. ZF50_WT_DS4b."""
    import pandas as pd

    return pd.Index(cell_ids).str.rsplit("_", n=1).str[0]


def ensure_dirs() -> None:
    """Create output directories used by the scripts."""
    for path in (PROCESSED, TABLES, REPORT_FIGURES, REPORT_TABLES):
        path.mkdir(parents=True, exist_ok=True)


def report_style() -> None:
    """Matplotlib defaults for figures included in the LaTeX report."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "hatch.linewidth": 0.6,
            "pdf.fonttype": 42,
        }
    )


def save_report_figure(fig, stem: str) -> None:
    """Save a matplotlib figure as PDF into report/figures."""
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_FIGURES / f"{stem}.pdf", bbox_inches="tight")


DROPSEQ_H5AD = PROCESSED / "dropseq_counts.h5ad"


def load_dropseq_counts() -> "ad.AnnData":
    """Drop-seq UMI counts as a sparse cells x genes AnnData with cell metadata.

    The first call streams the dense genes x cells TSV and caches the result
    in data/processed (delete the file to rebuild it). No cells or genes are
    removed.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    if DROPSEQ_H5AD.exists():
        return ad.read_h5ad(DROPSEQ_H5AD)
    print(f"streaming {RAW_DROPSEQ.name} into a sparse matrix ...")
    blocks, genes, cells = [], [], None
    for chunk in pd.read_csv(RAW_DROPSEQ, sep="\t", index_col=0, chunksize=2000):
        cells = chunk.columns
        genes.extend(chunk.index)
        blocks.append(sp.csr_matrix(chunk.to_numpy(dtype=np.float32)))
        print(f"  genes processed: {len(genes)}")
    counts = sp.vstack(blocks).T.tocsr()  # cells x genes
    meta = pd.read_csv(CELL_METADATA, index_col="cell_id").loc[cells]
    meta["library"] = library_of(meta.index)
    adata = ad.AnnData(
        X=counts, obs=meta, var=pd.DataFrame(index=pd.Index(genes, name="gene"))
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(DROPSEQ_H5AD, compression="gzip")
    return adata
