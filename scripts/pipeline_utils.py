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


def nb_overdispersion(counts, min_umis: int = 10, reliable_umis: int = 100,
                      bin_size: int = 200) -> "pd.DataFrame":
    """Negative-binomial overdispersion of every gene within one library.

    `counts` is a cells x genes sparse UMI matrix of a single library. A gene
    with the same expression in every cell has counts with mean s_i * m in
    cell i (s_i: the cell's depth relative to the library average, m: the
    gene's rate per average cell) and, under a negative binomial, variance
    s_i m + phi (s_i m)^2. phi = 0 is pure sampling (Poisson); bursting and
    cell size make phi > 0 even for genes expressed in every cell, while a
    gene restricted to a sub-population has a much larger phi. phi is the
    method-of-moments estimate.

    `typical` is log10 of the phi of a typical gene of the same mean, from a
    dispersion-mean trend phi_t(m) = a0 + a1 * m^-b (as in DESeq2's
    parametric trend a0 + a1 / m, with a free exponent): a0 is the plateau of
    well-expressed genes, a1 m^-b the extra variability of rarer ones. It is
    fitted (least squares on log10) to the medians of the raw estimates in
    consecutive bins of `bin_size` genes, using only genes with at least
    `reliable_umis` UMIs (below that, most estimates are dominated by
    sampling, and about half of them are <= 0); medians make the fit robust
    to heterogeneous genes. Rarer genes get the extrapolated curve. The fitted
    parameters are in `.attrs["trend"]` (with the bin medians). `residual` is
    log10(phi) - typical (phi floored at 1e-3), i.e. how much more variable
    a gene is than a typical gene of its expression level in this library
    (0 = typical, 1 = ten times more). Genes with fewer than `min_umis` UMIs
    in the library get NaN.
    """
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    depth = np.asarray(counts.sum(axis=1)).ravel()
    s = depth / depth.mean()
    total = np.asarray(counts.sum(axis=0)).ravel()
    m = total / s.sum()
    sum_y2 = np.asarray(counts.multiply(counts).sum(axis=0)).ravel()
    sum_sy = np.asarray(counts.T @ s).ravel()
    # sum_i (y_i - s_i m)^2 - s_i m, expanded to stay sparse
    excess = sum_y2 - 2 * m * sum_sy + m**2 * (s**2).sum() - m * s.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = excess / (m**2 * (s**2).sum())
    out = pd.DataFrame({"mean": m, "phi": phi, "typical": np.nan, "residual": np.nan})
    ok = np.flatnonzero(total >= min_umis)
    reliable = ok[total[ok] >= reliable_umis]
    reliable = reliable[np.argsort(m[reliable], kind="stable")]
    n_bins = max(1, len(reliable) // bin_size)
    bins = np.array_split(reliable, n_bins)
    bin_m = np.array([np.median(m[b]) for b in bins])
    bin_phi = np.clip([np.median(phi[b]) for b in bins], 1e-3, None)

    def trend(x, a0, a1, b):
        return np.log10(a0 + a1 * x ** (-b))

    if n_bins >= 4:
        (a0, a1, b), _ = curve_fit(trend, bin_m, np.log10(bin_phi), p0=[0.03, 0.05, 0.8],
                                   bounds=([1e-4, 0, 0], [10, 100, 3]), maxfev=20000)
    else:  # too few reliable genes for a trend: one typical value
        a0, a1, b = float(np.median(phi[reliable])) if len(reliable) else 0.1, 0.0, 0.0
    out.loc[ok, "typical"] = trend(m[ok], a0, a1, b)
    out.loc[ok, "residual"] = (np.log10(np.clip(phi[ok], 1e-3, None))
                               - out.loc[ok, "typical"].to_numpy())
    out.attrs["trend"] = dict(a0=a0, a1=a1, b=b, bin_mean=bin_m, bin_phi=bin_phi)
    return out


RESIDUAL_CLIP = 10.0


def pearson_residuals(counts, genes, centre: bool = True) -> "np.ndarray":
    """Pearson residuals of one library under its own NB model (dense cells x genes).

    residual = (y - s m) / sqrt(s m + phi_t (s m)^2), with s the cell's depth
    relative to the library average, m the gene's rate and phi_t the
    library's dispersion-mean trend (`nb_overdispersion`). The trend also
    covers genes too rare in the library for a typical phi; genes absent from
    the library get 0. Residuals are clipped at +-RESIDUAL_CLIP, the same
    limit for every library (a limit of sqrt(cells), as in sctransform,
    differs between libraries of different size and makes two unequal random
    halves of one library look different after alignment). With `centre`,
    every gene is centred within the library (alignment).
    `genes` is a boolean mask or an index array of columns.
    """
    import numpy as np

    disp = nb_overdispersion(counts)
    s = np.asarray(counts.sum(axis=1)).ravel()
    s = s / s.mean()
    m = disp["mean"].to_numpy()[genes]
    t = disp.attrs["trend"]
    with np.errstate(divide="ignore"):
        phi = t["a0"] + t["a1"] * m ** (-t["b"])
    mu = s[:, None] * m[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (counts[:, genes].toarray() - mu) / np.sqrt(mu + phi[None, :] * mu**2)
    r[:, m == 0] = 0
    r = np.clip(r, -RESIDUAL_CLIP, RESIDUAL_CLIP)
    return r - r.mean(axis=0) if centre else r
