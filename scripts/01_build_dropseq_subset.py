#!/usr/bin/env python3
"""Stage 01: extract the 3.3-6.0 hpf Drop-seq subset into an AnnData object.

Input : data/raw/GSE106587/GSE106474/GSE106474_UMICounts.txt.gz
        data/metadata/dropseq_cell_metadata.csv
Output: data/processed/dropseq_3.3-6.0hpf_raw.h5ad
        results/tables/01_subset_summary.tsv
        results/tables/01_cells_per_stage.tsv

The GEO matrix is genes x cells (TSV). Its header has no name for the gene
column, so pandas infers that column as the index when index_col=0. The full
matrix is read once and then subset; selecting columns with usecols is not
reliable because of the missing header name.
"""
from __future__ import annotations

import argparse
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pipeline_utils import (
    CELL_METADATA,
    RAW_DROPSEQ,
    STAGE_COLORS,
    STAGE_HPF,
    STAGE_ORDER,
    SUBSET_H5AD,
    TABLES,
    ensure_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild the output even if it already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if SUBSET_H5AD.exists() and not args.force:
        print(f"[skip] {SUBSET_H5AD} exists (use --force to rebuild)")
        return

    print("[01] reading cell metadata")
    meta = pd.read_csv(CELL_METADATA)
    meta = meta[meta["stage"].isin(STAGE_ORDER)].copy()
    meta["stage"] = pd.Categorical(
        meta["stage"], categories=STAGE_ORDER, ordered=True
    )
    cells = meta["cell_id"].tolist()
    if len(cells) != len(set(cells)):
        raise RuntimeError("duplicate cell IDs in metadata")
    print(f"[01] selected {len(cells):,} cells from {len(STAGE_ORDER)} stages")

    t0 = time.time()
    print(f"[01] reading full Drop-seq matrix: {RAW_DROPSEQ}")
    counts = pd.read_csv(RAW_DROPSEQ, sep="\t", index_col=0)
    print(
        f"[01] read matrix in {time.time() - t0:.1f}s: "
        f"{counts.shape[0]:,} genes x {counts.shape[1]:,} cells"
    )

    missing = set(cells) - set(counts.columns)
    if missing:
        raise RuntimeError(f"{len(missing)} selected cells are missing from matrix")

    genes = counts.index.to_numpy()
    counts = counts.loc[:, cells]
    if list(counts.columns) != cells:
        raise RuntimeError("cell order changed during subsetting")

    print("[01] converting to sparse AnnData (cells x genes)")
    X = sp.csr_matrix(counts.to_numpy(dtype=np.int32).T)
    del counts

    obs = meta.set_index("cell_id").loc[cells].copy()
    obs.index.name = "cell_id"
    obs["stage"] = pd.Categorical(
        obs["stage"], categories=STAGE_ORDER, ordered=True
    )
    for col in ("sequencing_batch", "replicate", "geo_sample"):
        obs[col] = obs[col].astype("category")
    obs["hours_post_fertilization"] = obs["hours_post_fertilization"].astype(float)

    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["stage_colors"] = STAGE_COLORS
    adata.uns["subset"] = {
        "source": "GSE106474 (Drop-seq, wild type)",
        "stages": ", ".join(STAGE_ORDER),
        "hpf_min": min(STAGE_HPF[s] for s in STAGE_ORDER),
        "hpf_max": max(STAGE_HPF[s] for s in STAGE_ORDER),
    }

    print(f"[01] writing {SUBSET_H5AD}")
    adata.write_h5ad(SUBSET_H5AD, compression="gzip")

    # --- summary tables ----------------------------------------------------
    total_counts = np.asarray(X.sum(axis=1)).ravel()
    genes_per_cell = np.diff(X.indptr)
    summary = pd.DataFrame(
        {
            "metric": [
                "source",
                "stages",
                "hpf_min",
                "hpf_max",
                "n_cells",
                "n_genes",
                "n_nonzero_entries",
                "matrix_density",
                "total_counts",
                "median_counts_per_cell",
                "median_genes_per_cell",
                "min_counts_per_cell",
                "max_counts_per_cell",
            ],
            "value": [
                "GSE106474 (Drop-seq, wild type)",
                "; ".join(STAGE_ORDER),
                min(STAGE_HPF[s] for s in STAGE_ORDER),
                max(STAGE_HPF[s] for s in STAGE_ORDER),
                adata.n_obs,
                adata.n_vars,
                int(X.nnz),
                float(X.nnz) / (adata.n_obs * adata.n_vars),
                int(total_counts.sum()),
                float(np.median(total_counts)),
                float(np.median(genes_per_cell)),
                int(total_counts.min()),
                int(total_counts.max()),
            ],
        }
    )
    summary.to_csv(TABLES / "01_subset_summary.tsv", sep="\t", index=False)

    stage_table = (
        adata.obs.groupby("stage", observed=True)
        .agg(
            n_cells=("stage", "size"),
            n_geo_samples=("geo_sample", "nunique"),
        )
        .reindex(STAGE_ORDER)
        .reset_index()
    )
    stage_table.insert(1, "hours_post_fertilization", stage_table["stage"].map(STAGE_HPF))
    stage_table.to_csv(TABLES / "01_cells_per_stage.tsv", sep="\t", index=False)

    print(
        f"[01] done: {adata.n_obs:,} cells x {adata.n_vars:,} genes, "
        f"{X.nnz:,} nonzero entries"
    )
    print(stage_table.to_string(index=False))


if __name__ == "__main__":
    main()
