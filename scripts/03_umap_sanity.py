#!/usr/bin/env python3
"""Stage 03: normalise, embed and sanity-check the 3.3-6.0 hpf subset with UMAP.

Input : data/processed/dropseq_3.3-6.0hpf_qc.h5ad
Output: data/processed/dropseq_3.3-6.0hpf_umap.h5ad
        results/tables/03_umap_coordinates.csv
        results/tables/03_umap_summary.tsv
        results/tables/03_leiden_cluster_stage_counts.csv
        results/tables/03_leiden_cluster_stage_fraction.csv
        results/figures/03_*.png/.pdf

This is a sanity-check embedding, not a final trajectory analysis. Cell filters
are deliberately mild (all 10,101 cells currently pass them); genes detected in
fewer than 3 cells are dropped. No batch correction is applied, so the batch
panel is important when interpreting the UMAP.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from pipeline_utils import (
    QC_H5AD,
    STAGE_COLORS,
    STAGE_HPF,
    STAGE_ORDER,
    TABLES,
    UMAP_H5AD,
    ensure_dirs,
    save_figure,
)

MIN_GENES = 200
MIN_COUNTS = 500
MIN_CELLS = 3
N_TOP_GENES = 2000
N_PCS = 50
N_PCS_NEIGHBORS = 30
N_NEIGHBORS = 15
LEIDEN_RESOLUTION = 0.5
RANDOM_STATE = 0

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute even if the output already exists",
    )
    return parser.parse_args()


def umap_panel(ax, adata, color, title, **kwargs):
    sc.pl.umap(
        adata,
        color=color,
        ax=ax,
        show=False,
        frameon=False,
        title=title,
        **kwargs,
    )


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if UMAP_H5AD.exists() and not args.force:
        print(f"[skip] {UMAP_H5AD} exists (use --force to rebuild)")
        return

    print(f"[03] loading {QC_H5AD}")
    adata = sc.read_h5ad(QC_H5AD)
    adata.obs["stage"] = pd.Categorical(
        adata.obs["stage"], categories=STAGE_ORDER, ordered=True
    )
    n_cells_input, n_genes_input = adata.shape

    # --- explicit, mild QC filtering -------------------------------------
    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    sc.pp.filter_cells(adata, min_counts=MIN_COUNTS)
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    n_cells_qc, n_genes_qc = adata.shape
    print(
        f"[03] after QC: {n_cells_qc:,} cells x {n_genes_qc:,} genes "
        f"({n_cells_input - n_cells_qc} cells, "
        f"{n_genes_input - n_genes_qc} genes removed)"
    )

    adata.layers["counts"] = adata.X.copy()

    # --- normalisation and feature selection -----------------------------
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=N_TOP_GENES,
        flavor="seurat",
        batch_key="sequencing_batch",
    )
    n_hvg = int(adata.var["highly_variable"].sum())
    print(f"[03] highly variable genes: {n_hvg:,}")

    # PCA is computed on scaled HVG values; the full log-normalised matrix is
    # retained in adata.X / layers['lognorm'].
    hvg = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(hvg, max_value=10)
    sc.tl.pca(
        hvg,
        n_comps=N_PCS,
        svd_solver="arpack",
        random_state=RANDOM_STATE,
    )
    adata.obsm["X_pca"] = hvg.obsm["X_pca"].copy()
    adata.uns["pca"] = dict(hvg.uns["pca"])
    adata.varm["PCs"] = np.zeros((adata.n_vars, hvg.varm["PCs"].shape[1]), dtype=float)
    adata.varm["PCs"][adata.var["highly_variable"].to_numpy(), :] = hvg.varm["PCs"]
    del hvg

    # --- neighbours, UMAP, Leiden ----------------------------------------
    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        n_pcs=N_PCS_NEIGHBORS,
        random_state=RANDOM_STATE,
    )
    sc.tl.umap(adata, min_dist=0.3, random_state=RANDOM_STATE)
    sc.tl.leiden(
        adata,
        resolution=LEIDEN_RESOLUTION,
        key_added="leiden",
        flavor="leidenalg",
        random_state=RANDOM_STATE,
    )
    adata.obs["leiden"] = adata.obs["leiden"].astype("category")
    n_clusters = adata.obs["leiden"].nunique()
    print(f"[03] Leiden clusters: {n_clusters}")

    # --- tables -----------------------------------------------------------
    coords = pd.DataFrame(
        adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata.obs_names
    )
    coords.index.name = "cell_id"
    coords = coords.join(
        adata.obs[
            [
                "stage",
                "hours_post_fertilization",
                "sequencing_batch",
                "geo_sample",
                "leiden",
                "total_counts",
                "n_genes_by_counts",
                "pct_counts_ribo",
            ]
        ]
    )
    coords.to_csv(TABLES / "03_umap_coordinates.csv")

    clusters = (
        pd.crosstab(adata.obs["leiden"], adata.obs["stage"])
        .reindex(columns=STAGE_ORDER, fill_value=0)
    )
    clusters = clusters.loc[sorted(clusters.index, key=lambda x: int(x))]
    clusters.index.name = "leiden"
    clusters.to_csv(TABLES / "03_leiden_cluster_stage_counts.csv")
    clusters.div(clusters.sum(axis=1), axis=0).to_csv(
        TABLES / "03_leiden_cluster_stage_fraction.csv"
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "n_cells_input",
                "n_genes_input",
                "n_cells_after_qc",
                "n_genes_after_qc",
                "min_genes",
                "min_counts",
                "min_cells",
                "n_highly_variable_genes",
                "n_pcs",
                "n_pcs_neighbors",
                "n_neighbors",
                "leiden_resolution",
                "n_leiden_clusters",
                "median_counts_per_cell",
                "median_genes_per_cell",
            ],
            "value": [
                n_cells_input,
                n_genes_input,
                n_cells_qc,
                n_genes_qc,
                MIN_GENES,
                MIN_COUNTS,
                MIN_CELLS,
                n_hvg,
                N_PCS,
                N_PCS_NEIGHBORS,
                N_NEIGHBORS,
                LEIDEN_RESOLUTION,
                n_clusters,
                float(np.median(adata.obs["total_counts"])),
                float(np.median(adata.obs["n_genes_by_counts"])),
            ],
        }
    )
    summary.to_csv(TABLES / "03_umap_summary.tsv", sep="\t", index=False)

    # --- figures ----------------------------------------------------------
    common = dict(size=7, alpha=0.75, legend_loc="right margin", legend_fontsize=7)

    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    umap_panel(
        ax, adata, "stage", "UMAP coloured by stage",
        palette=STAGE_COLORS, groups=STAGE_ORDER, **common,
    )
    save_figure(fig, "03_umap_by_stage")
    plt.close(fig)

    # Small multiples: each panel highlights one timepoint against all cells.
    umap = adata.obsm["X_umap"]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0), sharex=True, sharey=True)
    for ax, stage in zip(axes.ravel(), STAGE_ORDER):
        mask = (adata.obs["stage"] == stage).to_numpy()
        ax.scatter(
            umap[~mask, 0], umap[~mask, 1], s=2, c="#d9d9d9",
            alpha=0.35, linewidths=0,
        )
        ax.scatter(
            umap[mask, 0], umap[mask, 1], s=4,
            c=STAGE_COLORS[stage], alpha=0.8, linewidths=0,
        )
        ax.set_title(
            f"{stage} ({STAGE_HPF[stage]:.1f} hpf, n={int(mask.sum()):,})"
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Individual timepoints highlighted on the uncorrected UMAP", y=0.99)
    fig.tight_layout()
    save_figure(fig, "03_umap_stage_highlight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    umap_panel(
        ax, adata, "hours_post_fertilization", "UMAP coloured by hpf",
        cmap="viridis", **common,
    )
    save_figure(fig, "03_umap_by_hpf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    umap_panel(
        ax, adata, "sequencing_batch", "UMAP coloured by sequencing batch",
        **common,
    )
    save_figure(fig, "03_umap_by_batch")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    umap_panel(
        ax, adata, "leiden", "UMAP coloured by Leiden cluster",
        legend_loc="on data", legend_fontsize=7, size=7, alpha=0.75,
    )
    save_figure(fig, "03_umap_by_leiden")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    umap_panel(
        ax, adata, "pct_counts_ribo", "UMAP coloured by ribosomal fraction",
        cmap="magma", **common,
    )
    save_figure(fig, "03_umap_by_ribo")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))
    umap_panel(
        axes[0, 0], adata, "stage", "Stage",
        palette=STAGE_COLORS, groups=STAGE_ORDER,
        legend_loc="right margin", legend_fontsize=7, size=6, alpha=0.75,
    )
    umap_panel(
        axes[0, 1], adata, "hours_post_fertilization", "Hours post-fertilisation",
        cmap="viridis", size=6, alpha=0.75,
    )
    umap_panel(
        axes[1, 0], adata, "sequencing_batch", "Sequencing batch",
        legend_loc="right margin", legend_fontsize=7, size=6, alpha=0.75,
    )
    umap_panel(
        axes[1, 1], adata, "leiden", "Leiden cluster",
        legend_loc="on data", legend_fontsize=6, size=6, alpha=0.75,
    )
    fig.suptitle("UMAP sanity check: Drop-seq, 3.3-6.0 hpf", y=0.995)
    fig.tight_layout()
    save_figure(fig, "03_umap_overview")
    plt.close(fig)

    vr = adata.uns["pca"]["variance_ratio"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(np.arange(1, len(vr) + 1), vr * 100, "o-", ms=3, lw=1)
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance explained (%)")
    ax.set_title("PCA variance ratio (HVG, scaled)")
    save_figure(fig, "03_pca_variance_ratio")
    plt.close(fig)

    print(f"[03] writing {UMAP_H5AD}")
    adata.write_h5ad(UMAP_H5AD, compression="gzip")

    print("[03] cluster x stage counts:")
    print(clusters.to_string())
    print(f"[03] done: {adata.n_obs:,} cells, {n_clusters} clusters")


if __name__ == "__main__":
    main()
