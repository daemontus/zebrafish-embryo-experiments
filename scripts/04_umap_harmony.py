#!/usr/bin/env python3
"""Stage 04: batch-corrected (Harmony) UMAP sanity check.

Input : data/processed/dropseq_3.3-6.0hpf_umap.h5ad   (from stage 03)
Output: data/processed/dropseq_3.3-6.0hpf_umap_harmony.h5ad
        results/tables/04_umap_harmony_coordinates.csv
        results/tables/04_harmony_summary.tsv
        results/tables/04_embedding_silhouette.tsv
        results/tables/04_leiden_harmony_cluster_stage_counts.csv
        results/tables/04_leiden_harmony_cluster_stage_fraction.csv
        results/figures/04_*.png/.pdf

Stage 03 shows a strong sequencing-batch effect (high/oblong/shield are all
DS5 while 50% epiboly is DS2-DS4). Harmony is run on the existing PCA to check
whether developmental structure survives after batch correction.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score

from pipeline_utils import (
    HARMONY_H5AD,
    STAGE_COLORS,
    STAGE_ORDER,
    TABLES,
    UMAP_H5AD,
    ensure_dirs,
    save_figure,
)

N_NEIGHBORS = 15
LEIDEN_RESOLUTION = 0.5
RANDOM_STATE = 0
SILHOUETTE_N = 5000

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
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def panel(ax, adata, basis, color, title, **kwargs):
    sc.pl.embedding(
        adata,
        basis=basis,
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
    if HARMONY_H5AD.exists() and not args.force:
        print(f"[skip] {HARMONY_H5AD} exists (use --force to rebuild)")
        return

    print(f"[04] loading {UMAP_H5AD}")
    adata = sc.read_h5ad(UMAP_H5AD)
    adata.obs["stage"] = pd.Categorical(
        adata.obs["stage"], categories=STAGE_ORDER, ordered=True
    )

    print("[04] running Harmony on X_pca (batch = sequencing_batch)")
    # scanpy 1.12's harmony_integrate wrapper assumes harmonypy's old (d x N)
    # Z_corr layout, while harmonypy 2.x returns (N x d). Call harmonypy
    # directly and normalise the orientation so this works with either.
    import harmonypy

    harmony_out = harmonypy.run_harmony(
        adata.obsm["X_pca"].astype(np.float64),
        adata.obs,
        "sequencing_batch",
        max_iter_harmony=20,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    z_corr = np.asarray(harmony_out.Z_corr)
    if z_corr.shape[0] != adata.n_obs:
        z_corr = z_corr.T
    adata.obsm["X_pca_harmony"] = z_corr

    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        use_rep="X_pca_harmony",
        key_added="harmony",
        random_state=RANDOM_STATE,
    )
    sc.tl.umap(
        adata,
        neighbors_key="harmony",
        key_added="X_umap_harmony",
        min_dist=0.3,
        random_state=RANDOM_STATE,
    )
    sc.tl.leiden(
        adata,
        neighbors_key="harmony",
        key_added="leiden_harmony",
        resolution=LEIDEN_RESOLUTION,
        flavor="igraph",
        n_iterations=2,
        random_state=RANDOM_STATE,
    )
    adata.obs["leiden_harmony"] = adata.obs["leiden_harmony"].astype("category")
    n_clusters = adata.obs["leiden_harmony"].nunique()
    print(f"[04] Leiden clusters after Harmony: {n_clusters}")

    # --- tables -----------------------------------------------------------
    coords = pd.DataFrame(
        adata.obsm["X_umap_harmony"],
        columns=["UMAP1", "UMAP2"],
        index=adata.obs_names,
    )
    coords.index.name = "cell_id"
    coords = coords.join(
        adata.obs[
            [
                "stage",
                "hours_post_fertilization",
                "sequencing_batch",
                "geo_sample",
                "leiden_harmony",
                "total_counts",
                "n_genes_by_counts",
                "pct_counts_ribo",
            ]
        ]
    )
    coords.to_csv(TABLES / "04_umap_harmony_coordinates.csv")

    clusters = (
        pd.crosstab(adata.obs["leiden_harmony"], adata.obs["stage"])
        .reindex(columns=STAGE_ORDER, fill_value=0)
    )
    clusters = clusters.loc[sorted(clusters.index, key=lambda x: int(x))]
    clusters.index.name = "leiden_harmony"
    clusters.to_csv(TABLES / "04_leiden_harmony_cluster_stage_counts.csv")
    clusters.div(clusters.sum(axis=1), axis=0).to_csv(
        TABLES / "04_leiden_harmony_cluster_stage_fraction.csv"
    )

    # Silhouette scores on a common subsample, before vs after Harmony.
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(
        adata.n_obs, size=min(SILHOUETTE_N, adata.n_obs), replace=False
    )
    sil_rows = []
    for embedding in ("X_pca", "X_pca_harmony"):
        for label in ("stage", "sequencing_batch"):
            score = silhouette_score(
                adata.obsm[embedding][idx],
                adata.obs[label].to_numpy()[idx],
                random_state=RANDOM_STATE,
            )
            sil_rows.append(
                {
                    "embedding": embedding,
                    "labels": label,
                    "silhouette": float(score),
                    "n_cells": len(idx),
                }
            )
    sil = pd.DataFrame(sil_rows)
    sil.to_csv(TABLES / "04_embedding_silhouette.tsv", sep="\t", index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "n_cells",
                "n_hvg",
                "n_pcs",
                "batch_key",
                "harmony_max_iter",
                "n_neighbors",
                "leiden_resolution",
                "n_leiden_harmony_clusters",
            ],
            "value": [
                adata.n_obs,
                int(adata.var["highly_variable"].sum()),
                adata.obsm["X_pca"].shape[1],
                "sequencing_batch",
                20,
                N_NEIGHBORS,
                LEIDEN_RESOLUTION,
                n_clusters,
            ],
        }
    )
    summary.to_csv(TABLES / "04_harmony_summary.tsv", sep="\t", index=False)

    # --- figures ----------------------------------------------------------
    common = dict(size=7, alpha=0.75, legend_loc="right margin", legend_fontsize=7)

    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    panel(
        ax, adata, "X_umap_harmony", "stage",
        "Harmony UMAP coloured by stage",
        palette=STAGE_COLORS, groups=STAGE_ORDER, **common,
    )
    save_figure(fig, "04_umap_harmony_by_stage")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    panel(
        ax, adata, "X_umap_harmony", "hours_post_fertilization",
        "Harmony UMAP coloured by hpf", cmap="viridis", **common,
    )
    save_figure(fig, "04_umap_harmony_by_hpf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    panel(
        ax, adata, "X_umap_harmony", "sequencing_batch",
        "Harmony UMAP coloured by sequencing batch", **common,
    )
    save_figure(fig, "04_umap_harmony_by_batch")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    panel(
        ax, adata, "X_umap_harmony", "leiden_harmony",
        "Harmony UMAP, Leiden clusters",
        legend_loc="on data", legend_fontsize=6, size=7, alpha=0.75,
    )
    save_figure(fig, "04_umap_harmony_by_leiden")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))
    panel(
        axes[0, 0], adata, "X_umap_harmony", "stage", "Stage",
        palette=STAGE_COLORS, groups=STAGE_ORDER,
        legend_loc="right margin", legend_fontsize=7, size=6, alpha=0.75,
    )
    panel(
        axes[0, 1], adata, "X_umap_harmony", "hours_post_fertilization",
        "Hours post-fertilisation", cmap="viridis",
        size=6, alpha=0.75,
    )
    panel(
        axes[1, 0], adata, "X_umap_harmony", "sequencing_batch",
        "Sequencing batch",
        legend_loc="right margin", legend_fontsize=7, size=6, alpha=0.75,
    )
    panel(
        axes[1, 1], adata, "X_umap_harmony", "leiden_harmony",
        "Leiden cluster",
        legend_loc="on data", legend_fontsize=6, size=6, alpha=0.75,
    )
    fig.suptitle("Harmony-corrected UMAP sanity check (3.3-6.0 hpf)", y=0.995)
    fig.tight_layout()
    save_figure(fig, "04_umap_harmony_overview")
    plt.close(fig)

    # Direct before/after comparison for stage colouring.
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
    panel(
        axes[0], adata, "X_umap", "stage", "Before Harmony (stage)",
        palette=STAGE_COLORS, groups=STAGE_ORDER,
        legend_loc="right margin", legend_fontsize=7, size=6, alpha=0.75,
    )
    panel(
        axes[1], adata, "X_umap_harmony", "stage", "After Harmony (stage)",
        palette=STAGE_COLORS, groups=STAGE_ORDER,
        legend_loc="right margin", legend_fontsize=7, size=6, alpha=0.75,
    )
    fig.suptitle("Effect of Harmony batch correction", y=1.02)
    fig.tight_layout()
    save_figure(fig, "04_umap_before_after_stage")
    plt.close(fig)

    print(f"[04] writing {HARMONY_H5AD}")
    adata.write_h5ad(HARMONY_H5AD, compression="gzip")

    print(sil.to_string(index=False))
    print("[04] cluster x stage counts (Harmony):")
    print(clusters.to_string())
    print(f"[04] done: {adata.n_obs:,} cells, {n_clusters} clusters")


if __name__ == "__main__":
    main()
