#!/usr/bin/env python3
"""Stage 02: QC metrics, summary tables and histograms for the 3.3-6.0 hpf subset.

Input : data/processed/dropseq_3.3-6.0hpf_raw.h5ad
Output: data/processed/dropseq_3.3-6.0hpf_qc.h5ad   (raw counts + QC metrics)
        results/tables/02_qc_cell_metrics.csv
        results/tables/02_qc_summary_overall.csv
        results/tables/02_qc_summary_by_stage.csv
        results/tables/02_top_genes.csv
        results/tables/02_qc_notes.txt
        results/figures/02_*.png/.pdf

No cells are removed here; this stage only annotates QC metrics. Cell/gene
filtering is explicit in stage 03.
"""
from __future__ import annotations

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
    SUBSET_H5AD,
    TABLES,
    ensure_dirs,
    save_figure,
)

METRICS = [
    "total_counts",
    "n_genes_by_counts",
    "pct_counts_ribo",
    "pct_counts_in_top_50_genes",
]

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def describe(values: np.ndarray) -> dict[str, float]:
    """Summary statistics used in the QC tables."""
    v = np.asarray(values, dtype=float)
    return {
        "n": len(v),
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)),
        "min": float(np.min(v)),
        "p1": float(np.percentile(v, 1)),
        "p5": float(np.percentile(v, 5)),
        "p25": float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "p75": float(np.percentile(v, 75)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": float(np.max(v)),
    }


def hist_panel(
    ax,
    values: np.ndarray,
    xlabel: str,
    title: str,
    *,
    logx: bool = False,
    bins: int = 60,
    vline: float | None = None,
) -> None:
    if logx:
        values = values[values > 0]
        ax.hist(np.log10(values), bins=bins, color="#4c72b0", alpha=0.8)
        ax.set_xlabel(f"log10({xlabel})")
        if vline is not None and vline > 0:
            ax.axvline(np.log10(vline), color="black", ls="--", lw=1)
    else:
        ax.hist(values, bins=bins, color="#4c72b0", alpha=0.8)
        ax.set_xlabel(xlabel)
        if vline is not None:
            ax.axvline(vline, color="black", ls="--", lw=1)
    ax.set_ylabel("cells")
    ax.set_title(title)


def main() -> None:
    ensure_dirs()
    print(f"[02] loading {SUBSET_H5AD}")
    adata = sc.read_h5ad(SUBSET_H5AD)
    adata.obs["stage"] = pd.Categorical(
        adata.obs["stage"], categories=STAGE_ORDER, ordered=True
    )

    # The Drop-seq matrix contains only nuclear genes; no mt-/MT- genes were
    # found in the GEO feature list. Ribosomal fraction is a useful proxy for
    # technical/translational load and is computed instead.
    adata.var["ribo"] = adata.var_names.str.upper().str.startswith(("RPL", "RPS"))
    n_ribo = int(adata.var["ribo"].sum())
    print(f"[02] ribosomal genes: {n_ribo}")

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["ribo"],
        percent_top=[50],
        log1p=False,
        inplace=True,
    )

    # --- cell-level table -------------------------------------------------
    cell_cols = [
        "stage",
        "hours_post_fertilization",
        "sequencing_batch",
        "replicate",
        "geo_sample",
        *METRICS,
    ]
    cells = adata.obs[cell_cols].copy()
    cells.index.name = "cell_id"
    cells.to_csv(TABLES / "02_qc_cell_metrics.csv")

    # --- overall summary --------------------------------------------------
    rows = []
    for metric in METRICS:
        stats = describe(adata.obs[metric].to_numpy())
        for name, value in stats.items():
            rows.append({"metric": metric, "statistic": name, "value": value})
    pd.DataFrame(rows).to_csv(
        TABLES / "02_qc_summary_overall.csv", index=False
    )

    # --- per-stage summary ------------------------------------------------
    stage_rows = []
    for stage in STAGE_ORDER:
        sub = adata.obs[adata.obs["stage"] == stage]
        row = {
            "stage": stage,
            "hours_post_fertilization": STAGE_HPF[stage],
            "n_cells": len(sub),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_median"] = float(sub[metric].median())
            row[f"{metric}_sd"] = float(sub[metric].std(ddof=1))
            row[f"{metric}_min"] = float(sub[metric].min())
            row[f"{metric}_max"] = float(sub[metric].max())
        stage_rows.append(row)
    pd.DataFrame(stage_rows).to_csv(
        TABLES / "02_qc_summary_by_stage.csv", index=False
    )

    batch_table = pd.crosstab(adata.obs["stage"], adata.obs["sequencing_batch"])
    batch_table = batch_table.reindex(STAGE_ORDER).fillna(0).astype(int)
    batch_table.index.name = "stage"
    batch_table.to_csv(TABLES / "02_cells_by_stage_batch.tsv", sep="\t")

    # --- gene-level summaries --------------------------------------------
    genes = adata.var[["total_counts", "n_cells_by_counts"]].copy()
    genes["pct_cells_detected"] = 100 * genes["n_cells_by_counts"] / adata.n_obs
    genes.index.name = "gene"
    genes.sort_values("total_counts", ascending=False).head(50).to_csv(
        TABLES / "02_top_genes.csv"
    )

    with (TABLES / "02_qc_notes.txt").open("w", encoding="utf-8") as fh:
        fh.write("QC notes for dropseq_3.3-6.0hpf_raw.h5ad\n")
        fh.write("============================================\n")
        fh.write(f"cells: {adata.n_obs:,}\n")
        fh.write(f"genes: {adata.n_vars:,}\n")
        fh.write(f"ribosomal genes (RPL*/RPS*): {n_ribo}\n")
        fh.write(
            "mitochondrial genes: 0 in the GEO feature list; "
            "pct_counts_mt was not computed.\n"
        )
        fh.write(
            "No cells were removed at this stage. Stage 03 applies explicit "
            "min_genes/min_counts filters (which currently keep all cells) "
            "and a min_cells gene filter.\n"
        )

    # --- figures ----------------------------------------------------------
    total = adata.obs["total_counts"].to_numpy()
    ngene = adata.obs["n_genes_by_counts"].to_numpy()
    ribo = adata.obs["pct_counts_ribo"].to_numpy()
    top50 = adata.obs["pct_counts_in_top_50_genes"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    hist_panel(
        ax, total, "total UMI counts per cell", "Total counts per cell",
        logx=True, vline=float(np.median(total)),
    )
    ax.text(
        0.98, 0.95, f"median = {np.median(total):,.0f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
    )
    save_figure(fig, "02_hist_total_counts")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    hist_panel(
        ax, ngene, "genes detected per cell", "Genes per cell",
        logx=True, vline=float(np.median(ngene)),
    )
    ax.text(
        0.98, 0.95, f"median = {np.median(ngene):,.0f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
    )
    save_figure(fig, "02_hist_genes_per_cell")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    hist_panel(
        ax, ribo, "ribosomal counts (%)", "Ribosomal fraction per cell",
        logx=False,
    )
    ax.text(
        0.98, 0.95, f"median = {np.median(ribo):.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
    )
    save_figure(fig, "02_hist_pct_ribo")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    hist_panel(
        ax, top50, "counts in top 50 genes (%)",
        "Fraction of counts in top 50 genes", logx=False,
    )
    ax.text(
        0.98, 0.95, f"median = {np.median(top50):.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
    )
    save_figure(fig, "02_hist_top50_fraction")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    counts = (
        adata.obs["stage"].value_counts().reindex(STAGE_ORDER).fillna(0)
    )
    bars = ax.bar(
        STAGE_ORDER, counts.to_numpy(),
        color=[STAGE_COLORS[s] for s in STAGE_ORDER],
    )
    ax.set_ylabel("cells")
    ax.set_title("Cells per stage (3.3-6.0 hpf subset)")
    ax.tick_params(axis="x", rotation=30)
    for bar, value in zip(bars, counts.to_numpy()):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{int(value):,}", ha="center", va="bottom", fontsize=8,
        )
    save_figure(fig, "02_cells_per_stage")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for stage in STAGE_ORDER:
        mask = adata.obs["stage"] == stage
        ax.scatter(
            total[mask], ngene[mask], s=4, alpha=0.45,
            color=STAGE_COLORS[stage], label=stage, linewidths=0,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total UMI counts per cell")
    ax.set_ylabel("genes detected per cell")
    ax.set_title("Counts vs genes detected, coloured by stage")
    ax.legend(markerscale=3, fontsize=7, frameon=False)
    save_figure(fig, "02_scatter_counts_vs_genes")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharex=True)
    plot_df = adata.obs[["stage", "total_counts", "n_genes_by_counts", "pct_counts_ribo"]].copy()
    for ax, metric, ylabel, logy in zip(
        axes,
        ["total_counts", "n_genes_by_counts", "pct_counts_ribo"],
        ["total UMI counts", "genes detected", "ribosomal counts (%)"],
        [True, True, False],
    ):
        sns.violinplot(
            data=plot_df, x="stage", y=metric, order=STAGE_ORDER,
            hue="stage", palette=STAGE_COLORS, legend=False,
            cut=0, inner="quartile", ax=ax, linewidth=0.6,
        )
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("QC metrics by stage", y=1.02)
    fig.tight_layout()
    save_figure(fig, "02_qc_violin_by_stage")
    plt.close(fig)

    # Gene detection across cells (how many genes pass a min_cells threshold).
    detected = genes["n_cells_by_counts"].to_numpy()
    positive = detected[detected > 0]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(np.log10(positive), bins=60, color="#55a868", alpha=0.8)
    ax.axvline(np.log10(3), color="black", ls="--", lw=1,
               label="min_cells = 3")
    ax.set_xlabel("log10(number of cells expressing gene)")
    ax.set_ylabel("genes")
    ax.set_title("Gene detection across cells")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "02_hist_gene_detection")
    plt.close(fig)

    # 2x2 overview for a quick visual check.
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    hist_panel(
        axes[0, 0], total, "total UMI counts per cell",
        "Total counts per cell", logx=True, vline=float(np.median(total)),
    )
    hist_panel(
        axes[0, 1], ngene, "genes detected per cell", "Genes per cell",
        logx=True, vline=float(np.median(ngene)),
    )
    hist_panel(
        axes[1, 0], ribo, "ribosomal counts (%)",
        "Ribosomal fraction per cell",
    )
    axes[1, 1].bar(
        STAGE_ORDER, counts.to_numpy(),
        color=[STAGE_COLORS[s] for s in STAGE_ORDER],
    )
    axes[1, 1].set_ylabel("cells")
    axes[1, 1].set_title("Cells per stage")
    axes[1, 1].tick_params(axis="x", rotation=30)
    fig.suptitle("QC overview: Drop-seq, 3.3-6.0 hpf", y=1.01)
    fig.tight_layout()
    save_figure(fig, "02_qc_overview")
    plt.close(fig)

    print(f"[02] writing {QC_H5AD}")
    adata.write_h5ad(QC_H5AD, compression="gzip")

    print(
        f"[02] done: {adata.n_obs:,} cells; median counts "
        f"{np.median(total):,.0f}, median genes {np.median(ngene):,.0f}"
    )


if __name__ == "__main__":
    main()
