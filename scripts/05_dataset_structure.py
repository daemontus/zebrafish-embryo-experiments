#!/usr/bin/env python3
"""Stage 05: stage x batch x library structure of the full Drop-seq atlas.

Input : data/raw/GSE106587/GSE106474/GSE106474_UMICounts.txt.gz (streamed)
        data/metadata/dropseq_cell_metadata.csv
        data/metadata/soft/GSE106474_family.soft.gz
Output: results/tables/05_libraries.tsv          one row per GEO sample/library
        results/tables/05_stage_by_batch.tsv     cells per stage x batch
        results/tables/05_libraries_per_stage_batch.tsv stage x batch, number of libraries
        results/tables/05_cell_qc_full_atlas.csv per-cell UMIs / genes (39,505 cells)
        results/figures/05_dataset_structure.png/.pdf
        results/figures/05_library_qc.png/.pdf

Covers all 12 stages (3.3-12 hpf), not only the 3.3-6.0 hpf subset, so that
the batch design of the early window can be seen in context. Hierarchy:
stage -> sequencing batch (DS2-DS5) -> library (one GEO GSM; a second library
from the same stage and batch carries the "b" suffix in the cell name).
"""
from __future__ import annotations

import gzip

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_utils import (
    CELL_METADATA,
    FIGURES,
    METADATA,
    RAW_DROPSEQ,
    TABLES,
    ensure_dirs,
    save_figure,
)

SOFT = METADATA / "soft" / "GSE106474_family.soft.gz"
CELL_QC = TABLES / "05_cell_qc_full_atlas.csv"

ALL_STAGES = [
    "high", "oblong", "dome", "30% epiboly", "50% epiboly", "shield",
    "60% epiboly", "75% epiboly", "90% epiboly", "bud", "3-somite", "6-somite",
]
BATCHES = ["DS2", "DS3", "DS4", "DS5"]
BATCH_COLORS = {"DS2": "#4c72b0", "DS3": "#55a868", "DS4": "#c44e52", "DS5": "#8172b2"}
EARLY_WINDOW_HPF = 6.0

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read_soft_samples() -> pd.DataFrame:
    """GEO sample title, batch and SRA experiment per GSM."""
    rows, cur = [], None
    with gzip.open(SOFT, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE"):
                cur = {"geo_sample": line.split(" = ", 1)[1]}
                rows.append(cur)
            elif cur is None:
                continue
            elif line.startswith("!Sample_title"):
                cur["geo_title"] = line.split(" = ", 1)[1]
            elif line.startswith("!Sample_characteristics_ch1 = sequencing batch"):
                cur["geo_batch"] = line.split(": ", 1)[1]
            elif line.startswith("!Sample_relation = SRA"):
                cur["sra"] = line.rsplit("=", 1)[1]
    soft = pd.DataFrame(rows)
    soft["geo_rep"] = soft["geo_title"].str.extract(r"Rep (\d+)").astype(int)
    return soft


def per_cell_qc() -> pd.DataFrame:
    """Total UMIs and detected genes per cell, streaming the genes x cells TSV."""
    if CELL_QC.exists():
        return pd.read_csv(CELL_QC, index_col="cell_id")
    print(f"streaming {RAW_DROPSEQ.name} (genes x cells) ...")
    totals = detected = None
    for i, chunk in enumerate(
        pd.read_csv(RAW_DROPSEQ, sep="\t", index_col=0, chunksize=2000)
    ):
        values = chunk.to_numpy(dtype=np.float32)
        if totals is None:
            cells = chunk.columns
            totals = np.zeros(values.shape[1])
            detected = np.zeros(values.shape[1], dtype=np.int64)
        totals += values.sum(axis=0)
        detected += (values > 0).sum(axis=0)
        print(f"  genes processed: {(i + 1) * 2000}")
    qc = pd.DataFrame(
        {"total_counts": totals.astype(np.int64), "n_genes_by_counts": detected},
        index=pd.Index(cells, name="cell_id"),
    )
    qc.to_csv(CELL_QC)
    return qc


def library_table(meta: pd.DataFrame, qc: pd.DataFrame, soft: pd.DataFrame) -> pd.DataFrame:
    meta = meta.join(qc)
    meta["library"] = meta.index.str.rsplit("_", n=1).str[0]
    grouped = meta.groupby("library")
    libs = grouped.agg(
        stage=("stage", "first"),
        hpf=("hours_post_fertilization", "first"),
        batch=("sequencing_batch", "first"),
        replicate=("replicate", "first"),
        geo_sample=("geo_sample", "first"),
        n_cells=("stage", "size"),
        median_umis=("total_counts", "median"),
        median_genes=("n_genes_by_counts", "median"),
        total_umis=("total_counts", "sum"),
    )
    libs = libs.reset_index().merge(soft, on="geo_sample", how="left")
    assert (libs["batch"] == libs["geo_batch"]).all(), "batch mismatch vs GEO SOFT"
    libs["in_1_6hpf_window"] = libs["hpf"] <= EARLY_WINDOW_HPF
    libs["stage"] = pd.Categorical(libs["stage"], ALL_STAGES, ordered=True)
    libs = libs.sort_values(["stage", "batch", "replicate"]).reset_index(drop=True)
    cols = [
        "stage", "hpf", "batch", "replicate", "library", "geo_sample", "geo_rep",
        "sra", "n_cells", "median_umis", "median_genes", "total_umis",
        "in_1_6hpf_window",
    ]
    return libs[cols]


def plot_structure(libs: pd.DataFrame, stage_batch: pd.DataFrame) -> None:
    fig, (ax_bar, ax_grid) = plt.subplots(
        1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.4, 1]}
    )

    # Left: cells per stage, one stacked segment per library, coloured by batch.
    y = np.arange(len(ALL_STAGES))[::-1]
    for yi, stage in zip(y, ALL_STAGES):
        left = 0
        for _, lib in libs[libs["stage"] == stage].iterrows():
            ax_bar.barh(yi, lib["n_cells"], left=left, color=BATCH_COLORS[lib["batch"]],
                        edgecolor="white", linewidth=1.2, hatch="//" if lib["replicate"] == "b" else None)
            if lib["n_cells"] > 350:
                ax_bar.text(left + lib["n_cells"] / 2, yi, f"{lib['n_cells']:,}",
                            ha="center", va="center", fontsize=7, color="white")
            left += lib["n_cells"]
        ax_bar.text(left + 80, yi, f"{left:,}", va="center", fontsize=8)
    ax_bar.set_yticks(y, [f"{s} ({libs.loc[libs['stage'] == s, 'hpf'].iloc[0]:g} h)" for s in ALL_STAGES])
    ax_bar.axhline(y[ALL_STAGES.index("shield")] - 0.5, color="0.3", ls="--", lw=0.8)
    ax_bar.text(ax_bar.get_xlim()[1] * 0.98, y[ALL_STAGES.index("shield")] - 0.35,
                "end of 1–6 hpf window", ha="right", va="bottom", fontsize=7, color="0.3")
    ax_bar.set_xlabel("cells (segments = libraries; hatched = second library 'b')")
    ax_bar.set_title("Cells per stage, split by library")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BATCH_COLORS[b]) for b in BATCHES]
    ax_bar.legend(handles, BATCHES, title="batch", loc="upper right", frameon=False)

    # Right: stage x batch grid, bubble area = cells, label = libraries.
    nlib = libs.groupby(["stage", "batch"], observed=True).size()
    for yi, stage in zip(y, ALL_STAGES):
        for xi, batch in enumerate(BATCHES):
            n = stage_batch.loc[stage, batch]
            if n == 0:
                ax_grid.plot(xi, yi, marker=".", color="0.85")
                continue
            ax_grid.scatter(xi, yi, s=n / 6, color=BATCH_COLORS[batch], alpha=0.75)
            k = nlib.get((stage, batch), 0)
            ax_grid.text(xi + 0.22, yi, f"{n:,}" + (f" ({k} libs)" if k > 1 else ""),
                         va="center", fontsize=7)
    ax_grid.set_xticks(range(len(BATCHES)), BATCHES)
    ax_grid.set_yticks(y, ALL_STAGES)
    ax_grid.set_xlim(-0.5, len(BATCHES) - 0.1)
    ax_grid.axhline(y[ALL_STAGES.index("shield")] - 0.5, color="0.3", ls="--", lw=0.8)
    ax_grid.set_title("Stage × sequencing batch (area ∝ cells)")
    fig.tight_layout()
    save_figure(fig, "05_dataset_structure")
    plt.close(fig)


def plot_library_qc(meta: pd.DataFrame, qc: pd.DataFrame, libs: pd.DataFrame) -> None:
    cells = meta.join(qc)
    cells["library"] = cells.index.str.rsplit("_", n=1).str[0]
    order = libs["library"].tolist()
    data = [np.log10(cells.loc[cells["library"] == lib, "total_counts"]) for lib in order]
    colors = [BATCH_COLORS[b] for b in libs["batch"]]

    fig, ax = plt.subplots(figsize=(12, 4))
    parts = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, c in zip(parts["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    labels = [f"{s} {b}{'b' if r == 'b' else ''}" for s, b, r in
              zip(libs["stage"], libs["batch"], libs["replicate"])]
    ax.set_xticks(range(1, len(order) + 1), labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("log10 UMIs per cell")
    ax.set_title("Per-library sequencing depth (boxes without outliers; colour = batch)")
    n_early = int(libs["in_1_6hpf_window"].sum())
    ax.axvline(n_early + 0.5, color="0.3", ls="--", lw=0.8)
    fig.tight_layout()
    save_figure(fig, "05_library_qc")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    meta = pd.read_csv(CELL_METADATA, index_col="cell_id")
    qc = per_cell_qc()
    assert qc.index.equals(meta.index) or set(qc.index) == set(meta.index)
    qc = qc.loc[meta.index]
    soft = read_soft_samples()

    libs = library_table(meta, qc, soft)
    libs.to_csv(TABLES / "05_libraries.tsv", sep="\t", index=False, float_format="%.0f")

    stage_batch = (
        pd.crosstab(meta["stage"], meta["sequencing_batch"])
        .reindex(index=ALL_STAGES, columns=BATCHES, fill_value=0)
    )
    stage_batch.to_csv(TABLES / "05_stage_by_batch.tsv", sep="\t")
    nlib = (
        libs.groupby(["stage", "batch"], observed=True).size().unstack(fill_value=0)
        .reindex(index=ALL_STAGES, columns=BATCHES, fill_value=0)
    )
    nlib.to_csv(TABLES / "05_libraries_per_stage_batch.tsv", sep="\t")

    plot_structure(libs, stage_batch)
    plot_library_qc(meta, qc, libs)

    print(libs.to_string(index=False))
    print(f"\nwrote tables to {TABLES} and figures to {FIGURES}")


if __name__ == "__main__":
    main()
