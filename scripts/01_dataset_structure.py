#!/usr/bin/env python3
"""01: stage x batch x library structure of the Drop-seq atlas (GSE106474).

Input : data/metadata/dropseq_cell_metadata.csv
        data/metadata/soft/GSE106474_family.soft.gz
        results/tables/cell_qc_full_atlas.csv (per-cell UMIs / genes; if
          missing, recomputed by streaming the raw GEO matrix
          data/raw/GSE106587/GSE106474/GSE106474_UMICounts.txt.gz)
Output: results/tables/libraries.tsv          one row per GEO sample/library
        results/tables/stage_by_batch.tsv     cells per stage x batch
        results/tables/design_blocks.tsv      stage/batch blocks sharing no stage
        report/figures/dataset_structure.pdf  cells per stage/library/batch
        report/figures/library_depth.pdf      per-library UMIs and genes per cell
        report/tables/libraries.tex           library table (LaTeX tabular)

The atlas is a hierarchy stage -> sequencing batch (DS2-DS5) -> library (one
GEO GSM; a second library from the same stage and batch carries the "b" suffix
in the cell name) -> cells. No cells are removed here.
"""
from __future__ import annotations

import gzip

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd

from pipeline_utils import (
    BATCH_COLORS,
    BATCHES,
    CELL_METADATA,
    DROPSEQ_SOFT,
    RAW_DROPSEQ,
    REPORT_TABLES,
    REPORT_WIDTH_IN,
    STAGE_HPF,
    STAGES,
    TABLES,
    ensure_dirs,
    library_of,
    report_style,
    save_report_figure,
)

CELL_QC = TABLES / "cell_qc_full_atlas.csv"


def read_soft_samples() -> pd.DataFrame:
    """GEO sample title, batch and SRA experiment per GSM."""
    rows, cur = [], None
    with gzip.open(DROPSEQ_SOFT, "rt") as fh:
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


def library_table(cells: pd.DataFrame, soft: pd.DataFrame) -> pd.DataFrame:
    libs = cells.groupby("library").agg(
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
    libs["stage"] = pd.Categorical(libs["stage"], STAGES, ordered=True)
    libs = libs.sort_values(["stage", "batch", "replicate"]).reset_index(drop=True)
    cols = [
        "stage", "hpf", "batch", "replicate", "library", "geo_sample", "geo_rep",
        "sra", "n_cells", "median_umis", "median_genes", "total_umis",
    ]
    return libs[cols]


def design_blocks(stage_batch: pd.DataFrame) -> pd.DataFrame:
    """Connected components of the stage-batch graph (stages linked by batches).

    Within a block, stage and batch effects can (in principle) be separated;
    between blocks, batch is fully confounded with stage.
    """
    present = stage_batch > 0
    block_of, blocks = {}, []
    for stage in STAGES:
        if stage in block_of:
            continue
        stages, batches, todo = {stage}, set(), [stage]
        while todo:
            s = todo.pop()
            for b in present.columns[present.loc[s]]:
                if b not in batches:
                    batches.add(b)
                    for s2 in present.index[present[b]]:
                        if s2 not in stages:
                            stages.add(s2)
                            todo.append(s2)
        for s in stages:
            block_of[s] = len(blocks)
        blocks.append((stages, batches))
    rows = []
    for i, (stages, batches) in enumerate(blocks):
        stages = [s for s in STAGES if s in stages]
        rows.append(
            {
                "block": chr(ord("A") + i),
                "batches": ",".join(sorted(batches)),
                "stages": ",".join(stages),
                "n_cells": int(stage_batch.loc[stages].to_numpy().sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_structure(libs: pd.DataFrame, stage_batch: pd.DataFrame) -> None:
    fig, (ax_bar, ax_grid) = plt.subplots(
        1, 2, figsize=(REPORT_WIDTH_IN, 3.4), sharey=True,
        gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.08},
    )
    y = np.arange(len(STAGES))[::-1]

    # Left: cells per stage, one stacked segment per library, coloured by batch.
    for yi, stage in zip(y, STAGES):
        left = 0
        for _, lib in libs[libs["stage"] == stage].iterrows():
            ax_bar.barh(
                yi, lib["n_cells"], left=left, height=0.7,
                color=BATCH_COLORS[lib["batch"]], edgecolor="white", linewidth=1.0,
                hatch="////" if lib["replicate"] == "b" else None,
            )
            left += lib["n_cells"]
        ax_bar.text(left + 80, yi, f"{left:,}", va="center", fontsize=7)
    ax_bar.set_yticks(y, [f"{s} ({STAGE_HPF[s]:g} hpf)" for s in STAGES])
    ax_bar.tick_params(axis="y", length=0)
    ax_bar.set_xlim(0, 7600)
    ax_bar.set_xlabel("cells (segments = libraries, hatched = second library)")
    ax_bar.set_title("Cells per stage", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BATCH_COLORS[b]) for b in BATCHES]
    ax_bar.legend(handles, BATCHES, title="batch", loc="upper right", frameon=False)

    # Right: stage x batch grid, bubble area = cells, label = cells (libraries).
    nlib = libs.groupby(["stage", "batch"], observed=True).size()
    for yi, stage in zip(y, STAGES):
        for xi, batch in enumerate(BATCHES):
            n = stage_batch.loc[stage, batch]
            if n == 0:
                ax_grid.plot(xi, yi, marker=".", markersize=3, color="0.8")
                continue
            ax_grid.scatter(xi, yi, s=n / 40, color=BATCH_COLORS[batch],
                            edgecolor="white", linewidth=0.6, zorder=3)
            k = nlib.get((stage, batch), 0)
            ax_grid.text(xi + 0.17, yi, f"{n:,}" + (f" ({k})" if k > 1 else ""),
                         va="center", fontsize=6.5)
    ax_grid.set_xticks(range(len(BATCHES)), BATCHES)
    ax_grid.set_xlim(-0.3, len(BATCHES) - 0.15)
    ax_grid.tick_params(axis="y", length=0)
    ax_grid.spines["left"].set_visible(False)
    ax_grid.set_title("Stage × batch (area ∝ cells, (n) = libraries)", loc="left")
    save_report_figure(fig, "dataset_structure")
    plt.close(fig)


def plot_library_depth(cells: pd.DataFrame, libs: pd.DataFrame) -> None:
    order = libs["library"].tolist()
    colors = [BATCH_COLORS[b] for b in libs["batch"]]
    labels = [f"{s} {b}{'b' if r == 'b' else ''}" for s, b, r in
              zip(libs["stage"], libs["batch"], libs["replicate"])]
    # Separators between stages.
    bounds = np.flatnonzero(libs["stage"].to_numpy()[1:] != libs["stage"].to_numpy()[:-1]) + 1.5

    fig, axes = plt.subplots(2, 1, figsize=(REPORT_WIDTH_IN, 4.2), sharex=True)
    for ax, col, ylabel in zip(
        axes,
        ["total_counts", "n_genes_by_counts"],
        ["UMIs per cell", "genes per cell"],
    ):
        data = [cells.loc[cells["library"] == lib, col] for lib in order]
        parts = ax.boxplot(
            data, patch_artist=True, showfliers=False, widths=0.6,
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6},
            boxprops={"linewidth": 0.6},
        )
        for patch, c, r in zip(parts["boxes"], colors, libs["replicate"]):
            patch.set_facecolor(c)
            if r == "b":
                patch.set_hatch("////")
                patch.set_edgecolor("white")
        for x in bounds:
            ax.axvline(x, color="0.85", lw=0.6, zorder=0)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", which="major", color="0.92", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    axes[1].set_xticks(range(1, len(order) + 1), labels, rotation=60, ha="right",
                       rotation_mode="anchor")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BATCH_COLORS[b]) for b in BATCHES]
    axes[0].legend(handles, BATCHES, title="batch", ncol=4, loc="upper right",
                   frameon=False, bbox_to_anchor=(1.0, 1.25))
    fig.align_ylabels(axes)
    save_report_figure(fig, "library_depth")
    plt.close(fig)


def write_latex_rows(libs: pd.DataFrame) -> None:
    """Library table for the report; a new stage starts with its name."""
    lines, prev = [], None
    for _, lib in libs.iterrows():
        stage = lib["stage"].replace("%", r"\%")
        first = lib["stage"] != prev
        if first and prev is not None:
            lines.append(r"\addlinespace[2pt]")
        prev = lib["stage"]
        lines.append(
            " & ".join(
                [
                    stage if first else "",
                    f"{lib['hpf']:g}" if first else "",
                    lib["batch"],
                    lib["replicate"],
                    lib["geo_sample"],
                    f"{lib['n_cells']:,}".replace(",", "{,}"),
                    f"{lib['median_umis']:,.0f}".replace(",", "{,}"),
                    f"{lib['median_genes']:,.0f}".replace(",", "{,}"),
                ]
            )
            + r" \\"
        )
    # The whole tabular is generated, since \input inside a tabular breaks
    # the booktabs rules that follow it.
    header = [
        "% Generated by scripts/01_dataset_structure.py; do not edit.",
        r"\begin{tabular}{llcccrrr}",
        r"\toprule",
        r"Stage & hpf & Batch & Lib. & GEO sample & Cells & Median UMIs & Median genes \\",
        r"\midrule",
    ]
    footer = [r"\bottomrule", r"\end{tabular}"]
    (REPORT_TABLES / "libraries.tex").write_text("\n".join(header + lines + footer) + "\n")


def main() -> None:
    ensure_dirs()
    report_style()
    meta = pd.read_csv(CELL_METADATA, index_col="cell_id")
    qc = per_cell_qc()
    assert set(qc.index) == set(meta.index), "QC table does not match metadata"
    cells = meta.join(qc)
    cells["library"] = library_of(cells.index)

    libs = library_table(cells, read_soft_samples())
    libs.to_csv(TABLES / "libraries.tsv", sep="\t", index=False)

    stage_batch = (
        pd.crosstab(meta["stage"], meta["sequencing_batch"])
        .reindex(index=STAGES, columns=BATCHES, fill_value=0)
    )
    stage_batch.to_csv(TABLES / "stage_by_batch.tsv", sep="\t")
    blocks = design_blocks(stage_batch)
    blocks.to_csv(TABLES / "design_blocks.tsv", sep="\t", index=False)

    plot_structure(libs, stage_batch)
    plot_library_depth(cells, libs)
    write_latex_rows(libs)

    print(libs.to_string(index=False))
    print()
    print(stage_batch.to_string())
    print()
    print(blocks.to_string(index=False))
    for block in blocks["block"]:
        sel = libs["batch"].isin(blocks.loc[blocks["block"] == block, "batches"].iloc[0].split(","))
        print(f"block {block}: {sel.sum()} libraries")
    print(f"\ncells: {len(cells):,}, libraries: {len(libs)}")
    print(f"median UMIs/cell overall: {cells['total_counts'].median():.0f}, "
          f"median genes/cell: {cells['n_genes_by_counts'].median():.0f}")


if __name__ == "__main__":
    main()
