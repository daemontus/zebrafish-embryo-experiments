"""Which programs are co-active with which signalling readouts, at which stage?

For every stage, the smoothed scores of the curated programs (as in scripts
18 and 19) are correlated over the stage's cells (scores are centred per
library, so this is co-variation within libraries). Genes of the signalling
readouts (data/curated/signalling_programs.tsv) are first removed from all
other programs of the stage: many readouts are also identity markers (lft1,
ndr1 in the marginal mesendoderm, id1, bambia in the organizer and
prechordal plate programs), and shared genes would make correlations by
construction. A correlation means that a pathway's readouts are expressed
in the cells of a program at that stage; it does not show that the signal
causes the program. Smoothing over neighbouring cells inflates absolute
values; changes over time and signs are more informative.

Outputs:
  results/tables/signal_coupling.tsv          (stage, signal, program, r)
  results/tables/signal_coupling_removed.tsv  (genes removed from programs)
  report/figures/signal_coupling.pdf
"""
from __future__ import annotations

import importlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from pipeline_utils import (
    REPORT_WIDTH_IN,
    STAGES,
    TABLES,
    ensure_dirs,
    load_dropseq_counts,
    report_style,
    save_report_figure,
)

program_populations = importlib.import_module("18_program_populations")

MIN_R = 0.3  # programs shown: |r| >= MIN_R at some stage
SIGNALS = ["Nodal", "BMP", "Wnt", "FGF", "Notch", "RA"]


def without_readouts(curated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Curated programs with the readout genes of their stage removed."""
    is_signal = curated["program"].str.startswith("signal:")
    readouts = curated[is_signal].groupby("stage")["gene"].agg(set)
    shared = np.array([g in readouts.get(s, set())
                       for g, s in zip(curated["gene"], curated["stage"])])
    removed = curated[~is_signal & shared]
    return curated[is_signal | ~shared], removed


def coupling(adata, programs: pd.DataFrame, support: pd.DataFrame,
             genes_table: pd.DataFrame) -> pd.DataFrame:
    space = program_populations.ProgramSpace(programs, support, genes_table,
                                             supported_only=False)
    rows = []
    for stage in STAGES:
        sub = adata[adata.obs["stage"] == stage]
        libs = sorted(sub.obs["library"].unique())
        raw, _, _ = space(sub, libs, stage)
        names = list(space.stage_programs(stage).index)
        corr = pd.DataFrame(np.vstack([raw[lib] for lib in libs]), columns=names).corr()
        for signal in [n for n in names if n.startswith("signal:")]:
            for p in names:
                if not p.startswith("signal:"):
                    rows.append(dict(stage=stage, signal=signal.removeprefix("signal:"),
                                     program=p.removeprefix("state:"),
                                     r=corr.loc[signal, p]))
        print(stage, flush=True)
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame) -> None:
    panels = []
    for signal in SIGNALS:
        t = table[table["signal"] == signal].pivot(index="program", columns="stage", values="r")
        t = t.reindex(columns=STAGES)
        t = t[(t.abs() >= MIN_R).any(axis=1)]
        t = t.loc[t.abs().max(axis=1).sort_values(ascending=False).index]
        panels.append((signal, t))
    left, right = panels[0::2], panels[1::2]
    heights = [max(sum(len(t) for _, t in col), 1) for col in (left, right)]
    fig = plt.figure(figsize=(REPORT_WIDTH_IN, 0.13 * max(heights) + 1.6))
    for c, col in enumerate((left, right)):
        grid = fig.add_gridspec(len(col), 1, height_ratios=[max(len(t), 1) for _, t in col],
                                left=0.17 + 0.5 * c, right=0.47 + 0.5 * c, top=0.96,
                                bottom=0.15, hspace=0.45)
        for k, (signal, t) in enumerate(col):
            ax = fig.add_subplot(grid[k])
            ax.set_facecolor("#F2F2F2")
            im = ax.imshow(t.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r",
                           norm=TwoSlopeNorm(0, -1, 1))
            ax.set_yticks(range(len(t)), [p.replace("_", " ") for p in t.index], fontsize=5)
            last = k == len(col) - 1
            ax.set_xticks(range(len(STAGES)), STAGES if last else [], rotation=60, ha="right",
                          fontsize=5)
            ax.set_title(signal, fontsize=7, loc="left")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    cax = fig.add_axes([0.35, 0.025, 0.3, 0.01])
    fig.colorbar(im, cax=cax, orientation="horizontal").ax.tick_params(labelsize=5)
    cax.set_title("correlation with the readouts (within libraries)", fontsize=6)
    save_report_figure(fig, "signal_coupling")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    report_style()
    adata = load_dropseq_counts()
    programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    curated = programs[(programs["set"] == "curated") & programs["kept"]]
    stripped, removed = without_readouts(curated)
    removed.to_csv(TABLES / "signal_coupling_removed.tsv", sep="\t", index=False)
    print(f"{len(removed)} readout genes removed from other programs")
    support = pd.read_csv(TABLES / "program_support.tsv", sep="\t")
    genes_table = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                              usecols=["stage", "gene", "heterogeneous"])
    table = coupling(adata, stripped, support, genes_table)
    table.to_csv(TABLES / "signal_coupling.tsv", sep="\t", index=False, float_format="%.3g")
    plot(table)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["plot"]:
        report_style()
        plot(pd.read_csv(TABLES / "signal_coupling.tsv", sep="\t"))
    else:
        main()
