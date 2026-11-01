"""Pseudobulk strength of the curated programs (and controls) across libraries.

The timeline of script 24 shows where a program *varies between cells*
(heterogeneous genes). A program expressed evenly in every cell (maternal
transcripts, the blastomere state) is invisible there. This script shows how
strongly each library expresses a program as a whole, whether or not its genes
vary between cells.

Pseudobulk: the summed counts of all cells of a library divided by its total
UMIs (CP10k, UMIs per 10,000). Dividing by the library total controls for
library size (number of cells and sequencing depth); it does not control for
composition: a program becomes stronger both when its cells become more
numerous and when they express it more, which is what "overall strength"
should mean. Libraries are not pooled: everything is computed per library and
averaged over the libraries of a stage.

Strength of a program in a library: the mean over its genes of the gene's
CP10k relative to the gene's peak (its highest stage, mean of the stage's
libraries), so 1 = every gene at its peak. Every gene weighs the same: the
summed CP10k ("share", also in the tables) is dominated by a few abundant,
broadly expressed genes (e.g., npm1a in the hatching gland, hspb1 in the
neural plate). Genes whose peak is below MIN_PEAK CP10k are left out (their
relative levels are noise).

Every program uses its fixed gene list (results/tables/programs.tsv) at every
stage.

Figure (report figure `program_pseudobulk`), in the rows and layout of the
timeline of script 24, but with squares (to tell the two figures apart):
square area = strength (0 to 1); box: the stages at which ZFIN records the
program (results/tables/curated_zfin_windows.tsv), as in the timeline. The
strength is relative to each gene's peak, so it does not show how abundant a
program is. Ringed: the leaders of a stage, the programs that have the
highest level (mean CP10k of its genes; not the summed share, which grows
with the number of genes) among the programs of their kind (identity:
anatomy and the maternal and blastomere states; signalling; other cell
states; controls) in at least one library of the stage. Per library, not
within a margin of the stage mean: the level differs by a median of ~20%
between the libraries of a stage, so a fixed margin would mostly measure
replicate noise. It replaces the library levels of the state programs that scripts 17
and 19 computed before.

Reads results/tables/programs.tsv (16) and the Drop-seq counts; takes about a
minute.

Outputs:
  results/tables/program_pseudobulk.tsv   (program, stage, library, strength,
                                            genes used, share = sum and level =
                                            mean CP10k of its genes)
  results/tables/program_pseudobulk_stages.tsv   (program, stage, mean
                                            strength and share, libraries,
                                            spread = max - min strength over
                                            the libraries, in_window = in the
                                            ZFIN window, kind, leader =
                                            libraries in which it leads)
  report/figures/program_pseudobulk.pdf
"""
from __future__ import annotations

import importlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from pipeline_utils import (
    REPORT_WIDTH_IN,
    STAGES,
    TABLES,
    ensure_dirs,
    load_dropseq_counts,
    report_style,
    save_report_figure,
)

overview = importlib.import_module("24_program_overview")

SCALE = 1e4  # CP10k: UMIs per 10,000 of the library
MIN_PEAK = 0.1  # CP10k: genes with a lower peak are left out
SIZE_TICKS = (0.1, 0.5, 1.0)
EARLY_STATES = ("state:maternal", "state:blastomere")  # compared with the anatomy programs
MARKER = "s"  # squares: not to be confused with the circles of the timeline (24)


def gene_lists(programs: pd.DataFrame) -> pd.DataFrame:
    """Fixed gene list of every curated program and control."""
    kept = programs[programs["kept"] & programs["set"].isin(["curated", "controls"])]
    return kept[["program", "gene"]].drop_duplicates()


def library_cp10k(adata) -> pd.DataFrame:
    """Pseudobulk CP10k: genes x libraries."""
    library = adata.obs["library"].astype(str).to_numpy()
    X = adata.X.tocsr()
    cols = {}
    for lib in pd.unique(library):
        sums = np.asarray(X[library == lib].sum(axis=0)).ravel()
        cols[lib] = SCALE * sums / sums.sum()
    return pd.DataFrame(cols, index=adata.var_names)


def strengths(genes: pd.DataFrame, cp10k: pd.DataFrame, stage_of: dict) -> pd.DataFrame:
    """Strength and share of every program in every library."""
    stage_mean = cp10k.T.groupby(pd.Series(stage_of)).mean().T  # genes x stages
    peak = stage_mean.max(axis=1)
    relative = cp10k.div(peak, axis=0)
    rows = []
    for program, g in genes.groupby("program"):
        present = cp10k.index.intersection(g["gene"])
        used = peak.loc[present].index[peak.loc[present] >= MIN_PEAK]
        for lib in cp10k.columns:
            rows.append(dict(program=program, stage=stage_of[lib], library=lib,
                             strength=relative.loc[used, lib].mean() if len(used) else np.nan,
                             genes=len(used), share=cp10k.loc[present, lib].sum(),
                             level=cp10k.loc[used, lib].mean() if len(used) else np.nan))
    return pd.DataFrame(rows)


def per_stage(lib_table: pd.DataFrame) -> pd.DataFrame:
    lib_table = lib_table.assign(kind=lib_table["program"].map(kind))
    measured = lib_table[lib_table["level"].notna()]
    top = measured.loc[measured.groupby(["kind", "library"])["level"].idxmax()]
    leads = top.groupby(["program", "stage"]).size()
    g = lib_table.groupby(["program", "stage"])
    out = pd.DataFrame(dict(strength=g["strength"].mean(), share=g["share"].mean(),
                            level=g["level"].mean(), libraries=g.size(),
                            spread=g["strength"].max() - g["strength"].min())).reset_index()
    zfin = overview.zfin_spans()
    out["in_window"] = [STAGES.index(s) in zfin[p] if p in zfin else True
                        for p, s in zip(out["program"], out["stage"])]
    out["kind"] = out["program"].map(kind)
    out["leader"] = [int(leads.get((p, s), 0)) for p, s in zip(out["program"], out["stage"])]
    out["stage"] = pd.Categorical(out["stage"], STAGES, ordered=True)
    return out.sort_values(["program", "stage"])


def kind(program: str) -> str:
    """Programs compared for the leaders: identity (anatomy and the early
    states, so that the shift from maternal / blastomere to anatomy shows),
    signalling, the other cell states, controls."""
    if program.startswith(("signal:", "receiving:", "sending:", "blocking:")):
        return "signalling"
    if program in EARLY_STATES:
        return "identity"
    return {"state": "cell states", "control": "controls"}.get(program.split(":")[0], "identity")


def area(strength: float) -> float:
    return 1 + 75 * strength


def figure(stages: pd.DataFrame, notes: pd.DataFrame) -> None:
    stages = stages[stages["strength"].notna()]
    rows = overview.row_order(notes, set(stages["program"]))
    table = stages.set_index(["program", "stage"])
    fig, ax = plt.subplots(figsize=(REPORT_WIDTH_IN, 0.155 * len(rows) + 1.0))
    zfin = overview.zfin_spans()
    for i, (p, group, _) in enumerate(rows):
        color = overview.GROUP_COLOR[group]
        ax.plot([0, len(STAGES) - 1], [i, i], color=color, lw=0.5, alpha=0.5, zorder=1)
        if p in zfin:
            overview.zfin_box(ax, i, zfin[p], color)
        for j, stage in enumerate(STAGES):
            row = table.loc[(p, stage)]
            size = area(row["strength"])
            if row["leader"]:  # halo: a white gap and a dark ring, visible on any fill
                side = np.sqrt(size)
                for extra, c, z in ((3.2, "#222222", 2.8), (1.8, "white", 2.9)):
                    ax.scatter(j, i, s=(side + extra) ** 2, marker=MARKER, color=c,
                               linewidths=0, zorder=z, clip_on=False)
            ax.scatter(j, i, s=size, marker=MARKER, color=(*to_rgb(color), 0.8), linewidths=0,
                       zorder=3, clip_on=False)
    overview.rows_axis(ax, rows)
    grey = to_rgb("#555555")
    handles = [Line2D([], [], ls="", marker=MARKER, ms=np.sqrt(area(v)), mfc=(*grey, 0.8), mec="none",
                      label=f"{v:g}") for v in SIZE_TICKS]
    size_legend = ax.legend(handles=handles, frameon=False, ncol=len(SIZE_TICKS), fontsize=6,
                            title="strength (mean level of the genes relative to their peak)",
                            title_fontsize=6, loc="upper left", bbox_to_anchor=(0.0, -0.09))
    ax.add_artist(size_legend)
    leader = (Line2D([], [], ls="", marker=MARKER, ms=8.2, mfc="#222222", mec="none"),
              Line2D([], [], ls="", marker=MARKER, ms=6.8, mfc="white", mec="none"),
              Line2D([], [], ls="", marker=MARKER, ms=5, mfc=(*grey, 0.8), mec="none"))
    ax.legend(handles=[leader, overview.zfin_legend()],
              labels=["leads its kind in >= 1 library of the stage",
                      overview.zfin_legend().get_label()], frameon=False, fontsize=6,
              loc="upper right", bbox_to_anchor=(1.0, -0.09))
    save_report_figure(fig, "program_pseudobulk")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    report_style()
    programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    notes = pd.read_csv(overview.ANNOTATION, sep="\t", keep_default_na=False)
    genes = gene_lists(programs)
    adata = load_dropseq_counts()
    stage_of = (adata.obs.groupby("library", observed=True)["stage"].first().astype(str)
                .to_dict())
    lib_table = strengths(genes, library_cp10k(adata), stage_of)
    lib_table.to_csv(TABLES / "program_pseudobulk.tsv", sep="\t", index=False,
                     float_format="%.4g")
    stages = per_stage(lib_table)
    stages.to_csv(TABLES / "program_pseudobulk_stages.tsv", sep="\t", index=False,
                  float_format="%.4g")
    figure(stages, notes)
    pd.set_option("display.width", 250)
    wide = stages.pivot(index="program", columns="stage", values="strength")
    print(wide.round(2).to_string())
    print("spread between the libraries of a stage: median",
          round(stages["spread"].median(), 3), "max", round(stages["spread"].max(), 3))
    print(lib_table.groupby("program")["genes"].first().to_string())


if __name__ == "__main__":
    main()
