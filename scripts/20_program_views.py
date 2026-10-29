"""Four views of the curated programs (exploration, not in the report).

Uses the per-cell program scores and active calls of script 19, the robust
populations, tracks and per-stage UMAPs of script 14, and the counts.

  1. program_populations_dots.pdf   per stage: robust populations (rows) x
     programs (columns); colour = mean score in the population (scores
     divided by their standard deviation over the stage's cells), size =
     share of the population's cells with the program active.
  2. program_coactivity.pdf         per stage: correlation of the program
     scores over the stage's cells (scores are centred per library, so this
     is co-variation within libraries), clustered.
  3. program_tracks.pdf             per track of script 14 (populations linked
     across >= MIN_TRACK_STAGES stages): share of active cells of every
     program in every population of the track (programs active in at least
     MIN_TRACK_ACTIVE of some population's cells).
  4. program_genes.pdf              per program: for every gene and stage
     where the gene belongs to the program, log2 of its detection rate in the
     program's active cells over that in the other cells of the stage; dots
     mark stages where the gene is not part of the program.

Outputs: results/figures/program_{populations_dots,coactivity,tracks,genes}.pdf
"""
from __future__ import annotations

import importlib
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from pipeline_utils import ROOT, STAGES, TABLES, load_dropseq_counts

atlas = importlib.import_module("19_program_atlas")
program_populations = importlib.import_module("18_program_populations")

FIGURES = ROOT / "results" / "figures"
ROBUST = program_populations.ROBUST
MIN_TRACK_STAGES = 3
MIN_TRACK_ACTIVE = 0.2
PSEUDO = 0.005  # detection rate added before the log ratio


def load():
    scores = pd.read_csv(atlas.SCORES, sep="\t", index_col=0)
    active = pd.read_csv(atlas.ACTIVE, sep="\t", index_col=0)
    cells = pd.read_csv(TABLES / "replicated_populations_cells.tsv.gz", sep="\t", index_col=0,
                        low_memory=False)
    cons = pd.read_csv(TABLES / "replicated_populations.tsv", sep="\t")
    cons = cons[cons["n_resolutions"] >= ROBUST]
    programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    curated = programs[(programs["set"] == "curated") & programs["kept"]]
    tracks = pd.read_csv(TABLES / "replicated_populations_tracks.tsv", sep="\t")
    return scores, active, cells, cons, curated, tracks


def population_members(cells: pd.DataFrame, cons: pd.DataFrame) -> dict:
    """(stage, consensus) -> cell index."""
    out = {}
    for r in cons.itertuples():
        out[(r.stage, r.consensus)] = program_populations.members(
            cells[cells["stage"] == r.stage], r.representative)
    return out


def label(r) -> str:
    return f"{r.consensus} {','.join(r.top_genes.split(',')[:3])}"


def stage_programs(scores: pd.DataFrame, cells_of_stage, order: list[str]) -> list[str]:
    defined = scores.loc[cells_of_stage].notna().any()
    return [p for p in order if p in defined.index and defined[p]]


# --- 1. programs x populations -----------------------------------------------------------------

def dots(pdf, scores, active, cells, cons, members, order) -> None:
    for stage in STAGES:
        idx = cells.index[cells["stage"] == stage]
        progs = stage_programs(scores, idx, order)
        pops = cons[cons["stage"] == stage].sort_values("cells", ascending=False)
        if pops.empty or not progs:
            continue
        z = scores.loc[idx, progs] / scores.loc[idx, progs].std()
        mean = np.array([[z.loc[members[(stage, r.consensus)], p].mean() for p in progs]
                         for r in pops.itertuples()])
        share = np.array([[active.loc[members[(stage, r.consensus)], p].mean() for p in progs]
                          for r in pops.itertuples()])
        fig, ax = plt.subplots(figsize=(3 + 0.32 * len(progs), 1.6 + 0.3 * len(pops)))
        yy, xx = np.meshgrid(np.arange(len(pops)), np.arange(len(progs)), indexing="ij")
        sc_ = ax.scatter(xx.ravel(), yy.ravel(), s=10 + 220 * np.nan_to_num(share.ravel()),
                         c=mean.ravel(), cmap="RdBu_r", norm=TwoSlopeNorm(0, -2, 2),
                         edgecolors="#555555", linewidths=0.3)
        ax.set_xticks(range(len(progs)), progs, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(pops)), [f"{label(r)} ({r.cells})" for r in pops.itertuples()],
                      fontsize=7)
        ax.set_ylim(len(pops) - 0.5, -0.5)
        ax.set_xlim(-0.5, len(progs) - 0.5)
        ax.set_title(f"{stage}: robust populations x curated programs "
                     "(colour: mean score in SD; size: share active)", fontsize=8)
        plt.colorbar(sc_, ax=ax, fraction=0.02, pad=0.01)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


# --- 2. co-activity ---------------------------------------------------------------------------------

def coactivity(pdf, scores, cells, order) -> None:
    for stage in STAGES:
        idx = cells.index[cells["stage"] == stage]
        progs = stage_programs(scores, idx, order)
        if len(progs) < 3:
            continue
        corr = scores.loc[idx, progs].corr().fillna(0)
        dist = squareform(np.clip(1 - corr.to_numpy(), 0, 2), checks=False)
        leaf = leaves_list(linkage(dist, "average"))
        c = corr.iloc[leaf, leaf]
        n = len(progs)
        fig, ax = plt.subplots(figsize=(2.5 + 0.28 * n, 1.8 + 0.28 * n))
        im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n), c.columns, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(n), c.index, fontsize=7)
        for i in range(n):
            for j in range(n):
                if i != j and abs(c.iat[i, j]) >= 0.5:
                    ax.text(j, i, f"{c.iat[i, j]:.1f}", ha="center", va="center", fontsize=5)
        ax.set_title(f"{stage}: correlation of program scores (within libraries)", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


# --- 3. tracks -------------------------------------------------------------------------------------

def track_views(pdf, active, tracks, members, order) -> None:
    stage_pos = {s: i for i, s in enumerate(STAGES)}
    for t, g in tracks.groupby("track"):
        if g["stage"].nunique() < MIN_TRACK_STAGES:
            continue
        g = g.assign(_s=g["stage"].map(stage_pos)).sort_values(["_s", "cells"],
                                                               ascending=[True, False])
        nodes = list(zip(g["stage"], g["consensus"]))
        share = pd.DataFrame({f"{s}\n{c}": active.loc[members[(s, c)]].mean()
                              for s, c in nodes if (s, c) in members})
        keep = [p for p in order if p in share.index and share.loc[p].max() >= MIN_TRACK_ACTIVE]
        if not keep:
            continue
        share = share.loc[keep]
        fig, ax = plt.subplots(figsize=(max(8.0, 3 + 0.8 * share.shape[1]), 1.6 + 0.28 * len(keep)))
        ax.set_facecolor("#F2F2F2")
        im = ax.imshow(share.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0,
                       vmax=1)
        ax.set_xticks(range(share.shape[1]), share.columns, fontsize=6)
        ax.set_yticks(range(len(keep)), keep, fontsize=7)
        first = g.iloc[g["cells"].to_numpy().argmax()]
        ax.set_title(textwrap.fill(f"track {t}: {first['top_genes']} (largest population); "
                                   "share of cells with the program active (grey: not defined)",
                                   110), fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


# --- 4. gene contributions -------------------------------------------------------------------------

def gene_views(pdf, active, cells, curated, order) -> None:
    adata = load_dropseq_counts()
    genes = sorted(set(curated["gene"]))
    gi = adata.var_names.get_indexer(genes)
    detected = pd.DataFrame((adata.X[:, gi] > 0).toarray(), index=adata.obs_names,
                            columns=genes)
    stage_of = cells["stage"]
    for p in order:
        rows = curated[curated["program"] == p]
        pg = list(dict.fromkeys(rows.sort_values("rank")["gene"]))
        member = rows.groupby("gene")["stage"].agg(set)
        ratio = pd.DataFrame(np.nan, index=pg, columns=STAGES)
        for stage in STAGES:
            idx = stage_of.index[stage_of == stage]
            if p not in active or active.loc[idx, p].isna().all():
                continue
            on = active.loc[idx, p] == 1
            if on.sum() == 0:
                continue
            d = detected.loc[idx, pg]
            ratio[stage] = np.log2((d[on.to_numpy()].mean() + PSEUDO)
                                   / (d[~on.to_numpy()].mean() + PSEUDO))
        # order genes by the number of stages, then by mean enrichment
        n = member.reindex(pg).map(len)
        pg = sorted(pg, key=lambda g: (-n[g], -np.nan_to_num(ratio.loc[g].mean())))
        ratio = ratio.loc[pg]
        h = 1.5 + 0.13 * len(pg)
        fig, ax = plt.subplots(figsize=(7, min(h, 40)))
        ax.set_facecolor("#F2F2F2")
        show = ratio.copy()
        for g in pg:
            for stage in STAGES:
                if stage not in member[g]:
                    show.loc[g, stage] = np.nan
        im = ax.imshow(show.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r",
                       norm=TwoSlopeNorm(0, -4, 4))
        for i, g in enumerate(pg):
            for j, stage in enumerate(STAGES):
                if stage not in member[g] and np.isfinite(ratio.loc[g, stage]):
                    ax.plot(j, i, ".", ms=2, color="#777777")
        ax.set_xticks(range(len(STAGES)), STAGES, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(pg)), [g.lower() for g in pg], fontsize=5 if len(pg) > 60 else 6)
        ax.set_title(f"{p}: log2 detection in active vs other cells "
                     "(grey dot: gene not in the program at that stage)", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    scores, active, cells, cons, curated, tracks = load()
    order = atlas.program_order(curated["program"].unique())
    members = population_members(cells, cons)
    with PdfPages(FIGURES / "program_populations_dots.pdf") as pdf:
        dots(pdf, scores, active, cells, cons, members, order)
    with PdfPages(FIGURES / "program_coactivity.pdf") as pdf:
        coactivity(pdf, scores, cells, order)
    with PdfPages(FIGURES / "program_tracks.pdf") as pdf:
        track_views(pdf, active, tracks, members, order)
    with PdfPages(FIGURES / "program_genes.pdf") as pdf:
        gene_views(pdf, active, cells, curated, order)


if __name__ == "__main__":
    main()
