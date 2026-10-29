"""Where do the curated programs live, and how do they change over the stages?

An exploration figure (not in the report) for every curated program of
script 16, at every stage where it is defined:
  * the score of the program (mean centred Pearson residual of its genes,
    smoothed over each cell and its 10 nearest neighbours in its library,
    as in script 18) on the per-stage UMAPs of script 14 (aligned data, so
    the panels of one stage are comparable with the population pages of
    results/figures/replicated_populations.pdf);
  * the share of the stage's cells in which the program is active (smoothed
    score above the 99.9% quantile of a gene-wise permutation within the
    library; median over the stage's libraries), whether it is supported
    (script 17), and its genes.
The bottom row of every program page shows the program on the atlas UMAPs
of script 15 (aligned and unaligned; scores scaled per stage, since a
program's genes differ between stages), next to the stages of the unaligned
UMAP. These are pictures only: the aligned atlas blurs adjacent stages, the
unaligned one mixes stage with batch.
The first pages are overviews over all programs and stages: the share of
active cells, and the library-level expression (pseudobulk of the program's
genes at the stage, UMIs per 10,000, median over genes, relative to each
gene's maximum over all 28 libraries; median over the stage's libraries),
which shows programs that act on whole libraries (maternal, blastomere) and
are invisible within a library.

Two more files show only the atlas UMAP, one page per program (the stages
first): program_atlas_aligned.pdf and program_atlas_unaligned.pdf.

`19_program_atlas.py plot` redraws from the saved scores and table;
`19_program_atlas.py atlas` redraws only the two atlas files. A third,
program_atlas_combined.pdf, shows groups of programs expected to be mutually
exclusive (EXCLUSIVE_GROUPS) on one UMAP each, coloured by the active program
(`19_program_atlas.py combined` redraws it).

Outputs:
  results/figures/program_atlas.pdf
  results/figures/program_atlas_{aligned,unaligned,combined}.pdf
  results/tables/program_atlas.tsv            (program x stage: genes,
                                               supported, active share, level)
  data/processed/program_scores.tsv.gz        (cell x program smoothed scores)
  data/processed/program_active.tsv.gz        (cell x program active calls)
"""
from __future__ import annotations

import importlib
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

from pipeline_utils import PROCESSED, ROOT, STAGES, TABLES, ensure_dirs, load_dropseq_counts

warnings.filterwarnings("ignore", category=FutureWarning)
program_populations = importlib.import_module("18_program_populations")
alignment = importlib.import_module("15_alignment")

FIGURES = ROOT / "results" / "figures"
SCORES = PROCESSED / "program_scores.tsv.gz"
ACTIVE = PROCESSED / "program_active.tsv.gz"  # 1 / 0, empty where not defined
# Display order: programs roughly by germ layer, then signalling and states.
ORDER = ["maternal", "blastomere", "EVL", "PGC", "dorsal_organizer", "dorsal_forerunner",
         "marginal_mesendoderm", "endoderm", "prechordal_plate", "notochord", "ventral_bmp",
         "presomitic_mesoderm", "adaxial_somite", "tailbud", "lateral_plate",
         "intermediate_mesoderm", "heart", "neural_ectoderm", "forebrain_eye",
         "midbrain_hindbrain_boundary", "hindbrain", "floor_plate", "primary_neurons",
         "neural_crest", "placodes", "non_neural_ectoderm"]


def program_order(names) -> list[str]:
    names = set(names)
    signal = sorted(n for n in names if n.startswith("signal:"))
    state = sorted(n for n in names if n.startswith("state:"))
    known = [n for n in ORDER if n in names]
    rest = sorted(names - set(known) - set(signal) - set(state))
    return known + rest + signal + state


def library_levels(adata, programs: pd.DataFrame) -> pd.DataFrame:
    """Pseudobulk level of every program's stage genes in every library of the stage."""
    library = adata.obs["library"].astype(str).to_numpy()
    X = adata.X.tocsr()
    depth = np.asarray(X.sum(axis=1)).ravel()
    libs = np.unique(library)
    bulk = np.vstack([np.asarray(X[library == lib].sum(axis=0)).ravel()
                      / depth[library == lib].sum() * 1e4 for lib in libs])
    rel = pd.DataFrame(bulk / np.maximum(bulk.max(axis=0), 1e-12), index=libs,
                       columns=adata.var_names)
    stage_of = adata.obs.groupby("library", observed=True)["stage"].first().astype(str)
    rows = []
    for (stage, program), g in programs.groupby(["stage", "program"]):
        for lib in stage_of.index[stage_of == stage]:
            rows.append(dict(stage=stage, program=program, library=lib,
                             level=float(rel.loc[lib, g["gene"]].median())))
    return pd.DataFrame(rows)


def heatmap(ax, table: pd.DataFrame, programs: list[str], title: str, cmap: str,
            vmax: float, supported: pd.DataFrame | None = None) -> None:
    t = table.reindex(index=programs, columns=STAGES)
    ax.set_facecolor("#F2F2F2")
    im = ax.imshow(t.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(STAGES)), STAGES, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(programs)), programs, fontsize=7)
    ax.set_title(title, fontsize=9)
    if supported is not None:
        s = supported.reindex(index=programs, columns=STAGES)
        for i, p in enumerate(programs):
            for j, st in enumerate(STAGES):
                if np.isfinite(t.loc[p, st]) and not bool(s.loc[p, st]):
                    ax.text(j, i, "x", ha="center", va="center", fontsize=8, color="white",
                            weight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)


def scaled_scores(program: str, atlas: pd.DataFrame, scores: pd.DataFrame) -> pd.Series:
    """Scores divided by their 99.9% quantile within each stage (programs are
    scaled per stage); NaN where the program is not defined."""
    v = scores[program].reindex(atlas.index) if program in scores else \
        pd.Series(np.nan, index=atlas.index)
    scaled = v.copy()
    for stage, idx in atlas.groupby("stage").groups.items():
        lim = np.nanquantile(np.abs(v.loc[idx]), 0.999) if v.loc[idx].notna().any() else np.nan
        scaled.loc[idx] = v.loc[idx] / lim if lim else np.nan
    return scaled


def draw_program(ax, xy: np.ndarray, scaled: pd.Series, size: float) -> None:
    """Program on atlas coordinates: grey where not defined, strongest on top."""
    defined = scaled.notna().to_numpy()
    order = np.argsort(np.abs(np.nan_to_num(scaled.to_numpy())))
    order = order[defined[order]]
    ax.set_facecolor("#EBEBEB")
    ax.scatter(xy[~defined, 0], xy[~defined, 1], s=size * 0.6, c="#D0D0D0", linewidths=0,
               rasterized=True)
    ax.scatter(xy[order, 0], xy[order, 1], s=size, c=scaled.to_numpy()[order], cmap="RdBu_r",
               norm=TwoSlopeNorm(0, -1, 1), linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_stages(ax, xy: np.ndarray, atlas: pd.DataFrame, size: float, fontsize: float) -> None:
    stage_idx = atlas["stage"].map({s: i for i, s in enumerate(STAGES)}).to_numpy()
    shuffled = np.random.default_rng(0).permutation(len(atlas))
    ax.scatter(xy[shuffled, 0], xy[shuffled, 1], s=size,
               c=plt.get_cmap("viridis")(stage_idx[shuffled] / (len(STAGES) - 1)),
               linewidths=0, rasterized=True)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3,
                          color=plt.get_cmap("viridis")(i / (len(STAGES) - 1)), label=st)
               for i, st in enumerate(STAGES)]
    ax.legend(handles=handles, fontsize=fontsize, frameon=False, loc="center left",
              bbox_to_anchor=(1.0, 0.5))
    ax.set_xticks([])
    ax.set_yticks([])


def atlas_panels(fig, grid, program: str, atlas: pd.DataFrame, scores: pd.DataFrame) -> None:
    """Bottom row: the program on the aligned and the unaligned atlas UMAP, and
    the stages of the unaligned UMAP for orientation. Scores are divided by
    their 99.9% quantile within each stage, since programs differ between
    stages; cells of stages where the program is not defined are grey."""
    scaled = scaled_scores(program, atlas, scores)
    for k, (prefix, title) in enumerate([("aligned", "atlas, aligned"),
                                         ("unaligned", "atlas, unaligned")]):
        ax = fig.add_subplot(grid[3, 4 * k:4 * k + 4])
        draw_program(ax, atlas[[f"{prefix}_1", f"{prefix}_2"]].to_numpy(), scaled, 0.5)
        ax.set_title(f"{title} (scores scaled per stage; grey: not defined)", fontsize=8)
    ax = fig.add_subplot(grid[3, 8:12])
    draw_stages(ax, atlas[["unaligned_1", "unaligned_2"]].to_numpy(), atlas, 0.3, 5.5)
    ax.set_title("atlas, unaligned: stages", fontsize=8)


# Groups of programs expected to be (mostly) mutually exclusive in a cell at
# the stages where they coexist; drawn together by atlas_combined().
EXCLUSIVE_GROUPS = {
    "germ layers": ["EVL", "endoderm", "mesoderm", "neural_plate", "non_neural_ectoderm"],
    "dorsal / ventral (gastrula)": ["dorsal_organizer", "ventral_bmp", "presumptive_neural",
                                    "non_neural_ectoderm"],
    "axial and dorsal midline": ["dorsal_forerunner", "prechordal_plate", "axial_mesoderm",
                                 "notochord", "hatching_gland"],
    "mesoderm (segmentation)": ["presomitic_mesoderm", "adaxial_somite", "lateral_plate",
                                "intermediate_mesoderm", "tailbud", "notochord"],
    "neural regions": ["forebrain_eye", "midbrain_hindbrain_boundary", "hindbrain",
                       "neural_crest", "placodes", "primary_neurons"],
}
GROUP_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def atlas_combined(atlas: pd.DataFrame, active: pd.DataFrame) -> None:
    """results/figures/program_atlas_combined.pdf: per group of EXCLUSIVE_GROUPS,
    the aligned and the unaligned atlas UMAP with every cell coloured by the
    group's program active in it (active call of the cache), black if several
    are active; light grey: none active; pale: no program of the group defined
    at the cell's stage."""
    a = active.reindex(atlas.index)
    with PdfPages(FIGURES / "program_atlas_combined.pdf") as pdf:
        for name, programs in EXCLUSIVE_GROUPS.items():
            programs = [p for p in programs if p in a]
            on = a[programs].fillna(0).to_numpy() > 0
            defined = a[programs].notna().any(axis=1).to_numpy()
            n_on = on.sum(axis=1)
            colour = np.full(len(a), "#D9D9D9", dtype=object)
            colour[~defined] = "#F2F2F2"
            for k, p in enumerate(programs):
                colour[(n_on == 1) & on[:, k]] = GROUP_COLORS[k]
            colour[n_on > 1] = "black"
            # Inactive first, single programs above, several on top.
            order = np.argsort(n_on.clip(0, 2) + defined, kind="stable")
            stages = [s for s in STAGES if defined[(atlas["stage"] == s).to_numpy()].any()]
            fig, axes = plt.subplots(1, 2, figsize=(16, 7.8))
            for ax, prefix in zip(axes, ["aligned", "unaligned"]):
                xy = atlas[[f"{prefix}_1", f"{prefix}_2"]].to_numpy()
                ax.scatter(xy[order, 0], xy[order, 1], s=np.where(n_on[order] > 0, 2.0, 0.8),
                           c=colour[order], linewidths=0, rasterized=True)
                ax.set_title(f"atlas, {prefix}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            n_active = max(int((n_on > 0).sum()), 1)
            handles = [plt.Line2D([], [], ls="", marker="o", ms=6, color=GROUP_COLORS[k],
                                  label=f"{p} ({int(((n_on == 1) & on[:, k]).sum())} cells)")
                       for k, p in enumerate(programs)]
            handles += [plt.Line2D([], [], ls="", marker="o", ms=6, color="black",
                                   label=f"several ({int((n_on > 1).sum())} cells, "
                                         f"{(n_on > 1).sum() / n_active:.0%} of active)"),
                        plt.Line2D([], [], ls="", marker="o", ms=6, color="#D9D9D9",
                                   label="none active"),
                        plt.Line2D([], [], ls="", marker="o", ms=6, color="#F2F2F2",
                                   markeredgecolor="#BBBBBB", label="no program defined")]
            fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
            fig.suptitle(f"{name} ({stages[0]} - {stages[-1]}): cells by the active program "
                         "of the group (smoothed score above the 99.9% permutation quantile)",
                         fontsize=11)
            fig.tight_layout(rect=(0, 0.08, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)
            several = (n_on > 1).sum() / n_active
            print(f"{name}: {n_active} active cells, several {several:.1%}")


def atlas_pages(order: list[str], atlas: pd.DataFrame, scores: pd.DataFrame) -> None:
    """results/figures/program_atlas_{aligned,unaligned}.pdf: one atlas UMAP per
    page, the stages first, then every program (scaled per stage)."""
    for prefix in ["aligned", "unaligned"]:
        xy = atlas[[f"{prefix}_1", f"{prefix}_2"]].to_numpy()
        with PdfPages(FIGURES / f"program_atlas_{prefix}.pdf") as pdf:
            fig, ax = plt.subplots(figsize=(9, 7.5))
            draw_stages(ax, xy, atlas, 1.5, 9)
            ax.set_title(f"atlas, {prefix}: stages", fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            for p in order:
                fig, ax = plt.subplots(figsize=(9, 7.5))
                scaled = scaled_scores(p, atlas, scores)
                draw_program(ax, xy, scaled, 2.0)
                stages = [s for s in STAGES if scaled[atlas["stage"] == s].notna().any()]
                window = f"{stages[0]} - {stages[-1]}" if stages else "not defined"
                ax.set_title(f"{p}  ({window}; atlas, {prefix}; score scaled per stage, "
                             "grey: not defined)", fontsize=11)
                sm = plt.cm.ScalarMappable(norm=TwoSlopeNorm(0, -1, 1), cmap="RdBu_r")
                fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def program_page(pdf, program: str, cells: pd.DataFrame, scores: pd.DataFrame,
                 info: pd.DataFrame, genes: dict, atlas: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 17))
    grid = fig.add_gridspec(4, 12)
    axes = [fig.add_subplot(grid[i // 4, 3 * (i % 4):3 * (i % 4) + 3]) for i in range(12)]
    atlas_panels(fig, grid, program, atlas, scores)
    for ax, stage in zip(axes, STAGES):
        c = cells[cells["stage"] == stage]
        ax.set_xticks([])
        ax.set_yticks([])
        if program not in scores or (stage, program) not in info.index:
            ax.scatter(c["umap_1"], c["umap_2"], s=1, c="#DDDDDD", linewidths=0,
                       rasterized=True)
            ax.set_title(f"{stage}: not defined", fontsize=9, color="#999999")
            continue
        v = scores.loc[c.index, program].to_numpy()
        lim = np.nanquantile(np.abs(v), 0.999) or 1.0
        order = np.argsort(np.abs(np.nan_to_num(v)))
        size = float(np.clip(6000 / len(c), 1.2, 8))
        ax.set_facecolor("#EBEBEB")
        ax.scatter(c["umap_1"].to_numpy()[order], c["umap_2"].to_numpy()[order], s=size,
                   c=v[order], cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), linewidths=0,
                   rasterized=True)
        r = info.loc[(stage, program)]
        mark = "supported" if r["supported"] else "not supported"
        ax.set_title(f"{stage}: {int(r['genes'])} genes, {mark}\n"
                     f"active in {100 * r['active']:.0f}% of cells, library level "
                     f"{r['level']:.2f}", fontsize=8)
    # genes: how many stages each gene is part of
    counts = pd.Series([g for gs in genes.values() for g in gs]).str.lower().value_counts()
    text = ", ".join(f"{g} ({n})" for g, n in counts.items())
    fig.suptitle(program, fontsize=14, x=0.01, ha="left")
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap("Genes (number of stages): " + text, 230)),
             fontsize=6.5, va="bottom")
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    adata = load_dropseq_counts()
    programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    curated = programs[(programs["set"] == "curated") & programs["kept"]]
    support = pd.read_csv(TABLES / "program_support.tsv", sep="\t")
    support = support[support["set"] == "curated"].set_index(["stage", "program"])["supported"]
    genes_table = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                              usecols=["stage", "gene", "heterogeneous"])
    cells = pd.read_csv(TABLES / "replicated_populations_cells.tsv.gz", sep="\t", index_col=0,
                        usecols=["Unnamed: 0", "stage", "library", "umap_1", "umap_2"])
    space = program_populations.ProgramSpace(programs, support.reset_index().assign(
        set="curated"), genes_table, supported_only=False)

    columns, active_cols, active_rows = {}, {}, []
    for stage in STAGES:
        sub = adata[adata.obs["stage"] == stage]
        libs = sorted(sub.obs["library"].unique())
        raw, _, _ = space(sub, libs, stage)
        names = list(space.stage_programs(stage).index)
        lib_of = sub.obs["library"].to_numpy()
        for j, name in enumerate(names):
            col = columns.setdefault(name, pd.Series(np.nan, index=adata.obs_names))
            act = active_cols.setdefault(name, pd.Series(np.nan, index=adata.obs_names))
            for lib in libs:
                col.loc[sub.obs_names[lib_of == lib]] = raw[lib][:, j]
                act.loc[sub.obs_names[lib_of == lib]] = space.active[lib][:, j].astype(float)
                active_rows.append(dict(stage=stage, program=name, library=lib,
                                        active=space.active[lib][:, j].mean()))
        print(stage, flush=True)
    scores = pd.DataFrame(columns)
    scores.to_csv(SCORES, sep="\t", float_format="%.4g")
    pd.DataFrame(active_cols).to_csv(ACTIVE, sep="\t", float_format="%.0f")

    active = pd.DataFrame(active_rows).groupby(["stage", "program"])["active"].median()
    levels = library_levels(adata, curated).groupby(["stage", "program"])["level"].median()
    n_genes = curated.groupby(["stage", "program"]).size()
    info = pd.DataFrame(dict(genes=n_genes, active=active, level=levels,
                             supported=support.reindex(n_genes.index).fillna(False)))
    info.reset_index().to_csv(TABLES / "program_atlas.tsv", sep="\t", index=False,
                              float_format="%.4g")
    plot(curated, cells, scores, info, atlas_coordinates(adata, genes_table))


def atlas_coordinates(adata, genes_table) -> pd.DataFrame:
    """Atlas UMAPs of script 15 (cached in data/processed/atlas_umap.tsv.gz)."""
    if alignment.ATLAS_UMAP.exists():
        return pd.read_csv(alignment.ATLAS_UMAP, sep="\t", index_col=0)
    if adata is None:
        adata = load_dropseq_counts()
    libs_of = {s: sorted(adata.obs.loc[adata.obs["stage"] == s, "library"].unique())
               for s in STAGES}
    return alignment.atlas_umaps(adata, genes_table, libs_of)


def plot(curated, cells, scores, info, atlas) -> None:
    order = program_order(curated["program"].unique())
    stage_genes = {p: {s: list(g["gene"]) for s, g in curated[curated["program"] == p]
                       .groupby("stage")} for p in order}

    with PdfPages(FIGURES / "program_atlas.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(16, 11))
        heatmap(axes[0], info["active"].unstack("stage"), order,
                "share of cells with the program active (median library); x: not supported",
                "viridis", 0.3, info["supported"].unstack("stage"))
        heatmap(axes[1], info["level"].unstack("stage"), order,
                "library level (pseudobulk relative to the maximum over libraries)", "magma", 1.0)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        for p in order:
            program_page(pdf, p, cells, scores, info, stage_genes[p], atlas)
    atlas_pages(order, atlas, scores)
    atlas_combined(atlas, pd.read_csv(ACTIVE, sep="\t", index_col=0))
    print(info["active"].unstack("stage").reindex(index=order, columns=STAGES).round(2).to_string())


def replot() -> None:
    programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    curated = programs[(programs["set"] == "curated") & programs["kept"]]
    cells = pd.read_csv(TABLES / "replicated_populations_cells.tsv.gz", sep="\t", index_col=0,
                        usecols=["Unnamed: 0", "stage", "library", "umap_1", "umap_2"])
    scores = pd.read_csv(SCORES, sep="\t", index_col=0)
    info = pd.read_csv(TABLES / "program_atlas.tsv", sep="\t").set_index(["stage", "program"])
    genes_table = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                              usecols=["stage", "gene", "heterogeneous"])
    plot(curated, cells, scores, info, atlas_coordinates(None, genes_table))


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["combined"]:
        atlas_combined(atlas_coordinates(None, None), pd.read_csv(ACTIVE, sep="\t", index_col=0))
    elif sys.argv[1:] == ["atlas"]:
        programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
        names = programs.loc[(programs["set"] == "curated") & programs["kept"], "program"]
        atlas_pages(program_order(names.unique()), atlas_coordinates(None, None),
                    pd.read_csv(SCORES, sep="\t", index_col=0))
    else:
        replot() if sys.argv[1:] == ["plot"] else main()
