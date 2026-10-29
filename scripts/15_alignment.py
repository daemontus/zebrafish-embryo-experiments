"""Aligning libraries: does per-library centring of residuals do what we need?

Alignment (as in script 14): every library's counts become Pearson residuals
under its own NB model (pipeline_utils.pearson_residuals) on genes
heterogeneous at the stage, centred per library, and one PCA of all
libraries of the stage gives shared axes. Centring removes every difference
between library averages (batch, a/b, timing, but also composition), so
aligned data can compare cells across libraries but says nothing about
library averages.

Four checks, each against the unaligned joint embedding of script 02
(joint log-normalisation, scaling, PCA of the compared libraries):
  1. mixing      every pair of libraries of a stage: share of cells whose K
                 nearest neighbours are significantly unbalanced between the
                 two libraries (binomial test, as in 02), vs a random split of
                 the larger library (sizes as in the pair; the halves are
                 aligned as if they were two libraries).
  2. stages      pairs of libraries of different stages in the same batch
                 (so batch does not contribute), both subsampled to
                 STAGE_CELLS; genes heterogeneous at either stage.
  3. structure   per library: overlap of each cell's K nearest neighbours in
                 the shared PCs with those in the library's own PCA (same
                 residuals, centring does not change distances; only the
                 choice of axes differs). Unaligned reference: the joint
                 log-normalised PCA of the stage, restricted to the library.
  4. cell types  a few populations with unambiguous markers (MARKERS, at the
                 stages where they are unambiguous), called per library from
                 a marker score above the 99.9% quantile of matched random
                 gene sets. Marker genes are removed from the features. For
                 every ordered pair of libraries of a stage, each cell of B
                 takes its K nearest cells of A; AUROC of the share of
                 marker-positive neighbours for B's own call.

Figures: per-stage UMAPs of the aligned data (colour = batch), the marker
populations of check 4 in every library on these UMAPs, atlas UMAP of
the aligned data (colour = stage) and of the unaligned data (log-normalised,
same genes; colour = stage and batch). In the aligned atlas, stage averages
are removed too, so stages separate only where their cells differ, and
adjacent stages overlap (Section 'alignment' of the report).

Outputs:
  results/tables/alignment_{mixing,stages,structure,markers}.tsv
  data/processed/atlas_umap.tsv.gz (atlas UMAP coordinates, used by 19)
  report/figures/alignment_checks.pdf, alignment_stages.pdf, alignment_markers.pdf,
  alignment_atlas.pdf
`15_alignment.py markers` redraws alignment_stages, alignment_markers and
alignment_atlas only (atlas coordinates from the cache).
"""
from __future__ import annotations

import importlib
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from pipeline_utils import (
    BATCH_COLORS,
    PROCESSED,
    REPORT_WIDTH_IN,
    STAGE_HPF,
    STAGES,
    TABLES,
    ensure_dirs,
    load_dropseq_counts,
    pearson_residuals,
    report_style,
    save_report_figure,
)

warnings.filterwarnings("ignore", category=FutureWarning)
mixing = importlib.import_module("02_replicate_mixing")

K = mixing.K
N_PCS = 30
N_PERM = 50
STAGE_CELLS = 800
N_NULL = 20
SEED = 0
ATLAS_UMAP = PROCESSED / "atlas_umap.tsv.gz"
# Populations with unambiguous markers, and the stages where the markers are
# unambiguous (EVL keratins spread to the epidermis after shield).
MARKERS = {
    "EVL": (["KRT4", "KRT8", "KRT18", "CLDNE", "CYT1", "KRT5", "CLDNB"], STAGES[2:6]),
    "PGC": (["NANOS3", "DND1", "TDRD7A", "CA15B", "DDX4"], STAGES[4:]),
    "p53-like": (["MDM2", "CDKN1A", "PHLDA3", "SESN3", "BCL2L14", "GADD45AA"], STAGES[2:]),
    "endoderm": (["SOX17", "SOX32"], STAGES[4:]),
    "notochord": (["NTD5", "TWIST2", "COL8A1A", "PMP22B", "NOTO"], STAGES[7:]),
    "hatching gland": (["HE1A", "HE1B", "CTSLB"], STAGES[9:]),
}
MARKER_GENES = {g for genes, _ in MARKERS.values() for g in genes}


# --- representations ------------------------------------------------------------

def aligned(adata, libs: list[str], genes: np.ndarray, cells: dict | None = None) -> dict:
    """Shared PCs of the aligned residuals; `cells` optionally restricts each
    library (or pseudo-library) to given rows of `adata`."""
    X = adata.X.tocsr()
    lib_of = adata.obs["library"].to_numpy()
    rows = cells or {lib: np.flatnonzero(lib_of == lib) for lib in libs}
    res = {lib: pearson_residuals(X[rows[lib]], genes) for lib in libs}
    stacked = np.vstack([res[lib] for lib in libs])
    n = min(N_PCS, stacked.shape[1] - 1, stacked.shape[0] - 1)
    pca = PCA(n, random_state=SEED).fit(stacked)
    return {lib: pca.transform(res[lib]) for lib in libs}


def joint(adata, rows: dict, genes: set) -> dict:
    """Unaligned joint embedding of script 02 for the given row sets."""
    order = list(rows)
    sub = adata[np.concatenate([rows[k] for k in order])].copy()
    pcs = mixing.embed(sub, genes)
    out, start = {}, 0
    for k in order:
        out[k] = pcs[start:start + len(rows[k])]
        start += len(rows[k])
    return out


def unbalanced(pcs_a: np.ndarray, pcs_b: np.ndarray, rng) -> tuple[float, float]:
    """Share of cells with unbalanced neighbourhoods, and its permutation mean."""
    pcs = np.vstack([pcs_a, pcs_b])
    is_b = np.r_[np.zeros(len(pcs_a), bool), np.ones(len(pcs_b), bool)]
    idx = mixing.knn(pcs)
    obs = mixing.neighbourhood_stats(idx, is_b)["reject"]
    null = np.mean([mixing.neighbourhood_stats(idx, rng.permutation(is_b))["reject"]
                    for _ in range(N_PERM)])
    return obs, null


def stage_genes(genes_table: pd.DataFrame, stages: list[str], var_names) -> np.ndarray:
    het = genes_table.loc[genes_table["stage"].isin(stages) & genes_table["heterogeneous"], "gene"]
    return var_names.isin(set(het) - MARKER_GENES)


# --- marker calls ---------------------------------------------------------------------

def marker_calls(sub, genes: list[str], rng) -> np.ndarray:
    """Marker score above the 99.9% quantile of matched random gene sets."""
    ln = sub.copy()
    sc.pp.normalize_total(ln, target_sum=1e4)
    sc.pp.log1p(ln)
    present = [g for g in genes if g in ln.var_names]
    mean = np.asarray(ln.X.mean(axis=0)).ravel()
    bins = pd.qcut(pd.Series(mean).rank(method="first"), 50, labels=False).to_numpy()
    sc.tl.score_genes(ln, present, score_name="s", random_state=SEED)
    null = []
    for r in range(N_NULL):
        fake = {rng.choice(ln.var_names[bins == bins[ln.var_names.get_loc(g)]]) for g in present}
        sc.tl.score_genes(ln, list(fake), score_name="n", random_state=r)
        null.append(ln.obs["n"].to_numpy())
    return ln.obs["s"].to_numpy() > np.quantile(np.concatenate(null), 0.999)


def all_marker_calls(adata, libs_of: dict, rng) -> dict:
    """(type, library) -> marker call of every cell of the library."""
    lib_of = adata.obs["library"].to_numpy()
    calls = {}
    for ctype, (genes, stages) in MARKERS.items():
        for s in stages:
            for lib in libs_of[s]:
                calls[(ctype, lib)] = marker_calls(adata[lib_of == lib].copy(), genes, rng)
    return calls


def transfer_auroc(pcs_a, label_a, pcs_b, label_b) -> float:
    idx = NearestNeighbors(n_neighbors=K).fit(pcs_a).kneighbors(pcs_b, return_distance=False)
    vote = label_a[idx].mean(axis=1)
    return mixing.roc_auc_score(label_b, vote) if 0 < label_b.sum() < len(label_b) else np.nan


# --- checks -----------------------------------------------------------------------------

def check_stage(adata, stage, libs, genes_table, calls, rng) -> dict:
    """Checks 1, 3 and 4 for one stage."""
    sub = adata[adata.obs["stage"] == stage]
    lib_of = sub.obs["library"].to_numpy()
    mask = stage_genes(genes_table, [stage], sub.var_names)
    het_calls = set(sub.var_names[mask])
    pcs = aligned(sub, libs, mask)
    rows = {lib: np.flatnonzero(lib_of == lib) for lib in libs}
    joint_stage = joint(sub, rows, het_calls)
    out = dict(mixing=[], structure=[], markers=[], pcs=pcs)

    # 3. structure within each library
    for lib in libs:
        own = aligned(sub, [lib], mask, {lib: rows[lib]})[lib]
        n_own = NearestNeighbors(n_neighbors=K + 1).fit(own).kneighbors(return_distance=False)
        for name, emb in [("aligned", pcs[lib]), ("joint", joint_stage[lib])]:
            n_emb = NearestNeighbors(n_neighbors=K + 1).fit(emb).kneighbors(
                return_distance=False)
            overlap = np.mean([len(set(a[1:]) & set(b[1:])) / K for a, b in zip(n_own, n_emb)])
            out["structure"].append(dict(stage=stage, library=lib, space=name, overlap=overlap))

    # 1. mixing of library pairs and a split control
    batch = sub.obs.groupby("library", observed=True)["sequencing_batch"].first()
    for i, a in enumerate(libs):
        for b in libs[i + 1:]:
            kind = "a/b" if batch[a] == batch[b] else "other batch"
            pair_rows = {a: rows[a], b: rows[b]}
            j = joint(sub, pair_rows, het_calls)
            for space, (pa, pb) in [("aligned", (pcs[a], pcs[b])), ("joint", (j[a], j[b]))]:
                obs, null = unbalanced(pa, pb, rng)
                out["mixing"].append(dict(stage=stage, library_a=a, library_b=b, kind=kind,
                                          space=space, unbalanced=obs, null=null))
            big = a if len(rows[a]) >= len(rows[b]) else b
            perm = rng.permutation(rows[big])
            cut = round(len(perm) * len(rows[a]) / (len(rows[a]) + len(rows[b])))
            halves = {"h1": perm[:cut], "h2": perm[cut:]}
            al = aligned(sub, ["h1", "h2"], mask, halves)
            jh = joint(sub, halves, het_calls)
            for space, (pa, pb) in [("aligned", (al["h1"], al["h2"])),
                                    ("joint", (jh["h1"], jh["h2"]))]:
                obs, null = unbalanced(pa, pb, rng)
                out["mixing"].append(dict(stage=stage, library_a=big, library_b=big,
                                          kind="random split", space=space, unbalanced=obs,
                                          null=null))

    # 4. cell types across libraries
    for ctype, (genes, stages) in MARKERS.items():
        if stage not in stages or len(libs) < 2:
            continue
        label = {lib: calls[(ctype, lib)] for lib in libs}
        for a in libs:
            for b in libs:
                if a == b or label[a].sum() < 5 or label[b].sum() < 5:
                    continue
                j = joint(sub, {a: rows[a], b: rows[b]}, het_calls)
                for space, (pa, pb) in [("aligned", (pcs[a], pcs[b])), ("joint", (j[a], j[b]))]:
                    out["markers"].append(dict(
                        stage=stage, type=ctype, library_a=a, library_b=b, space=space,
                        positives_a=int(label[a].sum()), positives_b=int(label[b].sum()),
                        auroc=transfer_auroc(pa, label[a], pb, label[b])))
    return out


def check_stage_distance(adata, libs_of, genes_table, rng) -> pd.DataFrame:
    """Check 2: same-batch library pairs of different stages."""
    lib_info = adata.obs.groupby("library", observed=True)[["stage", "sequencing_batch"]].first()
    lib_of = adata.obs["library"].to_numpy()
    rows = []
    for batch, info in lib_info.groupby("sequencing_batch"):
        # one library per stage (the a library where there are two)
        per_stage = {s: sorted(info.index[info["stage"] == s])[0]
                     for s in STAGES if (info["stage"] == s).any()}
        stages = list(per_stage)
        for i, s1 in enumerate(stages):
            for s2 in stages[i + 1:]:
                a, b = per_stage[s1], per_stage[s2]
                ra, rb = np.flatnonzero(lib_of == a), np.flatnonzero(lib_of == b)
                n = min(len(ra), len(rb), STAGE_CELLS)
                pick = {a: rng.choice(ra, n, replace=False), b: rng.choice(rb, n, replace=False)}
                mask = stage_genes(genes_table, [s1, s2], adata.var_names)
                al = aligned(adata, [a, b], mask, pick)
                jt = joint(adata, pick, set(adata.var_names[mask]))
                for space, emb in [("aligned", al), ("joint", jt)]:
                    obs, null = unbalanced(emb[a], emb[b], rng)
                    rows.append(dict(batch=batch, stage_a=s1, stage_b=s2, library_a=a,
                                     library_b=b, hours=STAGE_HPF[s2] - STAGE_HPF[s1],
                                     cells=n, space=space, unbalanced=obs, null=null))
    return pd.DataFrame(rows)


# --- UMAPs -----------------------------------------------------------------------------

def umap(emb: np.ndarray) -> np.ndarray:
    a = sc.AnnData(np.zeros((len(emb), 1), dtype=np.float32))
    a.obsm["X_pca"] = emb
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca", random_state=SEED)
    sc.tl.umap(a, random_state=SEED)
    return a.obsm["X_umap"]


STAGE_CMAP = plt.get_cmap("viridis")


LIBRARY_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def plot_stages(adata, stage_pcs: dict) -> dict:
    """Aligned UMAP of every stage, one colour per library; returns the
    coordinates (stage -> {library: xy})."""
    coords = {}
    fig, axes = plt.subplots(3, 4, figsize=(REPORT_WIDTH_IN, REPORT_WIDTH_IN * 0.85))
    for ax, stage in zip(axes.ravel(), STAGES):
        pcs = stage_pcs[stage]
        libs = list(pcs)
        xy = umap(np.vstack([pcs[lib] for lib in libs]))
        ends = np.cumsum([0] + [len(pcs[lib]) for lib in libs])
        coords[stage] = {lib: xy[ends[k]:ends[k + 1]] for k, lib in enumerate(libs)}
        colour = np.concatenate([np.full(len(pcs[lib]), k) for k, lib in enumerate(libs)])
        order = np.random.default_rng(SEED).permutation(len(xy))
        size = float(np.clip(1500 / len(xy), 0.4, 4))
        ax.scatter(xy[order, 0], xy[order, 1], s=size, linewidths=0, rasterized=True,
                   c=[LIBRARY_COLORS[k] for k in colour[order]])
        handles = [plt.Line2D([], [], ls="", marker="o", ms=2.5, color=LIBRARY_COLORS[k],
                              label=lib.split("_")[-1]) for k, lib in enumerate(libs)]
        ax.legend(handles=handles, frameon=False, fontsize=5, loc="lower right",
                  handletextpad=0.1, borderpad=0.1, labelspacing=0.2)
        ax.set_title(stage, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    save_report_figure(fig, "alignment_stages")
    plt.close(fig)
    return coords


MARKER_COLORS = {"EVL": "#0072B2", "PGC": "#D55E00", "p53-like": "#CC79A7",
                 "endoderm": "#009E73", "notochord": "#E69F00", "hatching gland": "#56B4E9"}
MARKER_COLUMNS = 6


def plot_markers(coords: dict, calls: dict) -> None:
    """The marker-called populations of check 4 in every library, on the aligned
    UMAP of its stage (the same coordinates as alignment_stages; marker genes
    are not among the features). Grey: the stage's other libraries."""
    panels = [(stage, lib) for stage in STAGES for lib in coords[stage]
              if any(stage in stages for _, stages in MARKERS.values())]
    rows = -(-len(panels) // MARKER_COLUMNS)
    fig, axes = plt.subplots(rows, MARKER_COLUMNS,
                             figsize=(REPORT_WIDTH_IN, REPORT_WIDTH_IN / MARKER_COLUMNS * rows
                                      + 0.2))
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax, (stage, lib) in zip(axes.ravel(), panels):
        others = [xy for other, xy in coords[stage].items() if other != lib]
        if others:
            o = np.vstack(others)
            ax.scatter(o[:, 0], o[:, 1], s=0.3, c="#E4E4E4", linewidths=0, rasterized=True)
        xy = coords[stage][lib]
        ax.scatter(xy[:, 0], xy[:, 1], s=0.5, c="#A0A0A0", linewidths=0, rasterized=True)
        for ctype, (_, stages) in MARKERS.items():
            if stage not in stages:
                continue
            on = calls[(ctype, lib)]
            ax.scatter(xy[on, 0], xy[on, 1], s=2.5, c=MARKER_COLORS[ctype], linewidths=0,
                       rasterized=True)
        ax.set_title(f"{stage} {lib.split('_')[-1]}", fontsize=6, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color=c, label=t)
               for t, c in MARKER_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=6,
               frameon=False, handletextpad=0.1)
    fig.tight_layout(rect=(0, 0.17 / (REPORT_WIDTH_IN / MARKER_COLUMNS * rows + 0.2), 1, 1),
                     h_pad=0.3, w_pad=0.2)
    save_report_figure(fig, "alignment_markers")
    plt.close(fig)


def markers_only() -> None:
    """Redraw alignment_stages, _markers and _atlas without the checks (the
    marker calls use the same random stream as main, so they are identical)."""
    report_style()
    sc.settings.verbosity = 0
    rng = np.random.default_rng(SEED)
    adata = load_dropseq_counts()
    libs_of = {s: sorted(adata.obs.loc[adata.obs["stage"] == s, "library"].unique())
               for s in STAGES}
    genes_table = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                              usecols=["stage", "gene", "heterogeneous"])
    calls = all_marker_calls(adata, libs_of, rng)
    stage_pcs = {}
    for stage in STAGES:
        sub = adata[adata.obs["stage"] == stage]
        stage_pcs[stage] = aligned(sub, libs_of[stage],
                                   stage_genes(genes_table, [stage], sub.var_names))
    plot_markers(plot_stages(adata, stage_pcs), calls)
    plot_atlas(adata, genes_table, libs_of, calls)


def atlas_umaps(adata, genes_table, libs_of) -> pd.DataFrame:
    """Aligned and unaligned atlas UMAPs on the genes heterogeneous at any stage
    (cached in ATLAS_UMAP; delete the file to recompute)."""
    if ATLAS_UMAP.exists():
        return pd.read_csv(ATLAS_UMAP, sep="\t", index_col=0)
    mask = stage_genes(genes_table, STAGES, adata.var_names)
    libs = [lib for s in STAGES for lib in libs_of[s]]
    X = adata.X.tocsr()
    lib_of = adata.obs["library"].to_numpy()
    rows = np.concatenate([np.flatnonzero(lib_of == lib) for lib in libs])
    res = np.vstack([pearson_residuals(X[lib_of == lib], mask).astype(np.float32)
                     for lib in libs])
    al = PCA(N_PCS, random_state=SEED).fit_transform(res)
    del res
    norm = adata[rows].copy()
    sc.pp.normalize_total(norm, target_sum=1e4)
    sc.pp.log1p(norm)
    norm = norm[:, mask].copy()
    sc.pp.scale(norm, max_value=10)
    jt = PCA(N_PCS, random_state=SEED).fit_transform(norm.X)
    out = adata.obs.iloc[rows][["stage", "sequencing_batch", "library"]].copy()
    out[["aligned_1", "aligned_2"]] = umap(al)
    out[["unaligned_1", "unaligned_2"]] = umap(jt)
    out.to_csv(ATLAS_UMAP, sep="\t", float_format="%.4g")
    return out


def marker_labels(adata, calls: dict, index) -> pd.Series:
    """Per cell of `index`: the marker population it is called in (at the stages
    where the markers are unambiguous), or "" if none. A cell called in two
    populations keeps the later one of MARKERS."""
    lib_of = adata.obs["library"].to_numpy()
    label = pd.Series("", index=adata.obs_names)
    for (ctype, lib), on in calls.items():
        label[adata.obs_names[lib_of == lib][on]] = ctype
    return label.reindex(index)


def plot_atlas(adata, genes_table, libs_of, calls: dict) -> None:
    atlas = atlas_umaps(adata, genes_table, libs_of)
    stage_idx = atlas["stage"].map({s: i for i, s in enumerate(STAGES)}).to_numpy()
    batch = atlas["sequencing_batch"].to_numpy()
    label = marker_labels(adata, calls, atlas.index).to_numpy()
    order = np.random.default_rng(SEED).permutation(len(atlas))
    al = atlas[["aligned_1", "aligned_2"]].to_numpy()
    jt = atlas[["unaligned_1", "unaligned_2"]].to_numpy()
    fig, axes = plt.subplots(2, 3, figsize=(REPORT_WIDTH_IN, REPORT_WIDTH_IN * 0.72))
    for row, (xy, space) in enumerate([(al, "aligned"), (jt, "unaligned")]):
        for col, colour in enumerate(["stage", "batch", "markers"]):
            ax = axes[row, col]
            if colour == "markers":
                ax.scatter(xy[:, 0], xy[:, 1], s=0.2, c="#D0D0D0", rasterized=True,
                           linewidths=0)
                for ctype, c in MARKER_COLORS.items():
                    on = label == ctype
                    ax.scatter(xy[on, 0], xy[on, 1], s=0.8, c=c, rasterized=True,
                               linewidths=0)
            else:
                c = (STAGE_CMAP(stage_idx[order] / (len(STAGES) - 1)) if colour == "stage"
                     else [BATCH_COLORS[b] for b in batch[order]])
                ax.scatter(xy[order, 0], xy[order, 1], s=0.2, c=c, rasterized=True,
                           linewidths=0)
            ax.set_title(f"{'abcdef'[3 * row + col]}  {space} ({colour})", fontsize=7, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    legends = [
        ([plt.Line2D([], [], ls="", marker="o", ms=3, color=STAGE_CMAP(i / (len(STAGES) - 1)),
                     label=s) for i, s in enumerate(STAGES)], 3),
        ([plt.Line2D([], [], ls="", marker="o", ms=3, color=c, label=b)
          for b, c in BATCH_COLORS.items()], 2),
        ([plt.Line2D([], [], ls="", marker="o", ms=3, color=c, label=t)
          for t, c in MARKER_COLORS.items()], 2),
    ]
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ax, (handles, ncol) in zip(axes[1], legends):
        ax.legend(handles=handles, frameon=False, ncol=ncol, fontsize=5.5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.04), handletextpad=0.1, columnspacing=0.8)
    save_report_figure(fig, "alignment_atlas")
    plt.close(fig)


def plot_checks(mix: pd.DataFrame, dist: pd.DataFrame, struct: pd.DataFrame,
                marks: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(REPORT_WIDTH_IN, REPORT_WIDTH_IN * 0.62))
    axes = axes.ravel()
    colors = {"joint": "#E69F00", "aligned": "#0072B2"}
    # a: mixing by kind
    ax = axes[0]
    kinds = ["random split", "a/b", "other batch"]
    for k, kind in enumerate(kinds):
        for d, space in [(-0.18, "joint"), (0.18, "aligned")]:
            v = mix.loc[(mix["kind"] == kind) & (mix["space"] == space), "unbalanced"]
            ax.scatter(np.full(len(v), k + d) + np.random.default_rng(k).uniform(-0.06, 0.06, len(v)),
                       v, s=6, color=colors[space], label=space if k == 0 else None)
    ax.set_xticks(range(len(kinds)), ["split", "a/b", "other\nbatch"])
    ax.set_ylabel("unbalanced neighbourhoods")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("a  same stage", fontsize=7, loc="left")
    ax.legend(frameon=False, fontsize=6)
    # b: stage distance
    ax = axes[1]
    for space in ["joint", "aligned"]:
        d = dist[dist["space"] == space]
        ax.scatter(d["hours"], d["unbalanced"], s=6, color=colors[space])
    ax.set_xlabel("hours apart (same batch)")
    ax.set_ylabel("unbalanced neighbourhoods")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("b  different stages", fontsize=7, loc="left")
    # c: structure
    ax = axes[2]
    for k, space in enumerate(["joint", "aligned"]):
        v = struct.loc[struct["space"] == space, "overlap"]
        ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.12, 0.12, len(v)),
                   v, s=6, color=colors[space])
    ax.set_xticks([0, 1], ["joint", "aligned"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("neighbours shared with\nthe library's own PCA")
    ax.set_title("c  within a library", fontsize=7, loc="left")
    # d: marker cell types
    ax = axes[3]
    types = list(MARKERS)
    med = marks.groupby(["type", "space"])["auroc"].median().unstack()
    for k, t in enumerate(types):
        for d, space in [(-0.18, "joint"), (0.18, "aligned")]:
            v = marks.loc[(marks["type"] == t) & (marks["space"] == space), "auroc"]
            ax.scatter(np.full(len(v), k + d), v, s=4, color=colors[space], alpha=0.5)
            if t in med.index:
                ax.scatter(k + d, med.loc[t, space], s=22, marker="_", color="black")
    ax.set_xticks(range(len(types)), types, rotation=25, ha="right")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("AUROC, neighbours\nin the other library")
    ax.set_title("d  cell types across libraries", fontsize=7, loc="left")
    fig.tight_layout()
    save_report_figure(fig, "alignment_checks")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    report_style()
    sc.settings.verbosity = 0
    rng = np.random.default_rng(SEED)
    adata = load_dropseq_counts()
    libs_of = {s: sorted(adata.obs.loc[adata.obs["stage"] == s, "library"].unique())
               for s in STAGES}
    genes_table = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                              usecols=["stage", "gene", "heterogeneous"])
    calls = all_marker_calls(adata, libs_of, rng)
    mix, struct, marks, stage_pcs = [], [], [], {}
    for stage in STAGES:
        r = check_stage(adata, stage, libs_of[stage], genes_table, calls, rng)
        mix += r["mixing"]
        struct += r["structure"]
        marks += r["markers"]
        stage_pcs[stage] = r["pcs"]
        print(stage, flush=True)
    mix, struct, marks = pd.DataFrame(mix), pd.DataFrame(struct), pd.DataFrame(marks)
    dist = check_stage_distance(adata, libs_of, genes_table, rng)
    for name, table in [("mixing", mix), ("stages", dist), ("structure", struct),
                        ("markers", marks)]:
        table.to_csv(TABLES / f"alignment_{name}.tsv", sep="\t", index=False,
                     float_format="%.4g")
    print(mix.groupby(["kind", "space"])["unbalanced"].describe().round(2))
    print(struct.groupby("space")["overlap"].describe().round(2))
    print(marks.groupby(["type", "space"])["auroc"].median().unstack().round(3))
    plot_checks(mix, dist, struct, marks)
    plot_markers(plot_stages(adata, stage_pcs), calls)
    plot_atlas(adata, genes_table, libs_of, calls)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["markers"]:
        markers_only()
    else:
        main()
