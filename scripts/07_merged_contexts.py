#!/usr/bin/env python3
"""Stage 07: how per-library clusters behave when related libraries are merged.

Input : data/processed/dropseq_3.3-6.0hpf_qc.h5ad   (raw counts + QC metrics)
        results/tables/06_cell_clusters.csv          (per-library Leiden labels)
Output: data/processed/07_<context>_joint.h5ad
        results/tables/07_<context>_*.tsv
        results/figures/07_<context>_*.png/.pdf

Two merged contexts, both WITHOUT batch correction:

  ds5     all DS5 libraries in the 3.3-6.0 hpf window (high, oblong, dome,
          30% epiboly, shield; including the 'b' libraries). Merges across
          timepoints within one batch.
  50pct   the four 50% epiboly libraries (DS2, DS3, DS4, DS4b). Merges across
          batches/replicates within one timepoint.

Each context gets a joint embedding (HVGs chosen per library and combined via
batch_key="library", PCA, kNN, UMAP) and a joint Leiden clustering
(resolution 0.5). The per-library ("local") clusters from stage 06 are then
compared within that shared space:

  local_vs_joint        fraction of each local cluster in each joint cluster
  library_agreement     ARI/AMI of local vs joint labels, per library
  joint_composition     library make-up of each joint cluster, with a
                        size-corrected evenness (1 = all libraries contribute
                        proportionally, 0 = one library only)
  centroid_correlation  Pearson correlation of local-cluster mean expression
                        on joint HVGs; 'global' centres each gene over all
                        clusters, 'library' first subtracts each library's
                        mean profile (removes library/batch/stage offsets)
  knn_mapping           for cells of local cluster A (library L1), the mean
                        fraction of their 15 nearest L2 cells (joint PCA)
                        that belong to each L2 cluster; best matches and
                        mutual best matches are tabulated
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from cluster_compare import (
    KNN_MAPPING_K,
    best_matches,
    joint_composition,
    knn_mapping,
    library_colors,
)
from pipeline_utils import (
    PROCESSED,
    QC_H5AD,
    STAGE_ORDER,
    TABLES,
    ensure_dirs,
    library_of,
    save_figure,
)

LOCAL_CLUSTERS = TABLES / "06_cell_clusters.csv"

CONTEXTS = {
    "ds5": {
        "title": "DS5, 3.3–6.0 hpf (merged across timepoints)",
        "select": lambda obs: obs["sequencing_batch"] == "DS5",
    },
    "50pct": {
        "title": "50% epiboly (merged across batches/replicates)",
        "select": lambda obs: obs["stage"] == "50% epiboly",
    },
}

MIN_CELLS = 3
N_TOP_GENES = 2000
N_PCS = 50
N_PCS_NEIGHBORS = 30
N_NEIGHBORS = 15
LEIDEN_RESOLUTION = 0.5
MIN_CLUSTER_FOR_SIMILARITY = 10
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
    parser.add_argument("--force", action="store_true",
                        help="recompute joint embeddings even if they exist")
    return parser.parse_args()


def short_library(stage: str, batch: str, replicate: str) -> str:
    """'30% epiboly', 'DS5', 'b' -> '30% DS5b'."""
    return f"{stage.replace(' epiboly', '')} {batch}{'b' if replicate == 'b' else ''}"


def build_joint(adata) -> None:
    """Normalise, embed and cluster a merged context in place."""
    adata.uns.clear()
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, flavor="seurat",
                                batch_key="library")
    hvg = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(hvg, max_value=10)
    sc.tl.pca(hvg, n_comps=N_PCS, svd_solver="arpack", random_state=RANDOM_STATE)
    adata.obsm["X_pca"] = hvg.obsm["X_pca"]
    adata.uns["pca"] = dict(hvg.uns["pca"])
    del hvg
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS_NEIGHBORS,
                    random_state=RANDOM_STATE)
    sc.tl.umap(adata, min_dist=0.3, random_state=RANDOM_STATE)
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added="joint",
                 flavor="leidenalg", random_state=RANDOM_STATE)


def order_joint_clusters(obs: pd.DataFrame, libs: list[str]) -> list[str]:
    """Joint clusters ordered by mean library position (≈ time for DS5), then size."""
    pos = obs["library_short"].astype(str).map({lib: i for i, lib in enumerate(libs)}).astype(float)
    key = pos.groupby(obs["joint"], observed=True).mean()
    size = obs["joint"].value_counts()
    return sorted(key.index, key=lambda c: (round(key[c], 1), -size[c]))


def local_order(obs: pd.DataFrame, libs: list[str]) -> list[str]:
    out = []
    for lib in libs:
        clusters = obs.loc[obs["library_short"] == lib, "local_cluster"].unique()
        out += [f"{lib}:{c}" for c in sorted(clusters, key=int)]
    return out


# --- comparisons -------------------------------------------------------------

def local_vs_joint(obs, local_labels, joint_labels) -> pd.DataFrame:
    return pd.crosstab(obs["local"], obs["joint"]).reindex(
        index=local_labels, columns=joint_labels, fill_value=0)


def library_agreement(obs, libs) -> pd.DataFrame:
    rows = []
    for lib in libs:
        sub = obs[obs["library_short"] == lib]
        rows.append({
            "library": lib,
            "n_cells": len(sub),
            "n_local_clusters": sub["local_cluster"].nunique(),
            "n_joint_clusters_used": (sub["joint"].value_counts() >= 5).sum(),
            "ari_local_vs_joint": adjusted_rand_score(sub["local_cluster"], sub["joint"]),
            "ami_local_vs_joint": adjusted_mutual_info_score(sub["local_cluster"], sub["joint"]),
        })
    return pd.DataFrame(rows)


def centroid_correlation(adata, labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = adata.obs
    hv = adata.var["highly_variable"].to_numpy()
    X = adata.X[:, hv]
    cent = {}
    for lab in labels:
        mask = (obs["local"] == lab).to_numpy()
        cent[lab] = np.asarray(X[mask].mean(axis=0)).ravel()
    C = pd.DataFrame(cent).T  # clusters x genes
    lib_of = pd.Series({lab: lab.rsplit(":", 1)[0] for lab in labels})
    g = C - C.mean(axis=0)
    lib_mean = C.groupby(lib_of).transform("mean")
    l = C - lib_mean
    return (pd.DataFrame(np.corrcoef(g), index=labels, columns=labels),
            pd.DataFrame(np.corrcoef(l), index=labels, columns=labels))


# --- figures -----------------------------------------------------------------

def plot_overview(adata, name, title, libs, lib_colors) -> None:
    obs = adata.obs
    obs["log10_umis"] = np.log10(obs["total_counts"])
    panels = [("library_short", "library"), ("joint", "joint Leiden (r=0.5)"),
              ("log10_umis", "log10 UMIs"), ("pct_counts_ribo", "% ribosomal")]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (col, t) in zip(axes, panels):
        kw = dict(ax=ax, show=False, frameon=False, size=8, title=t)
        if col == "library_short":
            sc.pl.umap(adata, color=col, palette=lib_colors, legend_fontsize=7, **kw)
        elif col == "joint":
            sc.pl.umap(adata, color=col, legend_loc="on data", legend_fontsize=8, **kw)
        else:
            sc.pl.umap(adata, color=col, cmap="viridis" if col == "log10_umis" else "magma", **kw)
    obs.drop(columns="log10_umis", inplace=True)
    fig.suptitle(f"{title}: joint UMAP, no batch correction (n={adata.n_obs:,})", y=1.02)
    fig.tight_layout()
    save_figure(fig, f"07_{name}_umap_overview")
    plt.close(fig)


def plot_local_on_joint(adata, name, title, libs) -> None:
    obs = adata.obs
    umap = adata.obsm["X_umap"]
    ncol = 4
    nrow = int(np.ceil(len(libs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow), squeeze=False)
    palette = plt.get_cmap("tab10").colors
    for ax, lib in zip(axes.ravel(), libs):
        mask = (obs["library_short"] == lib).to_numpy()
        ax.scatter(umap[~mask, 0], umap[~mask, 1], s=2, c="#dddddd", linewidths=0)
        clusters = sorted(obs.loc[mask, "local_cluster"].unique(), key=int)
        size = max(4, min(30, 3000 / mask.sum()))
        for i, c in enumerate(clusters):
            m = mask & (obs["local_cluster"] == c).to_numpy()
            ax.scatter(umap[m, 0], umap[m, 1], s=size, color=palette[i % 10],
                       linewidths=0, alpha=0.85, label=f"{c} (n={m.sum()})")
        ax.set_title(f"{lib}: own clusters (n={mask.sum()})")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(fontsize=6, markerscale=1.5, frameon=False, loc="best",
                  handletextpad=0.2, borderaxespad=0.2)
    for ax in axes.ravel()[len(libs):]:
        ax.set_axis_off()
    fig.suptitle(f"{title}: per-library (stage 06) clusters on the joint UMAP", y=1.0)
    fig.tight_layout()
    save_figure(fig, f"07_{name}_local_clusters_on_joint")
    plt.close(fig)


def _library_separators(ax, labels, axis="y") -> None:
    libs = [lab.rsplit(":", 1)[0] for lab in labels]
    for i in range(1, len(libs)):
        if libs[i] != libs[i - 1]:
            (ax.axhline if axis == "y" else ax.axvline)(i - 0.5, color="0.2", lw=0.8)


def plot_local_vs_joint(table, name, title, comp) -> None:
    frac = table.div(table.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(0.45 * table.shape[1] + 3, 0.24 * table.shape[0] + 2.2))
    im = ax.imshow(frac.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            v = frac.iat[i, j]
            if v >= 0.1:
                ax.text(j, i, f"{v:.1f}"[1:] if v < 0.95 else "1", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.6 else "black")
    ax.set_yticks(range(len(frac)), [f"{lab} (n={table.loc[lab].sum()})" for lab in frac.index],
                  fontsize=7)
    ax.set_xticks(range(frac.shape[1]),
                  [f"{c}\n{e:.2f}" for c, e in zip(frac.columns, comp["library_evenness"])],
                  fontsize=7)
    ax.set_xlabel("joint cluster (second line: library evenness)")
    _library_separators(ax, list(frac.index))
    fig.colorbar(im, ax=ax, shrink=0.6, label="fraction of local cluster")
    ax.set_title(f"{title}: local clusters → joint clusters")
    fig.tight_layout()
    save_figure(fig, f"07_{name}_local_vs_joint")
    plt.close(fig)


def plot_similarity(glob, libc, name, title, lib_colors) -> None:
    labels = list(glob.index)
    dist = np.clip(1 - libc.to_numpy(), 0, None)
    np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), "average"))
    labels = [labels[i] for i in order]
    libs = [lab.rsplit(":", 1)[0] for lab in labels]
    strip = ListedColormap([lib_colors[lib] for lib in dict.fromkeys(libs)])
    lib_idx = {lib: i for i, lib in enumerate(dict.fromkeys(libs))}

    n = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(2 * (0.22 * n + 2.5), 0.22 * n + 2.5))
    for ax, mat, t in ((axes[0], libc, "library-centred"),
                       (axes[1], glob, "globally centred")):
        m = mat.loc[labels, labels].to_numpy()
        im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n), labels, rotation=90, fontsize=6)
        ax.set_yticks(range(n), labels, fontsize=6)
        for tick, lib in zip(ax.get_yticklabels(), libs):
            tick.set_color(lib_colors[lib])
        for tick, lib in zip(ax.get_xticklabels(), libs):
            tick.set_color(lib_colors[lib])
        ax.set_title(f"centroid correlation, {t}")
        ax.imshow(np.array([[lib_idx[l]] for l in libs]), cmap=strip, aspect="auto",
                  extent=(n - 0.5, n + 0.5, n - 0.5, -0.5))
        ax.set_xlim(-0.5, n + 0.5)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Pearson r")
    fig.suptitle(f"{title}: similarity of local clusters "
                 f"(≥{MIN_CLUSTER_FOR_SIMILARITY} cells; order from library-centred)", y=1.0)
    save_figure(fig, f"07_{name}_cluster_similarity")
    plt.close(fig)


def plot_knn_mapping(mapping, labels, name, title) -> None:
    M = (mapping.pivot(index="src_cluster", columns="tgt_cluster", values="fraction")
         .reindex(index=labels, columns=labels))
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.24 * n + 3, 0.24 * n + 2))
    im = ax.imshow(M.to_numpy(), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n), labels, fontsize=6)
    _library_separators(ax, labels, "y")
    _library_separators(ax, labels, "x")
    ax.set_xlabel("target cluster (its library's cells are the neighbour pool)")
    ax.set_ylabel("source cluster")
    fig.colorbar(im, ax=ax, shrink=0.6, label=f"mean fraction of {KNN_MAPPING_K} NN in target")
    ax.set_title(f"{title}: kNN mapping between local clusters\n"
                 "(rows sum to 1 within each target-library block; diagonal blocks empty)")
    fig.tight_layout()
    save_figure(fig, f"07_{name}_knn_mapping")
    plt.close(fig)


def plot_composition(comp, libs, name, title, lib_colors) -> None:
    counts = comp[libs]
    per_lib = counts / counts.sum(axis=0)
    p = per_lib.div(per_lib.sum(axis=1), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, data, t in ((axes[0], counts.div(counts.sum(axis=1), axis=0), "raw cell share"),
                        (axes[1], p, "size-corrected share (each library weighted equally)")):
        bottom = np.zeros(len(data))
        for lib in libs:
            ax.bar(range(len(data)), data[lib], bottom=bottom, color=lib_colors[lib], label=lib)
            bottom += data[lib].to_numpy()
        ax.set_xticks(range(len(data)), [f"{c}\n(n={n})" for c, n in zip(data.index, comp["n_cells"])],
                      fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(t)
        ax.set_xlabel("joint cluster")
    axes[1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    fig.suptitle(f"{title}: library composition of joint clusters", y=1.02)
    fig.tight_layout()
    save_figure(fig, f"07_{name}_joint_composition")
    plt.close(fig)


# --- driver ------------------------------------------------------------------

def run_context(full, local, name, spec, force) -> pd.DataFrame:
    path = PROCESSED / f"07_{name}_joint.h5ad"
    if path.exists() and not force:
        adata = sc.read_h5ad(path)
    else:
        adata = full[spec["select"](full.obs).to_numpy()].copy()
        print(f"[07] {name}: {adata.n_obs:,} cells from {adata.obs['library'].nunique()} libraries")
        build_joint(adata)
        adata.obs["local_cluster"] = local.loc[adata.obs_names, "leiden"].astype(str).to_numpy()
        adata.obs["local"] = adata.obs["library_short"].astype(str) + ":" + adata.obs["local_cluster"]
        adata.write_h5ad(path, compression="gzip")

    obs = adata.obs
    libs = [lib for lib in full.uns["library_order"] if lib in set(obs["library_short"])]
    obs["library_short"] = pd.Categorical(obs["library_short"], libs)
    lib_colors = library_colors(libs)
    joint_labels = order_joint_clusters(obs, libs)
    local_labels = local_order(obs, libs)
    title = spec["title"]

    lvj = local_vs_joint(obs, local_labels, joint_labels)
    comp = joint_composition(obs, libs, joint_labels)
    agree = library_agreement(obs, libs)
    sizes = obs["local"].value_counts()
    big = [lab for lab in local_labels if sizes[lab] >= MIN_CLUSTER_FOR_SIMILARITY]
    glob, libc = centroid_correlation(adata, big)
    mapping = knn_mapping(adata.obsm["X_pca"][:, :N_PCS_NEIGHBORS], obs, libs, local_labels)
    best = best_matches(mapping, sizes)

    lvj.to_csv(TABLES / f"07_{name}_local_vs_joint_counts.tsv", sep="\t")
    comp.to_csv(TABLES / f"07_{name}_joint_composition.tsv", sep="\t", float_format="%.3f")
    agree.to_csv(TABLES / f"07_{name}_library_agreement.tsv", sep="\t", index=False,
                 float_format="%.3f")
    glob.to_csv(TABLES / f"07_{name}_centroid_corr_global.tsv", sep="\t", float_format="%.3f")
    libc.to_csv(TABLES / f"07_{name}_centroid_corr_library_centred.tsv", sep="\t",
                float_format="%.3f")
    mapping.to_csv(TABLES / f"07_{name}_knn_mapping.tsv", sep="\t", index=False,
                   float_format="%.3f")
    best.to_csv(TABLES / f"07_{name}_best_matches.tsv", sep="\t", index=False,
                float_format="%.3f")

    plot_overview(adata, name, title, libs, lib_colors)
    plot_local_on_joint(adata, name, title, libs)
    plot_local_vs_joint(lvj, name, title, comp)
    plot_composition(comp, libs, name, title, lib_colors)
    plot_similarity(glob, libc, name, title, lib_colors)
    plot_knn_mapping(mapping, local_labels, name, title)

    print(f"[07] {name}: {len(joint_labels)} joint clusters")
    print(agree.to_string(index=False))
    print(comp[["n_cells", "library_evenness", "dominant_library", "dominant_share"]].to_string())
    return agree.assign(context=name)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    print(f"[07] loading {QC_H5AD}")
    full = sc.read_h5ad(QC_H5AD)
    full.obs["library"] = library_of(full.obs_names)
    full.obs["stage"] = pd.Categorical(full.obs["stage"], STAGE_ORDER, ordered=True)
    full.obs["library_short"] = [
        short_library(s, b, r) for s, b, r in
        zip(full.obs["stage"], full.obs["sequencing_batch"], full.obs["replicate"])
    ]
    order = (full.obs.drop_duplicates("library_short")
             .sort_values(["stage", "sequencing_batch", "replicate"])["library_short"])
    full.uns["library_order"] = list(order)
    local = pd.read_csv(LOCAL_CLUSTERS, index_col="cell_id")

    for name, spec in CONTEXTS.items():
        run_context(full, local, name, spec, args.force)
    print("[07] done")


if __name__ == "__main__":
    main()
