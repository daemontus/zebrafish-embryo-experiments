#!/usr/bin/env python3
"""Stage 06: Leiden clustering and UMAP for each Drop-seq library separately.

Input : data/processed/dropseq_3.3-6.0hpf_qc.h5ad   (raw counts + QC metrics)
Output: data/processed/libraries/<library>.h5ad     (one object per library)
        results/tables/06_library_clustering_summary.tsv
        results/tables/06_cluster_qc.tsv
        results/tables/06_cluster_markers.tsv
        results/tables/06_cell_clusters.csv
        results/figures/06_umap_leiden_all_libraries.png/.pdf
        results/figures/06_cluster_qc_overview.png/.pdf
        results/figures/06_libraries/06_umap_<library>.png/.pdf

Each library (one GEO sample = one stage x batch x replicate) is processed on
its own, so no batch correction is involved. The defaults are deliberately
mild: Leiden resolution 0.5, 15 neighbours (fewer for libraries with <150
cells), 20 PCs. Clustering runs on every library, including tiny ones, for
completeness. The cluster QC table is the basis for judging which clusters
are meaningful:

  n_markers          genes with padj < 0.05, log2FC > 1, detected in >= 25%
                     of the cluster (Wilcoxon, cluster vs rest of library)
  stability_jaccard  mean best-match Jaccard over bootstrap re-clusterings of
                     80% subsamples (>0.75 stable, 0.6-0.75 weak, <0.5 dissolves)
  silhouette         mean silhouette width in the PCA space used for the graph
  log2_depth_ratio   cluster median UMIs vs library median; large |values| hint
                     at a depth-driven (technical) cluster
  doublet_score      mean Scrublet score (continuous score only; no calls)
"""
from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from pipeline_utils import (
    FIGURES,
    PROCESSED,
    QC_H5AD,
    STAGE_ORDER,
    TABLES,
    ensure_dirs,
    library_of,
    save_figure,
)

LIBRARY_DIR = PROCESSED / "libraries"
LIBRARY_FIGURES = FIGURES / "06_libraries"

MIN_CELLS = 3
N_TOP_GENES = 2000
N_PCS = 30
N_PCS_NEIGHBORS = 20
N_NEIGHBORS = 15
LEIDEN_RESOLUTION = 0.5
RESOLUTION_SWEEP = [0.2, 0.5, 1.0]
N_SEEDS = 5
N_BOOTSTRAP = 20
BOOTSTRAP_FRACTION = 0.8
MARKER_PADJ = 0.05
MARKER_LOG2FC = 1.0
MARKER_MIN_PCT = 0.25
DOUBLET_RATE = 0.05
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute per-library objects even if they already exist",
    )
    return parser.parse_args()


def n_neighbors_for(n_cells: int) -> int:
    """15 neighbours, scaled down for tiny libraries (n/10, at least 5)."""
    return int(min(N_NEIGHBORS, max(5, n_cells // 10)))


def leiden(adata, resolution, seed, key="leiden") -> None:
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=key,
        flavor="leidenalg",
        random_state=seed,
    )


def doublet_scores(counts: ad.AnnData) -> np.ndarray:
    """Continuous Scrublet scores; NaN if the library is too small for it."""
    tmp = counts.copy()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.pp.scrublet(
                tmp,
                expected_doublet_rate=DOUBLET_RATE,
                n_prin_comps=min(30, tmp.n_obs - 2),
                threshold=0.25,
                random_state=RANDOM_STATE,
            )
        return tmp.obs["doublet_score"].to_numpy()
    except Exception as exc:  # noqa: BLE001 - tiny libraries break scrublet
        print(f"    scrublet failed ({exc.__class__.__name__}); scores set to NaN")
        return np.full(tmp.n_obs, np.nan)


def process_library(counts: ad.AnnData) -> ad.AnnData:
    """Normalise, embed and cluster one library (raw counts in X)."""
    adata = counts.copy()
    adata.uns.clear()  # drop full-subset plotting state (not h5ad-serialisable)
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    adata.layers["counts"] = adata.X.copy()
    adata.obs["doublet_score"] = doublet_scores(adata)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, flavor="seurat")

    n_comps = min(N_PCS, adata.n_obs - 2)
    hvg = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(hvg, max_value=10)
    sc.tl.pca(hvg, n_comps=n_comps, svd_solver="arpack", random_state=RANDOM_STATE)
    adata.obsm["X_pca"] = hvg.obsm["X_pca"]
    adata.uns["pca"] = dict(hvg.uns["pca"])
    del hvg

    n_pcs = min(N_PCS_NEIGHBORS, n_comps)
    k = n_neighbors_for(adata.n_obs)
    sc.pp.neighbors(adata, n_neighbors=k, n_pcs=n_pcs, random_state=RANDOM_STATE)
    sc.tl.umap(adata, min_dist=0.3, random_state=RANDOM_STATE)
    leiden(adata, LEIDEN_RESOLUTION, RANDOM_STATE)
    for res in RESOLUTION_SWEEP:
        leiden(adata, res, RANDOM_STATE, key=f"leiden_r{res}")
    adata.uns["library_params"] = {"n_pcs": n_pcs, "n_neighbors": k}
    return adata


def seed_ari(adata) -> float:
    """Mean ARI between the reference partition and other Leiden seeds."""
    ref = adata.obs["leiden"].to_numpy()
    scores = []
    for seed in range(1, N_SEEDS):
        leiden(adata, LEIDEN_RESOLUTION, seed, key="_seed")
        scores.append(adjusted_rand_score(ref, adata.obs["_seed"]))
    del adata.obs["_seed"]
    return float(np.mean(scores))


def bootstrap_stability(adata) -> pd.Series:
    """Mean best-match Jaccard per cluster over subsample re-clusterings."""
    params = adata.uns["library_params"]
    labels = adata.obs["leiden"].to_numpy()
    clusters = adata.obs["leiden"].cat.categories
    rng = np.random.default_rng(RANDOM_STATE)
    n_sub = int(round(BOOTSTRAP_FRACTION * adata.n_obs))
    scores = {c: [] for c in clusters}
    for b in range(N_BOOTSTRAP):
        idx = np.sort(rng.choice(adata.n_obs, size=n_sub, replace=False))
        sub = ad.AnnData(obs=pd.DataFrame(index=adata.obs_names[idx]))
        sub.obsm["X_pca"] = adata.obsm["X_pca"][idx]
        sc.pp.neighbors(
            sub,
            n_neighbors=min(params["n_neighbors"], n_sub - 1),
            n_pcs=params["n_pcs"],
            use_rep="X_pca",
            random_state=b,
        )
        leiden(sub, LEIDEN_RESOLUTION, b, key="boot")
        orig = labels[idx]
        boot = sub.obs["boot"].to_numpy()
        for c in clusters:
            in_c = orig == c
            if not in_c.any():
                continue
            best = max(
                (in_c & (boot == k)).sum() / (in_c | (boot == k)).sum()
                for k in np.unique(boot[in_c])
            )
            scores[c].append(best)
    return pd.Series({c: np.mean(v) if v else np.nan for c, v in scores.items()})


def markers(adata) -> pd.DataFrame:
    """Wilcoxon cluster-vs-rest markers for clusters with >= 3 cells."""
    sizes = adata.obs["leiden"].value_counts()
    groups = [c for c in adata.obs["leiden"].cat.categories if sizes[c] >= 3]
    if len(groups) < 2:
        return pd.DataFrame()
    sc.tl.rank_genes_groups(
        adata, "leiden", groups=groups, reference="rest",
        method="wilcoxon", pts=True, use_raw=False,
    )
    df = sc.get.rank_genes_groups_df(adata, group=groups)
    return df.rename(columns={"group": "cluster"})


def cluster_qc(adata, library: str, marker_df: pd.DataFrame) -> pd.DataFrame:
    obs = adata.obs
    n_pcs = adata.uns["library_params"]["n_pcs"]
    lib_median = obs["total_counts"].median()
    if obs["leiden"].nunique() > 1:
        sil = silhouette_samples(adata.obsm["X_pca"][:, :n_pcs], obs["leiden"])
    else:
        sil = np.full(adata.n_obs, np.nan)
    stability = bootstrap_stability(adata)

    rows = []
    for c in obs["leiden"].cat.categories:
        mask = (obs["leiden"] == c).to_numpy()
        m = marker_df[marker_df["cluster"] == c] if len(marker_df) else marker_df
        if len(m):
            good = m[
                (m["pvals_adj"] < MARKER_PADJ)
                & (m["logfoldchanges"] > MARKER_LOG2FC)
                & (m["pct_nz_group"] >= MARKER_MIN_PCT)
            ]
            n_markers = len(good)
            top = ", ".join(good["names"].head(8))
        else:
            n_markers, top = np.nan, ""
        rows.append(
            {
                "library": library,
                "cluster": c,
                "n_cells": int(mask.sum()),
                "frac_of_library": mask.mean(),
                "median_umis": obs.loc[mask, "total_counts"].median(),
                "median_genes": obs.loc[mask, "n_genes_by_counts"].median(),
                "median_pct_ribo": obs.loc[mask, "pct_counts_ribo"].median(),
                "log2_depth_ratio": np.log2(obs.loc[mask, "total_counts"].median() / lib_median),
                "doublet_score": obs.loc[mask, "doublet_score"].mean(),
                "silhouette": sil[mask].mean(),
                "stability_jaccard": stability.get(c, np.nan),
                "n_markers": n_markers,
                "top_markers": top,
            }
        )
    return pd.DataFrame(rows)


def plot_library(adata, library: str, title: str, qc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    size = max(8, min(60, 12000 / adata.n_obs))
    adata.obs["log10_umis"] = np.log10(adata.obs["total_counts"])
    sc.pl.umap(adata, color="leiden", ax=axes[0], show=False, frameon=False, size=size,
               legend_loc="on data", legend_fontsize=8, title="Leiden (r=0.5)")
    sc.pl.umap(adata, color="log10_umis", ax=axes[1], show=False, frameon=False,
               size=size, cmap="viridis", title="log10 UMIs")
    sc.pl.umap(adata, color="pct_counts_ribo", ax=axes[2], show=False, frameon=False,
               size=size, cmap="magma", title="% ribosomal")
    if adata.obs["doublet_score"].notna().any():
        sc.pl.umap(adata, color="doublet_score", ax=axes[3], show=False, frameon=False,
                   size=size, cmap="Reds", title="Scrublet score")
    else:
        axes[3].set_axis_off()
        axes[3].text(0.5, 0.5, "Scrublet n/a\n(library too small)", ha="center",
                     va="center", transform=axes[3].transAxes)

    ax = axes[4]
    y = np.arange(len(qc))[::-1]
    ax.barh(y, qc["stability_jaccard"], color="#4c72b0", alpha=0.8)
    ax.axvline(0.75, color="0.3", ls="--", lw=0.8)
    ax.axvline(0.5, color="0.3", ls=":", lw=0.8)
    ax.set_yticks(y, [f"{c} (n={n}, {int(m) if pd.notna(m) else '–'} mk)"
                      for c, n, m in zip(qc["cluster"], qc["n_cells"], qc["n_markers"])])
    ax.set_xlim(0, 1)
    ax.set_xlabel("bootstrap Jaccard stability")
    ax.set_title("Cluster stability (mk = markers)")
    adata.obs.drop(columns="log10_umis", inplace=True)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save_figure(fig, f"06_libraries/06_umap_{library}")
    plt.close(fig)


def plot_all_umaps(objs: dict[str, ad.AnnData], titles: dict[str, str]) -> None:
    ncol = 4
    nrow = int(np.ceil(len(objs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow))
    for ax, (lib, adata) in zip(axes.ravel(), objs.items()):
        size = max(8, min(60, 12000 / adata.n_obs))
        sc.pl.umap(adata, color="leiden", ax=ax, show=False, frameon=False, size=size,
                   legend_loc="on data", legend_fontsize=7, title=titles[lib])
    for ax in axes.ravel()[len(objs):]:
        ax.set_axis_off()
    fig.suptitle("Per-library Leiden clusters (resolution 0.5), 3.3–6.0 hpf", y=1.0)
    fig.tight_layout()
    save_figure(fig, "06_umap_leiden_all_libraries")
    plt.close(fig)


def plot_qc_overview(qc: pd.DataFrame, order: list[str], titles: dict[str, str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    ypos = {lib: i for i, lib in enumerate(order[::-1])}
    y = qc["library"].map(ypos) + np.random.default_rng(0).uniform(-0.15, 0.15, len(qc))
    for ax, col, label in (
        (axes[0], "stability_jaccard", "bootstrap Jaccard stability"),
        (axes[1], "silhouette", "mean silhouette (PCA space)"),
    ):
        pts = ax.scatter(
            qc[col], y, s=np.clip(qc["n_cells"], 5, None) / 2.5,
            c=qc["n_markers"].fillna(0).clip(upper=100), cmap="viridis",
            vmin=0, vmax=100, alpha=0.8, edgecolor="0.2", linewidth=0.4,
        )
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.3)
    axes[0].axvline(0.75, color="0.3", ls="--", lw=0.8)
    axes[0].axvline(0.5, color="0.3", ls=":", lw=0.8)
    axes[1].axvline(0, color="0.3", ls=":", lw=0.8)
    axes[0].set_yticks(list(ypos.values()), [titles[lib] for lib in ypos])
    cb = fig.colorbar(pts, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("marker genes (capped at 100)")
    fig.suptitle("Cluster QC per library (dot area ∝ cluster size)")
    save_figure(fig, "06_cluster_qc_overview")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_FIGURES.mkdir(parents=True, exist_ok=True)

    print(f"[06] loading {QC_H5AD}")
    full = sc.read_h5ad(QC_H5AD)
    full.obs["library"] = library_of(full.obs_names)
    full.obs["stage"] = pd.Categorical(full.obs["stage"], STAGE_ORDER, ordered=True)
    libs = (
        full.obs.groupby("library", observed=True)
        .agg(stage=("stage", "first"), batch=("sequencing_batch", "first"),
             replicate=("replicate", "first"), n_cells=("stage", "size"))
        .sort_values(["stage", "batch", "replicate"])
    )

    objs, titles, qcs, marker_tables, summaries = {}, {}, [], [], []
    for library, info in libs.iterrows():
        path = LIBRARY_DIR / f"{library}.h5ad"
        print(f"[06] {library}: {info['n_cells']} cells")
        if path.exists() and not args.force:
            adata = sc.read_h5ad(path)
        else:
            adata = process_library(full[full.obs["library"] == library].copy())

        marker_df = markers(adata)
        qc = cluster_qc(adata, library, marker_df)
        ari = seed_ari(adata)
        adata.write_h5ad(path, compression="gzip")

        n_clusters = adata.obs["leiden"].nunique()
        tag = f"{info['batch']}{'b' if info['replicate'] == 'b' else ''}"
        titles[library] = f"{info['stage']} {tag} (n={adata.n_obs}, k={n_clusters})"
        objs[library] = adata
        qcs.append(qc)
        if len(marker_df):
            marker_tables.append(
                marker_df.groupby("cluster", observed=True).head(20).assign(library=library)
            )
        stable = qc["stability_jaccard"] >= 0.75
        summaries.append(
            {
                "library": library,
                "stage": info["stage"],
                "batch": info["batch"],
                "replicate": info["replicate"],
                "n_cells": adata.n_obs,
                "n_genes_kept": adata.n_vars,
                "n_pcs": adata.uns["library_params"]["n_pcs"],
                "n_neighbors": adata.uns["library_params"]["n_neighbors"],
                **{f"n_clusters_r{r}": adata.obs[f"leiden_r{r}"].nunique()
                   for r in RESOLUTION_SWEEP},
                "seed_ari": ari,
                "n_stable_clusters": int(stable.sum()),
                "frac_cells_in_stable_clusters": qc.loc[stable, "n_cells"].sum() / adata.n_obs,
                "median_cluster_stability": qc["stability_jaccard"].median(),
            }
        )
        plot_library(adata, library, titles[library], qc)

    qc_all = pd.concat(qcs, ignore_index=True)
    qc_all.to_csv(TABLES / "06_cluster_qc.tsv", sep="\t", index=False, float_format="%.3f")
    pd.concat(marker_tables, ignore_index=True)[
        ["library", "cluster", "names", "logfoldchanges", "pvals_adj",
         "pct_nz_group", "pct_nz_reference", "scores"]
    ].to_csv(TABLES / "06_cluster_markers.tsv", sep="\t", index=False, float_format="%.4g")
    summary = pd.DataFrame(summaries)
    summary.to_csv(TABLES / "06_library_clustering_summary.tsv", sep="\t",
                   index=False, float_format="%.3f")
    cells = pd.concat(
        pd.DataFrame(
            {"library": lib, "leiden": a.obs["leiden"].astype(str),
             "UMAP1": a.obsm["X_umap"][:, 0], "UMAP2": a.obsm["X_umap"][:, 1]},
            index=a.obs_names,
        )
        for lib, a in objs.items()
    )
    cells.index.name = "cell_id"
    cells.to_csv(TABLES / "06_cell_clusters.csv", float_format="%.4f")

    plot_all_umaps(objs, titles)
    plot_qc_overview(qc_all, list(objs), titles)

    print(summary.to_string(index=False))
    print(qc_all.drop(columns="top_markers").to_string(index=False))
    print("[06] done")


if __name__ == "__main__":
    main()
