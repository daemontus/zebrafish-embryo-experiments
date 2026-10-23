#!/usr/bin/env python3
"""Stage 08: light batch correction of the 50% epiboly libraries (Harmony, scVI).

Input : data/processed/07_50pct_joint.h5ad   (stage-07 merged 50% epiboly object)
Output: data/processed/08_50pct_corrected.h5ad
        results/tables/08_50pct_population_labels.tsv
        results/tables/08_50pct_metrics.tsv
        results/tables/08_50pct_population_recovery.tsv
        results/tables/08_50pct_population_matches.tsv
        results/tables/08_50pct_joint_composition.tsv
        results/figures/08_50pct_*.png/.pdf

50% epiboly is the one stage sequenced in several batches (DS2, DS3, DS4,
DS4b), so batch and stage are not confounded and a correction is well-posed.
The batch key is the LIBRARY (DS4 and DS4b differ even though they share a
sequencing batch). Three embeddings are compared:

  uncorrected  stage-07 PCA (30 PCs)
  harmony      harmonypy on the same 30 PCs, default theta
  scvi         scVI on raw counts, 2,000 seurat_v3 HVGs, 10 latent dims,
               NB likelihood, one hidden layer; trained with 3 seeds (seed 0
               is used for figures, all seeds for metrics)

Each embedding gets a 15-NN graph, UMAP and Leiden (r=0.5). Scoring uses the
cross-library populations established in stage 07 (POPULATIONS below):

  batch mixing     iLISI (all cells, bulk ectoderm only); size-weighted
                   library evenness of bulk joint clusters
  biology kept     cLISI and silhouette on the confident populations + bulk;
                   per-population recovery (recall/purity of the best joint
                   cluster); mutual kNN matches between each population's
                   local clusters across libraries
"""
from __future__ import annotations

import argparse
import itertools
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors

from cluster_compare import best_matches, joint_composition, knn_mapping, library_colors
from pipeline_utils import PROCESSED, TABLES, ensure_dirs, save_figure

INPUT_H5AD = PROCESSED / "07_50pct_joint.h5ad"
OUTPUT_H5AD = PROCESSED / "08_50pct_corrected.h5ad"

N_PCS = 30
N_NEIGHBORS = 15
LEIDEN_RESOLUTION = 0.5
LISI_K = 30
SCVI_N_HVG = 2000
SCVI_N_LATENT = 10
SCVI_SEEDS = [0, 1, 2]
RANDOM_STATE = 0
METHODS = ["uncorrected", "harmony", "scvi"]
METHOD_COLORS = {"uncorrected": "#8c8c8c", "harmony": "#4c72b0", "scvi": "#c44e52"}

# Cross-library populations from stage 07 (results/07_merged_contexts.md and
# the per-library markers in stage 06). Keys are stage-06 local cluster IDs.
POPULATIONS = {
    "p53": {"50% DS2": ["6"], "50% DS3": ["6"], "50% DS4": ["7"], "50% DS4b": ["8"]},
    "endoderm": {"50% DS2": ["3"], "50% DS3": ["4"], "50% DS4": ["3"], "50% DS4b": ["5"]},
    "margin": {"50% DS2": ["4"], "50% DS3": ["3"], "50% DS4": ["2"], "50% DS4b": ["0"]},
    "EVL": {"50% DS2": ["5"], "50% DS3": ["7"], "50% DS4": ["6"], "50% DS4b": ["6"]},
    "prechordal": {"50% DS4": ["8"], "50% DS4b": ["7"]},
    "organizer": {"50% DS2": ["2"], "50% DS3": ["5"], "50% DS4": ["5"], "50% DS4b": ["3"]},
    "bulk": {"50% DS2": ["0"], "50% DS3": ["0", "1", "2"], "50% DS4": ["0", "1"],
             "50% DS4b": ["1", "2"]},
    "low_quality": {"50% DS2": ["1"], "50% DS4b": ["4"]},
    "heat_shock": {"50% DS4": ["4"]},
}
# Populations with a confident cross-library identity (used for bio metrics).
CONFIDENT = ["p53", "endoderm", "margin", "EVL", "prechordal"]
SCORED = CONFIDENT + ["bulk"]
POP_COLORS = {
    "p53": "#8c564b", "endoderm": "#2ca02c", "margin": "#ff7f0e", "EVL": "#1f77b4",
    "prechordal": "#9467bd", "organizer": "#e377c2", "bulk": "#d9d9d9",
    "low_quality": "#7f7f7f", "heat_shock": "#bcbd22",
}

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
                        help="recompute Harmony/scVI embeddings even if they exist")
    return parser.parse_args()


def assign_populations(obs: pd.DataFrame) -> pd.Series:
    lookup = {f"{lib}:{c}": pop for pop, libs in POPULATIONS.items()
              for lib, clusters in libs.items() for c in clusters}
    missing = sorted(set(obs["local"]) - set(lookup))
    assert not missing, f"local clusters without a population: {missing}"
    return obs["local"].map(lookup).astype("category")


# --- embeddings ---------------------------------------------------------------

def run_harmony(adata) -> np.ndarray:
    import harmonypy

    out = harmonypy.run_harmony(
        adata.obsm["X_pca"][:, :N_PCS].astype(np.float64), adata.obs, "library",
        max_iter_harmony=20, random_state=RANDOM_STATE, verbose=False,
    )
    z = np.asarray(out.Z_corr)
    return z if z.shape[0] == adata.n_obs else z.T  # harmonypy 2.x: (N x d)


def run_scvi(adata, seed: int) -> np.ndarray:
    import scvi

    scvi.settings.seed = seed
    sub = sc.AnnData(adata.layers["counts"].copy(), obs=adata.obs[["library"]].copy(),
                     var=pd.DataFrame(index=adata.var_names))
    sc.pp.highly_variable_genes(sub, n_top_genes=SCVI_N_HVG, flavor="seurat_v3",
                                batch_key="library", subset=True)
    scvi.model.SCVI.setup_anndata(sub, batch_key="library")
    model = scvi.model.SCVI(sub, n_latent=SCVI_N_LATENT, n_layers=1, gene_likelihood="nb")
    model.train(accelerator="cpu", early_stopping=True, enable_progress_bar=False)
    print(f"    scVI seed {seed}: {len(model.history['elbo_train'])} epochs")
    return model.get_latent_representation()


def embed(adata, rep: str, key: str, umap: bool) -> None:
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep=rep, key_added=key,
                    random_state=RANDOM_STATE)
    if umap:
        sc.tl.umap(adata, min_dist=0.3, neighbors_key=key, random_state=RANDOM_STATE)
        adata.obsm[f"X_umap_{key}"] = adata.obsm["X_umap"].copy()
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, neighbors_key=key,
                 key_added=f"joint_{key}", flavor="leidenalg", random_state=RANDOM_STATE)


def build(force: bool) -> sc.AnnData:
    if OUTPUT_H5AD.exists() and not force:
        return sc.read_h5ad(OUTPUT_H5AD)
    adata = sc.read_h5ad(INPUT_H5AD)
    adata.obs["population"] = assign_populations(adata.obs)

    adata.obsm["X_uncorrected"] = adata.obsm["X_pca"][:, :N_PCS].copy()
    print("[08] Harmony")
    adata.obsm["X_harmony"] = run_harmony(adata)
    for seed in SCVI_SEEDS:
        print(f"[08] scVI seed {seed}")
        adata.obsm[f"X_scvi_s{seed}"] = run_scvi(adata, seed)

    embed(adata, "X_uncorrected", "uncorrected", umap=True)
    embed(adata, "X_harmony", "harmony", umap=True)
    for seed in SCVI_SEEDS:
        embed(adata, f"X_scvi_s{seed}", f"scvi_s{seed}", umap=(seed == SCVI_SEEDS[0]))
    adata.write_h5ad(OUTPUT_H5AD, compression="gzip")
    return adata


# --- metrics ------------------------------------------------------------------

def inverse_simpson(X, labels: np.ndarray, k: int = LISI_K) -> np.ndarray:
    """Per-cell inverse Simpson index of `labels` among the k nearest neighbours."""
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    neigh = labels[idx]
    cats = np.unique(labels)
    p = np.stack([(neigh == c).mean(axis=1) for c in cats], axis=1)
    return 1.0 / (p ** 2).sum(axis=1)


def population_recovery(obs, key: str) -> pd.DataFrame:
    rows = []
    for pop in SCORED + ["organizer"]:
        mask = obs["population"] == pop
        counts = obs.loc[mask, key].value_counts()
        best = counts.idxmax()
        in_best = obs[key] == best
        per_lib = obs[mask].groupby("library_short", observed=True)[key].apply(
            lambda s: (s == best).mean())
        rows.append({
            "population": pop,
            "n_cells": int(mask.sum()),
            "best_joint_cluster": best,
            "recall": counts.max() / mask.sum(),
            "purity": (mask & in_best).sum() / in_best.sum(),
            "n_libraries": int(per_lib.size),
            "libraries_majority_in_best": int((per_lib >= 0.5).sum()),
        })
    return pd.DataFrame(rows)


def population_matches(adata, X, libs, local_labels) -> pd.DataFrame:
    obs = adata.obs
    mapping = knn_mapping(X, obs, libs, local_labels)
    best = best_matches(mapping, obs["local"].value_counts())
    frac = mapping.set_index(["src_cluster", "tgt_cluster"])["fraction"]
    mutual = best[best["mutual"]]
    mutual_pairs = {frozenset((a, b)) for a, b in zip(mutual["src_cluster"], mutual["tgt_cluster"])}
    rows = []
    for pop in CONFIDENT + ["organizer"]:
        clusters = [f"{lib}:{c}" for lib, cs in POPULATIONS[pop].items() for c in cs]
        for a, b in itertools.combinations(clusters, 2):
            rows.append({
                "population": pop, "cluster_a": a, "cluster_b": b,
                "frac_a_to_b": frac[(a, b)], "frac_b_to_a": frac[(b, a)],
                "mutual": frozenset((a, b)) in mutual_pairs,
            })
    return pd.DataFrame(rows)


def score(adata, method: str, seed: int | None, rep: str, key: str, libs, local_labels):
    obs = adata.obs
    X = adata.obsm[rep]
    lib = obs["library_short"].astype(str).to_numpy()
    pop = obs["population"].astype(str).to_numpy()
    n_lib = len(libs)

    ilisi = (inverse_simpson(X, lib) - 1) / (n_lib - 1)
    bulk = pop == "bulk"
    scored = np.isin(pop, SCORED)
    clisi = inverse_simpson(X[scored], pop[scored])
    clisi = (len(SCORED) - clisi) / (len(SCORED) - 1)
    # Bulk is ~60% of scored cells, so per-cell medians are saturated by it;
    # average per population instead so each population counts equally.
    clisi_by_pop = pd.Series(clisi).groupby(pop[scored]).mean()
    sil = silhouette_samples(X[scored], pop[scored])
    sil_by_pop = pd.Series(sil).groupby(pop[scored]).mean()

    jkey = f"joint_{key}"
    joint_labels = sorted(obs[jkey].unique(), key=int)
    comp = joint_composition(obs, libs, joint_labels, key=jkey)
    bulk_share = obs.groupby(jkey, observed=True)["population"].apply(
        lambda s: (s == "bulk").mean()).reindex(joint_labels)
    bulk_clusters = comp[bulk_share >= 0.5]
    bulk_evenness = np.average(bulk_clusters["library_evenness"], weights=bulk_clusters["n_cells"])

    matches = population_matches(adata, X, libs, local_labels)
    confident = matches[matches["population"].isin(CONFIDENT)]
    recovery = population_recovery(obs, jkey)

    metrics = {
        "method": method, "seed": seed, "n_dims": X.shape[1],
        "n_joint_clusters": len(joint_labels),
        "ilisi_all": float(np.median(ilisi)),
        "ilisi_bulk": float(np.median(ilisi[bulk])),
        "bulk_cluster_evenness": float(bulk_evenness),
        "clisi_scored": float(clisi_by_pop.mean()),
        **{f"clisi_{p}": float(clisi_by_pop.get(p, np.nan)) for p in SCORED},
        "silhouette_scored": float(sil_by_pop.mean()),
        **{f"silhouette_{p}": float(sil_by_pop.get(p, np.nan)) for p in SCORED},
        "mutual_pairs_confident": f"{int(confident['mutual'].sum())}/{len(confident)}",
        "mutual_frac_confident": float(confident["mutual"].mean()),
        "mean_min_match_confident": float(
            confident[["frac_a_to_b", "frac_b_to_a"]].min(axis=1).mean()),
        "mean_recall_confident": float(
            recovery.loc[recovery["population"].isin(CONFIDENT), "recall"].mean()),
        "mean_purity_confident": float(
            recovery.loc[recovery["population"].isin(CONFIDENT), "purity"].mean()),
    }
    tag = dict(method=method, seed=seed)
    return (metrics, recovery.assign(**tag), matches.assign(**tag),
            comp.reset_index().assign(**tag))


# --- figures ------------------------------------------------------------------

def plot_umaps(adata, libs) -> None:
    lib_colors = library_colors(libs)
    pops = [p for p in POP_COLORS if p in set(adata.obs["population"])]
    adata.obs["population"] = pd.Categorical(adata.obs["population"], pops)
    fig, axes = plt.subplots(3, 3, figsize=(17, 15))
    for row, (method, key) in enumerate(
            [("uncorrected", "uncorrected"), ("harmony", "harmony"),
             ("scvi", f"scvi_s{SCVI_SEEDS[0]}")]):
        adata.obsm["X_umap"] = adata.obsm[f"X_umap_{key}"]
        kw = dict(show=False, frameon=False, size=8)
        sc.pl.umap(adata, color="library_short", palette=lib_colors, ax=axes[row, 0],
                   title=f"{method}: library", legend_fontsize=7, **kw)
        sc.pl.umap(adata, color="population", palette=POP_COLORS, ax=axes[row, 1],
                   title=f"{method}: population (stage 07)", legend_fontsize=7, **kw)
        sc.pl.umap(adata, color=f"joint_{key}", ax=axes[row, 2], legend_loc="on data",
                   legend_fontsize=8, title=f"{method}: joint Leiden (r=0.5)", **kw)
    fig.suptitle("50% epiboly: uncorrected vs Harmony vs scVI (batch key = library)", y=1.0)
    fig.tight_layout()
    save_figure(fig, "08_50pct_umaps")
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame) -> None:
    panels = [
        ("ilisi_all", "iLISI, all cells\n(library mixing, 0–1)"),
        ("ilisi_bulk", "iLISI, bulk ectoderm\n(library mixing, 0–1)"),
        ("bulk_cluster_evenness", "library evenness of\nbulk joint clusters"),
        ("clisi_scored", "cLISI, mean over populations\n(1 = neighbours same population)"),
        ("silhouette_scored", "silhouette, mean over populations\n(separation; dims differ)"),
        ("mutual_frac_confident", "mutual kNN matches,\nconfident populations"),
        ("mean_recall_confident", "population recall\n(best joint cluster)"),
        ("mean_purity_confident", "population purity\n(best joint cluster)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, (col, title) in zip(axes.ravel(), panels):
        for i, m in enumerate(METHODS):
            vals = metrics.loc[metrics["method"] == m, col].to_numpy(dtype=float)
            ax.bar(i, vals.mean(), color=METHOD_COLORS[m], alpha=0.85)
            if len(vals) > 1:
                ax.scatter(np.full(len(vals), i), vals, color="black", s=10, zorder=3)
            ax.text(i, vals.mean(), f"{vals.mean():.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(METHODS)), METHODS)
        ax.set_title(title, fontsize=9)
        lo = min(0, metrics[col].min())
        ax.set_ylim(lo, max(1.0, metrics[col].max() * 1.15))
    fig.suptitle("Batch mixing (top row, first three) vs biology conservation "
                 "(scVI bar = mean of 3 seeds, dots = seeds)", y=1.0)
    fig.tight_layout()
    save_figure(fig, "08_50pct_metrics")
    plt.close(fig)


def plot_recovery(recovery: pd.DataFrame) -> None:
    pops = SCORED + ["organizer"]
    cols = METHODS
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.subplots_adjust(wspace=0.5)
    for ax, field in zip(axes, ["recall", "purity"]):
        mat = (recovery.groupby(["population", "method"])[field].mean()
               .unstack().reindex(index=pops, columns=cols))
        im = ax.imshow(mat.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 else "black")
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(len(pops)), pops)
        ax.set_title(f"{field} (best joint cluster)")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle("Population recovery by joint Leiden (r=0.5); scVI = mean of 3 seeds", y=1.02)
    save_figure(fig, "08_50pct_population_recovery")
    plt.close(fig)


def plot_composition(comps: pd.DataFrame, libs) -> None:
    lib_colors = library_colors(libs)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4), sharey=True)
    for ax, m in zip(axes, METHODS):
        c = comps[(comps["method"] == m) & (comps["seed"].fillna(0) == 0)]
        counts = c.set_index("joint_cluster")[libs]
        per_lib = counts / counts.sum(axis=0)
        p = per_lib.div(per_lib.sum(axis=1), axis=0)
        bottom = np.zeros(len(p))
        for lib in libs:
            ax.bar(range(len(p)), p[lib], bottom=bottom, color=lib_colors[lib], label=lib)
            bottom += p[lib].to_numpy()
        ax.set_xticks(range(len(p)), [f"{j}\n{e:.2f}" for j, e in
                                      zip(p.index, c["library_evenness"])], fontsize=7)
        ax.set_title(f"{m}: size-corrected library share per joint cluster")
        ax.set_xlabel("joint cluster (second line: evenness)")
    axes[-1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    fig.tight_layout()
    save_figure(fig, "08_50pct_joint_composition")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    warnings.filterwarnings("ignore", category=FutureWarning)
    adata = build(args.force)
    obs = adata.obs
    libs = list(obs["library_short"].cat.categories)
    local_labels = [f"{lib}:{c}" for lib in libs for c in sorted(
        obs.loc[obs["library_short"] == lib, "local_cluster"].unique(), key=int)]

    labels = (obs[["local", "population"]].drop_duplicates()
              .sort_values("local").rename(columns={"local": "local_cluster"}))
    labels["n_cells"] = labels["local_cluster"].map(obs["local"].value_counts())
    labels.to_csv(TABLES / "08_50pct_population_labels.tsv", sep="\t", index=False)

    runs = [("uncorrected", None, "X_uncorrected", "uncorrected"),
            ("harmony", None, "X_harmony", "harmony")]
    runs += [("scvi", s, f"X_scvi_s{s}", f"scvi_s{s}") for s in SCVI_SEEDS]
    metrics, recov, matches, comps = [], [], [], []
    for method, seed, rep, key in runs:
        print(f"[08] scoring {method}" + (f" seed {seed}" if seed is not None else ""))
        m, r, mt, c = score(adata, method, seed, rep, key, libs, local_labels)
        metrics.append(m)
        recov.append(r)
        matches.append(mt)
        comps.append(c)
    metrics = pd.DataFrame(metrics)
    recov = pd.concat(recov, ignore_index=True)
    matches = pd.concat(matches, ignore_index=True)
    comps = pd.concat(comps, ignore_index=True)

    fmt = dict(sep="\t", index=False, float_format="%.3f")
    metrics.to_csv(TABLES / "08_50pct_metrics.tsv", **fmt)
    recov.to_csv(TABLES / "08_50pct_population_recovery.tsv", **fmt)
    matches.to_csv(TABLES / "08_50pct_population_matches.tsv", **fmt)
    comps.to_csv(TABLES / "08_50pct_joint_composition.tsv", **fmt)

    plot_umaps(adata, libs)
    plot_metrics(metrics)
    plot_recovery(recov)
    plot_composition(comps, libs)

    pd.set_option("display.width", 250)
    print(metrics.to_string(index=False))
    print(recov.to_string(index=False))
    print("[08] done")


if __name__ == "__main__":
    main()
