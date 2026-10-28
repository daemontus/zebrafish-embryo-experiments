"""Do the a/b libraries of the same stage and batch mix?

Six stage/batch combinations have two libraries (a and b), each a separate
pool of embryos. For each pair, the two libraries are embedded together
(library-size normalisation, log1p, PCA of the genes heterogeneous in either
library, script 01b) and every cell's k nearest neighbours are inspected: if the libraries mix, the fraction of
neighbours from library b is binomial around the global fraction of b cells.

Each pair is calibrated against
  * a permutation null (a/b labels shuffled on the same kNN graph),
  * a negative control (a random split of one library, same sizes),
  * a positive control (an adjacent stage of the same batch, same sizes),
  * where available, the same stage from another batch (same sizes).

Two experiments are run on every comparison:
  * "normalized": full libraries, only library-size normalisation;
  * "thinned": the deeper library is binomially thinned first, so that the
    two libraries have the same median UMIs per cell.

Metrics per comparison: kBET-style rejection rate (per-cell binomial test of
the neighbourhood composition), a simplified iLISI (inverse Simpson index on
the uniform kNN neighbourhood), the number of locally enriched cells (BH
FDR < 0.05), and cross-validated AUC of a classifier predicting the library
from the PCs (plus the AUC of depth alone).

As a robustness check, the a/b comparisons are repeated after removing the
genes with the largest pseudobulk a/b fold change (top 50/200/1000), to see
whether a few genes drive the separation.

Genes: heterogeneous genes rather than HVGs, because HVGs selected on two
libraries jointly include genes that differ between the libraries as a whole,
which biases the test towards separation.

Outputs:
  results/tables/replicate_mixing.tsv
  results/tables/replicate_mixing_gene_removal.tsv
  report/figures/replicate_mixing.pdf (summary + 50% epiboly DS4 UMAP)
  results/figures/replicate_mixing_hist.{pdf,png}
  results/figures/replicate_mixing_umap_{normalized,thinned}.{pdf,png}
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import binom, false_discovery_control
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from pipeline_utils import (
    REPORT_WIDTH_IN,
    ROOT,
    TABLES,
    ensure_dirs,
    load_dropseq_counts,
    report_style,
    save_report_figure,
)

FIGURES = ROOT / "results" / "figures"

K = 20  # neighbours per cell
N_PCS = 30
N_PERM = 200  # label permutations for the kNN null
N_PERM_AUC = 20  # label permutations for the classifier null
SEED = 0

# (pair name, library a, library b, adjacent-stage control, other-batch control)
PAIRS = [
    ("high DS5", "ZFHIGH_WT_DS5", "ZFHIGH_WT_DS5b", "ZFOBLONG_WT_DS5b", None),
    ("oblong DS5", "ZFOBLONG_WT_DS5", "ZFOBLONG_WT_DS5b", "ZFDOME_WT_DS5", None),
    ("30% epiboly DS5", "ZF30_WT_DS5", "ZF30_WT_DS5b", "ZFDOME_WT_DS5", None),
    ("50% epiboly DS4", "ZF50_WT_DS4", "ZF50_WT_DS4b", "ZF60_WT_DS4", "ZF50_WT_DS3"),
    ("bud DS2", "ZFB_WT_DS2", "ZFB_WT_DS2b", "ZF90_WT_DS2", "ZFB_WT_DS3"),
    ("6-somite DS5", "ZF6S_WT_DS5", "ZF6S_WT_DS5b", "ZF3S_WT_DS5", None),
]
KINDS = ["replicate", "split", "stage", "batch"]
KIND_LABELS = {
    "replicate": "a vs b",
    "split": "random split (neg.)",
    "stage": "adjacent stage (pos.)",
    "batch": "other batch",
}
KIND_STYLE = {
    "replicate": dict(color="#1A1A1A", lw=1.4, ls="-"),
    "split": dict(color="#7A7A7A", lw=1.0, ls="--"),
    "stage": dict(color="#D55E00", lw=1.2, ls="-"),
    "batch": dict(color="#0072B2", lw=1.2, ls="-."),
}
EXPERIMENTS = ["normalized", "thinned"]
GENE_REMOVAL = [50, 200, 1000]
# Pair and gene shown in the report figure.
FIGURE_PAIR = "50% epiboly DS4"
FIGURE_GENE = "HSP90AA1.2"


def heterogeneous_genes(comp: dict, calls: pd.DataFrame) -> set:
    """Genes heterogeneous (weak tier) in either library of a comparison (script 01b)."""
    return set(calls.index[(calls[list(set(comp["libs"]))] > 0).any(axis=1)])


def comparisons(adata: sc.AnnData, rng: np.random.Generator) -> list[dict]:
    """Cell sets of all comparisons; controls are sized like their pair."""
    lib = adata.obs["library"].to_numpy()
    cells = {name: np.flatnonzero(lib == name) for name in np.unique(lib)}
    out = []
    for pair, lib_a, lib_b, lib_stage, lib_batch in PAIRS:
        a, b = cells[lib_a], cells[lib_b]
        n_a, n_b = len(a), len(b)
        out.append(dict(pair=pair, kind="replicate", a=a, b=b, libs=(lib_a, lib_b)))

        # Negative control: split the larger library with the pair's a:b ratio.
        big_name, big = (lib_a, a) if n_a >= n_b else (lib_b, b)
        perm = rng.permutation(big)
        cut = round(len(big) * n_a / (n_a + n_b))
        out.append(
            dict(pair=pair, kind="split", a=perm[:cut], b=perm[cut:],
                 libs=(big_name, big_name))
        )

        for kind, other in (("stage", lib_stage), ("batch", lib_batch)):
            if other is None:
                continue
            o = cells[other]
            out.append(
                dict(pair=pair, kind=kind, a=a,
                     b=rng.choice(o, size=min(n_b, len(o)), replace=False),
                     libs=(lib_a, other))
            )
    return out


def thin(counts: sp.csr_matrix, keep: float, rng: np.random.Generator) -> sp.csr_matrix:
    """Binomial thinning: keep each UMI independently with probability `keep`."""
    out = counts.copy()
    out.data = rng.binomial(out.data.astype(np.int64), keep).astype(np.float32)
    out.eliminate_zeros()
    return out


def build(adata: sc.AnnData, comp: dict, experiment: str,
          rng: np.random.Generator) -> sc.AnnData:
    """Counts of one comparison (optionally depth-matched), with a/b labels."""
    sub = adata[np.concatenate([comp["a"], comp["b"]])].copy()
    sub.obs["side"] = ["a"] * len(comp["a"]) + ["b"] * len(comp["b"])
    is_b = (sub.obs["side"] == "b").to_numpy()
    totals = np.asarray(sub.X.sum(axis=1)).ravel()
    med_a, med_b = np.median(totals[~is_b]), np.median(totals[is_b])
    sub.uns["median_umis"] = (med_a, med_b)
    if experiment == "thinned" and comp["kind"] != "split":
        deep = ~is_b if med_a > med_b else is_b
        keep = min(med_a, med_b) / max(med_a, med_b)
        X = sub.X.tocsr()
        X[deep] = thin(X[deep], keep, rng)
        sub.X = X
    sub.obs["total_counts"] = np.asarray(sub.X.sum(axis=1)).ravel()
    return sub


def embed(sub: sc.AnnData, genes: set) -> np.ndarray:
    """Joint normalisation and PCA of a comparison on the given genes;
    returns the PCs."""
    sc.pp.filter_genes(sub, min_cells=3)
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    sub = sub[:, sub.var_names.isin(list(genes))].copy()
    sc.pp.scale(sub, max_value=10)
    n_pcs = min(N_PCS, sub.n_obs - 1)
    sc.tl.pca(sub, n_comps=n_pcs, random_state=SEED)
    return sub.obsm["X_pca"]


def knn(pcs: np.ndarray) -> np.ndarray:
    """Indices of the K nearest neighbours of every cell (self excluded)."""
    nn = NearestNeighbors(n_neighbors=K + 1).fit(pcs)
    return nn.kneighbors(pcs, return_distance=False)[:, 1:]


def neighbourhood_stats(idx: np.ndarray, is_b: np.ndarray) -> dict:
    """Mixing statistics of a kNN graph for one labelling."""
    n_b = is_b[idx].sum(axis=1)
    p = is_b.mean()
    pval = np.minimum(1.0, 2 * np.minimum(binom.cdf(n_b, K, p), binom.sf(n_b - 1, K, p)))
    frac = n_b / K
    return dict(
        n_b=n_b,
        reject=(pval < 0.05).mean(),
        enriched=(false_discovery_control(pval) < 0.05).sum(),
        lisi=np.median(1.0 / (frac**2 + (1 - frac) ** 2)),
    )


def cv_auc(features: np.ndarray, is_b: np.ndarray, seed: int) -> float:
    """5-fold cross-validated AUC of a balanced logistic regression."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    model = LogisticRegression(class_weight="balanced", max_iter=2000)
    score = cross_val_predict(model, features, is_b, cv=folds,
                              method="decision_function")
    return roc_auc_score(is_b, score)


def analyse(sub: sc.AnnData, pcs: np.ndarray, rng: np.random.Generator) -> dict:
    is_b = (sub.obs["side"] == "b").to_numpy()
    idx = knn(pcs)
    obs = neighbourhood_stats(idx, is_b)
    null = [neighbourhood_stats(idx, rng.permutation(is_b)) for _ in range(N_PERM)]
    null_auc = [cv_auc(pcs, rng.permutation(is_b), seed=i) for i in range(N_PERM_AUC)]
    log_depth = np.log10(sub.obs["total_counts"].to_numpy())[:, None]
    depth_pc_corr = [abs(np.corrcoef(log_depth[:, 0], pcs[:, i])[0, 1]) for i in range(10)]
    return dict(
        n_b_obs=obs["n_b"],
        n_b_null=np.concatenate([s["n_b"] for s in null]),
        n_a=int((~is_b).sum()),
        n_b=int(is_b.sum()),
        reject=obs["reject"],
        reject_null=np.mean([s["reject"] for s in null]),
        enriched=int(obs["enriched"]),
        enriched_null_max=int(max(s["enriched"] for s in null)),
        lisi=obs["lisi"],
        lisi_null=np.mean([s["lisi"] for s in null]),
        auc=cv_auc(pcs, is_b, seed=SEED),
        auc_null_mean=float(np.mean(null_auc)),
        auc_null_max=float(np.max(null_auc)),
        auc_depth=roc_auc_score(is_b, log_depth[:, 0]),
        depth_pc_corr=max(depth_pc_corr),
        depth_pc=int(np.argmax(depth_pc_corr)) + 1,
    )


def plot_histograms(results: dict) -> None:
    """Fraction of neighbours from b: each pair, both experiments."""
    fig, axes = plt.subplots(
        len(PAIRS), 2, figsize=(REPORT_WIDTH_IN, 1.45 * len(PAIRS)),
        sharex=True, constrained_layout=True,
    )
    bins = np.arange(K + 2) - 0.5
    for row, (pair, *_) in enumerate(PAIRS):
        for col, exp in enumerate(EXPERIMENTS):
            ax = axes[row, col]
            rep = results[(pair, "replicate", exp)]
            null = np.histogram(rep["n_b_null"], bins=bins, density=True)[0]
            ax.bar(np.arange(K + 1) / K, null, width=1 / K, color="#DDDDDD",
                   label="permutation null")
            for kind in KINDS:
                res = results.get((pair, kind, exp))
                if res is None:
                    continue
                h = np.histogram(res["n_b_obs"], bins=bins, density=True)[0]
                ax.step(np.arange(K + 1) / K, h, where="mid", label=KIND_LABELS[kind],
                        **KIND_STYLE[kind])
            p = rep["n_b"] / (rep["n_a"] + rep["n_b"])
            ax.axvline(p, color="#7A7A7A", lw=0.6, ls=":")
            ax.set_xlim(-0.03, 1.03)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            if row == 0:
                ax.set_title(f"{exp}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{pair}\n(a {rep['n_a']}, b {rep['n_b']})", fontsize=7)
            if row == len(PAIRS) - 1:
                ax.set_xlabel(f"fraction of {K} nearest neighbours from b")
    handles, labels = axes[3, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    save(fig, "replicate_mixing_hist")


def plot_umaps(umaps: dict, exp: str) -> None:
    """Separate panels: cells of a, cells of b, and log10 UMIs."""
    fig, axes = plt.subplots(
        len(PAIRS), 3, figsize=(REPORT_WIDTH_IN, 1.75 * len(PAIRS)),
        constrained_layout=True,
    )
    for row, (pair, *_) in enumerate(PAIRS):
        sub = umaps[(pair, exp)]
        xy = sub.obsm["X_umap"]
        is_b = (sub.obs["side"] == "b").to_numpy()
        depth = np.log10(sub.obs["total_counts"].to_numpy())
        size = max(1.0, min(8.0, 3000 / sub.n_obs))
        for col, (mask, title) in enumerate(((~is_b, "library a"), (is_b, "library b"))):
            ax = axes[row, col]
            ax.scatter(*xy[~mask].T, s=size, c="#DDDDDD", lw=0, rasterized=True)
            ax.scatter(*xy[mask].T, s=size, c="#1A1A1A", lw=0, rasterized=True)
            ax.set_title(f"{pair}: {title} ({mask.sum()})", fontsize=7)
        ax = axes[row, 2]
        order = np.random.default_rng(SEED).permutation(sub.n_obs)
        points = ax.scatter(*xy[order].T, s=size, c=depth[order], cmap="viridis",
                            lw=0, rasterized=True)
        fig.colorbar(points, ax=ax, shrink=0.8, label="log10 UMIs")
        ax.set_title(f"{pair}: depth", fontsize=7)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
            for spine in ax.spines.values():
                spine.set_visible(False)
    save(fig, f"replicate_mixing_umap_{exp}")


def pseudobulk_lfc(adata: sc.AnnData, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """log2 fold change (b / a) of mean expression per 10k UMIs."""
    def profile(idx):
        X = adata.X[idx]
        return np.asarray(X.multiply(1e4 / X.sum(axis=1)).mean(axis=0)).ravel()
    return np.log2((profile(b) + 0.1) / (profile(a) + 0.1))


def gene_removal(adata: sc.AnnData, comps: list[dict], calls: pd.DataFrame,
                 rng: np.random.Generator) -> pd.DataFrame:
    """Repeat the a/b comparisons without the most differing genes."""
    rows = []
    for comp in comps:
        if comp["kind"] != "replicate":
            continue
        order = np.argsort(-np.abs(pseudobulk_lfc(adata, comp["a"], comp["b"])))
        for n_removed in [0, *GENE_REMOVAL]:
            keep = np.ones(adata.n_vars, dtype=bool)
            keep[order[:n_removed]] = False
            sub = build(adata[:, keep], comp, "normalized", rng)
            res = analyse(sub, embed(sub, heterogeneous_genes(comp, calls)), rng)
            rows.append({
                "pair": comp["pair"], "genes_removed": n_removed,
                "genes": sub.n_vars,
                **{key: res[key] for key in (
                    "reject", "reject_null", "lisi", "lisi_null", "auc", "auc_null_max")},
            })
            print(f"  {comp['pair']:16s} without top {n_removed:4d} genes: "
                  f"reject {res['reject']:.2f}  AUC {res['auc']:.2f}")
    return pd.DataFrame(rows)


def plot_report_figure(table: pd.DataFrame, umap: sc.AnnData, gene_expr: np.ndarray) -> None:
    """AUC summary of all pairs (top) and one UMAP of a thinned pair (bottom)."""
    fig = plt.figure(figsize=(REPORT_WIDTH_IN, 4.6), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 0.9])

    ax = top.subplots()
    pairs = [p[0] for p in PAIRS]
    y = {pair: len(pairs) - 1 - i for i, pair in enumerate(pairs)}
    offsets = {"normalized": 0.13, "thinned": -0.13}
    for kind in KINDS:
        for exp in EXPERIMENTS:
            rows = table[(table["kind"] == kind) & (table["experiment"] == exp)]
            color = KIND_STYLE[kind]["color"]
            ax.scatter(
                rows["auc"], [y[p] + offsets[exp] for p in rows["pair"]],
                s=22, marker="o" if kind != "batch" else "D",
                facecolors=color if exp == "thinned" else "white",
                edgecolors=color, linewidths=1.0, zorder=3,
            )
    null = table[(table["kind"] == "replicate")].groupby("pair")["auc_null_max"].max()
    for pair, value in null.items():
        ax.plot([0.3, value], [y[pair]] * 2, color="#CCCCCC", lw=3, solid_capstyle="butt",
                zorder=1)
    ax.set_yticks(list(y.values()), list(y.keys()))
    ax.set_xlim(0.3, 1.02)
    ax.set_xlabel("cross-validated AUC of predicting the library (0.5 = indistinguishable)")
    ax.grid(axis="x", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    marker = lambda color, shape="o", fill=True: Line2D(
        [], [], ls="", marker=shape, markersize=5, markeredgecolor=color,
        markerfacecolor=color if fill else "white")
    handles = [marker(KIND_STYLE[k]["color"], "D" if k == "batch" else "o") for k in KINDS]
    labels = [KIND_LABELS[k] for k in KINDS]
    handles += [marker("#555555", fill=False), marker("#555555"),
                Line2D([], [], color="#CCCCCC", lw=3)]
    labels += ["normalized", "thinned", "permutation null (max)"]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=6.5)

    axes = bottom.subplots(1, 4)
    xy = umap.obsm["X_umap"]
    is_b = (umap.obs["side"] == "b").to_numpy()
    order = np.random.default_rng(SEED).permutation(umap.n_obs)
    panels = [
        ("library a", ~is_b, None, None),
        ("library b", is_b, None, None),
        ("log10 UMIs (thinned)", None, np.log10(umap.obs["total_counts"].to_numpy()), "viridis"),
        (f"{FIGURE_GENE.lower()} (log1p per 10k)", None, gene_expr, "magma_r"),
    ]
    for ax, (title, mask, values, cmap) in zip(axes, panels):
        if mask is not None:
            ax.scatter(*xy[~mask].T, s=1.5, c="#DDDDDD", lw=0, rasterized=True)
            ax.scatter(*xy[mask].T, s=1.5, c="#1A1A1A", lw=0, rasterized=True)
        else:
            points = ax.scatter(*xy[order].T, s=1.5, c=values[order], cmap=cmap,
                                lw=0, rasterized=True)
            bottom.colorbar(points, ax=ax, shrink=0.7, location="bottom", pad=0.02)
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for spine in ax.spines.values():
            spine.set_visible(False)
    bottom.suptitle(f"{FIGURE_PAIR}, after thinning", fontsize=8)
    save_report_figure(fig, "replicate_mixing")
    plt.close(fig)


def save(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    report_style()
    sc.settings.verbosity = 0
    adata = load_dropseq_counts()
    calls = pd.read_csv(TABLES / "heterogeneous_genes_calls.tsv.gz", sep="\t", index_col=0)
    rng = np.random.default_rng(SEED)
    comps = comparisons(adata, rng)

    results, umaps, rows = {}, {}, []
    for exp in EXPERIMENTS:
        for comp in comps:
            genes = heterogeneous_genes(comp, calls)
            sub = build(adata, comp, exp, rng)
            pcs = embed(sub, genes)
            res = analyse(sub, pcs, rng)
            results[(comp["pair"], comp["kind"], exp)] = res
            if comp["kind"] == "replicate":
                sub.obsm["X_pca"] = pcs
                sc.pp.neighbors(sub, n_neighbors=15, use_rep="X_pca", random_state=SEED)
                sc.tl.umap(sub, random_state=SEED)
                umaps[(comp["pair"], exp)] = sub
            med_a, med_b = sub.uns["median_umis"]
            rows.append({
                "experiment": exp, "pair": comp["pair"], "kind": comp["kind"],
                "library_a": comp["libs"][0], "library_b": comp["libs"][1],
                "genes": len(genes), "n_a": res["n_a"], "n_b": res["n_b"],
                "median_umis_a": med_a, "median_umis_b": med_b,
                **{key: res[key] for key in (
                    "reject", "reject_null", "enriched", "enriched_null_max",
                    "lisi", "lisi_null", "auc", "auc_null_mean", "auc_null_max",
                    "auc_depth", "depth_pc_corr", "depth_pc")},
            })
            print(f"{exp:10s} {comp['pair']:16s} {comp['kind']:9s} "
                  f"reject {res['reject']:.2f} (null {res['reject_null']:.2f})  "
                  f"AUC {res['auc']:.2f} (null max {res['auc_null_max']:.2f})")

    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "replicate_mixing.tsv", sep="\t", index=False,
                 float_format="%.4g")
    plot_histograms(results)
    for exp in EXPERIMENTS:
        plot_umaps(umaps, exp)

    # embed() leaves the normalised log1p expression of all genes in X.
    fig_umap = umaps[(FIGURE_PAIR, "thinned")]
    gene_expr = np.asarray(fig_umap[:, FIGURE_GENE].X.todense()).ravel()
    plot_report_figure(table, fig_umap, gene_expr)

    print("gene removal:")
    removal = gene_removal(adata, comps, calls, rng)
    removal.to_csv(TABLES / "replicate_mixing_gene_removal.tsv", sep="\t", index=False,
                   float_format="%.4g")


if __name__ == "__main__":
    main()
