"""How many activity levels does a program need?

A Boolean (or multi-valued) model describes every program in a cell by one
of a few discrete levels. The question: how much of what the continuous
program scores say about the cells is kept when every program is reduced to
k levels, and which programs need more than two?

Prediction (the test of script 17, same evaluation genes, cells, split and
smoothing coordinates, read from its per-gene table). The curated programs'
smoothed scores are discretised per stage, on the pooled (aligned) scores of
the stage's libraries, i.e. one set of thresholds per program and stage, as
a model would use. Every level is one-hot encoded (k - 1 columns per
program, "level >= v", so the L2 penalty of the regression shrinks every
level towards the next lower one; a level may still have any effect).
One-hot columns ("level == v", shrunk towards the lowest level) span the same
models but lose with many levels (k3_onehot, k6_onehot, as a check).
Feature sets:
  curated        continuous scores (as in 17; a consistency check)
  k2 ... k6      k levels from 1-D k-means of the training cells' scores
                 (thresholds halfway between the centres; all cells get the
                 level of their nearest centre)
  active         Boolean: active if the score is above the 99.9% quantile of
                 the gene-wise permutation null of the library (the call of
                 scripts 18-19, recomputed on these coordinates)
  3state         depleted (below the 0.1% quantile of the same null) /
                 neither / active
  supported_*    active and k3 for the supported programs only (17)
The floor, the ceiling and the random reference are those of script 17 (the
ceiling is the best of all models of 17 and of this script), so the
explained fractions are on the same scale as in the report's program
section, and `curated` reproduces 17's numbers.

Fidelity (descriptive, all cells): per stage and program, the share of the
variance of the smoothed score kept by its k k-means levels (between-level
variance / total), and the share of cells in the top level, for k = 2-6.
A program whose scores are bimodal keeps most of its variance with 2
levels; a graded program (a gradient, or a population with graded marker
expression) needs more.

Outputs:
  results/tables/program_levels.tsv            (summary as 17's)
  results/tables/program_levels_genes.tsv.gz   (per gene losses)
  results/tables/program_levels_fidelity.tsv   (stage, program, k)
  report/figures/program_levels.pdf

`22_program_levels.py summarise` redraws from the tables.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import importlib
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from pipeline_utils import (
    REPORT_WIDTH_IN,
    STAGES,
    TABLES,
    ensure_dirs,
    load_dropseq_counts,
    report_style,
    save_report_figure,
)
import program_utils as prediction

tests = importlib.import_module("17_program_tests")

LEVELS = [2, 3, 4, 5, 6]
FEATURE_SETS = (["curated"] + [f"k{k}" for k in LEVELS] + ["k3_onehot", "k6_onehot", "active",
                                                            "3state", "supported_active",
                                                            "supported_k3"])
N_PERM = 5
ACTIVE_QUANTILE = 0.999
SEED = 0


# --- discretisation -----------------------------------------------------------------------

def kmeans_levels(values: np.ndarray, fit: np.ndarray, k: int) -> np.ndarray:
    """Level (0 .. k-1) of every value; centres from 1-D k-means of `fit`."""
    n = min(k, len(np.unique(fit)))
    if n < 2:
        return np.zeros(len(values), dtype=int)
    centres = np.sort(KMeans(n_clusters=n, n_init=5, random_state=SEED)
                      .fit(fit[:, None]).cluster_centers_.ravel())
    return np.searchsorted((centres[1:] + centres[:-1]) / 2, values)


def encode(levels: np.ndarray, k: int, how: str = "cumulative") -> np.ndarray:
    """k - 1 indicator columns per program: level >= v (cumulative) or
    level == v (one-hot), v = 1 .. k-1. Both span the same functions of the
    levels; they differ in what the L2 penalty shrinks towards (the next
    lower level vs the lowest level)."""
    if levels.shape[1] == 0:
        return np.zeros((len(levels), 0))
    op = np.greater_equal if how == "cumulative" else np.equal
    return np.hstack([np.column_stack([op(levels[:, j], v) for v in range(1, k)])
                      for j in range(levels.shape[1])]).astype(float)


def null_thresholds(res: np.ndarray, col: dict, programs: pd.DataFrame, library: np.ndarray,
                    coords: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Per cell and program: the 0.1% and 99.9% quantiles of the smoothed
    scores of gene-wise permuted residuals of the cell's library."""
    low = np.zeros((len(library), len(programs)))
    high = np.zeros_like(low)
    for lib in np.unique(library):
        rows = np.flatnonzero(library == lib)
        lib_vec = np.full(len(rows), lib)
        for j, genes in enumerate(programs["genes"]):
            r = res[rows][:, [col[g] for g in genes]]
            null = np.concatenate([prediction.smooth(
                rng.permuted(r, axis=0).mean(axis=1)[:, None], lib_vec, coords[rows])[:, 0]
                for _ in range(N_PERM)])
            low[rows, j], high[rows, j] = np.quantile(null, [1 - ACTIVE_QUANTILE,
                                                             ACTIVE_QUANTILE])
    return low, high


def fidelity(stage: str, names: list[str], score: np.ndarray, train: np.ndarray) -> list[dict]:
    rows = []
    for j, name in enumerate(names):
        v = score[:, j]
        total = v.var()
        for k in LEVELS:
            level = kmeans_levels(v, v[train], k)
            means = pd.Series(v).groupby(level).transform("mean").to_numpy()
            rows.append(dict(stage=stage, program=name.removeprefix("state:"), k=k,
                             retained=means.var() / total if total > 0 else np.nan,
                             top_share=(level == level.max()).mean()))
    return rows


# --- prediction --------------------------------------------------------------------------

def evaluate_stage(adata, stage: str, long: pd.DataFrame, support: pd.DataFrame,
                   het_genes: set, previous: pd.DataFrame, rng):
    sub = adata[adata.obs["stage"] == stage].copy()
    library = sub.obs["library"].astype(str).to_numpy()
    total = np.asarray(sub.X.sum(axis=1)).ravel()
    sets = tests.stage_programs(long, stage)
    supported = set(support.loc[(support["stage"] == stage) & (support["set"] == "curated")
                                & support["supported"], "program"])
    curated = sets["curated"]
    is_supported = curated.index.isin(supported)

    # Evaluation genes, split and coordinates exactly as in script 17.
    prev = previous[(previous["stage"] == stage) & (previous["target"] == "detection")]
    eval_genes, group = list(prev["gene"]), prev["group"].to_numpy()
    genes_b = sub.var_names.get_indexer(eval_genes)
    train = np.zeros(sub.n_obs, dtype=bool)
    tr, _ = train_test_split(np.arange(sub.n_obs), test_size=prediction.TEST_FRACTION,
                             stratify=library, random_state=tests.SEED)
    train[tr] = True
    coord_genes = sub.var_names.get_indexer(sorted(het_genes - set(eval_genes)))
    coords = PCA(prediction.N_PCS, random_state=tests.SEED).fit_transform(
        tests.library_residuals(sub, library, coord_genes))

    used = sorted({g for genes in curated["genes"] for g in genes})
    res = tests.library_residuals(sub, library, sub.var_names.get_indexer(used))
    col = {g: i for i, g in enumerate(used)}
    score = prediction.smooth(tests.scores(res, col, curated, weighted=False), library, coords)
    low, high = null_thresholds(res, col, curated, library, coords, rng)
    active = (score > high).astype(int)
    three = np.where(score > high, 2, np.where(score < low, 0, 1))
    levels = {k: np.column_stack([kmeans_levels(score[:, j], score[train, j], k)
                                  for j in range(score.shape[1])]) for k in LEVELS}
    features = {"curated": score, "active": active.astype(float), "3state": encode(three, 3),
                "supported_active": active[:, is_supported].astype(float),
                "supported_k3": encode(levels[3][:, is_supported], 3)}
    features.update({f"k{k}": encode(levels[k], k) for k in LEVELS})
    features.update({f"k{k}_onehot": encode(levels[k], k, "one-hot") for k in (3, 6)})
    available = [n for n in FEATURE_SETS if features[n].shape[1] > 0]

    norm = prediction.log_normalised(sub)
    detect_b = (sub.X[:, genes_b] > 0).toarray().astype(float)
    knn_p = prediction.ceiling_feature(detect_b, coords, train)
    log_depth = np.log10(total)
    smooth_b, smooth_depth = prediction.neighbourhood_target(norm, detect_b, genes_b, library,
                                                             log_depth)
    libs = pd.get_dummies(library, drop_first=True).to_numpy(dtype=float).reshape(len(library), -1)
    bases = {"detection": np.column_stack([log_depth, libs]),
             "neighbourhood": np.column_stack([smooth_depth, libs])}
    targets = {"detection": detect_b, "neighbourhood": smooth_b}
    tasks = [(target, y[:, j], knn_p[:, j]) for target, y in targets.items()
             for j in range(len(genes_b))]
    with ProcessPoolExecutor(max_workers=prediction.N_JOBS, initializer=prediction.init_worker,
                             initargs=(bases, [features[n] for n in available], train)) as pool:
        losses = list(pool.map(prediction.fit_gene, tasks, chunksize=8))
    per_gene = pd.DataFrame([[l.mean() for l in gene] for gene in losses],
                            columns=["floor_check"] + available + ["knn_check"])
    for n in FEATURE_SETS:
        if n not in per_gene:
            per_gene[n] = np.nan
    n_b = len(genes_b)
    per_gene.insert(0, "stage", stage)
    per_gene.insert(1, "target", np.repeat(prediction.TARGETS, n_b))
    per_gene.insert(2, "gene", np.tile(eval_genes, len(prediction.TARGETS)))
    per_gene.insert(3, "group", np.tile(group, len(prediction.TARGETS)))
    n_features = {n: features[n].shape[1] for n in FEATURE_SETS}
    n_features["n_programs"] = score.shape[1]
    return per_gene, n_features, fidelity(stage, list(curated.index), score, train)


def summarise(per_gene: pd.DataFrame, previous: pd.DataFrame, n_features: dict) -> pd.DataFrame:
    """Explained fractions on the floor and ceiling of script 17."""
    keys = ["stage", "target", "gene", "group"]
    merged = previous.merge(per_gene.drop(columns=["curated"]).rename(
        columns={n: f"lv_{n}" for n in FEATURE_SETS if n != "curated"}), on=keys)
    merged = merged.merge(per_gene[keys + ["curated"]].rename(columns={"curated": "lv_curated"}),
                          on=keys)
    previous_models = [m for m in tests.MODELS if m in previous]
    ours = [f"lv_{n}" for n in FEATURE_SETS]
    rows = []
    for (stage, target, genes), g in merged.groupby(["stage", "target", "group"], sort=False):
        fitted = [m for m in previous_models + ours if g[m].notna().all()]
        ceiling = min(fitted, key=lambda m: g[m].sum())
        gap = g["floor"].sum() - g[ceiling].sum()
        share = gap / g["floor"].sum()
        if share < prediction.MIN_SHARE:
            gap = np.nan
        explained = {m: (g["floor"].sum() - g[m].sum()) / gap if g[m].notna().all() else np.nan
                     for m in fitted}
        for name in FEATURE_SETS + ["curated_supported", "random_curated", "pcs"]:
            m = f"lv_{name}" if name in FEATURE_SETS else name
            rows.append(dict(stage=stage, target=target, genes=genes, feature_set=name,
                             n_eval_genes=len(g), n_features=n_features[stage].get(name, np.nan),
                             n_programs=n_features[stage]["n_programs"],
                             explained=explained.get(m, np.nan),
                             curated_17=explained.get("curated", np.nan),
                             ceiling=ceiling.removeprefix("lv_"), explainable_share=share,
                             floor_check=(g["floor_check"].sum() - g["floor"].sum())
                             / g["floor"].sum()))
    out = pd.DataFrame(rows)
    ref = out.set_index(["stage", "target", "genes", "feature_set"])["explained"]
    out["random"] = [ref.get((r.stage, r.target, r.genes, "random_curated"), np.nan)
                     for r in out.itertuples()]
    out["continuous"] = [ref.get((r.stage, r.target, r.genes, "curated"), np.nan)
                         for r in out.itertuples()]
    out["retained"] = out["explained"] / out["continuous"]
    return out


# --- figure -------------------------------------------------------------------------------

def plot(summary: pd.DataFrame, fid: pd.DataFrame) -> None:
    """(a) explained fraction kept vs levels, per stage; (b) explained per stage;
    (c) score variance kept by the levels (heterogeneous genes, neighbourhood)."""
    part = summary[(summary["target"] == "neighbourhood") & (summary["genes"] == "heterogeneous")]
    stages = [s for s in STAGES[2:] if part.loc[part["stage"] == s, "explained"].notna().any()]
    blue, orange, grey = "#0072B2", "#D55E00", "#7A7A7A"
    fig, axes = plt.subplots(1, 3, figsize=(REPORT_WIDTH_IN, 2.6), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1.4, 1]))
    ax = axes[0]
    kept = part.pivot(index="stage", columns="feature_set", values="retained").reindex(stages)
    cols = [f"k{k}" for k in LEVELS]
    for stage in stages:
        ax.plot(LEVELS, kept.loc[stage, cols], color="#BBBBBB", lw=0.6)
    ax.plot(LEVELS, kept[cols].median(), color=blue, lw=1.5, marker="s", ms=3,
            label="k-means levels")
    ax.plot([2], [kept["active"].median()], ls="", marker="^", ms=4, color=orange,
            label="Boolean (null)")
    ax.plot([3], [kept["3state"].median()], ls="", marker="v", ms=4, color=orange,
            label="3 states (null)")
    ax.axhline(1, color=grey, lw=0.5)
    ax.set_xticks(LEVELS)
    ax.set_xlabel("levels per program")
    ax.set_ylabel("explained, relative to continuous")
    ax.set_ylim(0.7, 1.06)
    ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    ax.set_title("a  kept (grey: stages)", fontsize=7, loc="left")

    ax = axes[1]
    t = part.pivot(index="stage", columns="feature_set", values="explained").reindex(stages)
    x = np.arange(len(stages))
    styles = [("curated", dict(color="black", marker="o", label="continuous")),
              ("k4", dict(color=blue, marker="s", alpha=0.4, label="4 levels")),
              ("k3", dict(color=blue, marker="s", alpha=0.7, label="3 levels")),
              ("k2", dict(color=blue, marker="s", label="2 levels")),
              ("active", dict(color=orange, marker="^", label="Boolean (null)")),
              ("random_curated", dict(color=grey, marker="x", ls=":", label="random, continuous"))]
    for name, style in styles:
        ax.plot(x, t[name], lw=1.0, ms=3, **style)
    ax.axhline(1, color=grey, lw=0.5)
    ax.set_ylim(0.5, 1.03)
    ax.set_xticks(x, stages, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("explained fraction")
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="lower center")
    ax.set_title("b  explained per stage", fontsize=7, loc="left")

    ax = axes[2]
    for k in LEVELS:
        v = fid.loc[(fid["k"] == k) & fid["stage"].isin(stages), "retained"].dropna()
        ax.boxplot(v, positions=[k], widths=0.5, showfliers=False,
                   medianprops=dict(color=blue), boxprops=dict(lw=0.7),
                   whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
    ax.set_xticks(LEVELS, [str(k) for k in LEVELS])
    ax.set_xlabel("levels per program")
    ax.set_ylabel("score variance kept")
    ax.set_ylim(0, 1.02)
    ax.set_title("c  programs x stages", fontsize=7, loc="left")
    save_report_figure(fig, "program_levels")
    plt.close(fig)


def report(summary: pd.DataFrame, fid: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    for genes in ("heterogeneous", "all"):
        t = summary[(summary["target"] == "neighbourhood") & (summary["genes"] == genes)]
        print(genes)
        print(t.pivot(index="stage", columns="feature_set", values="explained")
              .reindex(STAGES).round(3).to_string())
        print(t.pivot(index="stage", columns="feature_set", values="retained")
              .reindex(STAGES).round(3).to_string())
    print(summary.groupby("stage")["floor_check"].apply(lambda v: v.abs().max()))
    print(fid.groupby("k")["retained"].describe().round(2))


def main() -> None:
    ensure_dirs()
    report_style()
    rng = np.random.default_rng(SEED)
    adata = load_dropseq_counts()
    long = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    het = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                      usecols=["stage", "gene", "heterogeneous"])
    support = pd.read_csv(TABLES / "program_support.tsv", sep="\t")
    previous = pd.read_csv(TABLES / "program_tests_prediction_genes.tsv.gz", sep="\t")
    genes, fid, n_features = [], [], {}
    for stage in STAGES:
        stage_het = set(het.loc[(het["stage"] == stage) & het["heterogeneous"], "gene"])
        g, n, f = evaluate_stage(adata, stage, long, support, stage_het, previous, rng)
        genes.append(g)
        fid += f
        n_features[stage] = n
        print(stage, flush=True)
    per_gene = pd.concat(genes)
    per_gene.to_csv(TABLES / "program_levels_genes.tsv.gz", sep="\t", index=False,
                    float_format="%.4g")
    fid = pd.DataFrame(fid)
    fid.to_csv(TABLES / "program_levels_fidelity.tsv", sep="\t", index=False,
               float_format="%.4g")
    summary = summarise(per_gene, previous, n_features)
    summary.to_csv(TABLES / "program_levels.tsv", sep="\t", index=False, float_format="%.4g")
    report(summary, fid)
    plot(summary, fid)


if __name__ == "__main__":
    if sys.argv[1:] == ["summarise"]:
        report_style()
        s = pd.read_csv(TABLES / "program_levels.tsv", sep="\t")
        f = pd.read_csv(TABLES / "program_levels_fidelity.tsv", sep="\t")
        report(s, f)
        plot(s, f)
    else:
        main()
