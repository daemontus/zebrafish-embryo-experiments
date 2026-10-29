"""Do the program sets of script 16 describe the cells?

Scores. A program's score in a cell is the mean Pearson residual of its genes
under the library's own NB model (pipeline_utils.pearson_residuals, centred
per library; Farrell et al. modules weighted). Residuals already adjust for
sequencing depth, so no control genes are needed, and the score means the
same in aligned and per-library data. Centring makes the scores describe
differences between the cells of a library; a program that is on in a whole
library is a constant within it (library level, see below).

1. Coherence (are the programs real in the data?). In every library, the
   co-expression of a program's genes is
       R = n Var(score) / mean_g Var(residual_g),
   i.e. 1 + (n - 1) x the mean correlation between its genes (1 if the genes
   vary independently). Two nulls: (i) every gene's residuals shuffled
   independently across the cells (N_PERM times; p_perm), which nearly
   every set of heterogeneous genes passes, since they all vary along the
   main axes of the stage; (ii) N_RANDOM random sets of the same size drawn
   from the stage's heterogeneous genes. A program is coherent in a library
   if R exceeds the RANDOM_QUANTILE quantile of (ii), i.e. its genes vary
   together more than heterogeneous genes in general, and supported at a
   stage if coherent in at least half of the stage's libraries. Programs of
   a single gene are not tested.

2. Library level. For the state programs (maternal, blastomere, ...): the
   pseudobulk expression of the genes (UMIs per 10,000, median over genes)
   in each library relative to the gene's maximum over the atlas.

3. Prediction (as scripts 06 and 13, on all cells). Per stage, evaluation
   genes are drawn from genes not in any program of any set at the stage:
   N_EVAL "all" genes (detected in 2-98% of the stage's cells) and up to
   N_EVAL "heterogeneous" genes (script 01b). Each gene's detection
   (single cell) and neighbourhood detection frequency (smoothed) are
   predicted from the depth and library (floor) plus one feature set; the
   ceiling is the best model, including the detection frequency among a
   cell's nearest training cells (knn). Features are program scores
   smoothed over each cell and its neighbours in its library (PCA of the
   residuals of the stage's heterogeneous genes that are not evaluation
   genes). Random gene sets matched to each program (same number of genes,
   same expression bins, from non-evaluation genes) give the random
   reference. `curated_supported` keeps the coherent curated programs only.
   `curated` has one gene list per program for all stages; `curated_staged`
   is the same draft with stage-specific genes (script 16).

Outputs:
  results/tables/program_coherence.tsv      (per set, stage, program, library)
  results/tables/program_support.tsv        (per set, stage, program)
  results/tables/program_library_levels.tsv (state programs per library)
  results/tables/program_tests_prediction.tsv, ..._genes.tsv.gz
  report/figures/program_tests.pdf

`17_program_tests.py summarise` redraws the figure from the tables.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from pipeline_utils import (
    REPORT_WIDTH_IN,
    STAGES,
    TABLES,
    ensure_dirs,
    load_dropseq_counts,
    pearson_residuals,
    report_style,
    save_report_figure,
)
import program_utils as prediction


SETS = ["farrell", "zfin_pre", "zfin_all", "zfin_pre_het", "zfin_all_het", "signalling",
        "states", "curated_staged", "curated"]
RANDOM_OF = {"farrell": "random_farrell", "zfin_pre": "random_zfin", "zfin_all": "random_zfin",
             "zfin_pre_het": "random_zfin_het", "zfin_all_het": "random_zfin_het",
             "curated_staged": "random_curated_staged", "curated": "random_curated",
             "curated_supported": "random_curated"}
RANDOM_SOURCE = {"random_farrell": "farrell", "random_zfin": "zfin_all",
                 "random_zfin_het": "zfin_all_het", "random_curated_staged": "curated_staged",
                 "random_curated": "curated"}
FEATURE_SETS = SETS + ["curated_supported"] + list(RANDOM_SOURCE) + ["pcs", "all"]
MODELS = FEATURE_SETS + ["knn"]
N_PERM = 100
N_RANDOM = 500
RANDOM_QUANTILE = 0.95
N_EVAL = 400
N_BINS = 50
EVAL_GROUPS = ["all", "heterogeneous"]
SEED = 0


# --- scores ---------------------------------------------------------------------------

def library_residuals(adata, library: np.ndarray, genes: np.ndarray) -> np.ndarray:
    """Centred residuals of `genes` (index array) for every cell, per library."""
    X = adata.X.tocsr()
    out = np.zeros((adata.n_obs, len(genes)), dtype=np.float32)
    for lib in np.unique(library):
        rows = np.flatnonzero(library == lib)
        out[rows] = pearson_residuals(X[rows], genes)
    return out


def scores(res: np.ndarray, col: dict, programs: pd.DataFrame, weighted: bool) -> np.ndarray:
    """Mean (weighted) residual of every program's genes (cells x programs)."""
    out = []
    for genes, weights in zip(programs["genes"], programs["weights"]):
        idx = [col[g] for g in genes]
        w = np.asarray(weights, dtype=float) if weighted else np.ones(len(idx))
        w = np.where(np.isfinite(w), w, 1.0)
        out.append(res[:, idx] @ (w / w.sum()))
    return np.column_stack(out) if out else np.zeros((res.shape[0], 0))


def stage_programs(long: pd.DataFrame, stage: str) -> dict[str, pd.DataFrame]:
    use = long[(long["stage"] == stage) & long["kept"]]
    return {s: use[use["set"] == s].groupby("program", sort=False)
            .agg(genes=("gene", list), weights=("weight", list)) for s in SETS}


# --- 1. coherence -----------------------------------------------------------------------

def coherence(r: np.ndarray, rng, permute: bool = True) -> tuple[float, float]:
    """R = n Var(mean) / mean Var for one program in one library, and the
    p-value of the gene-wise permutation null."""
    n = r.shape[1]
    var = r.var(axis=0).mean()
    if n < 2 or var == 0:
        return np.nan, np.nan
    obs = n * r.mean(axis=1).var() / var
    if not permute:
        return obs, np.nan
    null = np.array([n * rng.permuted(r, axis=0).mean(axis=1).var() / var
                     for _ in range(N_PERM)])
    return obs, (1 + (null >= obs).sum()) / (N_PERM + 1)


def coherence_table(adata, long: pd.DataFrame, het: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for stage in STAGES:
        sub = adata[adata.obs["stage"] == stage]
        library = sub.obs["library"].astype(str).to_numpy()
        use = long[(long["stage"] == stage) & long["kept"]]
        stage_het = set(het.loc[(het["stage"] == stage) & het["heterogeneous"], "gene"])
        genes = sorted(set(use["gene"]) | stage_het)
        idx = sub.var_names.get_indexer(genes)
        res = library_residuals(sub, library, idx)
        col = {g: i for i, g in enumerate(genes)}
        het_cols = np.array([col[g] for g in sorted(stage_het)])
        for lib in np.unique(library):
            r_lib = res[library == lib]
            null_q = {}
            for (set_name, program), g in use.groupby(["set", "program"], sort=False):
                cols = [col[x] for x in g["gene"]]
                n = len(cols)
                if n not in null_q:
                    null = [coherence(r_lib[:, rng.choice(het_cols, n, replace=False)], rng,
                                      permute=False)[0] for _ in range(N_RANDOM)] \
                        if 2 <= n <= len(het_cols) else [np.nan]
                    null_q[n] = np.nanquantile(null, RANDOM_QUANTILE) if n >= 2 else np.nan
                obs, p = coherence(r_lib[:, cols], rng)
                rows.append(dict(set=set_name, stage=stage, program=program, library=lib,
                                 genes=n, coherence=obs, p_perm=p, random_q95=null_q[n],
                                 coherent=bool(obs > null_q[n]) if np.isfinite(obs) else False))
        print("coherence", stage, flush=True)
    return pd.DataFrame(rows)


def support_table(coh: pd.DataFrame) -> pd.DataFrame:
    t = (coh.groupby(["set", "stage", "program"], sort=False)
         .agg(genes=("genes", "first"), libraries=("library", "size"),
              coherent_libraries=("coherent", "sum"),
              median_coherence=("coherence", "median")).reset_index())
    t["supported"] = (t["coherent_libraries"] >= np.ceil(t["libraries"] / 2)) & (t["genes"] >= 2)
    return t


# --- 2. library level --------------------------------------------------------------------

def library_levels(adata, long: pd.DataFrame) -> pd.DataFrame:
    states = long[long["set"] == "states"].drop_duplicates(["program", "gene"])
    genes = sorted(set(states["gene"]))
    idx = adata.var_names.get_indexer(genes)
    library = adata.obs["library"].astype(str).to_numpy()
    X = adata.X.tocsr()
    depth = np.asarray(X.sum(axis=1)).ravel()
    bulk = {}
    for lib in np.unique(library):
        rows = library == lib
        bulk[lib] = np.asarray(X[rows][:, idx].sum(axis=0)).ravel() / depth[rows].sum() * 1e4
    bulk = pd.DataFrame(bulk, index=genes)
    rel = bulk.div(bulk.max(axis=1), axis=0)
    stage_of = adata.obs.groupby("library", observed=True)["stage"].first().astype(str)
    out = []
    for program, g in states.groupby("program"):
        level = rel.loc[g["gene"]].median()
        for lib, v in level.items():
            out.append(dict(program=program, library=lib, stage=stage_of[lib], level=v))
    return pd.DataFrame(out)


# --- 3. prediction ------------------------------------------------------------------------

def matched_random(genes: list[str], bins: pd.Series, pool: pd.Series, rng) -> list[str]:
    """Random genes from the same expression bins (from the pool)."""
    out = []
    for g in genes:
        candidates = pool.index[pool == bins[g]]
        out.append(rng.choice(candidates) if len(candidates) else rng.choice(pool.index))
    return out


def evaluate_stage(adata, stage: str, long: pd.DataFrame, support: pd.DataFrame,
                   het_genes: set, rng) -> tuple[list, pd.DataFrame]:
    sub = adata[adata.obs["stage"] == stage].copy()
    library = sub.obs["library"].astype(str).to_numpy()
    total = np.asarray(sub.X.sum(axis=1)).ravel()
    detected = pd.Series(np.asarray((sub.X > 0).mean(axis=0)).ravel(), index=sub.var_names)
    mean = pd.Series(np.asarray(sub.X.mean(axis=0)).ravel(), index=sub.var_names)
    sets = stage_programs(long, stage)
    supported = set(support.loc[(support["stage"] == stage) & (support["set"] == "curated")
                                & support["supported"], "program"])
    sets["curated_supported"] = sets["curated"][sets["curated"].index.isin(supported)]
    program_genes = set(long.loc[long["stage"] == stage, "gene"])

    eligible = detected[(detected >= prediction.MIN_DETECTED)
                        & (detected <= prediction.MAX_DETECTED)].index
    eligible = [g for g in eligible if g not in program_genes]
    het_pool = [g for g in eligible if g in het_genes]
    eval_het = list(rng.permutation(het_pool)[:N_EVAL])
    rest = [g for g in eligible if g not in set(eval_het)]
    eval_all = list(rng.permutation(rest)[:N_EVAL])
    eval_genes = eval_all + eval_het
    group = np.array(["all"] * len(eval_all) + ["heterogeneous"] * len(eval_het))
    genes_b = sub.var_names.get_indexer(eval_genes)

    train = np.zeros(sub.n_obs, dtype=bool)
    tr, _ = train_test_split(np.arange(sub.n_obs), test_size=prediction.TEST_FRACTION,
                             stratify=library, random_state=SEED)
    train[tr] = True
    in_train = np.asarray((sub.X[train][:, genes_b] > 0).mean(axis=0)).ravel()
    keep = (in_train > 0) & (in_train < 1)
    genes_b, group, eval_genes = genes_b[keep], group[keep], list(np.array(eval_genes)[keep])

    # Random sets (matched in expression bins, from genes that are not evaluation genes).
    measured = mean[mean > 0]
    bins = pd.Series(pd.qcut(measured.rank(method="first"), N_BINS, labels=False),
                     index=measured.index)
    pool = bins.drop([g for g in eval_genes if g in bins.index])
    for name, source in RANDOM_SOURCE.items():
        t = sets[source]
        sets[name] = pd.DataFrame(dict(
            genes=[matched_random(g, bins, pool, rng) for g in t["genes"]],
            weights=[[1.0] * len(g) for g in t["genes"]]), index=t.index)

    # Coordinates for smoothing: heterogeneous genes that are not evaluation genes.
    coord_genes = sub.var_names.get_indexer(sorted(het_genes - set(eval_genes)))
    coords = PCA(prediction.N_PCS, random_state=SEED).fit_transform(
        library_residuals(sub, library, coord_genes))
    used = sorted({g for t in sets.values() for genes in t["genes"] for g in genes})
    res = library_residuals(sub, library, sub.var_names.get_indexer(used))
    col = {g: i for i, g in enumerate(used)}
    features = {}
    for name in FEATURE_SETS:
        if name in ("pcs", "all"):
            continue
        s = scores(res, col, sets[name], weighted=name == "farrell")
        features[name] = prediction.smooth(s, library, coords) if s.shape[1] else None
    features["pcs"] = prediction.smooth(coords, library, coords)
    features["all"] = np.hstack([f for f in (features["pcs"], features["curated"],
                                             features["farrell"]) if f is not None])
    available = [n for n in FEATURE_SETS if features[n] is not None]

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
                             initargs=(bases, [features[n].astype(float) for n in available],
                                       train)) as pool_:
        losses = list(pool_.map(prediction.fit_gene, tasks, chunksize=8))
    models = available + ["knn"]
    per_gene = pd.DataFrame([[l.mean() for l in gene] for gene in losses],
                            columns=["floor"] + models)
    for n in MODELS:
        if n not in per_gene:
            per_gene[n] = np.nan
    n_b = len(genes_b)
    per_gene.insert(0, "stage", stage)
    per_gene.insert(1, "target", np.repeat(prediction.TARGETS, n_b))
    per_gene.insert(2, "gene", np.tile(eval_genes, len(prediction.TARGETS)))
    per_gene.insert(3, "group", np.tile(group, len(prediction.TARGETS)))
    n_programs = {n: (features[n].shape[1] if features[n] is not None else 0)
                  for n in FEATURE_SETS}
    return summarise(per_gene, n_programs), per_gene


def summarise(per_gene: pd.DataFrame, n_programs: dict) -> list[dict]:
    rows = []
    for (stage, target, group), g in per_gene.groupby(["stage", "target", "group"], sort=False):
        fitted = [m for m in MODELS if g[m].notna().all()]
        ceiling = min(fitted, key=lambda n: g[n].sum())
        gap = g["floor"].sum() - g[ceiling].sum()
        share = gap / g["floor"].sum()
        if share < prediction.MIN_SHARE:
            gap = np.nan
        for name in MODELS:
            loss = g[name].sum() if g[name].notna().all() else np.nan
            rows.append(dict(stage=stage, target=target, genes=group, feature_set=name,
                             n_eval_genes=len(g), n_programs=n_programs.get(name, np.nan),
                             explained=(g["floor"].sum() - loss) / gap,
                             ceiling=ceiling, explainable_share=share))
    out = pd.DataFrame(rows)
    explained = out.set_index(["stage", "target", "genes", "feature_set"])["explained"]
    out["random"] = [explained.get((r.stage, r.target, r.genes, RANDOM_OF.get(r.feature_set)),
                                   np.nan) for r in out.itertuples()]
    out["beyond_random"] = (out["explained"] - out["random"]) / (1 - out["random"])
    return out.to_dict("records")


# --- figure ---------------------------------------------------------------------------------

STYLES = {
    "farrell": dict(color="#D55E00", marker="^", ls="-", label="Farrell et al."),
    "zfin_all": dict(color="#0072B2", marker="o", ls="-", label="ZFIN terms"),
    "zfin_all_het": dict(color="#0072B2", marker="o", ls="--", mfc="white",
                         label="ZFIN terms, het. genes"),
    "curated_staged": dict(color="#009E73", marker="s", ls="--", mfc="white",
                           label="curated, stage-specific genes"),
    "curated": dict(color="#009E73", marker="s", ls="-", label="curated, fixed genes"),
    "random_curated": dict(color="#7A7A7A", marker="x", ls=":", label="random (curated)"),
}


def plot(summary: pd.DataFrame, support: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(REPORT_WIDTH_IN, 2.6), constrained_layout=True)
    x = np.arange(len(STAGES))
    for ax, (target, genes) in zip(axes[:2], [("neighbourhood", "all"),
                                               ("neighbourhood", "heterogeneous")]):
        part = summary[(summary["target"] == target) & (summary["genes"] == genes)]
        for name, style in STYLES.items():
            t = part[part["feature_set"] == name].set_index("stage").reindex(STAGES)
            ax.plot(x, t["explained"].clip(lower=prediction.Y_MIN), lw=1.0, ms=3, **style)
        ax.axhline(0, color="#7A7A7A", lw=0.5)
        ax.axhline(1, color="#7A7A7A", lw=0.5)
        ax.set_ylim(prediction.Y_MIN, 1.08)
        ax.set_xticks(x, STAGES, rotation=45, ha="right", fontsize=6)
        ax.set_title(f"{'a' if genes == 'all' else 'b'}  evaluation genes: {genes}", fontsize=7,
                     loc="left")
    axes[0].set_ylabel("explained fraction (neighbourhood)")
    axes[0].legend(frameon=False, fontsize=5.5)
    ax = axes[2]
    share = (support.groupby(["set", "stage"])["supported"].mean().unstack()
             .reindex(columns=STAGES))
    for name in ["farrell", "zfin_all", "zfin_all_het", "curated_staged", "curated"]:
        style = {k: v for k, v in STYLES[name].items()}
        ax.plot(x, share.loc[name], lw=1.0, ms=3, **style)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, STAGES, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("share of programs coherent\nin >= half of the libraries")
    ax.set_title("c  supported programs", fontsize=7, loc="left")
    save_report_figure(fig, "program_tests")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    report_style()
    rng = np.random.default_rng(SEED)
    adata = load_dropseq_counts()
    long = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    het = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                      usecols=["stage", "gene", "heterogeneous"])
    coh = coherence_table(adata, long, het, rng)
    coh.to_csv(TABLES / "program_coherence.tsv", sep="\t", index=False, float_format="%.4g")
    support = support_table(coh)
    support.to_csv(TABLES / "program_support.tsv", sep="\t", index=False, float_format="%.4g")
    print(support.groupby(["set"])["supported"].mean().round(2))
    library_levels(adata, long).to_csv(TABLES / "program_library_levels.tsv", sep="\t",
                                       index=False, float_format="%.4g")
    rows, genes = [], []
    for stage in STAGES:
        stage_het = set(het.loc[(het["stage"] == stage) & het["heterogeneous"], "gene"])
        r, g = evaluate_stage(adata, stage, long, support, stage_het, rng)
        rows += r
        genes.append(g)
        e = pd.DataFrame(r)
        e = e[(e["target"] == "neighbourhood")].set_index(["genes", "feature_set"])["explained"]
        print(stage, "  ".join(f"{k[0][:3]}:{k[1]} {v:.2f}" for k, v in e.items()
                               if k[1] in ("curated", "curated_staged", "farrell", "zfin_all",
                                           "random_curated")),
              flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "program_tests_prediction.tsv", sep="\t", index=False,
                   float_format="%.4g")
    pd.concat(genes).to_csv(TABLES / "program_tests_prediction_genes.tsv.gz", sep="\t",
                            index=False, float_format="%.4g")
    plot(summary, support)


if __name__ == "__main__":
    if sys.argv[1:] == ["summarise"]:
        report_style()
        plot(pd.read_csv(TABLES / "program_tests_prediction.tsv", sep="\t"),
             pd.read_csv(TABLES / "program_support.tsv", sep="\t"))
    else:
        main()
