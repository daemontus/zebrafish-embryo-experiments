"""Do the program sets of script 16 describe the cells?

Scores. A program's score in a cell is the mean Pearson residual of its genes
under the library's own NB model (pipeline_utils.pearson_residuals, centred
per library; Farrell et al. modules weighted). Residuals already adjust for
sequencing depth, so no control genes are needed, and the score means the
same in aligned and per-library data. Centring makes the scores describe
differences between the cells of a library; a program that is on in a whole
library is a constant within it (see 25_program_pseudobulk.py).

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

2. Prediction (on all cells). Per stage, evaluation genes are detected in
   2-98% of the stage's cells and in no program of the reference sets (all
   but `curated` and `controls`): up to N_EVAL heterogeneous genes (script 01b) and N_EVAL
   others, in a fixed random order (see stage_setup). Evaluation genes are
   left out of the curated and control programs in this test, so
   that changing them does not change the evaluation genes. Each gene's
   neighbourhood detection frequency is predicted from the (smoothed) depth
   and library (floor) plus one feature set; the ceiling is the best model,
   including the detection frequency among a cell's nearest training cells
   (knn). Features are program scores smoothed over each cell and its
   neighbours in its library (PCA of the residuals of the stage's
   heterogeneous genes that are not evaluation genes). Random gene sets
   matched to each program (same number of genes, same expression bins, from
   non-evaluation genes) give the random reference. `curated_supported`
   keeps the coherent curated programs only. `zfin_pre_het` (pre-2018
   records) is a sanity check of `zfin_all_het`.

Caching (data/processed/cache/17): the coherence of every set at every
stage, the random-set quantiles per library, the setup of every stage and
the losses of every model are stored with a digest of their inputs, and
every random draw has its own seed. A rerun after changing the curated
programs recomputes only the curated coherence and the curated models
(curated, curated_supported, random_curated, all); an interrupted run
resumes. Increase CACHE_VERSION when a code change changes the results.

Outputs:
  results/tables/program_coherence.tsv      (per set, stage, program, library)
  results/tables/program_support.tsv        (per set, stage, program)
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
import zlib
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
    cached,
    digest,
    ensure_dirs,
    load_cached,
    load_dropseq_counts,
    pearson_residuals,
    save_cached,
    report_style,
    save_report_figure,
)
import program_utils as prediction


SETS = ["farrell", "zfin_all", "zfin_all_het", "zfin_pre_het", "signalling", "states",
        "curated", "controls"]
RANDOM_OF = {"farrell": "random_farrell", "zfin_all": "random_zfin",
             "zfin_all_het": "random_zfin_het", "zfin_pre_het": "random_zfin_het",
             "curated": "random_curated", "curated_supported": "random_curated"}
RANDOM_SOURCE = {"random_farrell": "farrell", "random_zfin": "zfin_all",
                 "random_zfin_het": "zfin_all_het", "random_curated": "curated"}
FEATURE_SETS = SETS + ["curated_supported"] + list(RANDOM_SOURCE) + ["pcs", "all"]
MODELS = FEATURE_SETS + ["knn"]
N_PERM = 100
N_RANDOM = 500
RANDOM_QUANTILE = 0.95
N_EVAL = 400
N_BINS = 50
SEED = 0
# Sets that define the evaluation genes (rarely changed); the curated and
# control sets are not among them, so that changing them does not change the
# evaluation genes (evaluation genes are left out of their programs instead).
HELD_OUT_SETS = ["curated", "controls"]
REFERENCE_SETS = [s for s in SETS if s not in HELD_OUT_SETS]
CACHE = "17"  # data/processed/cache/17
CACHE_VERSION = 1  # increase when a change of the code changes cached results


def seed_of(*parts) -> list[int]:
    """A seed of its own for every unit of work (stage, library, set, ...), so
    that changing one unit does not change the random draws of the others."""
    return [SEED] + [zlib.crc32(str(p).encode()) for p in parts]


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


def random_quantiles(r_lib: np.ndarray, het_cols: np.ndarray, sizes: set, stage: str, lib: str,
                     het_key: str) -> dict:
    """RANDOM_QUANTILE quantile of the coherence of N_RANDOM random sets of
    heterogeneous genes, per set size (cached per library; sizes added as needed)."""
    key = digest(CACHE_VERSION, stage, lib, het_key, N_RANDOM, RANDOM_QUANTILE, SEED)
    name = f"random_q_{stage}_{lib}"
    q = load_cached(CACHE, name, key) or {}
    missing = sorted(n for n in sizes if n not in q)
    for n in missing:
        if 2 <= n <= len(het_cols):
            rng = np.random.default_rng(seed_of("random", stage, lib, n))
            null = [coherence(r_lib[:, rng.choice(het_cols, n, replace=False)], rng,
                              permute=False)[0] for _ in range(N_RANDOM)]
            q[n] = float(np.nanquantile(null, RANDOM_QUANTILE))
        else:
            q[n] = np.nan
    if missing:
        save_cached(CACHE, name, key, q)
    return q


def coherence_table(adata, long: pd.DataFrame, het: pd.DataFrame) -> pd.DataFrame:
    """Coherence of every program in every library; cached per stage and set,
    so that only changed sets are recomputed."""
    rows = []
    for stage in STAGES:
        use = long[(long["stage"] == stage) & long["kept"]]
        stage_het = sorted(set(het.loc[(het["stage"] == stage) & het["heterogeneous"], "gene"]))
        het_key = digest(stage_het)
        units = {}
        for set_name in SETS:
            progs = [(p, list(g["gene"])) for p, g in
                     use[use["set"] == set_name].groupby("program", sort=False)]
            key = digest(CACHE_VERSION, stage, set_name, progs, het_key, N_PERM, N_RANDOM,
                         RANDOM_QUANTILE, SEED)
            units[set_name] = (key, progs, load_cached(CACHE, f"coherence_{stage}_{set_name}", key))
        missing = [s for s, (_, _, v) in units.items() if v is None]
        if missing:
            sub = adata[adata.obs["stage"] == stage]
            library = sub.obs["library"].astype(str).to_numpy()
            genes = sorted({g for s in missing for _, gs in units[s][1] for g in gs}
                           | set(stage_het))
            res = library_residuals(sub, library, sub.var_names.get_indexer(genes))
            col = {g: i for i, g in enumerate(genes)}
            het_cols = np.array([col[g] for g in stage_het])
            sizes = {len(gs) for s in missing for _, gs in units[s][1]}
            for lib in np.unique(library):
                r_lib = res[library == lib]
                q = random_quantiles(r_lib, het_cols, sizes, stage, lib, het_key)
                for s in missing:
                    out = units[s][2] if units[s][2] is not None else []
                    for program, gs in units[s][1]:
                        rng = np.random.default_rng(seed_of("perm", stage, lib, s, program))
                        obs, p = coherence(r_lib[:, [col[g] for g in gs]], rng)
                        n = len(gs)
                        out.append(dict(set=s, stage=stage, program=program, library=lib,
                                        genes=n, coherence=obs, p_perm=p, random_q95=q[n],
                                        coherent=bool(obs > q[n]) if np.isfinite(obs) else False))
                    units[s] = (units[s][0], units[s][1], out)
            for s in missing:
                save_cached(CACHE, f"coherence_{stage}_{s}", units[s][0], units[s][2])
        for s in SETS:
            rows += units[s][2]
        print("coherence", stage, f"(computed: {', '.join(missing) or 'none'})", flush=True)
    return pd.DataFrame(rows)


def support_table(coh: pd.DataFrame) -> pd.DataFrame:
    t = (coh.groupby(["set", "stage", "program"], sort=False)
         .agg(genes=("genes", "first"), libraries=("library", "size"),
              coherent_libraries=("coherent", "sum"),
              median_coherence=("coherence", "median")).reset_index())
    t["supported"] = (t["coherent_libraries"] >= np.ceil(t["libraries"] / 2)) & (t["genes"] >= 2)
    return t


# --- 2. prediction ------------------------------------------------------------------------

def stage_setup(adata, stage: str, long: pd.DataFrame, het_genes: set) -> dict:
    """Everything of the prediction test of a stage that does not depend on the
    program sets being tested: evaluation genes, train/test split, smoothing
    coordinates, target, floor base, knn ceiling feature and the gene bins of
    the random reference (cached).

    Evaluation genes are detected in 2-98% of the stage's cells and in no
    program of the reference sets (all sets but `curated`) at the stage; they
    are taken in a fixed random order (seeded per stage): the first N_EVAL
    heterogeneous ones, then N_EVAL of the others. The curated programs are
    not excluded, so that changing them keeps the evaluation genes; instead,
    evaluation genes are left out of the curated programs in this test.
    """
    sub = adata[adata.obs["stage"] == stage]
    detected = pd.Series(np.asarray((sub.X > 0).mean(axis=0)).ravel(), index=sub.var_names)
    excluded = set(long.loc[(long["stage"] == stage) & long["set"].isin(REFERENCE_SETS), "gene"])
    eligible = sorted(g for g in detected.index[(detected >= prediction.MIN_DETECTED)
                                                & (detected <= prediction.MAX_DETECTED)]
                      if g not in excluded)
    order = list(np.random.default_rng(seed_of("eval", stage)).permutation(eligible))
    eval_het = [g for g in order if g in het_genes][:N_EVAL]
    chosen = set(eval_het)
    eval_all = [g for g in order if g not in chosen][:N_EVAL]
    key = digest(CACHE_VERSION, stage, eval_all, eval_het, sorted(het_genes), N_BINS,
                 prediction.N_PCS, prediction.K_SMOOTH, prediction.K_CEILING,
                 prediction.K_TARGET, prediction.TEST_FRACTION, SEED)

    def compute() -> dict:
        s = sub.copy()
        library = s.obs["library"].astype(str).to_numpy()
        eval_genes = eval_all + eval_het
        group = np.array(["all"] * len(eval_all) + ["heterogeneous"] * len(eval_het))
        genes_b = s.var_names.get_indexer(eval_genes)
        train = np.zeros(s.n_obs, dtype=bool)
        tr, _ = train_test_split(np.arange(s.n_obs), test_size=prediction.TEST_FRACTION,
                                 stratify=library, random_state=SEED)
        train[tr] = True
        in_train = np.asarray((s.X[train][:, genes_b] > 0).mean(axis=0)).ravel()
        keep = (in_train > 0) & (in_train < 1)
        genes_b, group = genes_b[keep], group[keep]
        eval_genes = list(np.array(eval_genes)[keep])
        coord_genes = s.var_names.get_indexer(sorted(het_genes - set(eval_genes)))
        coords = PCA(prediction.N_PCS, random_state=SEED).fit_transform(
            library_residuals(s, library, coord_genes))
        norm = prediction.log_normalised(s)
        detect_b = (s.X[:, genes_b] > 0).toarray().astype(float)
        log_depth = np.log10(np.asarray(s.X.sum(axis=1)).ravel())
        target, smooth_depth = prediction.neighbourhood_target(norm, detect_b, genes_b, library,
                                                               log_depth)
        libs = pd.get_dummies(library, drop_first=True).to_numpy(dtype=float) \
            .reshape(len(library), -1)
        mean = pd.Series(np.asarray(s.X.mean(axis=0)).ravel(), index=s.var_names)
        measured = mean[mean > 0]
        bins = pd.Series(pd.qcut(measured.rank(method="first"), N_BINS, labels=False),
                         index=measured.index)
        return dict(eval_genes=eval_genes, group=group, library=library, train=train,
                    coords=coords.astype(np.float32), target=target.astype(np.float32),
                    knn_p=prediction.ceiling_feature(detect_b, coords, train).astype(np.float32),
                    base=np.column_stack([smooth_depth, libs]), bins=bins)

    out = cached(CACHE, f"setup_{stage}", key, compute)
    return dict(out, key=key, stage=stage)


def definitions(long: pd.DataFrame, support: pd.DataFrame, setup: dict) -> dict:
    """Programs of every feature set at the stage (DataFrame: genes, weights per
    program), with the evaluation genes left out of the curated programs."""
    stage = setup["stage"]
    sets = stage_programs(long, stage)
    held_out = set(setup["eval_genes"])
    for name in HELD_OUT_SETS:
        t = sets[name].copy()
        t["weights"] = [[w for g, w in zip(gs, ws) if g not in held_out]
                        for gs, ws in zip(t["genes"], t["weights"])]
        t["genes"] = [[g for g in gs if g not in held_out] for gs in t["genes"]]
        sets[name] = t[t["genes"].map(len) > 0]
    supported = set(support.loc[(support["stage"] == stage) & (support["set"] == "curated")
                                & support["supported"], "program"])
    sets["curated_supported"] = sets["curated"][sets["curated"].index.isin(supported)]
    # random reference: genes of the same expression bins, never evaluation genes
    bins = setup["bins"]
    pool = bins.drop([g for g in held_out if g in bins.index])
    for name, source in RANDOM_SOURCE.items():
        t = sets[source]
        genes = []
        for program, gs in zip(t.index, t["genes"]):
            rng = np.random.default_rng(seed_of("random_set", stage, name, program))
            genes.append([matched_random_gene(g, bins, pool, rng) for g in gs])
        sets[name] = pd.DataFrame(dict(genes=genes, weights=[[1.0] * len(g) for g in genes]),
                                  index=t.index)
    return sets


def matched_random_gene(gene: str, bins: pd.Series, pool: pd.Series, rng) -> str:
    """A random gene of the same expression bin (from the pool)."""
    candidates = pool.index[pool == bins[gene]]
    return rng.choice(candidates) if len(candidates) else rng.choice(pool.index)


def definition_key(t: pd.DataFrame) -> list:
    return [(p, list(g), [float(w) if pd.notna(w) else None for w in ws])
            for p, g, ws in zip(t.index, t["genes"], t["weights"])]


def fit_losses(adata, setup: dict, specs: dict, group: str = CACHE) -> dict[str, dict]:
    """Mean test loss per evaluation gene of every model in `specs` (name ->
    (definition key, features() or None for floor/knn)); cached per model, and
    only missing models are fitted."""
    stage = setup["stage"]
    keys = {name: digest(CACHE_VERSION, setup["key"], name, d) for name, (d, _) in specs.items()}
    out = {name: load_cached(group, f"loss_{stage}_{name}", keys[name]) for name in specs}
    missing = [name for name, v in out.items() if v is None]
    if missing:
        features = {}
        for name in missing:
            make = specs[name][1]
            if make is not None:
                features[name] = make().astype(float)
        tasks = [(j, missing) for j in range(len(setup["eval_genes"]))]
        with ProcessPoolExecutor(max_workers=prediction.N_JOBS,
                                 initializer=prediction.init_worker,
                                 initargs=(setup["base"], setup["target"], setup["knn_p"],
                                           features, setup["train"])) as pool:
            losses = np.array(list(pool.map(prediction.fit_models, tasks, chunksize=4)))
        for k, name in enumerate(missing):
            out[name] = dict(loss=losses[:, k],
                             n_features=features[name].shape[1] if name in features else 0)
            save_cached(group, f"loss_{stage}_{name}", keys[name], out[name])
    print(f"  {stage}: fitted {', '.join(missing) or 'nothing (all cached)'}", flush=True)
    return out


def set_features(adata, setup: dict, sets: dict):
    """features(name) for the program sets of the stage: smoothed scores (the
    residuals are computed once, on first use)."""
    state = {}

    def residuals():
        if "res" not in state:
            sub = adata[adata.obs["stage"] == setup["stage"]]
            used = sorted({g for t in sets.values() for gs in t["genes"] for g in gs})
            state["res"] = library_residuals(sub, setup["library"],
                                             sub.var_names.get_indexer(used))
            state["col"] = {g: i for i, g in enumerate(used)}
        return state["res"], state["col"]

    def features(name: str) -> np.ndarray:
        if name == "pcs":
            return prediction.smooth(setup["coords"], setup["library"], setup["coords"])
        if name == "all":
            parts = [features(n) for n in ("pcs", "curated", "farrell")]
            return np.hstack([p for p in parts if p.shape[1]])
        res, col = residuals()
        s = scores(res, col, sets[name], weighted=name == "farrell")
        return prediction.smooth(s, setup["library"], setup["coords"]) if s.shape[1] else s
    return features


def evaluate_stage(adata, stage: str, long: pd.DataFrame, support: pd.DataFrame,
                   het_genes: set) -> tuple[list, pd.DataFrame]:
    setup = stage_setup(adata, stage, long, het_genes)
    sets = definitions(long, support, setup)
    features = set_features(adata, setup, sets)
    specs = {"floor": ([], None), "knn": ([], None), "pcs": ([], lambda: features("pcs"))}
    for name in SETS + ["curated_supported"] + list(RANDOM_SOURCE):
        if len(sets[name]):
            specs[name] = (definition_key(sets[name]), lambda n=name: features(n))
    specs["all"] = ([definition_key(sets[n]) for n in ("curated", "farrell")],
                    lambda: features("all"))
    fitted = fit_losses(adata, setup, specs)
    per_gene = pd.DataFrame({name: v["loss"] for name, v in fitted.items()})
    for n in MODELS:
        if n not in per_gene:
            per_gene[n] = np.nan
    per_gene.insert(0, "stage", stage)
    per_gene.insert(1, "target", prediction.TARGETS[0])
    per_gene.insert(2, "gene", setup["eval_genes"])
    per_gene.insert(3, "group", setup["group"])
    n_programs = {n: (v["n_features"] if n not in ("floor", "knn") else 0)
                  for n, v in fitted.items()}
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
    "curated": dict(color="#009E73", marker="s", ls="-", label="curated"),
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
    for name in ["farrell", "zfin_all", "zfin_all_het", "curated"]:
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
    adata = load_dropseq_counts()
    long = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    het = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                      usecols=["stage", "gene", "heterogeneous"])
    coh = coherence_table(adata, long, het)
    coh.to_csv(TABLES / "program_coherence.tsv", sep="\t", index=False, float_format="%.4g")
    support = support_table(coh)
    support.to_csv(TABLES / "program_support.tsv", sep="\t", index=False, float_format="%.4g")
    print(support.groupby(["set"])["supported"].mean().round(2))
    rows, genes = [], []
    for stage in STAGES:
        stage_het = set(het.loc[(het["stage"] == stage) & het["heterogeneous"], "gene"])
        r, g = evaluate_stage(adata, stage, long, support, stage_het)
        rows += r
        genes.append(g)
        e = pd.DataFrame(r)
        e = e[(e["target"] == "neighbourhood")].set_index(["genes", "feature_set"])["explained"]
        print(stage, "  ".join(f"{k[0][:3]}:{k[1]} {v:.2f}" for k, v in e.items()
                               if k[1] in ("curated", "zfin_all_het", "farrell", "zfin_all",
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
