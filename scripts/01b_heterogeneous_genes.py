"""Which genes vary between the cells of a library more than expected?

Every analysis that needs "interesting" genes (embeddings, evaluation
targets, gene filters) uses the genes selected here, instead of the usual
highly variable genes (HVGs).

Model (pipeline_utils.nb_overdispersion). In a library, cell i has relative
depth s_i (UMIs / library mean). A gene with the same expression in every
cell has UMI counts with mean s_i m and, under a negative binomial (gamma-
Poisson), variance s_i m + phi (s_i m)^2: Poisson sampling plus
cell-to-cell variation of the true expression (bursting, cell size), with
squared coefficient of variation phi. phi is estimated by the method of
moments; the typical phi of a mean is a fitted dispersion-mean trend
a0 + a1 m^-b (bin medians of genes with >= 100 UMIs, extrapolated to rarer
genes); `residual` is log10(phi) minus log10(typical phi). Genes with fewer
than MIN_UMIS UMIs in the library get no estimate.

The estimate of a gene with few UMIs is noisy (at 10 UMIs, most estimates
are <= 0), so a fixed cutoff calls many homogeneous low-expression genes.
The cutoff is therefore calibrated per library: the library is simulated
with every gene homogeneous (negative binomial with the typical phi of its
mean, the library's real depths and means; N_SIM times), and the running
NULL_QUANTILE quantile over genes of similar mean of the simulated residual
(simulated log10(phi) minus the typical phi it was simulated with) is the
chance level. The residual removes the trend of phi with the mean before
the quantile is taken: at the upper end, where means are sparse, a window
of simulated log10(phi) would include rarer genes with a higher typical phi
and overstate the chance level. A further, independent simulation estimates
the false positives that remain. Each library has its own random generator
(SEED and the library name), so the results are reproducible but depend on
SEED (a few percent of the calls change between seeds).

The simulation gives every homogeneous gene exactly the typical phi, but
real genes differ in their intrinsic noise even when uniform across cells:
among well-expressed genes, the residuals of the bulk are 3-4x wider than
sampling explains. Beyond chance is therefore not enough. The effect-size
floor of each tier comes from an empirical null: the residuals of the
well-expressed genes (>= WELL_UMIS UMIs, or the BULK_GENES most expressed)
are modelled as normal (median, MAD-based SD), the expected number of
well-expressed calls under this null divided by the observed number is the
estimated FDR, and the floor of a tier is the smallest one with an FDR of
at most TIERS[tier] (made monotone from above).

Heterogeneous genes:
  per library   residual >= max(floor of the tier, chance level); tiers
                weak / moderate / strong (FDR 50% / 10% / 1% among
                well-expressed genes); "heterogeneous" means weak;
  per stage     in the tier in at least half of the stage's libraries.
`01b_heterogeneous_genes.py plot` redraws the figures from the cache of the
last run (data/processed/heterogeneity_models.pkl).
`heterogeneity_calls(counts, rng)` applies the same calibration to any
library or part of one.

Checks:
  null          false positives in an independent homogeneous simulation.
  replication   for libraries of the same stage, the overlap (Jaccard) of
                the TOP_N genes ranked by residual vs. by Seurat's normalised
                dispersion (scanpy flavour "seurat", per library), and of the
                weak-tier genes vs. as many top Seurat genes.
  HVGs          per stage, the 2,000 Seurat HVGs of the pooled libraries (as
                used by scripts 04-13) against the heterogeneous genes.

Outputs:
  results/tables/heterogeneous_genes_libraries.tsv.gz  (gene x library residual)
  results/tables/heterogeneous_genes_calls.tsv.gz      (gene x library: 0 no,
                                                        1 weak, 2 moderate, 3 strong)
  results/tables/heterogeneous_genes.tsv.gz            (per stage and gene:
                                                        median residual, calls)
  results/tables/heterogeneity_libraries.tsv           (per library: counts,
                                                        null, replication)
  report/tables/heterogeneity_summary.tex
  report/figures/heterogeneity_model.pdf, heterogeneity_checks.pdf
  results/figures/heterogeneity_libraries.pdf          (panel a of the model
                                                        figure for every library)
"""
from __future__ import annotations

import pickle
import sys
import zlib
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import norm

from pipeline_utils import (
    BATCH_COLORS,
    ROOT,
    REPORT_TABLES,
    REPORT_WIDTH_IN,
    STAGES,
    TABLES,
    ensure_dirs,
    load_dropseq_counts,
    nb_overdispersion,
    report_style,
    save_report_figure,
)

MIN_UMIS = 10  # per library (nb_overdispersion default)
FIXED_CUTOFF = 0.5  # log10, ~3x the typical phi; only to show why a calibration is needed
TIERS = {"weak": 0.5, "moderate": 0.1, "strong": 0.01}  # target FDR among well-expressed genes
WELL_UMIS = 1000  # well-expressed genes, whose residuals define the empirical null
BULK_GENES = 300  # fallback: the most expressed genes, if fewer are well expressed
FLOOR_GRID = np.round(np.arange(0, 3.001, 0.01), 2)
N_HVG = 2000
TOP_N = 1000
N_SIM = 9  # simulations per library: N_SIM - 1 calibrate, 1 checks
NULL_QUANTILE = 0.999
NULL_WINDOW = 1001  # genes of similar mean per running quantile (x N_SIM - 1 simulated values)
EXAMPLE = "ZFS_WT_DS5"  # shield, the focus of the model
LABELLED = ["GSC", "TA", "CHD", "SOX3", "KRT4", "NANOS3", "ACTB1", "EEF1A1L1"]
SEED = 0
ZEROS_YMAX = 0.1  # y axis of the excess-zero panels; larger values are drawn at the top
EXTRAPOLATED = "#0072B2"  # typical phi extrapolated below the fitted bins
FIGURES = ROOT / "results" / "figures"
CACHE = ROOT / "data" / "processed" / "heterogeneity_models.pkl"  # for `plot`


def seurat_dispersion(counts) -> np.ndarray:
    """Seurat normalised dispersion of log-normalised expression (NaN -> -inf)."""
    a = sc.AnnData(counts.astype(np.float32))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat", inplace=False)
    return h["dispersions_norm"].fillna(-np.inf).to_numpy()


def typical_log_phi(disp: pd.DataFrame) -> pd.Series:
    """log10 of the typical phi at each measured gene's mean."""
    return disp["typical"].dropna()


def simulate_homogeneous(counts, disp: pd.DataFrame, rng) -> np.ndarray:
    """log10(phi) estimates (floored at 1e-3; NaN if < MIN_UMIS UMIs) in a
    library in which every gene is homogeneous: negative binomial with the
    typical phi of its mean, the real depths and means."""
    depth = np.asarray(counts.sum(axis=1)).ravel()
    s = depth / depth.mean()
    typical = typical_log_phi(disp)
    genes = typical.index.to_numpy()
    phi = 10 ** typical.to_numpy()
    m = disp["mean"].to_numpy()[genes]
    columns = []
    for start in range(0, len(genes), 2000):
        sl = slice(start, start + 2000)
        shape = 1 / phi[sl]
        rate = rng.gamma(shape, 1 / shape, size=(len(s), len(shape)))  # mean 1, CV^2 phi
        columns.append(sp.csc_matrix(rng.poisson(s[:, None] * m[sl] * rate)))
    sim = sp.hstack(columns).tocsr()
    disp_sim = nb_overdispersion(sim, min_umis=MIN_UMIS)
    log_phi = np.log10(disp_sim["phi"].clip(lower=1e-3)).to_numpy()
    return np.where(disp_sim["residual"].notna(), log_phi, np.nan)


def heterogeneity_calls(counts, rng) -> tuple[pd.DataFrame, dict]:
    """Residuals, calibrated cutoffs and calls of one library (cells x genes
    UMI counts). Also returns the chance level as a function of the mean and
    the false positives of an independent homogeneous simulation."""
    disp = nb_overdispersion(counts, min_umis=MIN_UMIS)
    typical = typical_log_phi(disp)
    genes = typical.index.to_numpy()
    log_m = np.log10(disp["mean"].to_numpy()[genes])
    sims = [simulate_homogeneous(counts, disp, rng) for _ in range(N_SIM)]
    # residuals of the simulated genes against the typical phi they were simulated with
    pool = pd.DataFrame(dict(log_m=np.tile(log_m, N_SIM - 1),
                             residual=np.concatenate([x - typical.to_numpy()
                                                      for x in sims[:-1]]))).dropna()
    pool = pool.sort_values("log_m", kind="stable")
    window = NULL_WINDOW * (N_SIM - 1)
    level = (pool["residual"].rolling(window, center=True, min_periods=window // 4)
             .quantile(NULL_QUANTILE))
    keep = level.notna().to_numpy()
    grid, chance = pool["log_m"].to_numpy()[keep], level.to_numpy()[keep]

    with np.errstate(divide="ignore"):
        all_log_m = np.log10(disp["mean"].to_numpy())
    res = disp["residual"].to_numpy()
    disp["chance"] = np.interp(all_log_m, grid, chance)
    floors, bulk = effect_floors(disp, counts.shape[0])
    disp["tier"] = 0
    for k, tier in enumerate(TIERS, start=1):
        disp[f"threshold_{tier}"] = np.maximum(floors[tier]["floor"], disp["chance"].to_numpy())
        disp.loc[res >= disp[f"threshold_{tier}"].to_numpy(), "tier"] = k
    disp["heterogeneous"] = disp["tier"].to_numpy() >= 1
    check = sims[-1] - typical.to_numpy()  # residuals of the independent simulation
    check_chance = np.interp(log_m, grid, chance)
    null = dict(fixed=int(np.nansum(check >= FIXED_CUTOFF)), grid=grid, chance=chance,
                floors=floors, bulk=bulk, cells=counts.shape[0], trend=disp.attrs["trend"],
                **{tier: int(np.nansum(check >= np.maximum(floors[tier]["floor"], check_chance)))
                   for tier in TIERS})
    return disp, null


def effect_floors(disp: pd.DataFrame, n_cells: int) -> tuple[dict, dict]:
    """Floor of each tier: the smallest residual floor at which the estimated
    FDR among well-expressed genes is at most the tier's target. The null is
    normal with the median and MAD-based SD of their residuals; a gene is
    called if its residual reaches max(floor, chance level)."""
    res = disp["residual"].to_numpy()
    ok = np.isfinite(res)
    well = ok & (disp["mean"].to_numpy() * n_cells >= WELL_UMIS)  # total UMIs = m * cells
    if well.sum() < BULK_GENES:
        top = np.flatnonzero(ok)[np.argsort(-disp["mean"].to_numpy()[ok])[:BULK_GENES]]
        well = np.zeros(len(res), dtype=bool)
        well[top] = True
    r, c = res[well], disp["chance"].to_numpy()[well]
    centre = float(np.median(r))
    sd = float(1.4826 * np.median(np.abs(r - centre)))
    cut = np.maximum(FLOOR_GRID[:, None], c[None, :])
    observed = (r[None, :] >= cut).sum(axis=1)
    expected = norm.sf(cut, centre, sd).sum(axis=1)
    fdr = np.where(observed > 0, np.minimum(1, expected / np.maximum(observed, 1)), 0.0)
    fdr = np.maximum.accumulate(fdr[::-1])[::-1]  # monotone: all higher floors also qualify
    floors = {}
    for tier, target in TIERS.items():
        i = int(np.argmax(fdr <= target)) if (fdr <= target).any() else len(FLOOR_GRID) - 1
        floors[tier] = dict(floor=float(FLOOR_GRID[i]), fdr=float(fdr[i]), called=int(observed[i]))
    bulk = dict(genes=int(well.sum()), median=centre, sd=sd, grid=FLOOR_GRID, fdr=fdr)
    return floors, bulk


def library_rng(lib: str, seed: int = SEED):
    """Random generator of one library: depends only on the seed and the
    library name, not on the order in which libraries are processed."""
    return np.random.default_rng([seed, zlib.crc32(lib.encode())])


def per_library(adata, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    library = adata.obs["library"].astype(str).to_numpy()
    X = adata.X.tocsr()
    residual, tiers, rows, ranks, models = {}, {}, [], {}, {}
    for lib in pd.unique(library):
        counts = X[library == lib]
        disp, null = heterogeneity_calls(counts, library_rng(lib, seed))
        res = disp["residual"].to_numpy()
        residual[lib] = res
        tiers[lib] = disp["tier"].to_numpy()
        dispersion = seurat_dispersion(counts)
        weak = set(np.flatnonzero(tiers[lib] >= 1))
        ranks[lib] = dict(residual=set(np.argsort(-np.nan_to_num(res, nan=-np.inf))[:TOP_N]),
                          seurat=set(np.argsort(-dispersion)[:TOP_N]),
                          weak=weak, seurat_matched=set(np.argsort(-dispersion)[:len(weak)]))
        models[lib] = dict(disp=disp, null=null)
        if lib == EXAMPLE:
            models[lib].update(counts=counts, seurat=np.argsort(-dispersion)[:N_HVG])
        stage = adata.obs.loc[library == lib, "stage"].astype(str).iat[0]
        row = dict(library=lib, stage=stage, batch=lib.split("_")[-1].rstrip("b"),
                   cells=counts.shape[0], measured=int(np.isfinite(res).sum()),
                   fixed_cutoff=int((res >= FIXED_CUTOFF).sum()), null_fixed_cutoff=null["fixed"],
                   bulk_genes=null["bulk"]["genes"], bulk_median=null["bulk"]["median"],
                   bulk_sd=null["bulk"]["sd"])
        for k, tier in enumerate(TIERS, start=1):
            row.update({tier: int((tiers[lib] >= k).sum()), f"null_{tier}": null[tier],
                        f"floor_{tier}": null["floors"][tier]["floor"],
                        f"fdr_{tier}": null["floors"][tier]["fdr"]})
        rows.append(row)
        print(f"{lib:18s} cells {counts.shape[0]:5d}  fixed {row['fixed_cutoff']:5d} "
              f"(null {null['fixed']:4d})  bulk sd {row['bulk_sd']:.2f}  " + "  ".join(
                  f"{t} {row[t]:4d} (floor {row[f'floor_{t}']:.2f}, null {row[f'null_{t}']})"
                  for t in TIERS), flush=True)
    libs = pd.DataFrame(rows).set_index("library")
    frames = {name: pd.DataFrame(d, index=adata.var_names)
              for name, d in [("residual", residual), ("tier", tiers)]}
    return frames, libs, ranks, models


def replication(libs: pd.DataFrame, ranks: dict) -> pd.DataFrame:
    rows = []
    for stage, group in libs.groupby("stage", sort=False):
        names = list(group.index)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                row = dict(stage=stage, a=a, b=b)
                for key in ("residual", "seurat", "weak", "seurat_matched"):
                    x, y = ranks[a][key], ranks[b][key]
                    row[key] = len(x & y) / len(x | y)
                rows.append(row)
    return pd.DataFrame(rows)


def per_stage(adata, frames: dict, libs: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Stage calls, and the comparison with the pooled HVGs of the stage."""
    residual = frames["residual"]
    genes, summary = [], []
    for stage in STAGES:
        names = list(libs.index[libs["stage"] == stage])
        med = residual[names].median(axis=1)
        counts = {tier: (frames["tier"][names] >= k).sum(axis=1)
                  for k, tier in enumerate(TIERS, start=1)}
        stage_calls = {tier: 2 * n >= len(names) for tier, n in counts.items()}
        is_het = stage_calls["weak"]
        rows = adata.obs["stage"].astype(str).to_numpy() == stage
        sub = adata[rows]
        stage_umis = pd.Series(np.asarray(sub.X.sum(axis=0)).ravel(), index=adata.var_names)
        hvg = set(adata.var_names[np.argsort(-seurat_dispersion(sub.X))[:N_HVG]])
        het = set(med.index[is_het])
        genes.append(pd.DataFrame(dict(stage=stage, gene=adata.var_names, median_residual=med,
                                       libraries=len(names), stage_umis=stage_umis,
                                       heterogeneous=is_het,
                                       **{f"libraries_{t}": n for t, n in counts.items()},
                                       **stage_calls, hvg=adata.var_names.isin(list(hvg))))
                     .query("stage_umis > 0"))
        only_hvg, only_het = list(hvg - het), list(het - hvg)
        summary.append(dict(stage=stage, libraries=len(names), cells=int(rows.sum()),
                            **{t: int(c.sum()) for t, c in stage_calls.items()},
                            fixed=int(libs.loc[names, "fixed_cutoff"].median()),
                            null_fixed=int(libs.loc[names, "null_fixed_cutoff"].median()),
                            null=int(libs.loc[names, "null_weak"].median()),
                            hvg_and_het=len(hvg & het), hvg_only=len(only_hvg),
                            hvg_only_umis=float(stage_umis[only_hvg].median()),
                            hvg_only_unmeasured=int(residual.loc[only_hvg, names].isna()
                                                    .all(axis=1).sum()),
                            het_only=len(only_het),
                            het_only_umis=float(stage_umis[only_het].median())))
    return pd.concat(genes, ignore_index=True), summary


def typical_phi_at(trend: dict, mean: np.ndarray) -> np.ndarray:
    """Typical phi (fitted dispersion-mean trend of nb_overdispersion) at any mean."""
    return trend["a0"] + trend["a1"] * np.asarray(mean, dtype=float) ** (-trend["b"])


def plot_typical(ax, grid: np.ndarray, trend: dict, lw: float, label: bool = False) -> None:
    """Typical line: black over the bins it is fitted to, blue where extrapolated
    to rarer genes; dots: the bin medians."""
    typical = np.log10(typical_phi_at(trend, grid))
    fitted = grid >= trend["bin_mean"].min()
    ax.plot(grid, np.where(fitted, np.nan, typical), color=EXTRAPOLATED, lw=lw,
            label="typical, extrapolated" if label else None)
    ax.plot(grid, np.where(fitted, typical, np.nan), color="#1A1A1A", lw=lw,
            label="typical (fitted trend)" if label else None)
    ax.scatter(trend["bin_mean"], np.log10(trend["bin_phi"]), s=lw * 4, color="#1A1A1A",
               lw=0, zorder=3, label="bin medians" if label else None)


def expected_zeros(s: np.ndarray, mean: np.ndarray, phi) -> np.ndarray:
    """Expected fraction of cells with 0 UMIs (NB; phi=0: Poisson)."""
    mu = s[:, None] * mean[None, :]
    phi = np.broadcast_to(phi, mean.shape)
    out = np.empty(len(mean))
    for j in range(0, len(mean), 2000):
        sl = slice(j, j + 2000)
        p = phi[sl][None, :]
        out[sl] = np.where(p > 0, (1 + p * mu[:, sl]) ** (-1 / np.maximum(p, 1e-12)),
                           np.exp(-mu[:, sl])).mean(axis=0)
    return out


def chance_cutoff(null: dict, mean: np.ndarray, tier: str) -> np.ndarray:
    """Cutoff of a tier on the residual at any mean."""
    return np.maximum(null["floors"][tier]["floor"],
                      np.interp(np.log10(mean), null["grid"], null["chance"]))


TIER_COLORS = {"weak": "#F5B78A", "moderate": "#E0662A", "strong": "#8C2D04"}
TIER_STYLES = {"weak": ":", "moderate": "--", "strong": "-."}


def plot_tiers(ax, disp: pd.DataFrame, null: dict, size: float, lw: float,
               label: bool = False) -> None:
    """Overdispersion against the mean: genes by tier, typical value and the
    cutoff of every tier."""
    ok = disp["residual"].notna().to_numpy()
    mean = disp["mean"].to_numpy()
    log_phi = np.log10(disp["phi"].clip(lower=1e-3).to_numpy())
    tier = disp["tier"].to_numpy()
    rest = ok & (tier == 0)
    ax.scatter(mean[rest], log_phi[rest], s=size, color="#BDBDBD", lw=0, rasterized=True)
    for k, (name, fdr) in enumerate(TIERS.items(), start=1):
        sel = ok & (tier == k)
        ax.scatter(mean[sel], log_phi[sel], s=size * 1.4, color=TIER_COLORS[name], lw=0,
                   rasterized=True,
                   label=f"{name}, FDR {100 * fdr:g}% ({(tier >= k).sum():,})" if label else None)
    grid = np.logspace(np.log10(mean[ok].min()), np.log10(mean[ok].max()), 300)
    typical = np.log10(typical_phi_at(null["trend"], grid))
    plot_typical(ax, grid, null["trend"], lw=lw, label=label)
    for name in TIERS:
        ax.plot(grid, typical + chance_cutoff(null, grid, name), color="#1A1A1A", lw=lw * 0.8,
                ls=TIER_STYLES[name], label=f"{name} cutoff" if label else None)
    ax.set_xscale("log")


def plot_model(models: dict, var_names) -> None:
    m = models[EXAMPLE]
    disp, counts, null = m["disp"], m["counts"], m["null"]
    ok = disp["residual"].notna().to_numpy()
    mean = disp["mean"].to_numpy()
    log_phi = np.log10(disp["phi"].clip(lower=1e-3).to_numpy())
    het = disp["heterogeneous"].to_numpy()
    hvg = np.zeros(len(mean), dtype=bool)
    hvg[m["seurat"]] = True
    present = mean > 0
    depth = np.asarray(counts.sum(axis=1)).ravel()
    s = depth / depth.mean()
    zeros = 1 - np.asarray((counts > 0).mean(axis=0)).ravel()
    excess = np.full(len(mean), np.nan)
    excess[present] = zeros[present] - expected_zeros(s, mean[present],
                                                      typical_phi_at(null["trend"], mean[present]))
    grey, orange, blue = "#BDBDBD", "#D55E00", "#0072B2"

    fig, axes = plt.subplots(1, 3, figsize=(REPORT_WIDTH_IN, 2.7), constrained_layout=True)
    ax = axes[0]
    plot_tiers(ax, disp, null, size=1, lw=1, label=True)
    for gene in LABELLED:
        if gene in var_names:
            j = var_names.get_loc(gene)
            if ok[j]:
                ax.annotate(gene.lower(), (mean[j], log_phi[j]), fontsize=5.5,
                            xytext=(2, 2), textcoords="offset points")
                ax.plot(mean[j], log_phi[j], "o", ms=2, mfc="none", mec="#1A1A1A", mew=0.5)
    ax.set_xlabel("mean UMIs per cell (m)")
    ax.set_ylabel(r"overdispersion $\log_{10}\varphi$")
    ax.set_title("a  overdispersion", loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(-0.25, -0.22),
              ncol=2, markerscale=4, columnspacing=0.8, handlelength=2.2)

    lin = np.logspace(np.log10(mean[ok].min()), np.log10(mean[ok].max()), 200)
    phi_lin = typical_phi_at(null["trend"], lin)
    base = expected_zeros(s, lin, phi_lin)
    # Genes are called on their variance (panel a), not on zeros, so no cutoff is drawn here:
    # variance from a few cells with many UMIs or from graded levels adds few zeros.
    curves = [("Poisson", expected_zeros(s, lin, 0.0) - base, ":")]
    for ax, title, highlight, color, label in [
            (axes[1], "b  zeros: heterogeneous", het, orange, "heterogeneous (weak)"),
            (axes[2], "c  zeros: HVGs only", hvg & present & ~het, blue,
             "HVGs, not heterogeneous")]:
        rest = present & ~highlight
        shown = np.minimum(excess, ZEROS_YMAX)  # genes above the axis are drawn at its top
        ax.scatter(mean[rest], shown[rest], s=1, color=grey, lw=0, rasterized=True)
        ax.scatter(mean[highlight], shown[highlight], s=2, color=color, lw=0, rasterized=True,
                   label=f"{label} ({highlight.sum():,})")
        ax.axhline(0, color="#1A1A1A", lw=1, label="NB, typical φ")
        for name, y, ls in curves:
            ax.plot(lin, y, color="#1A1A1A", lw=0.9, ls=ls, label=name)
        ax.set_xscale("log")
        ax.set_xlim(axes[0].get_xlim())
        ax.set_xlabel("mean UMIs per cell (m)")
        ax.set_title(title, loc="left", fontsize=8)
        ax.legend(frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(-0.05, -0.22),
                  markerscale=4)
    axes[1].set_ylabel("fraction of zeros − expected\n(NB with typical φ)")
    axes[2].sharey(axes[1])
    axes[1].set_ylim(-0.03, ZEROS_YMAX * 1.02)
    axes[2].tick_params(labelleft=False)
    save_report_figure(fig, "heterogeneity_model")
    plt.close(fig)


def plot_checks(libs: pd.DataFrame, rep: pd.DataFrame) -> None:
    order = sorted(libs.index, key=lambda l: (STAGES.index(libs.loc[l, "stage"]), l))
    fig, axes = plt.subplots(1, 3, figsize=(REPORT_WIDTH_IN, 2.7), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.5, 1.5, 0.8]))
    x = np.arange(len(order))
    t = libs.loc[order]
    colors = [BATCH_COLORS[b] for b in t["batch"]]
    labels = [f"{s} {l.split('_')[-1]}" for s, l in zip(t["stage"], order)]
    for ax, col, title in [(axes[0], "fixed_cutoff", "a  fixed cutoff (3.2× typical)"),
                           (axes[1], "weak", "b  calibrated, three tiers")]:
        null = "null_fixed_cutoff" if col == "fixed_cutoff" else "null_weak"
        if col == "weak":  # tiers as shades of the batch colour: light weak, full strong
            for tier, alpha in [("weak", 0.3), ("moderate", 0.6), ("strong", 1.0)]:
                ax.bar(x, t[tier], color=[matplotlib.colors.to_rgba(c, alpha) for c in colors],
                       width=0.8, lw=0)
                ax.bar([np.nan], [np.nan], color=(0.3, 0.3, 0.3, alpha), label=f"{tier} tier")
        else:
            ax.bar(x, t[col], color=colors, width=0.8, lw=0)
        ax.bar(x, t[null], color="#1A1A1A", width=0.8, lw=0,
               label="simulated homogeneous library" + (" (number)" if col == "weak" else ""))
        if col == "weak":  # the simulated bars are too small to see: print their height
            for xi, top, k in zip(x, t[col], t[null]):
                ax.text(xi, top + 25, f"{int(k)}", ha="center", va="bottom", fontsize=4)
        ax.set_xticks(x, labels, rotation=90, fontsize=4.5)
        ax.set_title(title, loc="left", fontsize=8)
    axes[1].sharey(axes[0])
    axes[0].set_ylabel("heterogeneous genes (colour: batch)")
    axes[0].legend(frameon=False, fontsize=6, loc="upper right")
    axes[1].legend(frameon=False, fontsize=6, loc="upper right")
    ax = axes[2]
    for _, r in rep.iterrows():
        ax.plot([0, 1], [r["seurat"], r["residual"]], color="#BDBDBD", lw=0.6)
    ax.scatter(np.zeros(len(rep)), rep["seurat"], s=8, color="#0072B2", zorder=3)
    ax.scatter(np.ones(len(rep)), rep["residual"], s=8, color="#D55E00", zorder=3)
    ax.set_xticks([0, 1], ["Seurat\ndisp.", "NB\nresidual"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel(f"overlap of top {TOP_N:,} genes (Jaccard)")
    ax.set_title("c  replication", loc="left", fontsize=8)
    save_report_figure(fig, "heterogeneity_checks")
    plt.close(fig)


def plot_libraries(models: dict, libs: pd.DataFrame) -> None:
    """Figure 5a for every library (supplementary, results/figures)."""
    order = sorted(libs.index, key=lambda l: (STAGES.index(libs.loc[l, "stage"]), l))
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(REPORT_WIDTH_IN * 1.25, 1.9 * nrow),
                             sharex=True, sharey=True, constrained_layout=True)
    for ax, lib in zip(axes.ravel(), order):
        plot_tiers(ax, models[lib]["disp"], models[lib]["null"], size=0.6, lw=0.8)
        r = libs.loc[lib]
        ax.set_title(f"{r['stage']} {lib.split('_')[-1]} ({int(r['cells']):,} cells)\n"
                     f"{int(r['weak']):,} / {int(r['moderate']):,} / {int(r['strong']):,} "
                     f"genes, null {int(r['null_weak'])}", fontsize=6, loc="left")
        ax.tick_params(labelsize=5.5)
    for ax in axes.ravel()[len(order):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10}\varphi$", fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("mean UMIs per cell", fontsize=6)
    fig.suptitle("Overdispersion per library. Solid: typical (fitted trend; black over the bin "
                 "medians (dots) it is fitted to, blue where extrapolated).\nCutoffs (max of the tier's floor and the "
                 "chance level): dotted weak (FDR 50%), dashed moderate (10%), dash-dot strong "
                 "(1%).\nPoints, light to dark orange: weak, moderate, strong; panel titles: "
                 "genes per tier (weak / moderate / strong), false positives of the weak tier in "
                 "a simulation.", fontsize=6.5)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "heterogeneity_libraries.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


def latex_table(summary: list) -> None:
    def n(x):
        return f"{int(x):,}".replace(",", "{,}")
    lines = [f"{r['stage'].replace('%', chr(92) + '%')} & {r['libraries']} & {n(r['cells'])} & "
             f"{n(r['weak'])} & {n(r['moderate'])} & {n(r['strong'])} & {n(r['hvg_and_het'])} & "
             f"{n(r['hvg_only'])} ({r['hvg_only_umis']:.0f}) & "
             f"{n(r['het_only'])} ({r['het_only_umis']:.0f}) \\\\" for r in summary]
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    (REPORT_TABLES / "heterogeneity_summary.tex").write_text(
        "% Generated by scripts/01b_heterogeneous_genes.py; do not edit.\n"
        "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
        "Stage & Lib. & Cells & Weak & Mod. & Strong & Both & "
        "HVG only & Het. only \\\\\n\\midrule\n" + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    ensure_dirs()
    report_style()
    sc.settings.verbosity = 0
    adata = load_dropseq_counts()
    frames, libs, ranks, models = per_library(adata)
    frames["residual"].rename_axis("gene").to_csv(
        TABLES / "heterogeneous_genes_libraries.tsv.gz", sep="\t", float_format="%.4g")
    calls = frames["tier"]
    calls.rename_axis("gene").to_csv(TABLES / "heterogeneous_genes_calls.tsv.gz", sep="\t")
    rep = replication(libs, ranks)
    libs = libs.join(rep.groupby("a")[["residual", "seurat", "weak", "seurat_matched"]].mean()
                     .rename(columns=lambda c: f"jaccard_{c}"))
    libs.to_csv(TABLES / "heterogeneity_libraries.tsv", sep="\t", float_format="%.4g")
    genes, summary = per_stage(adata, frames, libs)
    genes.to_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t", index=False, float_format="%.4g")
    print(pd.DataFrame(summary).to_string(index=False))
    print(rep.round(2).to_string(index=False))
    latex_table(summary)
    with open(CACHE, "wb") as fh:
        pickle.dump(dict(models=models, var_names=adata.var_names, rep=rep), fh)
    plot()


def plot() -> None:
    """Redraw the figures from the cache of the last run."""
    report_style()
    with open(CACHE, "rb") as fh:
        cache = pickle.load(fh)
    libs = pd.read_csv(TABLES / "heterogeneity_libraries.tsv", sep="\t", index_col=0)
    plot_model(cache["models"], cache["var_names"])
    plot_checks(libs, cache["rep"])
    plot_libraries(cache["models"], libs)


if __name__ == "__main__":
    if sys.argv[1:] == ["plot"]:
        plot()
    else:
        main()
