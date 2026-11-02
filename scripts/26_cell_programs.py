"""Which program does a cell have, and which cells are excluded before
population detection?

Program calls, per stage and library (libraries are never pooled):
  * Score: the mean Pearson residual of a program's genes, smoothed over the
    cell and its K_SMOOTH nearest neighbours in its library. The neighbours
    come from a PCA of the stage's heterogeneous genes without the genes of the
    scored programs (a program's own genes do not decide over which cells it is
    averaged) and without the genes of EXCLUDED_GENE_PROGRAMS (heat shock,
    dissociation stress, cell cycle: states on top of an identity, which are
    handled by leaving out their genes, not their cells).
  * Null: every gene's residuals permuted across the cells of the library
    (N_PERM times), scored and smoothed alike. Active = above the
    ACTIVE_QUANTILE of the null; significance = the standardised excess
    z = (score - null mean) / null sd, comparable between programs of
    different size and sparsity.
  * Competition: the identity programs (curated anatomy programs and the
    maternal and blastomere states) supported at the stage (script 17). Top =
    the active program with the largest z; margin = z of the top minus the
    largest z of the other competitors, the latter floored at 0 (= the null
    mean; with a single competitor the margin is the top's own z).
  * Reliability: every program's genes are split at random into two halves,
    and the top program is called from each half alone; the halves agree
    exactly, or up to a derivation (one derives from the other, directly or
    not; derives_from of curated_programs.tsv), by the margin of the full call.

Exclusion:
  * Stress: the same competition with the stress states added (p53 response,
    dissociation stress, heat shock, where supported). A cell is excluded if a
    stress state is its top program with a margin of at least EXCLUDE_MARGIN.
  * Yolk/YSL (nothing excluded): the YSL control of script 16 is scored as a
    program (not competing), and the yolk- and YSL-specific transcripts
    (YOLK_GENES) are counted per cell.
  * Quality (nothing excluded): depth (log UMIs vs the library median, in
    MADs), complexity (log detected genes given log UMIs, residual in MADs)
    and scrublet doublet scores per library (DOUBLET_THRESHOLD).

Stage UMAPs: the aligned cells of a stage (centred residuals per library) on
the stage's heterogeneous genes without EXCLUDED_GENE_PROGRAMS; PCA, UMAP.
Also used by script 19.

The calls of every library are cached (data/processed/cache/26); increase
CACHE_VERSION when a code change changes them. `26_cell_programs.py plot`
redraws the figures and report table from the saved tables.

Outputs:
  results/tables/cell_programs_cells.tsv.gz   cell: stage, library, UMAP,
      top, z_top, margin, active identity programs, stress top and margin,
      excluded (reason), YSL control active, yolk genes, quality measures
  results/tables/cell_programs.tsv            stage x program (kept cells,
      median library): active, top, top among the active, median margin
  results/tables/cell_programs_reliability.tsv   split-half agreement by
      stage and margin
  results/tables/cell_exclusions.tsv          per library
  data/processed/cell_programs_{z,active}.tsv.gz   cell x program z and
      active calls (all scored programs; empty where not scored)
  report/figures/cell_programs.pdf, report/figures/cell_programs_top.pdf,
  report/tables/cell_exclusions.tex, results/figures/cell_programs_atlas.pdf
"""
from __future__ import annotations

import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from pipeline_utils import (
    GZIP,
    PROCESSED,
    REPORT_TABLES,
    REPORT_WIDTH_IN,
    ROOT,
    STAGES,
    TABLES,
    cached,
    digest,
    ensure_dirs,
    load_dropseq_counts,
    pearson_residuals,
    report_style,
    save_report_figure,
)
from program_utils import K_SMOOTH, N_PCS

warnings.filterwarnings("ignore", category=FutureWarning)

CACHE_VERSION = 1
SEED = 0
N_PERM = 20
ACTIVE_QUANTILE = 0.999
IDENTITY_STATES = ("state:maternal", "state:blastomere")
NON_IDENTITY = ("state:", "signal:", "receiving:", "sending:", "blocking:", "control:")
STRESS = ("state:p53_response", "control:dissociation", "state:heat_shock")
EXCLUDED_GENE_PROGRAMS = ("state:heat_shock", "control:dissociation", "state:cell_cycle_s",
                          "state:cell_cycle_g2m")
YSL = "control:YSL"
YOLK_GENES = ["APOA1A", "APOA1B", "APOA2", "APOA4B.1", "APOBB.1", "AFP4", "ITIH2", "TFA"]
YOLK_MIN = 3  # yolk genes detected, at least, to count a cell as yolk-like
EXCLUDE_MARGIN = 2.0
MARGIN_BINS = [0, 1, 2, 3, 5, np.inf]
MARGIN_LEVELS = (1, 3)  # shares of cells with margins below / above (figure)
DOUBLET_THRESHOLD = 0.25
MAD_LIMIT = 3
UMAP_NEIGHBOURS = 15
ANNOTATION = ROOT / "data" / "curated" / "curated_programs.tsv"
FIGURES = ROOT / "results" / "figures"
CELLS = TABLES / "cell_programs_cells.tsv.gz"
Z_TABLE = PROCESSED / "cell_programs_z.tsv.gz"
ACTIVE_TABLE = PROCESSED / "cell_programs_active.tsv.gz"


def identity(program: str) -> bool:
    return program in IDENTITY_STATES or not program.startswith(NON_IDENTITY)


def stage_programs(programs: pd.DataFrame, support: pd.DataFrame, stage: str) -> dict:
    """Gene lists of the supported identity and stress programs of a stage,
    the YSL control and the genes of EXCLUDED_GENE_PROGRAMS."""
    ok = set(support.loc[(support["stage"] == stage) & support["supported"].astype(bool),
                         "program"])
    p = programs[(programs["stage"] == stage) & programs["kept"]
                 & programs["set"].isin(["curated", "controls"])]
    genes = p.groupby("program")["gene"].agg(list)
    ident = [q for q in genes.index if identity(q) and q in ok
             and q in set(p.loc[p["set"] == "curated", "program"])]
    stress = [q for q in STRESS if q in ok and q in genes.index]
    excluded = sorted(set(p.loc[p["program"].isin(EXCLUDED_GENE_PROGRAMS), "gene"]))
    return dict(identity={q: genes[q] for q in ident}, stress={q: genes[q] for q in stress},
                ysl=genes.get(YSL, []), excluded=excluded)


def library_calls(counts, var_names: pd.Index, het: set, progs: dict, seed) -> dict:
    """Smoothed scores against the permutation null of one library: z and
    active of every scored program, and the top identity program of each
    random half of the identity programs' genes."""
    rng = np.random.default_rng(seed)
    n = counts.shape[0]
    scored = {**progs["identity"], **progs["stress"]}
    if progs["ysl"]:
        scored[YSL] = progs["ysl"]
    genes = sorted({g for gs in scored.values() for g in gs})
    graph = sorted(het - set(genes) - set(progs["excluded"]))
    pcs = PCA(min(N_PCS, n - 1), random_state=SEED).fit_transform(
        pearson_residuals(counts, var_names.get_indexer(graph)))
    nb = NearestNeighbors(n_neighbors=min(K_SMOOTH, n - 1)).fit(pcs).kneighbors(
        return_distance=False)
    nb = np.column_stack([np.arange(n), nb])
    res = pearson_residuals(counts, var_names.get_indexer(genes))
    col = {g: k for k, g in enumerate(genes)}

    def call(gs):
        r = res[:, [col[g] for g in gs]]
        score = r.mean(axis=1)[nb].mean(axis=1)
        null = np.concatenate([rng.permuted(r, axis=0).mean(axis=1)[nb].mean(axis=1)
                               for _ in range(N_PERM)])
        sd = null.std() or 1.0
        return (score - null.mean()) / sd, score > np.quantile(null, ACTIVE_QUANTILE)

    z, active = {}, {}
    for name, gs in scored.items():
        z[name], active[name] = call(gs)
    halves = []
    for part in (0, 1):
        hz, ha = {}, {}
        for name, gs in progs["identity"].items():
            perm = np.random.default_rng([seed[0], seed[1], len(gs), part]).permutation(gs)
            half = list(perm[: len(gs) // 2]) if part == 0 else list(perm[len(gs) // 2:])
            hz[name], ha[name] = call(half) if half else (np.full(n, -np.inf), np.zeros(n, bool))
        halves.append(top_program(pd.DataFrame(hz), pd.DataFrame(ha))[0])
    return dict(z=pd.DataFrame(z), active=pd.DataFrame(active), halves=halves)


def top_program(z: pd.DataFrame, active: pd.DataFrame) -> tuple[np.ndarray, np.ndarray,
                                                                 np.ndarray]:
    """Top active program ("" if none), its z and its margin over the others."""
    if z.shape[1] == 0:
        empty = np.full(len(z), np.nan)
        return np.full(len(z), "", dtype=object), empty, empty
    zz = z.to_numpy()
    za = np.where(active.to_numpy(), zz, -np.inf)
    k = za.argmax(axis=1)
    rows = np.arange(len(zz))
    on = np.isfinite(za[rows, k])
    others = zz.copy()
    others[rows, k] = -np.inf
    second = np.maximum(others.max(axis=1), 0) if zz.shape[1] > 1 else np.zeros(len(zz))
    top = np.where(on, z.columns.to_numpy()[k], "")
    return top, np.where(on, zz[rows, k], np.nan), np.where(on, zz[rows, k] - second, np.nan)


def quality(adata) -> pd.DataFrame:
    """Depth and complexity in MADs of the library, scrublet doublet scores."""
    out = []
    X = adata.X.tocsr()
    lib_of = adata.obs["library"].astype(str).to_numpy()
    for lib in sorted(set(lib_of)):
        rows = np.flatnonzero(lib_of == lib)
        c = X[rows]
        umis = np.asarray(c.sum(axis=1)).ravel()
        detected = np.asarray((c > 0).sum(axis=1)).ravel()
        lu, lg = np.log10(umis), np.log10(detected)

        def mads(v):
            return (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))))
        fit = np.polyfit(lu, lg, 1)
        sub = adata[rows].copy()
        sc.pp.scrublet(sub, threshold=DOUBLET_THRESHOLD, random_state=SEED, verbose=False)
        out.append(pd.DataFrame(dict(umis=umis, depth_mad=mads(lu),
                                     complexity_mad=mads(lg - np.polyval(fit, lu)),
                                     doublet_score=sub.obs["doublet_score"].to_numpy()),
                                index=adata.obs_names[rows]))
    return pd.concat(out)


def stage_umap(sub, het: set, excluded: list) -> np.ndarray:
    genes = np.flatnonzero(sub.var_names.isin(sorted(het - set(excluded))))
    X = sub.X.tocsr()
    lib_of = sub.obs["library"].astype(str).to_numpy()
    res = np.zeros((sub.n_obs, len(genes)), dtype=np.float32)
    for lib in sorted(set(lib_of)):
        rows = np.flatnonzero(lib_of == lib)
        res[rows] = pearson_residuals(X[rows], genes, centre=True)
    a = sc.AnnData(np.zeros((sub.n_obs, 1), dtype=np.float32))
    a.obsm["X_pca"] = PCA(N_PCS, random_state=SEED).fit_transform(res)
    sc.pp.neighbors(a, n_neighbors=UMAP_NEIGHBOURS, use_rep="X_pca", random_state=SEED)
    sc.tl.umap(a, random_state=SEED)
    return a.obsm["X_umap"]


def ancestors_of() -> dict:
    d = pd.read_csv(ANNOTATION, sep="\t").dropna(subset=["derives_from"])
    parents = {r.program: r.derives_from.split(";") for r in d.itertuples()}

    def up(p):
        out = set()
        for q in parents.get(p, []):
            out |= {q} | up(q)
        return out
    return {p: up(p) for p in pd.read_csv(ANNOTATION, sep="\t")["program"]}


def related(a: str, b: str, anc: dict) -> bool:
    return a == b or a in anc.get(b, set()) or b in anc.get(a, set())


def compute() -> None:
    adata = load_dropseq_counts()
    programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    support = load_support()
    het_table = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                            usecols=["stage", "gene", "heterogeneous"])
    anc = ancestors_of()
    qc = cached("26", "quality", digest(CACHE_VERSION, DOUBLET_THRESHOLD, SEED),
                lambda: quality(adata))
    yolk = np.asarray((adata[:, YOLK_GENES].X > 0).sum(axis=1)).ravel()
    cells, zs, acts, halves = [], [], [], []
    for s, stage in enumerate(STAGES):
        sub = adata[adata.obs["stage"] == stage]
        het = set(het_table.loc[(het_table["stage"] == stage) & het_table["heterogeneous"],
                                "gene"])
        progs = stage_programs(programs, support, stage)
        xy = cached("26", f"umap-{stage}", digest(CACHE_VERSION, sorted(het), progs["excluded"]),
                    lambda: stage_umap(sub, het, progs["excluded"]))
        lib_of = sub.obs["library"].astype(str).to_numpy()
        X = sub.X.tocsr()
        ident, stress = list(progs["identity"]), list(progs["stress"])
        for lib in sorted(set(lib_of)):
            rows = np.flatnonzero(lib_of == lib)
            key = digest(CACHE_VERSION, N_PERM, ACTIVE_QUANTILE, K_SMOOTH, sorted(het), progs)
            r = cached("26", f"calls-{lib}", key, lambda: library_calls(
                X[rows], sub.var_names, het, progs, (SEED, s)))
            names = sub.obs_names[rows]
            z, act = r["z"].set_index(names), r["active"].set_index(names)
            top, z_top, margin = top_program(z[ident], act[ident])
            s_top, _, s_margin = top_program(z[ident + stress], act[ident + stress])
            is_stress = np.isin(s_top, stress)
            cells.append(pd.DataFrame(dict(
                stage=stage, library=lib, umap_1=xy[rows, 0], umap_2=xy[rows, 1], top=top,
                z_top=z_top, margin=margin, active_identity=act[ident].sum(axis=1).to_numpy(),
                stress_top=np.where(is_stress, s_top, ""),
                stress_margin=np.where(is_stress, s_margin, np.nan),
                excluded=np.where(is_stress & (s_margin >= EXCLUDE_MARGIN), s_top, ""),
                ysl_active=act[YSL].to_numpy() if YSL in act else False,
                yolk_genes=yolk[adata.obs_names.get_indexer(names)]), index=names))
            zs.append(z)
            acts.append(act)
            a, b = r["halves"]
            ok = (a != "") & (b != "") & (top != "")
            halves.append(pd.DataFrame(dict(
                stage=stage, library=lib, margin=margin[ok], exact=a[ok] == b[ok],
                lineage=[related(x, y, anc) for x, y in zip(a[ok], b[ok])])))
        print(stage, flush=True)
    cells = pd.concat(cells).join(qc)
    cells.index.name = "cell"
    cells.to_csv(CELLS, sep="\t", float_format="%.4g", compression=GZIP)
    pd.concat(zs).to_csv(Z_TABLE, sep="\t", float_format="%.3g", compression=GZIP)
    pd.concat(acts).astype(float).to_csv(ACTIVE_TABLE, sep="\t", float_format="%.0f",
                                         compression=GZIP)
    halves = pd.concat(halves)
    halves["margin_bin"] = pd.cut(halves["margin"], MARGIN_BINS, right=False).astype(str)
    halves.groupby(["stage", "margin_bin"], sort=False).agg(
        cells=("exact", "size"), exact=("exact", "mean"), lineage=("lineage", "mean")) \
        .reset_index().to_csv(TABLES / "cell_programs_reliability.tsv", sep="\t", index=False,
                              float_format="%.3f")
    program_table(cells, pd.concat(acts), programs, support).to_csv(
        TABLES / "cell_programs.tsv", sep="\t", index=False, float_format="%.3f")
    exclusions(cells).to_csv(TABLES / "cell_exclusions.tsv", sep="\t", index=False,
                             float_format="%.4f")


def load_support() -> pd.DataFrame:
    support = pd.read_csv(TABLES / "program_support.tsv", sep="\t")
    return support[support["set"].isin(["curated", "controls"])]


def program_table(cells, active, programs, support) -> pd.DataFrame:
    """Per stage and identity program, over the kept cells (median library):
    share active, share on top, share of the active cells with it on top, and
    the median margin of its top cells."""
    kept = cells[cells["excluded"] == ""]
    rows = []
    for (stage, lib), c in kept.groupby(["stage", "library"], sort=False):
        a = active.loc[c.index]
        for p in stage_programs(programs, support, stage)["identity"]:
            on, tops = a[p] == 1, c["top"] == p
            rows.append(dict(stage=stage, library=lib, program=p, active=on.mean(),
                             top=tops.mean(), top_of_active=tops.sum() / on.sum() if on.any()
                             else np.nan, margin=c.loc[tops, "margin"].median()))
    return pd.DataFrame(rows).groupby(["stage", "program"], sort=False)[
        ["active", "top", "top_of_active", "margin"]].median().reset_index()


def exclusions(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stage, lib), c in cells.groupby(["stage", "library"], sort=False):
        low_cx = c["complexity_mad"] < -MAD_LIMIT
        rows.append(dict(
            stage=stage, library=lib, cells=len(c),
            **{f"excluded_{q.split(':')[1]}": int((c["excluded"] == q).sum()) for q in STRESS},
            excluded=(c["excluded"] != "").mean(),
            stress_top_any_margin=int((c["stress_top"] != "").sum()),
            ysl_active=int(c["ysl_active"].sum()),
            ysl_active_with_identity=int((c["ysl_active"] & (c["active_identity"] > 0)).sum()),
            yolk_like=int((c["yolk_genes"] >= YOLK_MIN).sum()),
            doublets=int((c["doublet_score"] >= DOUBLET_THRESHOLD).sum()),
            low_depth=int((c["depth_mad"] < -MAD_LIMIT).sum()),
            high_depth=int((c["depth_mad"] > MAD_LIMIT).sum()),
            low_complexity=int(low_cx.sum()),
            low_complexity_evl=int((low_cx & (c["top"] == "EVL")).sum())))
    return pd.DataFrame(rows)


# --- figures -----------------------------------------------------------------------------------

LEVEL_COLOURS = ["#DDDDDD", "#C6DBEF", "#6BAED6", "#08519C"]


def plot() -> None:
    report_style()
    cells = pd.read_csv(CELLS, sep="\t", index_col=0, keep_default_na=False,
                        na_values={c: [""] for c in ("margin", "z_top", "stress_margin")})
    rel = pd.read_csv(TABLES / "cell_programs_reliability.tsv", sep="\t")
    table = pd.read_csv(TABLES / "cell_programs.tsv", sep="\t")
    fig_main(cells, rel)
    fig_top(table)
    latex_exclusions(pd.read_csv(TABLES / "cell_exclusions.tsv", sep="\t"))
    atlas(cells, pd.read_csv(ACTIVE_TABLE, sep="\t", index_col=0), table)


def fig_main(cells: pd.DataFrame, rel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(REPORT_WIDTH_IN, 2.4), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axes[0]
    late = rel[rel["stage"].isin(STAGES[STAGES.index("shield"):])]
    bins = [b for b in late["margin_bin"].unique() if b != "nan"]
    bins = sorted(bins, key=lambda b: float(b[1:].split(",")[0]))
    g = late.groupby("margin_bin")
    for col, colour, label in (("exact", "#0072B2", "same program"),
                               ("lineage", "#E69F00", "same up to a derivation")):
        share = g.apply(lambda d: (d[col] * d["cells"]).sum() / d["cells"].sum()).reindex(bins)
        ax.plot(range(len(bins)), share, marker="o", ms=3, color=colour, label=label)
    ax.set_xticks(range(len(bins)), [b.replace("inf", "∞") for b in bins], rotation=30)
    ax.set_xlabel("margin of the top program")
    ax.set_ylabel("agreement of the two gene halves")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("(a) split-half reliability (shield on)", loc="left", fontsize=8)
    ax = axes[1]
    kept = cells[cells["excluded"] == ""]
    lo, hi = MARGIN_LEVELS
    level = np.select([kept["top"] == "", kept["margin"] < lo, kept["margin"] < hi],
                      [0, 1, 2], 3)
    shares = pd.crosstab([kept["stage"], kept["library"]], level, normalize="index") \
        .groupby(level=0).mean().reindex(STAGES).fillna(0)
    bottom = np.zeros(len(STAGES))
    labels = ["no active program", f"margin < {lo}", f"margin {lo}–{hi}", f"margin ≥ {hi}"]
    for k in range(4):
        v = shares[k].to_numpy() if k in shares else np.zeros(len(STAGES))
        ax.bar(range(len(STAGES)), v, bottom=bottom, color=LEVEL_COLOURS[k], width=0.75,
               label=labels[k], edgecolor="white", linewidth=0.5)
        bottom += v
    ax.set_xticks(range(len(STAGES)), STAGES, rotation=45, ha="right")
    ax.set_ylabel("share of cells (mean library)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("(b) top identity program (kept cells)", loc="left", fontsize=8)
    save_report_figure(fig, "cell_programs")
    plt.close(fig)


def program_order(names) -> list[str]:
    order = list(IDENTITY_STATES) + list(pd.read_csv(ANNOTATION, sep="\t")["program"])
    return [p for p in dict.fromkeys(order) if p in set(names)]


def fig_top(table: pd.DataFrame) -> None:
    progs = program_order(table["program"].unique())
    fig, ax = plt.subplots(figsize=(REPORT_WIDTH_IN, 0.13 * len(progs) + 1.0),
                           constrained_layout=True)
    x = table["stage"].map({s: k for k, s in enumerate(STAGES)})
    y = table["program"].map({p: k for k, p in enumerate(progs)})
    sc_ = ax.scatter(x, y, s=400 * table["top"].clip(upper=0.5), c=table["top_of_active"],
                     cmap="viridis", vmin=0, vmax=1, linewidths=0)
    ax.scatter(x, y, s=4, facecolors="none", edgecolors="#999999", linewidths=0.4)
    ax.set_xticks(range(len(STAGES)), STAGES, rotation=45, ha="right")
    ax.set_yticks(range(len(progs)), [p.removeprefix("state:").replace("_", " ") for p in progs])
    ax.set_ylim(len(progs) - 0.5, -0.5)
    ax.set_xlim(-0.5, len(STAGES) - 0.5)
    fig.colorbar(sc_, ax=ax, shrink=0.4, label="share of its active cells\nwith it on top")
    for share in (0.05, 0.2, 0.5):
        ax.scatter([], [], s=400 * share, c="#777777", label=f"{share:.0%}")
    ax.legend(title="cells with it\non top", frameon=False, loc="upper left",
              bbox_to_anchor=(1.0, 0.35), labelspacing=1.2)
    save_report_figure(fig, "cell_programs_top")
    plt.close(fig)


def latex_exclusions(t: pd.DataFrame) -> None:
    head = ("Stage & Library & Cells & p53 & Dissoc. & Heat sh. & YSL ctrl. & Yolk & "
            "Doublet & Low depth & Low compl. \\\\")
    lines = ["% Generated by scripts/26_cell_programs.py; do not edit.",
             "\\begin{longtable}{@{}llrrrrrrrrr@{}}",
             "\\caption{Cells per library (\\protect\\nolinkurl{scripts/26_cell_programs.py}): "
             "excluded as stress cells (top program p53 response, dissociation stress or heat "
             f"shock with a margin of at least {EXCLUDE_MARGIN:g}); with the YSL control active "
             "(in brackets: of these, with an identity program active as well); with at least "
             f"{YOLK_MIN} yolk/YSL-specific transcripts detected; scrublet doublet score of at "
             f"least {DOUBLET_THRESHOLD:g}; depth and complexity more than {MAD_LIMIT} MADs below "
             "the library's median (in brackets: of these, with the EVL on top). Only the stress "
             "cells are excluded.}\\label{tab:cell-exclusions}\\\\",
             "\\toprule", head, "\\midrule", "\\endfirsthead", "\\toprule", head, "\\midrule",
             "\\endhead", "\\bottomrule", "\\endlastfoot"]
    for i, r in enumerate(t.itertuples()):
        lines.append(("\\rowcolor{gray!10}" if i % 2 else "") + " & ".join([
            r.stage.replace("%", "\\%"), r.library.split("_")[-1], f"{r.cells:,}",
            str(r.excluded_p53_response), str(r.excluded_dissociation),
            str(r.excluded_heat_shock),
            f"{r.ysl_active} ({r.ysl_active_with_identity})", str(r.yolk_like),
            str(r.doublets), str(r.low_depth),
            f"{r.low_complexity} ({r.low_complexity_evl})"]) + " \\\\")
    lines.append("\\end{longtable}")
    (REPORT_TABLES / "cell_exclusions.tex").write_text("\n".join(lines) + "\n")


EXCLUDED_COLOURS = {"state:p53_response": "#CC79A7", "control:dissociation": "#D55E00",
                    "state:heat_shock": "#E69F00"}


def atlas(cells: pd.DataFrame, active: pd.DataFrame, table: pd.DataFrame) -> None:
    """Exploration: first the excluded cells per stage, then per program and
    stage its active cells (light) and the cells with it on top (dark; black
    ring: margin >= MARGIN_LEVELS[1])."""
    def frame(ax, title):
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    with PdfPages(FIGURES / "cell_programs_atlas.pdf") as pdf:
        fig, axes = plt.subplots(3, 4, figsize=(12, 9.5))
        for ax, stage in zip(axes.ravel(), STAGES):
            c = cells[cells["stage"] == stage]
            ax.scatter(c["umap_1"], c["umap_2"], s=1, c="#DDDDDD", linewidths=0, rasterized=True)
            y = c[c["yolk_genes"] >= YOLK_MIN]
            ax.scatter(y["umap_1"], y["umap_2"], s=6, c="#009E73", linewidths=0, rasterized=True)
            for q, colour in EXCLUDED_COLOURS.items():
                e = c[c["excluded"] == q]
                ax.scatter(e["umap_1"], e["umap_2"], s=3, c=colour, linewidths=0,
                           rasterized=True)
            frame(ax, f"{stage}: {(c['excluded'] != '').mean():.1%} excluded")
        handles = [plt.Line2D([], [], ls="", marker="o", color=v, label=k.split(":")[1])
                   for k, v in EXCLUDED_COLOURS.items()]
        handles.append(plt.Line2D([], [], ls="", marker="o", color="#009E73",
                                  label=f">= {YOLK_MIN} yolk genes (kept)"))
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
        fig.suptitle("Excluded stress cells on the stage UMAPs")
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        kept = cells[cells["excluded"] == ""]
        for p in program_order(table["program"].unique()):
            stages = [s for s in STAGES if ((table["stage"] == s) & (table["program"] == p)).any()]
            fig, axes = plt.subplots(3, 4, figsize=(12, 9.5))
            for ax, stage in zip(axes.ravel(), STAGES):
                c = kept[kept["stage"] == stage]
                ax.scatter(c["umap_1"], c["umap_2"], s=1, c="#E5E5E5", linewidths=0,
                           rasterized=True)
                if stage not in stages:
                    frame(ax, f"{stage}: not supported")
                    continue
                on = active.loc[c.index, p] == 1
                top = c["top"] == p
                sure = top & (c["margin"] >= MARGIN_LEVELS[1])
                ax.scatter(c.loc[on, "umap_1"], c.loc[on, "umap_2"], s=2, c="#9ECAE1",
                           linewidths=0, rasterized=True)
                ax.scatter(c.loc[top, "umap_1"], c.loc[top, "umap_2"], s=3, c="#08519C",
                           linewidths=0, rasterized=True)
                ax.scatter(c.loc[sure, "umap_1"], c.loc[sure, "umap_2"], s=3, c="black",
                           linewidths=0, rasterized=True)
                r = table[(table["stage"] == stage) & (table["program"] == p)].iloc[0]
                frame(ax, f"{stage}: active {r['active']:.0%}, top {r['top']:.0%}")
            handles = [plt.Line2D([], [], ls="", marker="o", color=v, label=k) for k, v in (
                ("active", "#9ECAE1"), ("top", "#08519C"),
                (f"top, margin >= {MARGIN_LEVELS[1]}", "black"))]
            fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
            fig.suptitle(f"{p}: active vs top (kept cells; shares: median library)")
            fig.tight_layout(rect=(0, 0.04, 1, 0.97))
            pdf.savefig(fig, dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    ensure_dirs()
    if sys.argv[1:] != ["plot"]:
        compute()
    plot()
