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
    maternal and blastomere states) defined at the stage, whether supported
    (script 17) or not: a program can mark a small group of cells (e.g.
    progenitors) without being coherent over the whole library. Top =
    the active program with the largest z; margin = z of the top minus the
    largest z of the other competitors, the latter floored at 0 (= the null
    mean; with a single competitor the margin is the top's own z).
  * Chance level: N_RANDOM random sets of the stage's heterogeneous genes that
    are in no program (sizes drawn from the identity programs' sizes), scored
    alike; how often they are active, and how often one would be the top
    program if they competed.
  * Reliability: every program's genes are split at random into two halves,
    and the top program is called from each half alone; the halves agree
    exactly, or up to a derivation (one derives from the other, directly or
    not; derives_from of curated_programs.tsv), by the margin of the full call.

Exclusion:
  * Stress: the same competition with the stress states added (p53 response,
    dissociation stress, heat shock; only where supported, to exclude no more
    than needed). A cell is excluded if a
    stress state is its top program with a margin of at least EXCLUDE_MARGIN.
  * Yolk/YSL (nothing excluded): the YSL control of script 16 is scored as a
    program (not competing), and the yolk- and YSL-specific transcripts
    (YOLK_GENES) are counted per cell.
  * Quality (nothing excluded): depth (log UMIs vs the library median, in
    MADs), complexity (log detected genes given log UMIs, residual in MADs)
    and scrublet doublet scores per library (DOUBLET_THRESHOLD).

Stage UMAPs: the aligned cells of a stage (centred residuals per library) on
the stage's heterogeneous genes without EXCLUDED_GENE_PROGRAMS; PCA, UMAP.
Also used by script 19. Program trajectories: for every identity program, the
kept cells in which it is active, at any stage, on a UMAP of their aligned
residuals (genes heterogeneous at any stage, without EXCLUDED_GENE_PROGRAMS;
centred per library over all its cells), coloured by the call and by stage.
Centring removes the stage averages, so the stages separate only where the
cells differ; a picture, not a time line.

The calls of every library are cached (data/processed/cache/26); increase
CACHE_VERSION when a code change changes them. `26_cell_programs.py plot`
redraws the figures and report table from the saved tables.

Outputs:
  results/tables/cell_programs_cells.tsv.gz   cell: stage, library, UMAP,
      top, z_top, margin, active identity programs, stress top and margin,
      excluded (reason), YSL control active, yolk genes, quality measures
  results/tables/cell_programs.tsv            stage x program (kept cells,
      median library): supported, active, top, top among the active, median
      margin
  results/tables/cell_programs_chance.tsv     per library: random sets active,
      on top if competing
  results/tables/cell_programs_zfin{,_stages}.tsv   top programs vs the ZFIN
      windows: per program (onset vs the window) and per stage (top programs
      inside their window, observed vs expected)
  results/tables/cell_programs_reliability.tsv   split-half agreement by
      stage and margin
  results/tables/cell_exclusions.tsv          per library
  results/tables/cell_programs_winners.tsv    stage x program: in its active
      cells without it on top, the relation of the top program to it
  data/processed/cell_programs_{z,active}.tsv.gz   cell x program z and
      active calls (all scored programs; empty where not scored)
  report/figures/cell_programs.pdf, report/figures/cell_programs_top.pdf,
  report/figures/cell_programs_winners.pdf,
  report/tables/cell_exclusions.tex, results/figures/cell_programs_atlas.pdf,
  results/figures/cell_programs_trajectories.pdf (UMAPs cached in
      data/processed/cell_programs_trajectories.tsv.gz)
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

CACHE_VERSION = 2
SEED = 0
N_PERM = 20
N_RANDOM = 10
MIN_TRAJECTORY = 50  # active cells, at least, for a trajectory UMAP
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
ONSET_SHARE = 0.01  # on top in this share of the cells (median library): the program has started
MIN_LOST = 10  # active-but-not-top cells of a library, at least, for the winner shares
RELATIONS = ["descendant", "ancestor", "sibling", "unrelated"]
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
TRAJECTORIES = PROCESSED / "cell_programs_trajectories.tsv.gz"
ZFIN_WINDOWS = TABLES / "curated_zfin_windows.tsv"


def identity(program: str) -> bool:
    return program in IDENTITY_STATES or not program.startswith(NON_IDENTITY)


def stage_programs(programs: pd.DataFrame, support: pd.DataFrame, stage: str) -> dict:
    """Gene lists of the identity programs of a stage (and which of them are
    supported), the supported stress programs, the YSL control and the genes
    of EXCLUDED_GENE_PROGRAMS."""
    ok = set(support.loc[(support["stage"] == stage) & support["supported"].astype(bool),
                         "program"])
    p = programs[(programs["stage"] == stage) & programs["kept"]
                 & programs["set"].isin(["curated", "controls"])]
    genes = p.groupby("program")["gene"].agg(list)
    ident = [q for q in genes.index if identity(q)
             and q in set(p.loc[p["set"] == "curated", "program"])]
    stress = [q for q in STRESS if q in ok and q in genes.index]
    excluded = sorted(set(p.loc[p["program"].isin(EXCLUDED_GENE_PROGRAMS), "gene"]))
    return dict(identity={q: genes[q] for q in ident}, stress={q: genes[q] for q in stress},
                ysl=genes.get(YSL, []), excluded=excluded,
                supported=[q for q in ident if q in ok])


def random_sets(programs: pd.DataFrame, progs: dict, het: set, s: int) -> dict:
    """N_RANDOM sets of heterogeneous genes of the stage that are in no
    program, sizes drawn from the identity programs' sizes."""
    rng = np.random.default_rng([SEED, s, 1])
    in_any = set(programs.loc[programs["stage"] == STAGES[s], "gene"])
    pool = np.array(sorted(het - in_any - set(progs["excluded"])))
    sizes = rng.choice([len(g) for g in progs["identity"].values()], N_RANDOM)
    return {f"random:{k}": sorted(rng.choice(pool, min(size, len(pool)), replace=False))
            for k, size in enumerate(sizes)}


def library_calls(counts, var_names: pd.Index, het: set, progs: dict, seed) -> dict:
    """Smoothed scores against the permutation null of one library: z and
    active of every scored program, and the top identity program of each
    random half of the identity programs' genes."""
    rng = np.random.default_rng(seed)
    n = counts.shape[0]
    scored = {**progs["identity"], **progs["stress"], **progs["random"]}
    if progs["ysl"]:
        scored[YSL] = progs["ysl"]
    genes = sorted({g for gs in scored.values() for g in gs})
    graph = sorted(het - set(genes) - set(progs["excluded"]))
    pcs = PCA(min(N_PCS, n - 1, len(graph) - 1), random_state=SEED).fit_transform(
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
    cells, zs, acts, halves, chance = [], [], [], [], []
    for s, stage in enumerate(STAGES):
        sub = adata[adata.obs["stage"] == stage]
        het = set(het_table.loc[(het_table["stage"] == stage) & het_table["heterogeneous"],
                                "gene"])
        progs = stage_programs(programs, support, stage)
        progs["random"] = random_sets(programs, progs, het, s)
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
            rand = list(progs["random"])
            r_top = top_program(z[ident + rand], act[ident + rand])[0]
            chance.append(dict(stage=stage, library=lib, cells=len(rows),
                               random_active=act[rand].mean().mean(),
                               random_active_max=act[rand].mean().max(),
                               random_top=np.isin(r_top, rand).mean(),
                               top=(top != "").mean()))
            zs.append(z.drop(columns=rand))
            acts.append(act.drop(columns=rand))
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
    winners(cells, pd.concat(acts), programs, support, anc).to_csv(
        TABLES / "cell_programs_winners.tsv", sep="\t", index=False, float_format="%.3f")
    pd.DataFrame(chance).to_csv(TABLES / "cell_programs_chance.tsv", sep="\t", index=False,
                                float_format="%.4f")
    stages_zfin, programs_zfin = zfin_windows(cells, program_table(cells, pd.concat(acts),
                                                                   programs, support))
    stages_zfin.to_csv(TABLES / "cell_programs_zfin_stages.tsv", sep="\t", index=False,
                       float_format="%.3f")
    programs_zfin.to_csv(TABLES / "cell_programs_zfin.tsv", sep="\t", index=False,
                         float_format="%.3f")
    trajectories(adata, cells, pd.concat(acts), het_table, programs)


def zfin_windows(cells: pd.DataFrame, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top programs vs the ZFIN windows (stages with >= 3 recorded genes, script 16).
    Per stage and library: of the kept cells whose top program has a window, the
    share whose top program's window contains the stage, vs the share expected
    if the top program were drawn uniformly from the stage's competitors with a
    window (mean over libraries; also for supported and unsupported tops). Per
    program: onset = the first stage at which it is on top in >= ONSET_SHARE of
    the cells (median library) at that and the next stage, vs the window's
    first stage (in stages), and the share of its top cells inside the window."""
    idx = {st: k for k, st in enumerate(STAGES)}
    win = pd.read_csv(ZFIN_WINDOWS, sep="\t").set_index("program")
    first, last = win["first"].map(idx), win["last"].map(idx)

    def inside(p, st):
        return p in win.index and first[p] <= idx[st] <= last[p]
    kept = cells[cells["excluded"] == ""]
    rows = []
    for (stage, lib), c in kept.groupby(["stage", "library"], sort=False):
        t = table[table["stage"] == stage]
        comp = [p for p in t["program"] if p in win.index]
        sup = set(t.loc[t["supported"].astype(bool), "program"])
        tops = c["top"][c["top"].isin(comp)]
        hit = tops.map(lambda p: inside(p, stage))
        rows.append(dict(stage=stage, library=lib, cells=len(tops),
                         observed=hit.mean() if len(tops) else np.nan,
                         expected=np.mean([inside(p, stage) for p in comp]),
                         supported=hit[tops.isin(sup)].mean() if tops.isin(sup).any() else np.nan,
                         unsupported=hit[~tops.isin(sup)].mean() if (~tops.isin(sup)).any()
                         else np.nan, unsupported_share=(~tops.isin(sup)).mean()))
    stages = pd.DataFrame(rows).groupby("stage", sort=False).mean(numeric_only=True) \
        .reindex(STAGES).reset_index()
    counts = kept[kept["top"] != ""].groupby(["top", "stage"]).size()
    prog_rows = []
    for p in program_order([q for q in table["program"].unique() if q in win.index]):
        share = table[table["program"] == p].set_index("stage")["top"].reindex(STAGES)
        on = (share >= ONSET_SHARE).to_numpy()
        both = [k for k in range(len(STAGES)) if on[k] and (k == len(STAGES) - 1 or on[k + 1])]
        n = counts.get(p, pd.Series(dtype=float))
        inw = n[[first[p] <= idx[st] <= last[p] for st in n.index]].sum() / n.sum() \
            if n.sum() else np.nan
        prog_rows.append(dict(program=p, zfin_first=win.loc[p, "first"],
                              zfin_last=win.loc[p, "last"],
                              onset=STAGES[both[0]] if both else "",
                              onset_offset=both[0] - first[p] if both else np.nan,
                              top_cells=int(n.sum()), top_in_window=inw))
    return stages, pd.DataFrame(prog_rows)


def trajectories(adata, cells, active, het_table, programs) -> None:
    """UMAP of the kept cells in which a program is active, per program."""
    kept = cells.index[cells["excluded"] == ""]
    excluded = set(programs.loc[programs["program"].isin(EXCLUDED_GENE_PROGRAMS), "gene"])
    genes = sorted(set(het_table.loc[het_table["heterogeneous"], "gene"]) - excluded)
    idx = np.flatnonzero(adata.var_names.isin(genes))
    X = adata.X.tocsr()
    lib_of = adata.obs["library"].astype(str).to_numpy()
    res = np.zeros((adata.n_obs, len(idx)), dtype=np.float32)
    for lib in sorted(set(lib_of)):
        rows = np.flatnonzero(lib_of == lib)
        res[rows] = pearson_residuals(X[rows], idx, centre=True)
    pos = pd.Series(np.arange(adata.n_obs), index=adata.obs_names)
    out = []
    for p in program_order([c for c in active.columns if identity(c)]):
        on = active.loc[kept, p] == 1
        members = on.index[on.to_numpy()]
        if len(members) < MIN_TRAJECTORY:
            continue

        def embed():
            a = sc.AnnData(np.zeros((len(members), 1), dtype=np.float32))
            a.obsm["X_pca"] = PCA(N_PCS, random_state=SEED).fit_transform(
                res[pos[members].to_numpy()])
            sc.pp.neighbors(a, n_neighbors=UMAP_NEIGHBOURS, use_rep="X_pca", random_state=SEED)
            sc.tl.umap(a, random_state=SEED)
            return a.obsm["X_umap"]
        xy = cached("26", f"trajectory-{p}", digest(CACHE_VERSION, genes, list(members)), embed)
        out.append(pd.DataFrame(dict(program=p, umap_1=xy[:, 0], umap_2=xy[:, 1]),
                                index=pd.Index(members, name="cell")))
        print("trajectory", p, len(members), flush=True)
    pd.concat(out).to_csv(TRAJECTORIES, sep="\t", float_format="%.4g", compression=GZIP)


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
        progs = stage_programs(programs, support, stage)
        for p in progs["identity"]:
            on, tops = a[p] == 1, c["top"] == p
            rows.append(dict(stage=stage, library=lib, program=p,
                             supported=p in progs["supported"], active=on.mean(),
                             top=tops.mean(), top_of_active=tops.sum() / on.sum() if on.any()
                             else np.nan, margin=c.loc[tops, "margin"].median()))
    return pd.DataFrame(rows).groupby(["stage", "program"], sort=False).agg(
        supported=("supported", "first"), active=("active", "median"), top=("top", "median"),
        top_of_active=("top_of_active", "median"), margin=("margin", "median")).reset_index()


def relation(winner: str, program: str, anc: dict) -> str:
    """How the top program of a cell relates to an active program that lost."""
    if program in anc.get(winner, set()):
        return "descendant"
    if winner in anc.get(program, set()):
        return "ancestor"
    if (anc.get(winner, set()) & anc.get(program, set())) - set(IDENTITY_STATES):
        return "sibling"
    return "unrelated"


def winners(cells, active, programs, support, anc) -> pd.DataFrame:
    """For every identity program and stage: in the kept cells in which it is
    active but not on top, which program is on top, by its relation to it
    (descendant, ancestor, sibling = a common ancestor other than the maternal
    and blastomere states, from which every lineage derives; unrelated); shares per
    library (libraries with at least MIN_LOST such cells), median over the
    stage's libraries, and the most frequent winners (pooled)."""
    kept = cells[cells["excluded"] == ""]
    rows = []
    for stage, cs in kept.groupby("stage", sort=False):
        for p in stage_programs(programs, support, stage)["identity"]:
            per_lib, pooled = [], []
            for lib, c in cs.groupby("library"):
                on = active.loc[c.index, p] == 1
                lost = c.loc[on & (c["top"] != p), "top"]
                pooled.append(lost)
                if len(lost) >= MIN_LOST:
                    kind = lost.map(lambda w: relation(w, p, anc))
                    per_lib.append(dict(lost=len(lost) / on.sum(), **{
                        k: (kind == k).mean() for k in RELATIONS}))
            lost = pd.concat(pooled)
            if not per_lib:
                continue
            med = pd.DataFrame(per_lib).median()
            top = lost.value_counts(normalize=True).head(3)
            rows.append(dict(stage=stage, program=p, libraries=len(per_lib), cells=len(lost),
                             not_top=med["lost"], **{k: med[k] for k in RELATIONS},
                             winners=", ".join(f"{w} {x:.0%}" for w, x in top.items())))
    return pd.DataFrame(rows)


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
    fig_winners(pd.read_csv(TABLES / "cell_programs_winners.tsv", sep="\t"))
    latex_exclusions(pd.read_csv(TABLES / "cell_exclusions.tsv", sep="\t"))
    active = pd.read_csv(ACTIVE_TABLE, sep="\t", index_col=0)
    atlas(cells, active, table)
    trajectory_pages(cells, pd.read_csv(TRAJECTORIES, sep="\t", index_col=0), table)


def trajectory_pages(cells: pd.DataFrame, traj: pd.DataFrame, table: pd.DataFrame) -> None:
    """Per program: its active cells (all stages) on their own UMAP, by the
    call (active / top / top with margin >= MARGIN_LEVELS[1]) and by stage."""
    cmap = plt.get_cmap("viridis", len(STAGES))
    with PdfPages(FIGURES / "cell_programs_trajectories.pdf") as pdf:
        for p in program_order(traj["program"].unique()):
            t = traj[traj["program"] == p].join(cells[["stage", "top", "margin"]])
            rng = np.random.default_rng(SEED)
            t = t.iloc[rng.permutation(len(t))]  # no stage drawn systematically on top
            top = (t["top"] == p).to_numpy()
            sure = top & (t["margin"] >= MARGIN_LEVELS[1]).to_numpy()
            size = float(np.clip(3000 / len(t), 0.5, 6))
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
            ax = axes[0]
            for mask, colour, label in ((~top, "#9ECAE1", "active, another program on top"),
                                        (top & ~sure, "#08519C", "on top"),
                                        (sure, "black", f"on top, margin >= {MARGIN_LEVELS[1]}")):
                ax.scatter(t.loc[mask, "umap_1"], t.loc[mask, "umap_2"], s=size, c=colour,
                           linewidths=0, rasterized=True, label=f"{label} ({mask.sum()})")
            ax.legend(frameon=False, fontsize=7, markerscale=3, loc="best")
            ax = axes[1]
            code = t["stage"].map({st: k for k, st in enumerate(STAGES)}).to_numpy()
            im = ax.scatter(t["umap_1"], t["umap_2"], s=size, c=code, cmap=cmap, vmin=-0.5,
                            vmax=len(STAGES) - 0.5, linewidths=0, rasterized=True)
            cb = fig.colorbar(im, ax=ax, ticks=range(len(STAGES)), shrink=0.8)
            cb.ax.set_yticklabels(STAGES, fontsize=7)
            for a in axes:
                a.set_xticks([])
                a.set_yticks([])
            sup = table[(table["program"] == p) & table["supported"].astype(bool)]["stage"]
            per = t["stage"].value_counts().reindex(STAGES).dropna().astype(int)
            fig.suptitle(f"{p}: {len(t)} active cells (kept), supported at: "
                         f"{', '.join(sup) if len(sup) else 'no stage'}\n"
                         + ", ".join(f"{st} {k}" for st, k in per.items()), fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            pdf.savefig(fig, dpi=150)
            plt.close(fig)


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
    win = pd.read_csv(ZFIN_WINDOWS, sep="\t").set_index("program")
    for k, p in enumerate(progs):
        if p in win.index:
            a, b = STAGES.index(win.loc[p, "first"]), STAGES.index(win.loc[p, "last"])
            ax.add_patch(plt.Rectangle((a - 0.45, k - 0.42), b - a + 0.9, 0.84, fill=False,
                                       edgecolor="#BBBBBB", linewidth=0.6, zorder=0))
    sup = table["supported"].astype(bool).to_numpy()
    sc_ = ax.scatter(x, y, s=400 * table["top"].clip(upper=0.5), c=table["top_of_active"],
                     cmap="viridis", vmin=0, vmax=1, linewidths=np.where(sup, 0, 0.7),
                     edgecolors="black")
    ax.scatter(x[sup], y[sup], s=4, facecolors="none", edgecolors="#999999", linewidths=0.4)
    ax.set_xticks(range(len(STAGES)), STAGES, rotation=45, ha="right")
    ax.set_yticks(range(len(progs)), [p.removeprefix("state:").replace("_", " ") for p in progs])
    ax.set_ylim(len(progs) - 0.5, -0.5)
    ax.set_xlim(-0.5, len(STAGES) - 0.5)
    fig.colorbar(sc_, ax=ax, shrink=0.4, label="share of its active cells\nwith it on top")
    for share in (0.05, 0.2, 0.5):
        ax.scatter([], [], s=400 * share, c="#777777", label=f"{share:.0%}")
    ax.scatter([], [], s=400 * 0.2, c="#777777", edgecolors="black", linewidths=0.7,
               label="not supported")
    ax.legend(title="cells with it\non top", frameon=False, loc="upper left",
              bbox_to_anchor=(1.0, 0.35), labelspacing=1.2)
    save_report_figure(fig, "cell_programs_top")
    plt.close(fig)


RELATION_COLOURS = {"descendant": "#0072B2", "ancestor": "#56B4E9", "sibling": "#E69F00",
                    "unrelated": "#BBBBBB"}


def fig_winners(w: pd.DataFrame) -> None:
    """Per program and stage (from shield on): the share of its active cells
    in which it is not on top (bar length), split by how the winner relates to it."""
    w = w[w["stage"].isin(STAGES[STAGES.index("shield"):])]
    progs = program_order(w["program"].unique())
    stages = [s for s in STAGES if s in set(w["stage"])]
    fig, axes = plt.subplots(1, len(stages), figsize=(REPORT_WIDTH_IN, 0.13 * len(progs) + 1.1),
                             sharey=True, constrained_layout=True)
    for ax, stage in zip(axes, stages):
        d = w[w["stage"] == stage].set_index("program").reindex(progs)
        left = np.zeros(len(progs))
        for k in RELATIONS:
            v = (d[k] * d["not_top"]).fillna(0).to_numpy()
            ax.barh(range(len(progs)), v, left=left, color=RELATION_COLOURS[k], height=0.75,
                    label=k)
            left += v
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_title(stage, fontsize=6, rotation=30, ha="left", loc="left")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(progs)),
                       [p.removeprefix("state:").replace("_", " ") for p in progs])
    axes[0].set_ylim(len(progs) - 0.5, -0.5)
    fig.supxlabel("share of the active cells in which the program is not on top", fontsize=7)
    axes[-1].legend(title="top program is a", frameon=False, loc="upper left",
                    bbox_to_anchor=(1.0, 1.0))
    save_report_figure(fig, "cell_programs_winners")
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
                    frame(ax, f"{stage}: no genes measured")
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
                note = "" if r.get("supported", True) else " (unsupported)"
                frame(ax, f"{stage}{note}: active {r['active']:.0%}, top {r['top']:.0%}")
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
