"""Cell populations that replicate across the libraries of a stage.

Every library is clustered on its own, and a cluster is kept as a population
only if an equivalent cluster is found independently in another library.
Clustering a library always yields clusters; a cluster that reflects sampling
noise or one library's peculiarities should not reappear elsewhere.

1. Features. Genes heterogeneous at the stage (script 01b: weak tier in at
   least half of the stage's libraries). Each library's counts become Pearson
   residuals under that library's own NB model, (y - s m) / sqrt(s m +
   phi_t (s m)^2), with phi_t the library's dispersion-mean trend
   (pipeline_utils.nb_overdispersion), clipped at +-RESIDUAL_CLIP, and are
   centred per library. Centring removes the mean differences between
   libraries (batch, a/b, timing) along with differences in composition. One
   PCA of all libraries of the window gives shared axes.
2. Window. A stage with several libraries is compared within itself. A stage
   with one library (dome, shield, 3-somite) is compared with the libraries
   of the adjacent stages (weaker evidence: a match also needs the population
   to persist over ~1 h of development).
3. Clusters. Per library: kNN graph on the shared PCs, Leiden at several
   resolutions.
4. Matching (neighbour voting, as MetaNeighbor; Crow et al. 2018). Every
   cell of library B takes its VOTE_K nearest cells of library A; the vote
   of A's cluster a is the fraction of them in a. For each B cluster b, the
   AUROC of the vote (b's cells vs the rest of B) says how well a picks out
   b. The best b is compared with the runner-up (one-vs-best AUROC: b's
   cells vs the runner-up's cells only), which separates a true counterpart
   from a merely related cluster. a and b match if each is the other's best
   counterpart and both one-vs-best AUROCs are >= MIN_ONE_VS_BEST.
   Populations of a resolution are the connected components of matches
   that contain a cluster of the stage.
5. Consensus over resolutions. Populations of different resolutions that
   share >= CONSENSUS_JACCARD of their cells (of the stage) are the same
   consensus population; nested populations (e.g. mesoderm and its ventral
   part) stay separate. Each cell gets the smallest consensus population
   that contains it ("most specific").
6. Description. Top genes: largest mean residual in the population vs the
   rest of its library. ZFIN: anatomy terms of pre-2018 in situ records at
   the stage that are enriched among the top ZFIN_TOP genes (hypergeometric,
   background: the stage's heterogeneous genes with a localised record).
7. Controls. Negative: arbitrary but spatially coherent clusters (Voronoi
   cells around random seed cells, as many as the real clusters) matched the
   same way, N_NULL_REPS times; matches between them come from the geometry
   alone. Positive: EVL cells (marker score above the 99.9% quantile of
   matched random gene sets).
8. Links between stages. For consecutive stages, joint residual PCA of both
   (genes heterogeneous at either); every cell of the later stage votes for
   the consensus populations of the earlier one (fraction of its VOTE_K
   nearest earlier cells in the population), and vice versa. AUROC of the
   vote for each later population; the best earlier population is its
   predecessor.

Usage: 14_replicated_populations.py [stage ...] [--seed N] [--tag T]
(default: all stages, seed 0; a tag is appended to the output names)

Outputs (results/tables/replicated_populations*):
  _summary.tsv        per stage and resolution: clusters, matches, null
  _matches.tsv        all reciprocal-best cluster pairs
  _resolution.tsv     populations per resolution
  .tsv                consensus populations (genes, ZFIN terms, EVL)
  _cells.tsv.gz       cell -> clusters, populations, consensus, UMAP
  _links.tsv          consensus populations of consecutive stages
  results/figures/replicated_populations.pdf (UMAPs per stage)
"""
from __future__ import annotations

import argparse
import importlib
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import hypergeom, rankdata
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from pipeline_utils import BATCH_COLORS, RESIDUAL_CLIP, ROOT, STAGES, TABLES, ensure_dirs, \
    load_dropseq_counts, nb_overdispersion, pearson_residuals  # noqa: F401

warnings.filterwarnings("ignore", category=FutureWarning)
zfin_check = importlib.import_module("01c_heterogeneity_zfin")

FIGURES = ROOT / "results" / "figures"
N_PCS = 30
K = 15  # neighbours for the clustering graph
VOTE_K = 10  # neighbours in the other library for voting
RESOLUTIONS = [0.5, 1.0, 2.0, 4.0]
MIN_ONE_VS_BEST = 0.8
CONSENSUS_JACCARD = 0.5
N_NULL_REPS = 3
TOP_GENES = 8
ZFIN_TOP = 20
ZFIN_MAX_P = 1e-3
ZFIN_MIN_GENES = 3
MIN_LINK_AUROC = 0.9
# Robust: found at >= 2 resolutions (these reproduced with another seed in
# 95-97% of cases, populations of a single resolution in 56%).
ROBUST_RESOLUTIONS = 2

EVL = ["KRT4", "KRT8", "KRT18", "CLDNE", "CYT1", "KRT5", "ANXA1C", "CLDNB"]
N_EVL_NULL = 20


# --- features ------------------------------------------------------------------

def residuals(counts, genes: np.ndarray) -> np.ndarray:
    """Centred, clipped Pearson residuals of `genes` (column mask) in one library."""
    return pearson_residuals(counts, genes, centre=True)


def shared_pcs(adata, libs: list[str], genes: np.ndarray, seed: int):
    """Residuals and shared PCs of every library (genes: column mask)."""
    X = adata.X
    lib_of = adata.obs["library"].to_numpy()
    res = {lib: residuals(X[lib_of == lib], genes) for lib in libs}
    stacked = np.vstack([res[lib] for lib in libs])
    n = min(N_PCS, stacked.shape[1] - 1, stacked.shape[0] - 1)
    pca = PCA(n, random_state=seed).fit(stacked)
    return res, {lib: pca.transform(res[lib]) for lib in libs}


def evl_cells(sub, rng, seed: int) -> np.ndarray:
    """EVL marker score above the 99.9% quantile of matched random gene sets."""
    ln = sub.copy()
    sc.pp.normalize_total(ln, target_sum=1e4)
    sc.pp.log1p(ln)
    present = [g for g in EVL if g in ln.var_names]
    mean = np.asarray(ln.X.mean(axis=0)).ravel()
    bins = pd.qcut(pd.Series(mean).rank(method="first"), 50, labels=False).to_numpy()
    sc.tl.score_genes(ln, present, score_name="s", random_state=seed)
    null = []
    for r in range(N_EVL_NULL):
        fake = {rng.choice(ln.var_names[bins == bins[ln.var_names.get_loc(g)]])
                for g in present}
        sc.tl.score_genes(ln, list(fake), score_name="n", random_state=r)
        null.append(ln.obs["n"].to_numpy())
    return ln.obs["s"].to_numpy() > np.quantile(np.concatenate(null), 0.999)


# --- clustering and matching -----------------------------------------------------

def leiden(pcs: np.ndarray, resolution: float, seed: int) -> np.ndarray:
    a = sc.AnnData(np.zeros((len(pcs), 1), dtype=np.float32))
    a.obsm["X_pca"] = pcs
    sc.pp.neighbors(a, n_neighbors=min(K, len(pcs) - 1), use_rep="X_pca", random_state=seed)
    sc.tl.leiden(a, resolution=resolution, flavor="igraph", n_iterations=2,
                 random_state=seed)
    return a.obs["leiden"].astype(int).to_numpy()


def voronoi(pcs: np.ndarray, n_clusters: int, rng) -> np.ndarray:
    """Arbitrary, spatially coherent clusters: nearest of n random seed cells."""
    seeds = pcs[rng.choice(len(pcs), size=n_clusters, replace=False)]
    d = ((pcs[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    n1 = positive.sum()
    n0 = len(positive) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(score)
    return (ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auroc_per_group(score: np.ndarray, groups: np.ndarray, names: np.ndarray) -> np.ndarray:
    """AUROC of `score` for every group (one vs rest)."""
    ranks = rankdata(score)
    n = len(score)
    out = np.full(len(names), np.nan)
    for j, g in enumerate(names):
        pos = groups == g
        n1 = pos.sum()
        if 0 < n1 < n:
            out[j] = (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * (n - n1))
    return out


def best_counterparts(pcs_a, labels_a, pcs_b, labels_b) -> pd.DataFrame:
    """For each A cluster: best and runner-up B cluster, one-vs-rest and
    one-vs-best AUROC of A's votes."""
    clusters_a = np.unique(labels_a)
    clusters_b = np.unique(labels_b)
    idx = NearestNeighbors(n_neighbors=min(VOTE_K, len(pcs_a))).fit(pcs_a).kneighbors(
        pcs_b, return_distance=False)
    nbr = labels_a[idx]
    rows = []
    for a in clusters_a:
        v = (nbr == a).mean(axis=1)
        rest = auroc_per_group(v, labels_b, clusters_b)
        order = np.argsort(-np.nan_to_num(rest, nan=-1))
        b1 = clusters_b[order[0]]
        if len(clusters_b) > 1:
            b2 = clusters_b[order[1]]
            pair = np.isin(labels_b, [b1, b2])
            best = auroc(v[pair], labels_b[pair] == b1)
        else:
            b2, best = -1, np.nan
        rows.append(dict(cluster=a, best=b1, runner_up=b2, auroc_rest=rest[order[0]],
                         auroc_best=best))
    return pd.DataFrame(rows).set_index("cluster")


def match(pairs: list[tuple[str, str]], pcs: dict, labels: dict) -> pd.DataFrame:
    """Reciprocal best matches between the clusters of the given library pairs."""
    rows = []
    for A, B in pairs:
        ab = best_counterparts(pcs[A], labels[A], pcs[B], labels[B])
        ba = best_counterparts(pcs[B], labels[B], pcs[A], labels[A])
        for a, r in ab.iterrows():
            back = ba.loc[r["best"]]
            rows.append(dict(library_a=A, cluster_a=a, library_b=B, cluster_b=r["best"],
                             reciprocal=back["best"] == a,
                             auroc_rest_ab=r["auroc_rest"], auroc_best_ab=r["auroc_best"],
                             auroc_rest_ba=back["auroc_rest"],
                             auroc_best_ba=back["auroc_best"]))
    m = pd.DataFrame(rows)
    # A missing runner-up (B has one cluster) cannot confirm a match.
    m["match"] = (m["reciprocal"] & (m["auroc_best_ab"].fillna(0) >= MIN_ONE_VS_BEST)
                  & (m["auroc_best_ba"].fillna(0) >= MIN_ONE_VS_BEST))
    return m


def components(m: pd.DataFrame, focal: list[str]) -> list[set]:
    """Connected components of matches that contain a cluster of a focal library."""
    g = nx.Graph()
    for r in m[m["match"]].itertuples():
        g.add_edge((r.library_a, r.cluster_a), (r.library_b, r.cluster_b))
    comps = [c for c in nx.connected_components(g) if any(lib in focal for lib, _ in c)]
    return sorted(comps, key=lambda c: -len(c))


def cluster_profiles(res: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Mean residual in the cluster minus the rest of the library (clusters x genes)."""
    out = {}
    for c in np.unique(labels):
        inside = labels == c
        out[c] = res[inside].mean(axis=0) - res[~inside].mean(axis=0)
    return pd.DataFrame(out).T


# --- ZFIN -----------------------------------------------------------------------------

def zfin_terms(records: pd.DataFrame, stage: str, background: set[str]):
    """Localised anatomy term -> genes with an in situ record at the stage."""
    hpf = zfin_check.ATLAS_HPF[stage]
    at = records[(records["start_hpf"] <= hpf) & (records["stop"] > hpf)
                 & (records["kind"] == "restricted") & records["gene"].isin(background)]
    return at.groupby("term")["gene"].agg(set)


def enriched_terms(top: list[str], terms: pd.Series) -> str:
    universe = set().union(*terms) if len(terms) else set()
    hits = [g for g in top if g in universe]
    if not hits:
        return ""
    rows = []
    for term, genes in terms.items():
        k = len(genes & set(hits))
        if k >= ZFIN_MIN_GENES:
            p = hypergeom.sf(k - 1, len(universe), len(genes), len(hits))
            if p <= ZFIN_MAX_P:
                rows.append((p, term, k, len(hits)))
    rows.sort()
    return "; ".join(f"{t} ({k}/{n})" for _, t, k, n in rows[:3])


# --- one stage -----------------------------------------------------------------------

def window(stage: str, libs_of: dict) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Focal libraries, all libraries of the window and the pairs to compare."""
    focal = libs_of[stage]
    if len(focal) > 1:
        pairs = [(a, b) for i, a in enumerate(focal) for b in focal[i + 1:]]
        return focal, focal, pairs
    i = STAGES.index(stage)
    near = [lib for s in STAGES[max(i - 1, 0):i + 2] if s != stage for lib in libs_of[s]]
    return focal, focal + near, [(focal[0], b) for b in near]


def run_stage(adata, stage: str, libs_of: dict, genes_table: pd.DataFrame,
              zfin: pd.DataFrame | None, seed: int, embed=None) -> dict:
    """Populations of one stage. `embed(sub, libs, stage)`, if given, replaces
    the residual PCs by another feature space: it returns per-library
    features for the profiles, per-library coordinates and the feature names
    (script 18: program scores)."""
    rng = np.random.default_rng([seed, STAGES.index(stage)])
    focal, libs, pairs = window(stage, libs_of)
    sub = adata[adata.obs["library"].isin(libs)]
    het = genes_table.loc[(genes_table["stage"] == stage) & genes_table["heterogeneous"], "gene"]
    lib_of = sub.obs["library"].to_numpy()
    if embed is None:
        mask = sub.var_names.isin(het)
        gene_names = sub.var_names[mask]
        res, pcs = shared_pcs(sub, libs, mask, seed)
    else:
        res, pcs, gene_names = embed(sub, libs, stage)
        gene_names = pd.Index(gene_names)
    batch = sub.obs.groupby("library", observed=True)["sequencing_batch"].first()
    in_focal = np.isin(lib_of, focal)

    cells = pd.DataFrame(index=sub.obs_names[in_focal])
    cells["stage"], cells["library"] = stage, lib_of[in_focal]
    cells["batch"] = cells["library"].map(batch)
    cells["evl"] = False
    for lib in focal:
        cells.loc[cells["library"] == lib, "evl"] = evl_cells(
            sub[lib_of == lib].copy(), rng, seed)
    terms = zfin_terms(zfin, stage, set(het)) if zfin is not None else pd.Series(dtype=object)

    match_rows, pop_rows, summary, members_of = [], [], [], {}
    for resolution in RESOLUTIONS:
        tag = f"{resolution:g}"
        labels = {lib: leiden(pcs[lib], resolution, seed) for lib in libs}
        m = match(pairs, pcs, labels).assign(stage=stage, resolution=resolution)
        match_rows.append(m)
        comps = components(m, focal)
        null = [match(pairs, pcs, {lib: voronoi(pcs[lib], len(np.unique(labels[lib])), rng)
                                   for lib in libs}) for _ in range(N_NULL_REPS)]
        cells[f"cluster_{tag}"] = -1
        cells[f"population_{tag}"] = -1
        for lib in focal:
            cells.loc[cells["library"] == lib, f"cluster_{tag}"] = labels[lib]
        profiles = {lib: cluster_profiles(res[lib], labels[lib]) for lib in focal}
        for p, comp in enumerate(comps):
            node_libs = {lib for lib, _ in comp}
            inside = np.zeros(len(cells), dtype=bool)
            for lib, c in comp:
                if lib in focal:
                    inside |= ((cells["library"] == lib) & (cells[f"cluster_{tag}"] == c)).to_numpy()
            cells.loc[inside, f"population_{tag}"] = p
            profile = pd.concat([profiles[lib].loc[[c]] for lib, c in comp if lib in focal]).mean()
            order = np.argsort(-profile.to_numpy())
            pid = f"r{tag}_p{p}"
            members_of[pid] = inside
            pop_rows.append(dict(
                stage=stage, resolution=resolution, population=pid,
                libraries=len(node_libs), focal_libraries=len(node_libs & set(focal)),
                batches=len({batch[lib] for lib in node_libs}),
                adjacent_only=len(focal) == 1, clusters=len(comp), cells=int(inside.sum()),
                share=inside.mean(),
                evl_precision=cells.loc[inside, "evl"].mean(),
                evl_recall=cells.loc[inside, "evl"].sum() / max(cells["evl"].sum(), 1),
                top_genes=",".join(gene_names[order[:TOP_GENES]].str.lower()),
                zfin=enriched_terms(list(gene_names[order[:ZFIN_TOP]]), terms)))
        n_clusters = sum(len(np.unique(labels[lib])) for lib in focal)
        null_matches = [int(nm["match"].sum()) for nm in null]
        summary.append(dict(
            stage=stage, resolution=resolution, libraries=len(focal), window=len(libs),
            clusters=n_clusters, populations=len(comps),
            cell_coverage=(cells[f"population_{tag}"] >= 0).mean(),
            matches=int(m["match"].sum()), pairs=len(m),
            null_matches=np.mean(null_matches), null_pairs=np.mean([len(nm) for nm in null]),
            null_populations=np.mean([len(components(nm, focal)) for nm in null])))

    pops = pd.DataFrame(pop_rows)
    consensus, cells = build_consensus(pops, members_of, cells)
    consensus["stage"] = stage
    cells = add_umap(cells, pcs, libs, focal, lib_of, seed)
    return dict(cells=cells, matches=pd.concat(match_rows), pops=pops, consensus=consensus,
                summary=pd.DataFrame(summary))


def build_consensus(pops: pd.DataFrame, members_of: dict, cells: pd.DataFrame):
    """Merge populations of different resolutions with the same cells."""
    if pops.empty:  # no replicated population (e.g. too few programs in 18)
        cells["consensus"] = ""
        cells["covered_any"] = False
        return pd.DataFrame(columns=["consensus", "representative", "resolutions",
                                     "n_resolutions", "population", "cells"]), cells
    g = nx.Graph()
    g.add_nodes_from(pops["population"])
    ids = list(pops["population"])
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            x, y = members_of[a], members_of[b]
            union = (x | y).sum()
            if union and (x & y).sum() / union >= CONSENSUS_JACCARD:
                g.add_edge(a, b)
    p = pops.set_index("population")
    rows = []
    for comp in nx.connected_components(g):
        group = p.loc[sorted(comp)]
        rep = group.sort_values(["focal_libraries", "libraries", "cells"],
                                ascending=False).index[0]
        r = p.loc[rep]
        rows.append(dict(representative=rep, resolutions=",".join(
            f"{x:g}" for x in sorted(group["resolution"].unique())),
            n_resolutions=group["resolution"].nunique(), **r.drop("stage").to_dict()))
    consensus = pd.DataFrame(rows).sort_values("cells", ascending=False).reset_index(drop=True)
    consensus.insert(0, "consensus", [f"c{i}" for i in range(len(consensus))])
    # Most specific: the smallest consensus population containing the cell.
    label = np.full(len(cells), "", dtype=object)
    size = np.full(len(cells), np.inf)
    for r in consensus.itertuples():
        inside = members_of[r.representative]
        better = inside & (r.cells < size)
        label[better], size[better] = r.consensus, r.cells
    cells["consensus"] = label
    cells["covered_any"] = cells[[c for c in cells if c.startswith("population_")]].ge(0).any(axis=1)
    return consensus, cells


def add_umap(cells, pcs, libs, focal, lib_of, seed):
    emb = np.vstack([pcs[lib] for lib in focal])
    a = sc.AnnData(np.zeros((len(emb), 1), dtype=np.float32))
    a.obsm["X_pca"] = emb
    sc.pp.neighbors(a, n_neighbors=K, use_rep="X_pca", random_state=seed)
    sc.tl.umap(a, random_state=seed)
    # cells are in library order of `focal` within the stage subset
    order = np.concatenate([np.flatnonzero(cells["library"].to_numpy() == lib) for lib in focal])
    xy = np.empty_like(a.obsm["X_umap"])
    xy[order] = a.obsm["X_umap"]
    cells["umap_1"], cells["umap_2"] = xy.T
    return cells


# --- links between consecutive stages -------------------------------------------------

def link_stages(adata, libs_of: dict, genes_table: pd.DataFrame, out: dict,
                seed: int) -> pd.DataFrame:
    rows = []
    stages = [s for s in STAGES if s in out]
    for early, late in zip(stages, stages[1:]):
        if STAGES.index(late) - STAGES.index(early) != 1:
            continue
        libs = libs_of[early] + libs_of[late]
        sub = adata[adata.obs["library"].isin(libs)]
        het = genes_table.loc[genes_table["stage"].isin([early, late])
                              & genes_table["heterogeneous"], "gene"]
        _, pcs = shared_pcs(sub, libs, sub.var_names.isin(het), seed)
        lib_of = sub.obs["library"].to_numpy()
        stage_of = {early: np.vstack([pcs[l] for l in libs_of[early]]),
                    late: np.vstack([pcs[l] for l in libs_of[late]])}
        names = {s: np.concatenate([sub.obs_names[lib_of == l] for l in libs_of[s]])
                 for s in (early, late)}
        members = {}
        for s in (early, late):
            c = out[s]["cells"]
            cons = out[s]["consensus"]
            for r in cons.itertuples():
                tag, pop = r.representative[1:].split("_p")
                pop = int(pop)
                inside = c[f"population_{tag}"].reindex(names[s]).to_numpy() == pop
                members[(s, r.consensus)] = inside
        for src, dst, direction in [(early, late, "forward"), (late, early, "backward")]:
            idx = NearestNeighbors(n_neighbors=VOTE_K).fit(stage_of[src]).kneighbors(
                stage_of[dst], return_distance=False)
            for (s, P), inside in members.items():
                if s != src:
                    continue
                v = inside[idx].mean(axis=1)
                for (t, Q), target in members.items():
                    if t != dst or target.sum() == 0:
                        continue
                    rows.append(dict(early=early, late=late, direction=direction,
                                     source=P, target=Q, auroc=auroc(v, target)))
    links = pd.DataFrame(rows)
    if links.empty:
        return links
    f = links[links.direction == "forward"].rename(columns={"source": "early_pop",
                                                            "target": "late_pop",
                                                            "auroc": "auroc_forward"})
    b = links[links.direction == "backward"].rename(columns={"source": "late_pop",
                                                             "target": "early_pop",
                                                             "auroc": "auroc_backward"})
    both = f.drop(columns="direction").merge(
        b.drop(columns="direction"), on=["early", "late", "early_pop", "late_pop"])
    both["auroc_min"] = both[["auroc_forward", "auroc_backward"]].min(axis=1)
    both["best_for_late"] = both.groupby(["early", "late", "late_pop"])["auroc_min"] \
        .transform("max") == both["auroc_min"]
    both["link"] = both["best_for_late"] & (both["auroc_min"] >= MIN_LINK_AUROC)
    return both


# --- figures -----------------------------------------------------------------------------

def short_name(r) -> str:
    return r.top_genes.split(",")[0]


def plot(out: dict, pdf: PdfPages) -> None:
    for stage, r in out.items():
        cells = r["cells"]
        cons = r["consensus"]
        libs = sorted(cells["library"].unique())
        fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
        ax = axes[0, 0]
        markers = ["o", "s", "^", "D"]
        for i, lib in enumerate(libs):
            c = cells[cells["library"] == lib]
            ax.scatter(c["umap_1"], c["umap_2"], s=2, color=BATCH_COLORS[c["batch"].iloc[0]],
                       marker=markers[i % 4], alpha=0.5, label=lib, rasterized=True)
        ax.legend(markerscale=4, fontsize=6, loc="best")
        ax.set_title(f"{stage}: libraries (colour = batch)")
        ax = axes[0, 1]
        ax.scatter(cells["umap_1"], cells["umap_2"], s=2, c="#DDDDDD", rasterized=True)
        e = cells[cells["evl"]]
        ax.scatter(e["umap_1"], e["umap_2"], s=4, c="#0072B2", rasterized=True)
        ax.set_title(f"EVL marker cells ({len(e)})")
        ax = axes[0, 2]
        ax.scatter(cells["umap_1"], cells["umap_2"], s=2, c="#DDDDDD", rasterized=True)
        cmap = plt.get_cmap("tab20")
        for i, row in enumerate(cons.itertuples()):
            c = cells[cells["consensus"] == row.consensus]
            if len(c):
                ax.scatter(c["umap_1"], c["umap_2"], s=2, color=cmap(i % 20), rasterized=True)
                ax.annotate(f"{row.consensus} {short_name(row)}",
                            (c["umap_1"].median(), c["umap_2"].median()), fontsize=6,
                            ha="center", weight="bold")
        ax.set_title(f"consensus, most specific ({(cells['consensus'] != '').mean():.0%})")
        ax = axes[0, 3]
        ax.axis("off")
        text = "\n".join(f"{row.consensus} [{row.resolutions}; {row.focal_libraries}/"
                         f"{row.libraries} lib] {','.join(row.top_genes.split(',')[:5])}"
                         for row in cons.itertuples())
        ax.text(0, 1, text, fontsize=5, va="top", family="monospace")
        for ax, resolution in zip(axes[1], RESOLUTIONS):
            col = f"population_{resolution:g}"
            ax.scatter(cells["umap_1"], cells["umap_2"], s=2, c="#DDDDDD", rasterized=True)
            pops = sorted(p for p in cells[col].unique() if p >= 0)
            for p in pops:
                c = cells[cells[col] == p]
                ax.scatter(c["umap_1"], c["umap_2"], s=2, color=cmap(p % 20), rasterized=True)
            ax.set_title(f"resolution {resolution:g}: {len(pops)} populations "
                         f"({(cells[col] >= 0).mean():.0%} of cells)")
        for ax in axes.ravel()[:3].tolist() + axes[1].tolist():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        pdf.savefig(fig, dpi=200)
        plt.close(fig)


def tracks(name: str) -> None:
    """Robust populations linked across consecutive stages (table and figure).

    A later population's predecessor is the robust earlier population with
    the highest AUROC (minimum of the two directions); it is a link if that
    AUROC is >= MIN_LINK_AUROC. Tracks are the connected components of links.
    """
    cons = pd.read_csv(TABLES / f"{name}.tsv", sep="\t")
    links = pd.read_csv(TABLES / f"{name}_links.tsv", sep="\t")
    cells = pd.read_csv(TABLES / f"{name}_cells.tsv.gz", sep="\t", index_col=0,
                        low_memory=False)
    cons["robust"] = cons["n_resolutions"] >= ROBUST_RESOLUTIONS
    robust = cons[cons["robust"]].copy()
    keys = set(zip(robust["stage"], robust["consensus"]))
    links = links[[(a, x) in keys and (b, y) in keys for a, x, b, y in
                   zip(links["early"], links["early_pop"], links["late"], links["late_pop"])]]
    best = links.loc[links.groupby(["late", "late_pop"])["auroc_min"].idxmax()]
    best = best[best["auroc_min"] >= MIN_LINK_AUROC]
    g = nx.Graph()
    g.add_nodes_from(keys)
    for r in best.itertuples():
        g.add_edge((r.early, r.early_pop), (r.late, r.late_pop), auroc=r.auroc_min)
    stage_pos = {s: i for i, s in enumerate(STAGES)}
    comps = sorted(nx.connected_components(g),
                   key=lambda c: (min(stage_pos[s] for s, _ in c),
                                  -len(c), -max(stage_pos[s] for s, _ in c)))
    pred = {(r.late, r.late_pop): (r.early_pop, r.auroc_min) for r in best.itertuples()}
    robust = robust.set_index(["stage", "consensus"])
    # Tracks are packed into rows (first fit); a row is free one stage after
    # a track ends, leaving room for the labels.
    rows, y_of, free_from = [], {}, []
    for t, comp in enumerate(comps):
        per_stage = {}
        for node in sorted(comp, key=lambda n: -robust.loc[n, "cells"]):
            per_stage.setdefault(node[0], []).append(node)
        height = max(len(v) for v in per_stage.values())
        first = min(stage_pos[s] for s in per_stage)
        last = max(stage_pos[s] for s in per_stage)
        y = next((i for i in range(len(free_from) + 1)
                  if all(free_from[j] <= first for j in range(i, min(i + height, len(free_from))))),
                 len(free_from))
        free_from += [0] * max(0, y + height - len(free_from))
        for j in range(y, y + height):
            free_from[j] = last + 1
        for stage, nodes in per_stage.items():
            for j, node in enumerate(nodes):
                y_of[node] = y + j
        for node in comp:
            r = robust.loc[node]
            p = pred.get(node, (None, np.nan))
            rows.append(dict(track=t, stage=node[0], consensus=node[1],
                             cells=r["cells"], share=r["share"],
                             focal_libraries=r["focal_libraries"], libraries=r["libraries"],
                             resolutions=r["resolutions"], predecessor=p[0],
                             link_auroc=p[1], top_genes=r["top_genes"], zfin=r["zfin"]))
    y = len(free_from)
    table = pd.DataFrame(rows)
    table["_s"] = table["stage"].map(stage_pos)
    table = table.sort_values(["track", "_s", "cells"], ascending=[True, True, False]) \
        .drop(columns="_s")
    table.to_csv(TABLES / f"{name}_tracks.tsv", sep="\t", index=False, float_format="%.4g")
    # Share of cells in some robust population, per stage.
    covered = []
    for stage, c in cells.groupby("stage", sort=False):
        inside = np.zeros(len(c), dtype=bool)
        for r in robust.loc[stage].itertuples() if stage in robust.index.get_level_values(0) else []:
            res, pop = r.representative[1:].split("_p")
            inside |= (c[f"population_{res}"] == int(pop)).to_numpy()
        covered.append(dict(stage=stage, robust_populations=int((robust.index.get_level_values(0) == stage).sum()),
                            robust_coverage=inside.mean()))
    print(pd.DataFrame(covered).round(2).to_string(index=False))

    fig, ax = plt.subplots(figsize=(20, max(5, 0.28 * y + 1.5)))
    for a, b, d in g.edges(data=True):
        (s1, _), (s2, _) = sorted([a, b], key=lambda n: stage_pos[n[0]])
        ax.plot([stage_pos[s1], stage_pos[s2]], [y_of[a], y_of[b]], color="#555555",
                lw=0.6 + 2 * (d["auroc"] - MIN_LINK_AUROC) / (1 - MIN_LINK_AUROC), alpha=0.7,
                zorder=1)
    for node, yy in y_of.items():
        r = robust.loc[node]
        x = stage_pos[node[0]]
        ax.scatter(x, yy, s=10 + 600 * r["share"], color="#0072B2", zorder=2,
                   edgecolor="white", linewidth=0.5)
        genes = ",".join(r["top_genes"].split(",")[:2])
        ax.annotate(f"{genes}", (x, yy), xytext=(5, 0), textcoords="offset points",
                    fontsize=6.5, va="center")
    ax.set_xticks(range(len(STAGES)), STAGES, rotation=30)
    ax.set_xlim(-0.3, len(STAGES) - 0.2)
    ax.set_ylim(y, -1)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_title("Robust replicated populations (found at >= 2 resolutions) and links "
                 f"between consecutive stages (AUROC >= {MIN_LINK_AUROC}); node area = share "
                 "of the stage's cells", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_tracks.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="*", default=STAGES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--tracks-only", action="store_true",
                        help="only redraw the tracks from the saved tables")
    args = parser.parse_args()
    name = f"replicated_populations{args.tag}"
    if args.tracks_only:
        tracks(name)
        return
    ensure_dirs()
    adata = load_dropseq_counts()
    libs_of = {s: sorted(adata.obs.loc[adata.obs["stage"] == s, "library"].unique())
               for s in STAGES}
    genes_table = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                              usecols=["stage", "gene", "heterogeneous"])
    zfin, _ = zfin_check.zfin_genes(adata)
    out = {}
    for stage in args.stages:
        out[stage] = run_stage(adata, stage, libs_of, genes_table, zfin, args.seed)
        print(out[stage]["summary"].to_string(index=False), flush=True)
    links = link_stages(adata, libs_of, genes_table, out, args.seed)

    tables = {"_summary.tsv": pd.concat([r["summary"] for r in out.values()]),
              "_resolution.tsv": pd.concat([r["pops"] for r in out.values()]),
              ".tsv": pd.concat([r["consensus"] for r in out.values()]),
              "_matches.tsv": pd.concat([r["matches"] for r in out.values()]),
              "_links.tsv": links}
    for suffix, table in tables.items():
        table.to_csv(TABLES / f"{name}{suffix}", sep="\t", index=False, float_format="%.4g")
    pd.concat([r["cells"] for r in out.values()]).to_csv(
        TABLES / f"{name}_cells.tsv.gz", sep="\t", float_format="%.4g")
    with PdfPages(FIGURES / f"{name}.pdf") as pdf:
        plot(out, pdf)
    tracks(name)


if __name__ == "__main__":
    main()
