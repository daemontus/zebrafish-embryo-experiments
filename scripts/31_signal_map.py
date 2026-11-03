"""Do the signals lie where the map of Section 2 puts them?

Cells are placed on two coordinates computed from identity programs only, and
the signalling profiles (which share no gene with them) are overlaid. Per
library (kept cells of script 26), every program and profile is scored as the
mean Pearson residual of its genes, smoothed over the cell and its K_SMOOTH
neighbours (neighbours from the stage's heterogeneous genes without the scored
genes and the excluded state genes of script 26), and standardised over the
library's cells. Coordinates, mean of one group minus mean of the other:
  gastrula (dome to 90% epiboly)
    axis 1, animal-vegetal: ectoderm programs (+, animal) - margin-side programs
    axis 2, dorsal-ventral: dorsal programs (+) - ventral programs
  bud to 6-somite
    axis 1, anterior-posterior: anterior programs (+) - posterior programs
    axis 2, medial-lateral: midline programs (+) - lateral and ventral programs
High and oblong are left out (too few programs to place the cells).

For every profile and axis: the Spearman correlation of the profile's score
with the coordinate, relative to N_RANDOM random sets of heterogeneous genes in
no program, with the profile's size (0 = like random genes, +1 / -1 at their
95% / 5% quantile, as in script 21), median over the stage's libraries; beyond
the random sets on the same side in every library = consistent. EXPECTED: the
direction expected from Table 2 of the report (tab:expected-signals) where
it states a location; profiles without one have no expectation. Grids:
GRID_1 x GRID_2 quantile bins of the two coordinates, mean profile score,
median over the libraries.

Every library's scores are cached (data/processed/cache/31); increase
CACHE_VERSION when a change alters them. `31_signal_map.py plot` redraws.

Outputs:
  results/tables/signal_map.tsv          stage x library x profile: r, relative
  results/tables/signal_map_stages.tsv   stage x profile: median relative,
      consistent, expected, agreement
  results/tables/signal_map_grid.tsv     stage x library x profile x bins
  report/figures/signal_map.pdf, report/figures/signal_map_grids.pdf
"""
from __future__ import annotations

import importlib
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from pipeline_utils import (
    REPORT_WIDTH_IN,
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
cell_programs = importlib.import_module("26_cell_programs")

CACHE_VERSION = 1
SEED = 0
N_RANDOM = 100
GRID_1, GRID_2 = 4, 6
MAP_STAGES = STAGES[STAGES.index("dome"):]
LATE = STAGES[STAGES.index("bud"):]
GROUPS = {  # (axis 1 +, axis 1 -, axis 2 +, axis 2 -)
    "gastrula": (["state:blastomere", "neural_plate", "forebrain_eye",
                  "midbrain_hindbrain_boundary", "non_neural_ectoderm"],
                 ["margin", "mesoderm", "endoderm", "dorsal_organizer", "presomitic_mesoderm",
                  "axial_mesoderm", "tailbud"],
                 ["dorsal_organizer", "axial_mesoderm", "prechordal_plate", "neural_plate",
                  "forebrain_eye"],
                 ["non_neural_ectoderm", "tailbud", "lateral_plate"]),
    "late": (["forebrain_eye", "midbrain_hindbrain_boundary", "prechordal_plate",
              "hatching_gland", "heart"],
             ["tailbud", "presomitic_mesoderm", "margin"],
             ["notochord", "axial_mesoderm", "neural_plate", "forebrain_eye", "hindbrain"],
             ["non_neural_ectoderm", "lateral_plate", "placodes"]),
}
AXES = {"gastrula": ("animal (+) - vegetal", "dorsal (+) - ventral"),
        "late": ("anterior (+) - posterior", "medial (+) - lateral")}
PROFILES = ["signal:Nodal", "signal:FGF", "signal:Wnt", "signal:BMP", "blocking:BMP",
            "sending:BMP_organizer", "blocking:Wnt", "sending:Wnt/PCP", "sending:Hedgehog",
            "receiving:Notch", "sending:Notch", "sending:RA", "receiving:RA", "blocking:RA"]
# expected sign per (period, profile, axis), from Table 2 (tab:expected-signals)
EXPECTED = {
    ("gastrula", "signal:Nodal", 1): -1, ("gastrula", "signal:Nodal", 2): 1,
    ("gastrula", "signal:FGF", 1): -1, ("gastrula", "signal:Wnt", 1): -1,
    ("gastrula", "signal:BMP", 2): -1, ("gastrula", "blocking:BMP", 2): 1,
    ("gastrula", "sending:BMP_organizer", 2): 1, ("gastrula", "blocking:Wnt", 2): 1,
    ("gastrula", "sending:Hedgehog", 2): 1,
    ("late", "signal:Wnt", 1): -1, ("late", "signal:FGF", 1): -1,
    ("late", "signal:BMP", 2): -1, ("late", "sending:Hedgehog", 2): 1,
    ("late", "receiving:Notch", 1): -1, ("late", "sending:RA", 1): -1,
    ("late", "blocking:Wnt", 1): 1,
}
GRID_PROFILES = ["signal:Nodal", "signal:FGF", "signal:Wnt", "signal:BMP", "blocking:BMP",
                 "sending:Hedgehog"]
GRID_STAGES = ["50% epiboly", "shield", "75% epiboly", "bud", "6-somite"]


def period(stage: str) -> str:
    return "late" if stage in LATE else "gastrula"


def library_scores(counts, var_names, sets: dict, rand: dict, graph_genes: list) -> pd.DataFrame:
    n = counts.shape[0]
    pcs = PCA(min(N_PCS, n - 1), random_state=SEED).fit_transform(
        pearson_residuals(counts, var_names.get_indexer(graph_genes)))
    nb = NearestNeighbors(n_neighbors=min(K_SMOOTH, n - 1)).fit(pcs).kneighbors(
        return_distance=False)
    nb = np.column_stack([np.arange(n), nb])
    genes = sorted({g for v in list(sets.values()) + list(rand.values()) for g in v})
    res = pearson_residuals(counts, var_names.get_indexer(genes))
    col = {g: k for k, g in enumerate(genes)}

    def score(gs):
        v = res[:, [col[g] for g in gs]].mean(axis=1)[nb].mean(axis=1)
        return (v - v.mean()) / (v.std() or 1.0)
    return pd.DataFrame({p: score(v) for p, v in {**sets, **rand}.items() if v})


def relative(r: float, null: np.ndarray) -> float:
    med = np.median(null)
    hi, lo = np.quantile(null, 0.95), np.quantile(null, 0.05)
    return (r - med) / (hi - med) if r >= med else -(r - med) / (lo - med)


def compute() -> None:
    adata = load_dropseq_counts()
    programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    programs = programs[programs["kept"]]
    curated = programs[programs["set"] == "curated"]
    het = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                      usecols=["stage", "gene", "heterogeneous"])
    cells = pd.read_csv(cell_programs.CELLS, sep="\t", index_col=0, keep_default_na=False)
    kept = cells[cells["excluded"] == ""]
    rows, grids = [], []
    for s, stage in enumerate(MAP_STAGES):
        per = period(stage)
        g1p, g1n, g2p, g2n = GROUPS[per]
        genes = curated[curated["stage"] == stage].groupby("program")["gene"].agg(list)
        hs = set(het.loc[(het["stage"] == stage) & het["heterogeneous"], "gene"])
        excluded = set(programs.loc[(programs["stage"] == stage) & programs["program"].isin(
            cell_programs.EXCLUDED_GENE_PROGRAMS), "gene"])
        pool = np.array(sorted(hs - set(programs.loc[programs["stage"] == stage, "gene"])
                               - excluded))
        for lib in sorted(kept.loc[kept["stage"] == stage, "library"].unique()):
            idx = kept.index[kept["library"] == lib]
            sub = adata[idx]
            names = [p for p in set(g1p + g1n + g2p + g2n + PROFILES) if p in genes.index]
            sets = {p: [x for x in genes[p] if x in sub.var_names] for p in names}
            rng = np.random.default_rng([SEED, s, int(digest(lib), 16) % 2**31])
            sizes = [len(sets[p]) for p in PROFILES if sets.get(p)]
            rand = {f"random:{k}": sorted(rng.choice(pool, min(sizes[k % len(sizes)], len(pool)),
                                                     replace=False))
                    for k in range(N_RANDOM)}
            graph = sorted(hs - {x for v in list(sets.values()) + list(rand.values())
                                 for x in v} - excluded)
            key = digest(CACHE_VERSION, sets, rand, graph, K_SMOOTH)
            S = cached("31", f"scores-{lib}", key, lambda: library_scores(
                sub.X.tocsr(), sub.var_names, sets, rand, graph))
            S.index = idx
            axis1 = S[[p for p in g1p if p in S]].mean(axis=1) - S[[p for p in g1n if p in S]].mean(axis=1)
            axis2 = S[[p for p in g2p if p in S]].mean(axis=1) - S[[p for p in g2n if p in S]].mean(axis=1)
            rcols = [c for c in S if c.startswith("random:")]
            null1 = np.array([spearmanr(S[c], axis1).statistic for c in rcols])
            null2 = np.array([spearmanr(S[c], axis2).statistic for c in rcols])
            b1 = pd.qcut(axis1.rank(method="first"), GRID_1, labels=False)
            b2 = pd.qcut(axis2.rank(method="first"), GRID_2, labels=False)
            for p in PROFILES:
                if p not in S:
                    continue
                r1 = spearmanr(S[p], axis1).statistic
                r2 = spearmanr(S[p], axis2).statistic
                rows.append(dict(stage=stage, library=lib, profile=p, cells=len(idx),
                                 axes_correlation=spearmanr(axis1, axis2).statistic,
                                 r1=r1, r2=r2, rel1=relative(r1, null1), rel2=relative(r2, null2)))
                if p in GRID_PROFILES:
                    grid = S[p].groupby([b1, b2]).mean()
                    for (a, d), v in grid.items():
                        grids.append(dict(stage=stage, library=lib, profile=p, bin1=a, bin2=d,
                                          mean=v))
        print(stage, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "signal_map.tsv", sep="\t", index=False, float_format="%.3f")
    pd.DataFrame(grids).to_csv(TABLES / "signal_map_grid.tsv", sep="\t", index=False,
                               float_format="%.3f")
    stages_table(t).to_csv(TABLES / "signal_map_stages.tsv", sep="\t", index=False,
                           float_format="%.3f")


def stages_table(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stage, p), d in t.groupby(["stage", "profile"], sort=False):
        row = dict(stage=stage, profile=p, libraries=len(d),
                   axes_correlation=d["axes_correlation"].median())
        for k in (1, 2):
            rel = d[f"rel{k}"]
            side = 1 if (rel > 1).all() else (-1 if (rel < -1).all() else 0)
            exp = EXPECTED.get((period(stage), p, k), 0)
            row.update({f"rel{k}": rel.median(), f"consistent{k}": side, f"expected{k}": exp,
                        f"agrees{k}": "" if exp == 0 else ("yes" if side == exp else
                                                           ("opposite" if side == -exp else "no"))})
        rows.append(row)
    return pd.DataFrame(rows)


# --- figures -----------------------------------------------------------------------------------

def plot() -> None:
    report_style()
    st = pd.read_csv(TABLES / "signal_map_stages.tsv", sep="\t", keep_default_na=False)
    figure_summary(st)
    figure_grids(pd.read_csv(TABLES / "signal_map_grid.tsv", sep="\t"),
                 pd.read_csv(TABLES / "signal_map.tsv", sep="\t"))


def figure_summary(st: pd.DataFrame) -> None:
    stages = [s for s in MAP_STAGES if s in set(st["stage"])]
    fig, axes = plt.subplots(1, 2, figsize=(REPORT_WIDTH_IN, 3.2), sharey=True,
                             constrained_layout=True)
    titles = ["(a) animal (+) / margin;\nfrom bud: anterior (+) / posterior",
              "(b) dorsal (+) / ventral;\nfrom bud: medial (+) / lateral"]
    for k, ax in zip((1, 2), axes):
        m = st.pivot(index="profile", columns="stage", values=f"rel{k}").reindex(
            index=PROFILES, columns=stages)
        im = ax.imshow(m.to_numpy(dtype=float).clip(-4, 4), cmap="RdBu_r", vmin=-4, vmax=4,
                       aspect="auto")
        for i, p in enumerate(PROFILES):
            for j, s in enumerate(stages):
                r = st[(st["profile"] == p) & (st["stage"] == s)]
                if r.empty:
                    continue
                r = r.iloc[0]
                if int(r[f"consistent{k}"]) != 0:  # beyond the random sets in every library
                    ax.scatter(j, i, s=5, color="black", linewidths=0, zorder=3)
                exp = int(r[f"expected{k}"])
                if exp:  # expected direction: outline, sign in the top-left corner
                    ok = r[f"agrees{k}"] == "yes"
                    colour = "black" if ok else "#C0392B"
                    ax.add_patch(plt.Rectangle((j - 0.46, i - 0.42), 0.92, 0.84, fill=False,
                                               edgecolor=colour, linewidth=0.9 if ok else 0.7,
                                               linestyle="-" if ok else (0, (2, 1.2)), zorder=4))
                    ax.text(j - 0.4, i - 0.36, "+" if exp > 0 else "−", ha="left", va="top",
                            fontsize=4.2, color=colour, fontweight="bold", zorder=5)
        ax.axvline(stages.index("bud") - 0.5, color="black", lw=0.8)
        ax.set_xticks(range(len(stages)), stages, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(PROFILES)), [p.replace("signal:", "").replace(":", ": ")
                                             for p in PROFILES], fontsize=6)
        ax.set_title(titles[k - 1], fontsize=6.5, loc="left")
    fig.colorbar(im, ax=axes, shrink=0.7, label="correlation with the axis,\nrelative to random genes")
    save_report_figure(fig, "signal_map")
    plt.close(fig)


def figure_grids(g: pd.DataFrame, t: pd.DataFrame) -> None:
    stages = [s for s in GRID_STAGES if s in set(g["stage"])]
    fig, axes = plt.subplots(len(stages), len(GRID_PROFILES),
                             figsize=(REPORT_WIDTH_IN, 0.95 * len(stages) + 0.7),
                             constrained_layout=True)
    for i, s in enumerate(stages):
        late = period(s) == "late"
        for j, p in enumerate(GRID_PROFILES):
            ax = axes[i, j]
            d = g[(g["stage"] == s) & (g["profile"] == p)]
            x = d.groupby(["bin1", "bin2"])["mean"].median().unstack()
            im = ax.imshow(x.to_numpy(), cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto",
                           origin="lower")
            ax.set_xticks([0, GRID_2 - 1], ["lateral", "medial"] if late else ["ventral", "dorsal"],
                          fontsize=5)
            ax.set_yticks([0, GRID_1 - 1], ["post.", "ant."] if late else ["margin", "animal"],
                          fontsize=5)
            if i == 0:
                ax.set_title(p.replace("signal:", "").replace(":", ":\n"), fontsize=6)
            if j == 0:
                ax.set_ylabel(s, fontsize=6)
    fig.colorbar(im, ax=axes, shrink=0.5, label="mean standardised score")
    save_report_figure(fig, "signal_map_grids")
    plt.close(fig)


if __name__ == "__main__":
    ensure_dirs()
    if sys.argv[1:] != ["plot"]:
        compute()
    plot()
