"""How homogeneous are the cells that share a top program?

Populations: the kept cells of a library with the same top identity program
(script 26), at least MIN_CELLS of them. Each is tested with the calibrated
test of script 01b (`heterogeneity_calls`, run on the population's cells
only): a gene is heterogeneous if it varies more than typical genes of the
same mean within these cells. Control: as many random kept cells of the same
library. A call counts only at the moderate tier and if at least MIN_DRIVERS
cells carry it (Pearson residual > DRIVER_RESIDUAL under the subset's own NB
model): in populations of ~100 cells, a third of the calls are carried by
three cells or fewer (doublets, contamination, rare cells), in the controls as
well. Per program and stage, a gene is heterogeneous if it is called in at
least half of the libraries with a population.

Power: the random control is a sample of the library, so it should recover
the library's own strongly heterogeneous genes (script 01b, strong tier), as
far as they can reach MIN_DRIVERS driver cells at the population's size.
Detectable genes: the library's strong genes whose driver rate in the library
(share of its cells with a residual > DRIVER_RESIDUAL, under the library's
model) times the population size is at least DETECTABLE_FACTOR * MIN_DRIVERS;
power: the share of them the control calls. A population with fewer than
MIN_DETECTABLE detectable genes is inconclusive (shallow libraries, small
populations): few heterogeneous genes there do not mean homogeneity. Medians,
ubiquitous genes and the report table use the conclusive populations.

Program membership: the share of the called genes (without ubiquitous and
state genes) that belong to a curated program or profile, in the population
and in its control (heterogeneous genes are enriched in programs by
construction, script 16, so the control is the baseline).

Ubiquitous genes: called in at least UBIQUITOUS_SHARE of the conclusive library
populations, of at least UBIQUITOUS_PROGRAMS different programs. They vary
within almost any population and say nothing about which one; they are listed
and left out of the per-population gene lists (as are the genes of
EXCLUDED_GENE_PROGRAMS of script 26: heat shock, dissociation, cell cycle). What
they vary with inside the populations (median correlation over the conclusive
populations) tells what they are: positional gradients that run through many
populations, or states and technical axes (UBIQUITOUS_CLASSES, read from
population_ubiquitous_partners.tsv).

Every test is cached (data/processed/cache/28); increase CACHE_VERSION when a
change alters the calls. `28_population_heterogeneity.py plot` redraws the
figure and table from the saved tables.

Outputs:
  results/tables/population_heterogeneity.tsv        per program, stage and
      library: cells, calls (all, with >= MIN_DRIVERS drivers) in the
      population and the control
  results/tables/population_heterogeneity_genes.tsv  per program, stage and
      gene: libraries, median driver cells, ubiquitous, owner program
  results/tables/population_ubiquitous_genes.tsv     the ubiquitous genes
  results/tables/population_ubiquitous_partners.tsv  their co-varying genes, class
  report/figures/population_heterogeneity.pdf, report/tables/population_genes.tex
"""
from __future__ import annotations

import importlib
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_utils import (
    REPORT_TABLES,
    REPORT_WIDTH_IN,
    STAGES,
    TABLES,
    digest,
    ensure_dirs,
    load_cached,
    load_dropseq_counts,
    pearson_residuals,
    report_style,
    save_cached,
    save_report_figure,
)

warnings.filterwarnings("ignore", category=FutureWarning)
cell_programs = importlib.import_module("26_cell_programs")

CACHE_VERSION = 1
SEED = 0
MIN_CELLS = 100
TIER = 2  # moderate (estimated FDR 10%)
MIN_DRIVERS = 10
DRIVER_RESIDUAL = 3.0
DETECTABLE_FACTOR = 1.5
MIN_DETECTABLE = 5
UBIQUITOUS_SHARE = 0.25
UBIQUITOUS_PROGRAMS = 4
N_WORKERS = 8
LIST_GENES = 12  # genes per population in the report table
PARTNERS = 8  # co-varying genes listed per ubiquitous gene
MIN_DETECTING = 10  # cells of a population detecting a gene, at least, for its correlations
# What a ubiquitous gene varies with inside the populations (population_ubiquitous_partners
# .tsv, read by hand): position (ventral BMP targets, posterior, anterior, paraxial
# maturation), a state or technical axis, or unclear. Unlisted genes are "unclear".
UBIQUITOUS_CLASSES = {
    "APOC1": "position", "APOEB": "position", "VED": "position", "CDX4": "position",
    "IRX7": "position", "MSGN1": "position",
    "HISTH1L": "state or technical", "FOSAB": "state or technical",
    "ZGC:110425": "state or technical",
}
CLASS_COLOURS = {"position": "#0072B2", "state or technical": "#999999", "unclear": "#DDDDDD"}
_adata = None


def init_worker() -> None:
    global _adata
    warnings.filterwarnings("ignore")
    _adata = load_dropseq_counts()


def test_cells(task) -> dict:
    """01b's test on one set of cells; the driver cells of every called gene."""
    cells, seed = task
    het = importlib.import_module("01b_heterogeneous_genes")
    X = _adata[cells].X.tocsr()
    disp, null = het.heterogeneity_calls(X, np.random.default_rng(seed))
    tier = disp["tier"].to_numpy()
    called = np.flatnonzero(tier >= 1)
    r = pearson_residuals(X, called, centre=False)
    return dict(genes=list(_adata.var_names[called]), tier=tier[called].tolist(),
                drivers=(r > DRIVER_RESIDUAL).sum(axis=0).tolist(),
                null_moderate=int(null["moderate"]))


def populations() -> list[dict]:
    cells = pd.read_csv(cell_programs.CELLS, sep="\t", index_col=0, keep_default_na=False)
    kept = cells[cells["excluded"] == ""]
    out = []
    for (p, stage, lib), d in kept[kept["top"] != ""].groupby(["top", "stage", "library"]):
        if len(d) < MIN_CELLS:
            continue
        seed = [SEED, STAGES.index(stage), int(digest(p, lib), 16) % 2**31]
        pool = kept.index[kept["library"] == lib]
        control = sorted(np.random.default_rng(seed).choice(pool, len(d), replace=False))
        out.append(dict(program=p, stage=stage, library=lib, population=sorted(d.index),
                        control=control, seed=seed))
    return out


def compute() -> None:
    pops = populations()
    jobs = []  # (population, kind, cache key, task)
    for q in pops:
        for kind in ("population", "control"):
            key = digest(CACHE_VERSION, TIER, DRIVER_RESIDUAL, q[kind], q["seed"], kind)
            jobs.append((q, kind, key, (q[kind], q["seed"] + [int(kind == "control")])))
    todo = [j for j in jobs if load_cached("28", "test", j[2]) is None]
    print(len(jobs), "tests,", len(todo), "to compute", flush=True)
    if todo:
        with ProcessPoolExecutor(N_WORKERS, initializer=init_worker) as ex:
            for n, (job, r) in enumerate(zip(todo, ex.map(test_cells, [j[3] for j in todo]))):
                save_cached("28", "test", job[2], r)
                if n % 20 == 0:
                    print(n, job[0]["program"], job[0]["stage"], flush=True)
    results = {(q["program"], q["stage"], q["library"], kind): load_cached("28", "test", key)
               for q, kind, key, _ in jobs}
    summarise(pops, results, driver_rates(sorted({q["library"] for q in pops})))
    partners(pops)


def partners(pops: list[dict]) -> None:
    """For every ubiquitous gene: its median correlation (Pearson residuals of the
    population's own model) with the stage's heterogeneous genes, over the
    conclusive populations; genes detected in >= MIN_DETECTING of a population's
    cells, and correlated in >= a third of the populations."""
    t = pd.read_csv(TABLES / "population_heterogeneity.tsv", sep="\t")
    ubi = pd.read_csv(TABLES / "population_ubiquitous_genes.tsv", sep="\t")["gene"].tolist()
    het = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                      usecols=["stage", "gene", "heterogeneous"])
    conclusive = set(zip(t.loc[t["conclusive"], "program"], t.loc[t["conclusive"], "library"]))
    adata = load_dropseq_counts()
    corr = {g: [] for g in ubi}
    for q in pops:
        if (q["program"], q["library"]) not in conclusive:
            continue
        sub = adata[q["population"]]
        genes = sorted((set(het.loc[(het["stage"] == q["stage"]) & het["heterogeneous"], "gene"])
                        | set(ubi)) & set(sub.var_names))
        X = sub.X.tocsr()
        idx = sub.var_names.get_indexer(genes)
        keep = np.asarray((X[:, idx] > 0).sum(axis=0)).ravel() >= MIN_DETECTING
        names = np.array(genes)[keep]
        r = pearson_residuals(X, idx[keep])
        r = (r - r.mean(axis=0)) / (r.std(axis=0) + 1e-9)
        c = pd.DataFrame(r.T @ r / len(r), index=names, columns=names)
        for g in ubi:
            if g in c.index:
                corr[g].append(c[g].drop(g))
    rows = []
    for g in ubi:
        if not corr[g]:
            continue
        m = pd.concat(corr[g], axis=1)
        med = m.median(axis=1)[m.notna().sum(axis=1) >= max(5, len(corr[g]) // 3)]
        top, bottom = med.sort_values(ascending=False), med.sort_values()
        rows.append(dict(gene=g, populations=len(corr[g]),
                         class_=UBIQUITOUS_CLASSES.get(g, "unclear"),
                         positive=", ".join(f"{x}:{v:.2f}" for x, v in top.head(PARTNERS).items()),
                         negative=", ".join(f"{x}:{v:.2f}" for x, v in bottom.head(4).items())))
    pd.DataFrame(rows).rename(columns={"class_": "class"}).to_csv(
        TABLES / "population_ubiquitous_partners.tsv", sep="\t", index=False)


def driver_rates(libraries: list[str]) -> dict:
    """Per library: the share of its cells with a residual > DRIVER_RESIDUAL for
    each of its strongly heterogeneous genes (script 01b)."""
    calls = pd.read_csv(TABLES / "heterogeneous_genes_calls.tsv.gz", sep="\t", index_col=0)
    adata = None
    out = {}
    for lib in libraries:
        strong = sorted(calls.index[calls[lib] >= 3])

        def rates():
            nonlocal adata
            if adata is None:
                adata = load_dropseq_counts()
            sub = adata[adata.obs["library"] == lib]
            keep = [g for g in strong if g in sub.var_names]
            r = pearson_residuals(sub.X.tocsr(), sub.var_names.get_indexer(keep), centre=False)
            return pd.Series((r > DRIVER_RESIDUAL).mean(axis=0), index=keep)
        out[lib] = cached_value(f"rates-{lib}", digest(CACHE_VERSION, DRIVER_RESIDUAL, strong),
                                rates)
    return out


def cached_value(name: str, key: str, compute):
    value = load_cached("28", name, key)
    if value is None:
        value = compute()
        save_cached("28", name, key, value)
    return value


def summarise(pops: list[dict], results: dict, rates: dict) -> None:
    programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    programs = programs[programs["kept"] & programs["set"].isin(["curated", "controls"])]
    state_genes = set(programs.loc[programs["program"].isin(
        cell_programs.EXCLUDED_GENE_PROGRAMS), "gene"])

    def calls(key):
        r = results[key]
        return {g: d for g, t, d in zip(r["genes"], r["tier"], r["drivers"])
                if t >= TIER and d >= MIN_DRIVERS}
    owner_at = programs[~programs["program"].isin(cell_programs.EXCLUDED_GENE_PROGRAMS)
                        & (programs["set"] == "curated")].groupby(["stage", "gene"]).size()
    rows, gene_rows = [], []
    for q in pops:
        k = (q["program"], q["stage"], q["library"])
        pop, ctl = calls(k + ("population",)), calls(k + ("control",))
        n = len(q["population"])
        rate = rates[q["library"]]
        detectable = set(rate.index[rate * n >= DETECTABLE_FACTOR * MIN_DRIVERS])
        rows.append(dict(program=q["program"], stage=q["stage"], library=q["library"],
                         cells=n, detectable=len(detectable),
                         power=len(detectable & set(ctl)) / len(detectable) if detectable
                         else np.nan, conclusive=len(detectable) >= MIN_DETECTABLE,
                         _pop=set(pop), _ctl=set(ctl),
                         calls_all=sum(t >= TIER for t in results[k + ("population",)]["tier"]),
                         calls=len(pop),
                         control_all=sum(t >= TIER for t in results[k + ("control",)]["tier"]),
                         control=len(ctl),
                         null_moderate=results[k + ("population",)]["null_moderate"]))
        for g, d in pop.items():
            gene_rows.append(dict(program=q["program"], stage=q["stage"], library=q["library"],
                                  gene=g, drivers=d))
    t = pd.DataFrame(rows)
    t["ratio"] = t["calls"] / t["control"].clip(lower=1)
    g = pd.DataFrame(gene_rows)
    # ubiquitous: called in many conclusive library populations of several programs
    conclusive = set(t.loc[t["conclusive"], "library"] + "|" + t.loc[t["conclusive"], "program"])
    gc = g[(g["library"] + "|" + g["program"]).isin(conclusive)]
    per_gene = gc.groupby("gene").agg(populations=("library", "size"),
                                      programs=("program", "nunique"))
    per_gene["share"] = per_gene["populations"] / t["conclusive"].sum()
    ubiquitous = per_gene[(per_gene["share"] >= UBIQUITOUS_SHARE)
                          & (per_gene["programs"] >= UBIQUITOUS_PROGRAMS)]
    owner = programs.groupby("gene")["program"].agg(lambda s: ";".join(sorted(set(s))))
    ubiquitous = ubiquitous.assign(program=owner.reindex(ubiquitous.index).fillna("")) \
        .sort_values("share", ascending=False)
    ubiquitous.reset_index().to_csv(TABLES / "population_ubiquitous_genes.tsv", sep="\t",
                                    index=False, float_format="%.3f")
    skip = set(ubiquitous.index) | state_genes

    def in_program(genes, stage):
        genes = [x for x in genes if x not in skip]
        return np.mean([(stage, x) in owner_at.index for x in genes]) if genes else np.nan
    t["in_program"] = [in_program(x, st) for x, st in zip(t["_pop"], t["stage"])]
    t["control_in_program"] = [in_program(x, st) for x, st in zip(t["_ctl"], t["stage"])]
    t.drop(columns=["_pop", "_ctl"]).to_csv(TABLES / "population_heterogeneity.tsv", sep="\t",
                                            index=False, float_format="%.3f")
    libs = t.groupby(["program", "stage"])["library"].nunique()
    conclusive_libs = t[t["conclusive"]].groupby(["program", "stage"])["library"].nunique()
    stage_owner = programs.groupby(["stage", "gene"])["program"].agg(
        lambda s: ";".join(sorted(set(s))))
    out = []
    for (p, st, gene), d in g.groupby(["program", "stage", "gene"]):
        n = libs[(p, st)]
        if len(d) < int(np.ceil(n / 2)):
            continue
        out.append(dict(program=p, stage=st, gene=gene, libraries=len(d), of=n,
                        conclusive_libraries=int(conclusive_libs.get((p, st), 0)),
                        drivers=float(d["drivers"].median()),
                        ubiquitous=gene in ubiquitous.index, state=gene in state_genes,
                        owner=stage_owner.get((st, gene), "")))
    genes = pd.DataFrame(out)
    genes["_o"] = genes["stage"].map(STAGES.index)
    genes = genes.sort_values(["program", "_o", "libraries", "drivers"],
                              ascending=[True, True, False, False]).drop(columns="_o")
    genes.to_csv(TABLES / "population_heterogeneity_genes.tsv", sep="\t", index=False,
                 float_format="%.1f")


# --- figure and table --------------------------------------------------------------------------

def plot() -> None:
    report_style()
    t = pd.read_csv(TABLES / "population_heterogeneity.tsv", sep="\t")
    genes = pd.read_csv(TABLES / "population_heterogeneity_genes.tsv", sep="\t",
                        keep_default_na=False)
    ubi = pd.read_csv(TABLES / "population_ubiquitous_genes.tsv", sep="\t", keep_default_na=False)
    figure(t, ubi)
    latex_genes(genes, t)


def figure(t: pd.DataFrame, ubi: pd.DataFrame) -> None:
    progs = cell_programs.program_order(t["program"].unique())
    colours = {p: plt.get_cmap("tab20")(k % 20) for k, p in enumerate(progs)}
    fig, axes = plt.subplots(1, 3, figsize=(REPORT_WIDTH_IN, 2.6), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    ax = axes[0]
    lim = max(t["control"].max(), t["calls"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#999999", lw=0.6, ls="--")
    for p in progs:
        d = t[t["program"] == p]
        ok = d["conclusive"].astype(bool)
        ax.scatter(d.loc[ok, "control"], d.loc[ok, "calls"], s=9, color=colours[p],
                   linewidths=0)
        ax.scatter(d.loc[~ok, "control"], d.loc[~ok, "calls"], s=9, facecolors="none",
                   edgecolors=colours[p], linewidths=0.6)
    ax.set_xlabel("random cells (same library, same number)")
    ax.set_ylabel("population")
    ax.set_title("(a) heterogeneous genes per library", loc="left", fontsize=8)
    ax = axes[1]
    conc = t[t["conclusive"].astype(bool)]
    order = conc.groupby("program")["ratio"].median().reindex(progs).dropna().sort_values().index
    order = list(order) + [p for p in progs if p not in set(order)]
    for k, p in enumerate(order):
        d = t[t["program"] == p]
        ok = d["conclusive"].astype(bool)
        ax.scatter(d.loc[ok, "ratio"].clip(upper=2), np.full(ok.sum(), k), s=7,
                   color=colours[p], linewidths=0)
        ax.scatter(d.loc[~ok, "ratio"].clip(upper=2), np.full((~ok).sum(), k), s=7,
                   facecolors="none", edgecolors=colours[p], linewidths=0.6)
        if ok.any():
            ax.plot([d.loc[ok, "ratio"].median()] * 2, [k - 0.35, k + 0.35], color="black", lw=1)
    ax.axvline(1, color="#999999", lw=0.6, ls="--")
    ax.set_yticks(range(len(order)), [p.removeprefix("state:").replace("_", " ") for p in order],
                  fontsize=6)
    ax.set_xlabel("population / random (clipped at 2)")
    ax.set_title("(b) per program", loc="left", fontsize=8)
    ax = axes[2]
    u = ubi.head(20).iloc[::-1]
    kind = u["gene"].map(lambda g: UBIQUITOUS_CLASSES.get(g, "unclear"))
    ax.barh(range(len(u)), u["share"], height=0.7, color=kind.map(CLASS_COLOURS))
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=k)
                       for k, c in CLASS_COLOURS.items() if k in set(kind)],
              frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.4, -0.22),
              ncol=1)
    ax.set_yticks(range(len(u)), [x.lower() for x in u["gene"]], fontsize=6)
    ax.set_xlabel("share of library populations")
    ax.set_title("(c) ubiquitous genes", loc="left", fontsize=8)
    save_report_figure(fig, "population_heterogeneity")
    plt.close(fig)


def latex_genes(genes: pd.DataFrame, t: pd.DataFrame) -> None:
    """Per program and stage: the heterogeneous genes without ubiquitous and
    state genes (at most LIST_GENES), and how many there are."""
    def tex(s):
        return str(s).replace("%", "\\%").replace("_", " ")
    g = genes[~genes["ubiquitous"].astype(bool) & ~genes["state"].astype(bool)]
    lines = ["% Generated by scripts/28_population_heterogeneity.py; do not edit.",
             "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{2.6cm}"
             ">{\\raggedright\\arraybackslash}p{2.1cm}r>{\\raggedright\\arraybackslash}p{9.2cm}@{}}",
             "\\caption{Heterogeneous genes within top-program populations "
             "(\\protect\\nolinkurl{scripts/28_population_heterogeneity.py}): called in at least "
             f"half of the libraries with a population (at least {MIN_CELLS} cells), at the "
             f"moderate tier and carried by at least {MIN_DRIVERS} cells; ubiquitous genes and "
             "the heat shock, dissociation and cell-cycle genes are left out. Genes: listed "
             "of all heterogeneous genes; *: no library of the population has the power to "
             f"detect heterogeneity (fewer than {MIN_DETECTABLE} detectable genes), so few "
             "genes do not mean homogeneity. The first "
             f"{LIST_GENES} genes, by libraries and driver cells; in brackets the curated "
             "program or profile they belong to.}\\label{tab:population-genes}\\\\",
             "\\toprule", "Program & Stage & Genes & Heterogeneous genes \\\\", "\\midrule",
             "\\endfirsthead", "\\toprule", "Program & Stage & Genes & Heterogeneous genes \\\\",
             "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot"]
    pairs = t[["program", "stage"]].drop_duplicates()
    pairs = pairs.assign(_p=pairs["program"].map(
        {p: k for k, p in enumerate(cell_programs.program_order(pairs["program"].unique()))}),
        _s=pairs["stage"].map(STAGES.index)).sort_values(["_p", "_s"])
    total = genes.groupby(["program", "stage"]).size()
    conclusive = t.groupby(["program", "stage"])["conclusive"].any()
    for i, r in enumerate(pairs.itertuples()):
        d = g[(g["program"] == r.program) & (g["stage"] == r.stage)]
        count = f"{len(d)}/{total.get((r.program, r.stage), 0)}" \
            + ("" if conclusive[(r.program, r.stage)] else "*")
        names = ", ".join(f"\\emph{{{x.gene.lower()}}}"
                          + (f" ({tex(x.owner.split(';')[0].split(':')[-1])})" if x.owner else "")
                          for x in d.head(LIST_GENES).itertuples())
        lines.append(("\\rowcolor{gray!10}" if i % 2 else "") + " & ".join([
            tex(r.program.removeprefix("state:")), tex(r.stage), count, names or "--"])
            + " \\\\")
    lines.append("\\end{longtable}")
    (REPORT_TABLES / "population_genes.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ensure_dirs()
    if sys.argv[1:] != ["plot"]:
        compute()
    plot()
