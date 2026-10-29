"""Cell populations defined by program activity.

Script 14 finds populations from the genes and names them afterwards. Here,
every cell is described by the scores of the curated programs (script 16)
that are supported at its stage (coherent in at least half of the stage's
libraries, script 17), and the populations are found in that space with the
procedure of script 14: clusters of each library (Leiden at several
resolutions), kept if they match reciprocally in another library (neighbour
voting, one-vs-best AUROC), merged across resolutions; robust = found at
>= 2 resolutions. Single-library stages are matched against the libraries of
the adjacent stages (scored with the programs of the focal stage).

Features: a program's score is the mean centred Pearson residual of its genes
(as in 17), smoothed over each cell and its K_SMOOTH nearest neighbours in
its library (PCA of the residuals of the stage's heterogeneous genes), and
scaled to unit variance over the window. A population is described by its
top programs (largest mean score vs the rest of its library).

Comparison with the gene-based populations of script 14 (robust ones): a
gene population is captured if a robust program population shares at least
CAPTURED_JACCARD of its cells. A gene population that is not captured is a
candidate missing program; it is "named" by the curated programs if at least
NAMED_GENES of its top 8 genes belong to one curated program of the stage
(program present but not separating the cells), otherwise "unnamed".

Coverage per stage and library: share of cells with at least one active
program (smoothed score above the 99.9% quantile of N_PERM gene-wise
permutations within the library, as script 04), and share of cells in a
robust program-based and in a robust gene-based population.

Outputs (results/tables/program_populations*):
  .tsv              consensus populations (top programs)
  _summary.tsv      per stage and resolution (clusters, matches, null)
  _cells.tsv.gz     cell -> clusters, populations, consensus, active programs
  _comparison.tsv   robust gene populations of script 14: captured / named
  _coverage.tsv     per stage and library
  report/figures/program_populations.pdf, report/tables/missing_programs.tex
"""
from __future__ import annotations

import importlib
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pipeline_utils import (
    REPORT_TABLES,
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

warnings.filterwarnings("ignore", category=FutureWarning)
populations = importlib.import_module("14_replicated_populations")

PROGRAM_SET = "curated"
N_PCS = 30
N_PERM = 5
ACTIVE_QUANTILE = 0.999
CAPTURED_JACCARD = 0.3
NAMED_GENES = 2
ROBUST = populations.ROBUST_RESOLUTIONS
SEED = 0


class ProgramSpace:
    """Smoothed program scores per library (the `embed` of script 14)."""

    def __init__(self, programs: pd.DataFrame, support: pd.DataFrame, genes_table: pd.DataFrame,
                 supported_only: bool = True):
        self.supported_only = supported_only
        self.programs = programs
        self.support = support
        self.genes_table = genes_table
        self.active: dict[str, np.ndarray] = {}
        self.names: dict[str, list[str]] = {}

    def stage_programs(self, stage: str) -> pd.Series:
        ok = set(self.support.loc[(self.support["set"] == PROGRAM_SET)
                                  & (self.support["stage"] == stage)
                                  & self.support["supported"], "program"])
        p = self.programs[(self.programs["set"] == PROGRAM_SET) & (self.programs["stage"] == stage)
                          & self.programs["kept"]]
        if self.supported_only:
            p = p[p["program"].isin(ok)]
        return p.groupby("program", sort=False)["gene"].agg(list)

    def __call__(self, sub, libs: list[str], stage: str):
        rng = np.random.default_rng(SEED)
        progs = self.stage_programs(stage)
        het = self.genes_table.loc[(self.genes_table["stage"] == stage)
                                   & self.genes_table["heterogeneous"], "gene"]
        het_idx = np.flatnonzero(sub.var_names.isin(het))
        genes = sorted({g for gs in progs for g in gs})
        col = {g: i for i, g in enumerate(genes)}
        g_idx = sub.var_names.get_indexer(genes)
        X = sub.X.tocsr()
        lib_of = sub.obs["library"].to_numpy()
        raw, smoothed = {}, {}
        for lib in libs:
            rows = np.flatnonzero(lib_of == lib)
            counts = X[rows]
            coords = PCA(min(N_PCS, len(rows) - 1), random_state=SEED).fit_transform(
                pearson_residuals(counts, het_idx))
            res = pearson_residuals(counts, g_idx)
            library = np.full(len(rows), lib)
            score = np.column_stack([res[:, [col[g] for g in gs]].mean(axis=1) for gs in progs])
            smoothed[lib] = prediction.smooth(score, library, coords)
            # active: above the gene-wise permutation null of the library
            threshold = []
            for gs in progs:
                r = res[:, [col[g] for g in gs]]
                null = np.concatenate([prediction.smooth(
                    rng.permuted(r, axis=0).mean(axis=1)[:, None], library, coords)[:, 0]
                    for _ in range(N_PERM)])
                threshold.append(np.quantile(null, ACTIVE_QUANTILE))
            self.active[lib] = smoothed[lib] > np.array(threshold)[None, :]
            self.names[lib] = list(progs.index)
            raw[lib] = smoothed[lib]
        sd = np.vstack([smoothed[lib] for lib in libs]).std(axis=0)
        sd[sd == 0] = 1
        coords = {lib: smoothed[lib] / sd for lib in libs}
        return raw, coords, [p.removeprefix("state:") for p in progs.index]


def members(cells: pd.DataFrame, representative: str) -> pd.Index:
    res, pop = representative[1:].split("_p")
    return cells.index[cells[f"population_{res}"] == int(pop)]


def compare(gene_cons: pd.DataFrame, gene_cells: pd.DataFrame, prog_cons: pd.DataFrame,
            prog_cells: pd.DataFrame, programs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in gene_cons[gene_cons["n_resolutions"] >= ROBUST].itertuples():
        g_cells = set(members(gene_cells[gene_cells["stage"] == r.stage], r.representative))
        best, best_id = 0.0, ""
        pc = prog_cells[prog_cells["stage"] == r.stage]
        for q in prog_cons[(prog_cons["stage"] == r.stage)
                           & (prog_cons["n_resolutions"] >= ROBUST)].itertuples():
            p_cells = set(members(pc, q.representative))
            j = len(g_cells & p_cells) / len(g_cells | p_cells)
            if j > best:
                best, best_id = j, q.consensus
        top = [g.upper() for g in r.top_genes.split(",")[:8]]
        stage_programs = programs[(programs["set"] == PROGRAM_SET) & (programs["stage"] == r.stage)
                                  & programs["kept"]]
        hits = stage_programs[stage_programs["gene"].isin(top)].groupby("program").size()
        named = hits[hits >= NAMED_GENES]
        rows.append(dict(stage=r.stage, gene_population=r.consensus, cells=r.cells,
                         top_genes=r.top_genes, zfin=r.zfin, best_program_population=best_id,
                         jaccard=best, captured=best >= CAPTURED_JACCARD,
                         named_by=";".join(named.sort_values(ascending=False).index)))
    out = pd.DataFrame(rows)
    out["status"] = np.where(out["captured"], "captured",
                             np.where(out["named_by"] != "", "named, not separated", "unnamed"))
    return out


def coverage(prog_cells: pd.DataFrame, prog_cons: pd.DataFrame, gene_cells: pd.DataFrame,
             gene_cons: pd.DataFrame) -> pd.DataFrame:
    def robust_members(cells, cons):
        out = set()
        for r in cons[cons["n_resolutions"] >= ROBUST].itertuples():
            out |= set(members(cells[cells["stage"] == r.stage], r.representative))
        return out
    in_prog = robust_members(prog_cells, prog_cons)
    in_gene = robust_members(gene_cells, gene_cons)
    rows = []
    for (stage, lib), c in prog_cells.groupby(["stage", "library"], sort=False):
        rows.append(dict(stage=stage, library=lib, cells=len(c),
                         active_program=(c["active_programs"] > 0).mean(),
                         program_population=c.index.isin(list(in_prog)).mean(),
                         gene_population=c.index.isin(list(in_gene)).mean(),
                         either_population=(c.index.isin(list(in_prog))
                                            | c.index.isin(list(in_gene))).mean()))
    return pd.DataFrame(rows)


def plot(cov: pd.DataFrame, comp: pd.DataFrame, prog_cons: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(REPORT_WIDTH_IN, 2.5), constrained_layout=True)
    x = np.arange(len(STAGES))
    ax = axes[0]
    med = cov.groupby("stage")[["active_program", "program_population", "gene_population"]] \
        .median().reindex(STAGES)
    for column, style, label in [
            ("active_program", dict(color="#009E73", marker="s"), ">= 1 active program"),
            ("program_population", dict(color="#0072B2", marker="o"), "program population"),
            ("gene_population", dict(color="#D55E00", marker="^"), "gene population (14)")]:
        ax.plot(x, med[column], lw=1, ms=3, label=label, **style)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("share of cells (median library)")
    ax.set_xticks(x, STAGES, rotation=45, ha="right", fontsize=6)
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("a  coverage", fontsize=7, loc="left")
    ax = axes[1]
    counts = comp.groupby(["stage", "status"]).size().unstack(fill_value=0) \
        .reindex(index=STAGES, columns=["captured", "named, not separated", "unnamed"],
                 fill_value=0)
    bottom = np.zeros(len(STAGES))
    for status, color in [("captured", "#0072B2"), ("named, not separated", "#56B4E9"),
                          ("unnamed", "#D55E00")]:
        ax.bar(x, counts[status], bottom=bottom, color=color, label=status, width=0.7)
        bottom += counts[status].to_numpy()
    n_prog = prog_cons[prog_cons["n_resolutions"] >= ROBUST].groupby("stage").size() \
        .reindex(STAGES, fill_value=0)
    ax.plot(x, n_prog, color="black", marker="o", ms=3, lw=1, label="robust program populations")
    ax.set_xticks(x, STAGES, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("robust populations")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("b  gene-based populations (script 14)", fontsize=7, loc="left")
    save_report_figure(fig, "program_populations")
    plt.close(fig)


def latex_missing(comp: pd.DataFrame) -> None:
    t = comp[comp["status"] == "unnamed"]
    lines = ["\\begin{tabular}{llrp{0.55\\linewidth}}", "\\toprule",
             "Stage & Pop. & Cells & Top genes \\\\", "\\midrule"]
    for r in t.itertuples():
        genes = ", ".join(r.top_genes.split(",")[:6]).replace("_", "\\_")
        lines.append(f"{r.stage.replace('%', '\\%')} & {r.gene_population} & {r.cells} & "
                     f"\\emph{{{genes}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPORT_TABLES / "missing_programs.tex").write_text(
        "% Generated by scripts/18_program_populations.py; do not edit.\n"
        + "\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    report_style()
    adata = load_dropseq_counts()
    libs_of = {s: sorted(adata.obs.loc[adata.obs["stage"] == s, "library"].unique())
               for s in STAGES}
    genes_table = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                              usecols=["stage", "gene", "heterogeneous"])
    programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    support = pd.read_csv(TABLES / "program_support.tsv", sep="\t")
    space = ProgramSpace(programs, support, genes_table)
    out = {}
    for stage in STAGES:
        out[stage] = populations.run_stage(adata, stage, libs_of, genes_table, None, SEED,
                                           embed=space)
        cells = out[stage]["cells"]
        n_active = np.zeros(len(cells), dtype=int)
        for lib in cells["library"].unique():
            sel = (cells["library"] == lib).to_numpy()
            n_active[sel] = space.active[lib].sum(axis=1)
        cells["active_programs"] = n_active
        print(out[stage]["summary"][["stage", "resolution", "clusters", "populations",
                                     "cell_coverage", "null_populations"]].to_string(index=False),
              flush=True)
    cons = pd.concat([r["consensus"] for r in out.values()])
    cells = pd.concat([r["cells"] for r in out.values()])
    summary = pd.concat([r["summary"] for r in out.values()])
    gene_cons = pd.read_csv(TABLES / "replicated_populations.tsv", sep="\t")
    gene_cells = pd.read_csv(TABLES / "replicated_populations_cells.tsv.gz", sep="\t",
                             index_col=0, low_memory=False)
    comp = compare(gene_cons, gene_cells, cons, cells, programs)
    cov = coverage(cells, cons, gene_cells, gene_cons)
    cons = cons.rename(columns={"top_genes": "top_programs"}).drop(
        columns=["zfin", "evl_precision", "evl_recall"], errors="ignore")
    cons.to_csv(TABLES / "program_populations.tsv", sep="\t", index=False, float_format="%.4g")
    summary.to_csv(TABLES / "program_populations_summary.tsv", sep="\t", index=False,
                   float_format="%.4g")
    cells.to_csv(TABLES / "program_populations_cells.tsv.gz", sep="\t", float_format="%.4g")
    comp.to_csv(TABLES / "program_populations_comparison.tsv", sep="\t", index=False,
                float_format="%.4g")
    cov.to_csv(TABLES / "program_populations_coverage.tsv", sep="\t", index=False,
               float_format="%.4g")
    print(comp.groupby(["stage", "status"], sort=False).size().unstack(fill_value=0))
    print(cov.groupby("stage", sort=False)[["active_program", "program_population",
                                            "gene_population"]].median().round(2))
    plot(cov, comp, cons)
    latex_missing(comp)


if __name__ == "__main__":
    main()
