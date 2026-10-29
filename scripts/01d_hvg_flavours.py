"""How much does the choice of HVG method matter? (report section 4)

Script 01b compares the heterogeneous genes with the HVGs of scanpy's flavour
"seurat" (dispersion of log-normalised expression in bins of mean, as in
Seurat v1/v2). Here the same comparisons are repeated for the newer methods:
  seurat            as in 01b (log-normalised, top normalised dispersion);
  seurat_filtered   the same after dropping genes detected in < 3 cells, the
                    usual preprocessing (all flavours below are filtered too);
  seurat_v3         Seurat's current default "vst": variance of raw counts
                    standardised by a loess fit of the mean-variance trend;
  pearson           analytic Pearson residuals (Lause et al. 2021): variance
                    of NB Pearson residuals with a fixed theta = 100;
  *_batch           the same with batch_key = library (genes ranked within
                    every library, then combined; the recommended use for
                    several batches);
  nb_residual       the ranking of 01b (median over the stage's libraries of
                    the log10 overdispersion relative to the typical phi);
  heterogeneous     the calls of 01b (weak tier), as a set.
Per stage, on the pooled cells of the stage:
  top 2,000         genes also heterogeneous, genes too rare to measure (< 10
                    UMIs in every library of the stage), median UMIs in the
                    stage;
  ZFIN              share of the measured genes with a restricted in situ
                    pattern at the stage (pre-2018, as 01c) that are in the
                    top k, k = number of heterogeneous genes (matched size)
                    or 2,000, and the same for whole-organism genes.
Per library: Jaccard overlap of the top TOP_N genes of two libraries of the
same stage (replication, as in 01b).

Outputs:
  results/tables/hvg_flavours.tsv              (per stage, flavour and size)
  results/tables/hvg_flavours_replication.tsv  (per library pair and flavour)
  report/tables/hvg_flavours.tex
  report/figures/hvg_flavours.pdf
"""
from __future__ import annotations

import importlib
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from pipeline_utils import (
    REPORT_TABLES,
    REPORT_WIDTH_IN,
    STAGES,
    TABLES,
    ensure_dirs,
    load_dropseq_counts,
    report_style,
    save_report_figure,
)

heterogeneity = importlib.import_module("01b_heterogeneous_genes")
zfin = importlib.import_module("01c_heterogeneity_zfin")

N_HVG = heterogeneity.N_HVG
TOP_N = heterogeneity.TOP_N
MIN_CELLS = 3  # genes detected in fewer cells are dropped (all but "seurat")
SPANS = (0.3, 0.5, 1.0)  # loess span of seurat_v3 (scanpy default first)
FLAVOURS = ["seurat", "seurat_filtered", "seurat_v3", "seurat_v3_batch", "pearson",
            "pearson_batch", "nb_residual"]
LIBRARY_FLAVOURS = ["seurat", "seurat_filtered", "seurat_v3", "pearson", "nb_residual"]
COLOURS = {"seurat": "#999999", "seurat_filtered": "#999999", "seurat_v3": "#E69F00", "seurat_v3_batch": "#E69F00",
           "pearson": "#56B4E9", "pearson_batch": "#56B4E9", "nb_residual": "#000000",
           "heterogeneous": "#D55E00"}
LABELS = {"seurat": "Seurat (v1/v2)", "seurat_filtered": "Seurat (v1/v2), >= 3 cells",
          "seurat_v3": "Seurat v3 (vst)",
          "seurat_v3_batch": "Seurat v3, per library", "pearson": "Pearson residuals",
          "pearson_batch": "Pearson res., per library", "nb_residual": "NB residual (01b)",
          "heterogeneous": "heterogeneous (01b)"}


def hvg(counts, flavour: str, n: int, batch=None) -> np.ndarray:
    """Boolean mask of the top n genes of a flavour (raw counts in).

    All flavours except "seurat" (as in 01b) first drop genes detected in
    fewer than MIN_CELLS cells, the usual preprocessing; seurat_v3's loess
    fit fails on many genes with near-identical tiny means otherwise, and is
    retried with a wider span if it still fails."""
    base = flavour.removesuffix("_batch").removesuffix("_filtered")
    keep = np.ones(counts.shape[1], dtype=bool)
    if flavour != "seurat":
        keep = np.asarray((counts > 0).sum(axis=0)).ravel() >= MIN_CELLS
    a = sc.AnnData(counts[:, keep].astype(np.float32))
    key = None
    if flavour.endswith("_batch"):
        a.obs["library"] = pd.Categorical(batch)
        key = "library"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if base == "seurat":
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
            h = sc.pp.highly_variable_genes(a, n_top_genes=n, flavor="seurat", inplace=False)
        elif base == "seurat_v3":
            for span in SPANS:
                try:
                    h = sc.pp.highly_variable_genes(a, n_top_genes=n, flavor="seurat_v3",
                                                    batch_key=key, span=span, inplace=False)
                    break
                except ValueError:
                    print(f"  seurat_v3: loess failed with span {span}", flush=True)
            else:
                raise RuntimeError("seurat_v3 failed for every span")
        else:
            h = sc.experimental.pp.highly_variable_genes(a, n_top_genes=n, batch_key=key,
                                                         inplace=False)
    out = np.zeros(counts.shape[1], dtype=bool)
    out[keep] = np.asarray(h["highly_variable"], dtype=bool)
    return out


def selections(counts, batch, n: int, residual: pd.Series, names) -> dict:
    """Gene sets of every flavour for one stage and size."""
    out = {}
    for f in FLAVOURS:
        if f == "nb_residual":
            out[f] = set(residual.fillna(-np.inf).sort_values(ascending=False).index[:n])
        else:
            out[f] = set(names[hvg(counts, f, n, batch)])
    return out


def per_stage(adata, genes: pd.DataFrame, residual: pd.DataFrame, libs: pd.DataFrame,
              records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in STAGES:
        names_lib = list(libs.index[libs["stage"] == stage])
        measured = set(residual.index[residual[names_lib].notna().any(axis=1)])
        g = genes[genes["stage"] == stage].set_index("gene")
        het = set(g.index[g["heterogeneous"].astype(str) == "True"]) & measured
        sub = adata[adata.obs["stage"].astype(str) == stage]
        umis = pd.Series(np.asarray(sub.X.sum(axis=0)).ravel(), index=adata.var_names)
        keep = (umis > 0).to_numpy()
        counts = sub.X[:, keep]
        names = adata.var_names[keep]
        batch = sub.obs["library"].astype(str).to_numpy()
        groups = {k: v & measured for k, v in zfin.zfin_groups(records, stage).items()}
        for size, n in (("2000", N_HVG), ("matched", len(het))):
            sets = selections(counts, batch, n, g["median_residual"].reindex(names), names)
            if size == "matched":
                sets["heterogeneous"] = het
            for f, s in sets.items():
                rows.append(dict(
                    stage=stage, size=size, n=n, flavour=f,
                    heterogeneous=len(s & het), unmeasured=len(s - measured),
                    median_umis=float(umis[list(s)].median()),
                    restricted=len(s & groups["restricted"]) / max(len(groups["restricted"]), 1),
                    whole_organism=len(s & groups["whole organism"])
                    / max(len(groups["whole organism"]), 1),
                    restricted_genes=len(groups["restricted"])))
        print(stage, flush=True)
    return pd.DataFrame(rows)


def replication(adata, residual: pd.DataFrame, libs: pd.DataFrame) -> pd.DataFrame:
    library = adata.obs["library"].astype(str).to_numpy()
    top = {}
    for lib in libs.index:
        counts = adata.X[library == lib]
        keep = np.asarray(counts.sum(axis=0)).ravel() > 0
        names = adata.var_names[keep]
        top[lib] = {f: set(names[hvg(counts[:, keep], f, TOP_N)])
                    for f in LIBRARY_FLAVOURS if f != "nb_residual"}
        top[lib]["nb_residual"] = set(residual[lib].dropna().sort_values(ascending=False)
                                      .index[:TOP_N])
    rows = []
    for stage, group in libs.groupby("stage", sort=False):
        names = list(group.index)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                for f in LIBRARY_FLAVOURS:
                    x, y = top[a][f], top[b][f]
                    rows.append(dict(stage=stage, a=a, b=b, flavour=f,
                                     jaccard=len(x & y) / len(x | y)))
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, rep: pd.DataFrame) -> None:
    stages = [s for s in STAGES if s not in ("high", "oblong")]
    x = np.arange(len(stages))
    fig, axes = plt.subplots(1, 3, figsize=(REPORT_WIDTH_IN, 2.3))
    panels = [
        (axes[0], table[table["size"] == "2000"], "unmeasured",
         "a  top 2,000: genes too rare\nto measure", FLAVOURS),
        (axes[1], table[table["size"] == "matched"], "restricted",
         "b  ZFIN restricted genes in\ntop k (k = heterogeneous)", FLAVOURS + ["heterogeneous"]),
    ]
    for ax, t, col, title, flavours in panels:
        for f in flavours:
            v = t[t["flavour"] == f].set_index("stage").reindex(stages)[col]
            ax.plot(x, v, color=COLOURS[f], lw=1.2, ls=":" if f.endswith("_filtered") else "--" if f.endswith("_batch") else "-",
                    marker="o", ms=2.5, label=LABELS[f])
        ax.set_title(title, fontsize=7, loc="left")
    ax = axes[2]
    for f in LIBRARY_FLAVOURS:
        v = rep[rep["flavour"] == f].groupby("stage")["jaccard"].median().reindex(stages)
        ax.plot(x, v, color=COLOURS[f], lw=1.2, marker="o", ms=2.5)
    ax.set_title(f"c  top {TOP_N:,} of two libraries:\noverlap (Jaccard, median pair)",
                 fontsize=7, loc="left")
    for ax in axes:
        ax.set_xticks(x, stages, rotation=60, ha="right", fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.36, wspace=0.3)
    save_report_figure(fig, "hvg_flavours")
    plt.close(fig)


def latex_table(table: pd.DataFrame, rep: pd.DataFrame) -> None:
    stages = [s for s in STAGES if s not in ("high", "oblong")]

    def rng(v):
        return f"{v.min():.0f}--{v.max():.0f}"

    def prng(v):
        return f"{100 * v.min():.0f}--{100 * v.max():.0f}"

    lines = []
    for f in FLAVOURS + ["heterogeneous"]:
        t2 = table[(table["size"] == "2000") & (table["flavour"] == f) & table["stage"].isin(stages)]
        tm = table[(table["size"] == "matched") & (table["flavour"] == f)
                   & table["stage"].isin(stages)]
        r = rep[(rep["flavour"] == f) & rep["stage"].isin(stages)]
        jac = r.groupby("stage")["jaccard"].median()
        cells = [LABELS[f].replace(">=", "$\\geq$"),
                 rng(t2["heterogeneous"]) if len(t2) else "--",
                 rng(t2["unmeasured"]) if len(t2) else "--",
                 prng(tm["restricted"]), prng(tm["whole_organism"]),
                 f"{jac.min():.2f}--{jac.max():.2f}" if len(jac) else "--"]
        lines.append(" & ".join(cells) + " \\\\")
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    (REPORT_TABLES / "hvg_flavours.tex").write_text(
        "% Generated by scripts/01d_hvg_flavours.py; do not edit.\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Top 2{,}000} & \\multicolumn{2}{c}{ZFIN, top $k$} & \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "Method & Het. & Unmeas. & Restr.\\,\\% & Whole\\,\\% & Jaccard \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    ensure_dirs()
    report_style()
    sc.settings.verbosity = 0
    adata = load_dropseq_counts()
    genes = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t", low_memory=False)
    residual = pd.read_csv(TABLES / "heterogeneous_genes_libraries.tsv.gz", sep="\t",
                           index_col=0)
    libs = pd.read_csv(TABLES / "heterogeneity_libraries.tsv", sep="\t", index_col=0)
    records, _ = zfin.zfin_genes(adata)
    rep = replication(adata, residual, libs)
    rep.to_csv(TABLES / "hvg_flavours_replication.tsv", sep="\t", index=False,
               float_format="%.4g")
    print(rep.groupby(["flavour", "stage"], sort=False)["jaccard"].median().unstack(0).round(2))
    table = per_stage(adata, genes, residual, libs, records)
    table.to_csv(TABLES / "hvg_flavours.tsv", sep="\t", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    latex_table(table, rep)
    plot(table, rep)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["plot"]:
        report_style()
        t = pd.read_csv(TABLES / "hvg_flavours.tsv", sep="\t")
        r = pd.read_csv(TABLES / "hvg_flavours_replication.tsv", sep="\t")
        latex_table(t, r)
        plot(t, r)
    else:
        main()
