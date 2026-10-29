"""Do the heterogeneous genes (script 01b) reflect known expression patterns?

ZFIN wild-type expression records (data/raw/zfin/, scripts/download_zfin.sh)
are reduced to:
  * mRNA in situ hybridisation (the assay that shows where a gene is
    expressed; RT-PCR and similar record the whole embryo);
  * publications before 2018 (independent of Farrell et al. 2018; year as in
    zfin_utils.zfin_records);
  * genes of the Drop-seq matrix (ZFIN gene id -> matrix symbol via current
    symbols and unambiguous aliases, zfin_utils.matrix_symbols).
A record covers a time interval: from the start of its first ZFIN stage to
the start of the stage after its last one (ZFIN stages are intervals, e.g.
Shield is 6-8 hpf). It covers an atlas stage if the interval contains the
atlas stage's time (ATLAS_HPF; the start of the corresponding ZFIN stage).
At each atlas stage, a gene is
  restricted        if a covering record names a localised anatomy term:
                    any term except the broad ones of
                    data/curated/zfin_anatomy_programs.tsv ("(broad)": whole
                    organism, unspecified, ...) and the yolk/YSL terms (the
                    atlas embryos were deyolked);
  whole organism    if all its covering records name broad terms only.
Genes measurable at a stage (>= 10 UMIs in at least one of its libraries) are
compared with the heterogeneous genes of the stage (script 01b; weak tier, and
in the TSV also the moderate and strong tiers), with as many top Seurat HVGs
of the pooled stage, and with the usual 2,000 HVGs.

Outputs:
  results/tables/heterogeneity_zfin.tsv   (per stage and gene group)
  report/tables/heterogeneity_zfin.tex
"""
from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd

from pipeline_utils import REPORT_TABLES, STAGES, TABLES, load_dropseq_counts
from zfin_utils import MAPPING, MAX_YEAR, ZFIN_STAGE_HPF, matrix_symbols, zfin_records

warnings.filterwarnings("ignore", category=FutureWarning)
heterogeneity = importlib.import_module("01b_heterogeneous_genes")

ASSAY = "mRNA in situ hybridization"
ATLAS_HPF = {"high": 3.33, "oblong": 3.67, "dome": 4.33, "30% epiboly": 4.67,
             "50% epiboly": 5.25, "shield": 6.0, "60% epiboly": 7.0, "75% epiboly": 8.0,
             "90% epiboly": 9.0, "bud": 10.0, "3-somite": 11.0, "6-somite": 12.0}
GROUPS = ["restricted", "whole organism"]


def zfin_genes(adata, max_year: int | None = MAX_YEAR) -> tuple[pd.DataFrame, dict]:
    """In situ records with matrix gene, kind (restricted / broad / yolk) and
    the covered hpf interval [start, stop)."""
    records, symbols = zfin_records(max_year)
    mapping = pd.read_csv(MAPPING, sep="\t").set_index("term")["program"]
    starts = np.array(sorted(set(ZFIN_STAGE_HPF.values())))
    r = records[records["assay"] == ASSAY].copy()
    info = dict(records=len(r), genes=r["gene_id"].nunique(), pubs=r["pub"].nunique())
    r["gene"] = r["gene_id"].map(matrix_symbols(adata, symbols))
    r = r[r["gene"].notna()].copy()
    info.update(matched_records=len(r), matched_genes=r["gene"].nunique())
    following = np.searchsorted(starts, r["end_hpf"].to_numpy(), side="right")
    r["stop"] = np.where(following < len(starts), starts[np.minimum(following, len(starts) - 1)],
                         np.inf)
    term = r["term"].map(mapping)
    r["kind"] = np.where(term == "(broad)", "broad", "restricted")
    yolk = (term == "YSL") | r["term"].str.contains("yolk")
    r.loc[yolk.to_numpy(), "kind"] = "yolk"
    return r, info


def zfin_groups(records: pd.DataFrame, stage: str) -> dict[str, set]:
    """Restricted and whole-organism genes of the records covering the stage."""
    hpf = ATLAS_HPF[stage]
    at = records[(records["start_hpf"] <= hpf) & (records["stop"] > hpf)]
    kinds = at.groupby("gene")["kind"].agg(set)
    return {"restricted": set(kinds.index[kinds.map(lambda k: "restricted" in k)]),
            "whole organism": set(kinds.index[kinds.map(lambda k: k == {"broad"})])}


def main() -> None:
    adata = load_dropseq_counts()
    records, info = zfin_genes(adata)
    print(info)
    genes = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t", low_memory=False)
    residual = pd.read_csv(TABLES / "heterogeneous_genes_libraries.tsv.gz", sep="\t",
                           index_col=0)
    libs = pd.read_csv(TABLES / "heterogeneity_libraries.tsv", sep="\t", index_col=0)
    rows = []
    for stage in STAGES:
        names = list(libs.index[libs["stage"] == stage])
        measured = set(residual.index[residual[names].notna().any(axis=1)])
        g = genes[genes["stage"] == stage]
        tiers = {t: set(g.loc[g[t].astype(str) == "True", "gene"]) & measured
                 for t in heterogeneity.TIERS}
        het = tiers["weak"]
        sub = adata[adata.obs["stage"].astype(str) == stage]
        order = adata.var_names[np.argsort(-heterogeneity.seurat_dispersion(sub.X))]
        hvg_matched, hvg_2000 = set(order[:len(het)]), set(order[:heterogeneity.N_HVG])
        group = zfin_groups(records, stage)
        for name in GROUPS:
            m = group[name] & measured
            n = max(len(m), 1)
            rows.append(dict(stage=stage, group=name, zfin_genes=len(group[name]),
                             measured_genes=len(m), heterogeneous_genes=len(het),
                             all_measured=len(measured), background=len(het) / len(measured),
                             heterogeneous=len(m & het) / n, hvg_matched=len(m & hvg_matched) / n,
                             hvg_2000=len(m & hvg_2000) / n,
                             **{f"{t}_genes": len(x) for t, x in tiers.items()},
                             **{t: len(m & x) / n for t, x in tiers.items()}))
    out = pd.DataFrame(rows)
    out["enrichment"] = out["heterogeneous"] / out["background"]
    out.to_csv(TABLES / "heterogeneity_zfin.tsv", sep="\t", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    latex_table(out)


def latex_table(out: pd.DataFrame) -> None:
    def n(x):
        return f"{int(x):,}".replace(",", "{,}")

    def pct(x):
        return f"{100 * x:.0f}"

    wide = out.set_index(["stage", "group"])
    lines = []
    for stage in STAGES:
        r, w = wide.loc[(stage, "restricted")], wide.loc[(stage, "whole organism")]
        lines.append(
            f"{stage.replace('%', chr(92) + '%')} & {n(r['all_measured'])} & {pct(r['background'])} & "
            f"{n(r['measured_genes'])} & {pct(r['heterogeneous'])} & {pct(r['hvg_matched'])} & "
            f"{pct(r['hvg_2000'])} & {n(w['measured_genes'])} & {pct(w['heterogeneous'])} & "
            f"{pct(w['hvg_matched'])} & {pct(w['hvg_2000'])} \\\\")
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    (REPORT_TABLES / "heterogeneity_zfin.tex").write_text(
        "% Generated by scripts/01c_heterogeneity_zfin.py; do not edit.\n"
        "\\begin{tabular}{lrr|rrrr|rrrr}\n\\toprule\n"
        " & \\multicolumn{2}{c|}{All measured} & \\multicolumn{4}{c|}{ZFIN: restricted} & "
        "\\multicolumn{4}{c}{ZFIN: whole organism only} \\\\\n"
        "Stage & Genes & Het.\\,\\% & Genes & Het.\\,\\% & HVG$_k$\\,\\% & HVG$_{2000}$\\,\\% & "
        "Genes & Het.\\,\\% & HVG$_k$\\,\\% & HVG$_{2000}$\\,\\% \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    main()
