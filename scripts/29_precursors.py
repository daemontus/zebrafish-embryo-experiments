"""Where do populations emerge? The closest programs of a program's first cells.

Precursors: the kept cells with an identity program on top (script 26) at
stages before the program is supported (script 17), at least MIN_CELLS of them
per stage: cells that carry the program before it is coherent over a library.
The maternal state (no parent) is left out.

For the precursors of a program and stage:
  * runner-up: the identity program with the largest z after the top one
    (competitors of the stage), classified by its relation to the program
    (derives_from of curated_programs.tsv: ancestor, descendant, sibling =
    a common ancestor other than the maternal and blastomere states,
    unrelated). Parent and child programs share genes by design, which pulls
    the parent up wherever the child is high; the baseline is therefore the
    same share in the program's top cells at its supported stages. A parent
    that is much more often the runner-up in the precursors than later marks
    emergence, not only shared genes.
  * neighbours: the top programs of their K nearest neighbours in the library
    (PCA of the stage's heterogeneous genes without the excluded state genes
    of script 26), as a share and relative to the library's composition (cells
    with the program itself or no top program left out).

`29_precursors.py plot` rewrites the report table from the saved tables.

Outputs:
  results/tables/precursors.tsv             program x precursor stage: cells,
      runner-ups, ancestor share (precursors, supported stages), neighbours
  results/tables/precursor_neighbours.tsv   program x stage x library x
      neighbour program: share, enrichment, relation
  report/tables/precursors.tex
"""
from __future__ import annotations

import importlib
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from pipeline_utils import (
    REPORT_TABLES,
    STAGES,
    TABLES,
    ensure_dirs,
    load_dropseq_counts,
    pearson_residuals,
)
from program_utils import N_PCS

warnings.filterwarnings("ignore", category=FutureWarning)
cell_programs = importlib.import_module("26_cell_programs")

MIN_CELLS = 20
K = 15
LEFT_OUT = ("state:maternal",)
SEED = 0
def relation(winner: str, program: str, anc: dict) -> str:
    return cell_programs.relation(winner, program, anc)


def first_supported(table: pd.DataFrame) -> pd.Series:
    sup = table[table["supported"].astype(bool)]
    return sup.groupby("program")["stage"].agg(lambda s: min(s, key=STAGES.index))


def library_neighbours(adata, kept: pd.DataFrame, lib: str, stage: str, het: pd.DataFrame,
                       programs: pd.DataFrame) -> pd.DataFrame:
    cells = kept.index[kept["library"] == lib]
    genes = set(het.loc[(het["stage"] == stage) & het["heterogeneous"], "gene"]) - set(
        programs.loc[(programs["stage"] == stage)
                     & programs["program"].isin(cell_programs.EXCLUDED_GENE_PROGRAMS), "gene"])
    sub = adata[cells]
    res = pearson_residuals(sub.X.tocsr(), np.flatnonzero(sub.var_names.isin(sorted(genes))))
    pcs = PCA(min(N_PCS, len(cells) - 1), random_state=SEED).fit_transform(res)
    nb = NearestNeighbors(n_neighbors=K + 1).fit(pcs).kneighbors(return_distance=False)[:, 1:]
    return pd.DataFrame(kept.loc[cells, "top"].to_numpy()[nb], index=cells)


def compute() -> None:
    table = pd.read_csv(TABLES / "cell_programs.tsv", sep="\t")
    cells = pd.read_csv(cell_programs.CELLS, sep="\t", index_col=0, keep_default_na=False)
    z = pd.read_csv(cell_programs.Z_TABLE, sep="\t", index_col=0)
    programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    programs = programs[programs["kept"]]
    het = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                      usecols=["stage", "gene", "heterogeneous"])
    anc = cell_programs.ancestors_of()
    kept = cells[cells["excluded"] == ""]
    first = first_supported(table)
    counts = kept[kept["top"] != ""].groupby(["top", "stage"]).size()
    adata = load_dropseq_counts()
    neighbours = {}

    def runner_ups(idx, stage, p):
        comp = [q for q in table.loc[table["stage"] == stage, "program"] if q != p]
        return z.loc[idx, comp].idxmax(axis=1)
    rows, nrows = [], []
    for (p, stage), n in counts.items():
        f = first.get(p)
        if p in LEFT_OUT or n < MIN_CELLS or (f is not None
                                             and STAGES.index(stage) >= STAGES.index(f)):
            continue
        idx = kept.index[(kept["stage"] == stage) & (kept["top"] == p)]
        runner = runner_ups(idx, stage, p)
        later = [s for s in STAGES if f is not None and STAGES.index(s) >= STAGES.index(f)]
        late_idx = kept.index[(kept["top"] == p) & kept["stage"].isin(later)]
        late = pd.concat([runner_ups(i, s, p) for s, i in
                          kept.loc[late_idx].groupby("stage").groups.items()]) \
            if len(late_idx) else pd.Series(dtype=str)
        for lib, lib_idx in kept.loc[idx].groupby("library").groups.items():
            if lib not in neighbours:
                neighbours[lib] = library_neighbours(adata, kept, lib, stage, het, programs)
            nb = neighbours[lib].loc[lib_idx].to_numpy().ravel()
            nb = nb[(nb != p) & (nb != "")]
            comp = kept.loc[(kept["library"] == lib) & (kept["top"] != "")
                            & (kept["top"] != p), "top"].value_counts(normalize=True)
            share = pd.Series(nb).value_counts(normalize=True)
            for q, x in share.items():
                nrows.append(dict(program=p, stage=stage, library=lib, cells=len(lib_idx),
                                  neighbour=q, share=x, enrichment=x / comp.get(q, np.nan),
                                  relation=relation(q, p, anc)))
        rv = runner.value_counts()
        rows.append(dict(program=p, stage=stage, cells=n,
                         runner_ups=";".join(f"{q}:{k}" for q, k in rv.items()),
                         ancestor=runner.map(lambda q: relation(q, p, anc) == "ancestor").mean(),
                         ancestor_supported=late.map(
                             lambda q: relation(q, p, anc) == "ancestor").mean()
                         if len(late) else np.nan,
                         supported_cells=len(late)))
    pd.DataFrame(rows).to_csv(TABLES / "precursors.tsv", sep="\t", index=False,
                              float_format="%.3f")
    pd.DataFrame(nrows).to_csv(TABLES / "precursor_neighbours.tsv", sep="\t", index=False,
                               float_format="%.4g")


# --- figure and table --------------------------------------------------------------------------

def plot() -> None:
    pre = pd.read_csv(TABLES / "precursors.tsv", sep="\t")
    nb = pd.read_csv(TABLES / "precursor_neighbours.tsv", sep="\t")
    latex(pre, nb)


def latex(pre: pd.DataFrame, nb: pd.DataFrame) -> None:
    def tex(s):
        return str(s).replace("%", "\\%").replace("_", " ").replace("state:", "")
    lines = ["% Generated by scripts/29_precursors.py; do not edit.",
             "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{2.6cm}"
             ">{\\raggedright\\arraybackslash}p{1.6cm}r>{\\raggedright\\arraybackslash}p{4.3cm}"
             "rr>{\\raggedright\\arraybackslash}p{3.8cm}@{}}",
             "\\caption{Precursors: cells with a program on top before the program is supported "
             "(\\protect\\nolinkurl{scripts/29_precursors.py}). Runner-up: the program with the "
             "next largest $z$ in these cells (three most frequent). Ancestor: share of the "
             "precursors whose runner-up is an ancestor of the program, and the same share in "
             "the program's top cells at its supported stages (parent and child programs share "
             f"genes, so the latter is the baseline). Neighbours: top programs of their {K} "
             "nearest neighbours in the library (share, mean over libraries weighted by their "
             "precursor cells; in brackets the enrichment over the library's composition).}\\label{tab:precursors}\\\\",
             "\\toprule", "Program & Stage & Cells & Runner-up & \\multicolumn{2}{c}{Ancestor} "
             "& Neighbours \\\\", "& & & & prec. & later & \\\\", "\\midrule", "\\endfirsthead",
             "\\toprule", "Program & Stage & Cells & Runner-up & \\multicolumn{2}{c}{Ancestor} "
             "& Neighbours \\\\", "& & & & prec. & later & \\\\", "\\midrule", "\\endhead",
             "\\bottomrule", "\\endlastfoot"]
    order = {p: k for k, p in enumerate(cell_programs.program_order(pre["program"].unique()))}
    pre = pre.assign(_p=pre["program"].map(order), _s=pre["stage"].map(STAGES.index)) \
        .sort_values(["_p", "_s"])
    for i, r in enumerate(pre.itertuples()):
        cnt = pd.Series({q: int(k) for q, k in (s.rsplit(":", 1)
                                                for s in r.runner_ups.split(";"))})
        ru = ", ".join(f"{tex(q)} {k / cnt.sum():.0%}" for q, k in cnt.head(3).items())
        d = nb[(nb["program"] == r.program) & (nb["stage"] == r.stage)]
        total = d.drop_duplicates("library")["cells"].sum()
        g = d.assign(w=d["share"] * d["cells"], we=d["enrichment"] * d["cells"]) \
            .groupby("neighbour")[["w", "we"]].sum() / total
        g = g.rename(columns={"w": "share", "we": "enr"}).sort_values("share",
                                                                       ascending=False).head(3)
        nbs = ", ".join(f"{tex(q)} {v.share:.0%} ({v.enr:.1f}x)" for q, v in g.iterrows())
        later = "--" if pd.isna(r.ancestor_supported) else f"{r.ancestor_supported:.0%}"
        lines.append(("\\rowcolor{gray!10}" if i % 2 else "") + " & ".join([
            tex(r.program), tex(r.stage), str(r.cells), tex(ru), f"{r.ancestor:.0%}".replace(
                "%", "\\%"), later.replace("%", "\\%"), tex(nbs)]) + " \\\\")
    lines.append("\\end{longtable}")
    (REPORT_TABLES / "precursors.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ensure_dirs()
    if sys.argv[1:] != ["plot"]:
        compute()
    plot()
