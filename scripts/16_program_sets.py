"""Stage-specific sets of expression programs.

A program is a set of genes associated with one cell type, tissue, state or
signalling pathway at one stage. Sets:

  farrell          the NMF gene modules of Farrell et al. (Table S2): the
                   30 listed genes of each module of the stage (weights
                   kept). Found in these cells.
  zfin_pre         one program per ZFIN anatomy term at the stage: genes with
  zfin_all         an mRNA in situ record in the term whose stage interval
                   covers the atlas stage (01c_heterogeneity_zfin.zfin_genes;
                   pre-2018 publications or all). Broad terms (whole organism,
                   unspecified, ...) and yolk/YSL terms are left out; a gene
                   needs MIN_PUBS publications in the term; terms with fewer
                   than MIN_GENES genes are dropped, and a term whose genes are
                   mostly (DUPLICATE_OVERLAP) those of a larger term of the
                   stage is merged into it.
  zfin_pre_het     the same, restricted to the genes heterogeneous at the
  zfin_all_het     stage (script 01b, weak tier); MIN_GENES applies after the
                   restriction.
  signalling       readouts of signalling pathways (feedback inhibitors and
                   direct targets, data/curated/signalling_programs.tsv),
                   restricted to genes heterogeneous at the stage, at least
                   MIN_SIGNAL_GENES of them.
  states           maternal, blastomere, cell cycle S / G2M, heat shock and
                   p53 response (data/curated/archive/expression_programs.tsv;
                   not anatomy, so absent from ZFIN's anatomy terms). The same
                   genes at every stage; maternal and blastomere are expected
                   to act on whole libraries.
  curated_staged   first draft, stage-specific: ZFIN terms (all years)
                   grouped into programs by data/curated/zfin_anatomy_programs.tsv
                   (e.g. shield, embryonic shield and organizer ->
                   dorsal_organizer), so that a program keeps its name across
                   stages; a gene belongs to a group if it has MIN_PUBS
                   publications in the group's terms at the stage and the
                   group holds at least MIN_PUB_SHARE of its publications in
                   specific terms at the stage; restricted to genes
                   heterogeneous at the stage; plus the signalling and state
                   programs. ZFIN's maternal and blastomere groups are left
                   out (EXCLUDED_GROUPS). Its genes change from stage to stage
                   (median program: 23 genes over its stages, 1 in all).
  curated          the same programs with one gene list for all stages of a
                   window, so that a program means the same at every stage:
                   * groups whose identity changes are split (column
                     fixed_program of the mapping table: margin / mesoderm,
                     axial mesoderm / notochord, prechordal plate / hatching
                     gland, presumptive neural / neural plate);
                   * window: the stages from the first to the last at which
                     the stage-specific group exists (as in curated_staged);
                   * genes: the group's evidence pooled over the records
                     overlapping the window (MIN_PUBS, MIN_PUB_SHARE as
                     above), heterogeneous at >= 1 stage of the window,
                     ranked by their mean heterogeneity tier over the window
                     (0 = none, 1-3 = weak / moderate / strong; script 01b),
                     then by publications, and capped at CAP;
                   * signalling: the pathway's readouts heterogeneous at >= 1
                     stage of its window (stages from the first to the last
                     with >= MIN_SIGNAL_GENES heterogeneous readouts);
                   * states: as in `states` (already the same genes).
                   At every stage of a window, genes not measured at the stage
                   (< MIN_UMIS) are left out; otherwise the list is fixed.
Every set is restricted to genes with at least MIN_UMIS UMIs in the stage and
capped at CAP genes per program (most publications first; Farrell et al.:
highest weight first; curated: see above).

Outputs:
  results/tables/programs.tsv           (set, stage, program, gene, weight,
                                         pubs, rank, kept)
  results/tables/programs_summary.tsv   (per set and stage)
  report/tables/programs.tex
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from pipeline_utils import REPORT_TABLES, ROOT, STAGES, TABLES, ensure_dirs, load_dropseq_counts
from program_utils import farrell_modules

zfin_check = importlib.import_module("01c_heterogeneity_zfin")

MAPPING = ROOT / "data" / "curated" / "zfin_anatomy_programs.tsv"
SIGNALLING = ROOT / "data" / "curated" / "signalling_programs.tsv"
STATES_TSV = ROOT / "data" / "curated" / "archive" / "expression_programs.tsv"
STATES = ["maternal", "blastomere", "cell_cycle_s", "cell_cycle_g2m", "heat_shock",
          "p53_response"]
MIN_UMIS = 10
MIN_PUBS = 1
MIN_GENES = 3
MIN_SIGNAL_GENES = 2
DUPLICATE_OVERLAP = 0.8
MIN_PUB_SHARE = 1 / 3
CAP = 30
# Groups left out of the curated set: broad terms, the yolk syncytial layer
# (deyolked embryos), and ZFIN's maternal / blastomere terms, which at later
# stages collect ventral and marginal genes; the literature state programs
# describe these states instead.
EXCLUDED_GROUPS = ["(broad)", "program", "YSL", "maternal", "blastomere"]
SETS = ["farrell", "zfin_pre", "zfin_all", "zfin_pre_het", "zfin_all_het", "signalling",
        "states", "curated_staged", "curated"]
TIERS = ["weak", "moderate", "strong"]


def at_stage(records: pd.DataFrame, stage: str) -> pd.DataFrame:
    hpf = zfin_check.ATLAS_HPF[stage]
    return records[(records["start_hpf"] <= hpf) & (records["stop"] > hpf)
                   & (records["kind"] == "restricted")]


def term_programs(records: pd.DataFrame, allowed: set) -> pd.DataFrame:
    """One program per term; duplicates merged into the larger term."""
    ev = (records[records["gene"].isin(allowed)].groupby(["term", "gene"])
          .agg(pubs=("pub", "nunique")).reset_index())
    ev = ev[ev["pubs"] >= MIN_PUBS]
    sizes = ev.groupby("term").size()
    terms = sizes[sizes >= MIN_GENES].sort_values(ascending=False).index
    genes = {t: set(ev.loc[ev["term"] == t, "gene"]) for t in terms}
    kept = []
    for t in terms:
        if not any(len(genes[t] & genes[k]) >= DUPLICATE_OVERLAP * len(genes[t]) for k in kept):
            kept.append(t)
    ev = ev[ev["term"].isin(kept)].rename(columns={"term": "program"})
    return ev.sort_values(["program", "pubs"], ascending=[True, False])


def group_programs(records: pd.DataFrame, mapping: pd.Series, allowed: set) -> pd.DataFrame:
    """Terms grouped by the mapping table; genes assigned to their main groups."""
    r = records.assign(program=records["term"].map(mapping))
    r = r[r["program"].notna() & ~r["program"].isin(EXCLUDED_GROUPS)]
    ev = r.groupby(["program", "gene"]).agg(pubs=("pub", "nunique")).reset_index()
    share = ev["pubs"] / ev.groupby("gene")["pubs"].transform("sum")
    ev = ev[(ev["pubs"] >= MIN_PUBS) & (share >= MIN_PUB_SHARE) & ev["gene"].isin(allowed)]
    sizes = ev.groupby("program").size()
    ev = ev[ev["program"].map(sizes) >= MIN_GENES]
    return ev.sort_values(["program", "pubs"], ascending=[True, False])


def windows(present: dict[str, list[int]]) -> dict[str, list[str]]:
    """Stages from the first to the last stage (indices) of every program."""
    return {p: STAGES[min(i):max(i) + 1] for p, i in present.items() if i}


def fixed_group_programs(records: pd.DataFrame, mapping: pd.Series, tier: pd.DataFrame,
                         measured: dict[str, set]) -> pd.DataFrame:
    """Anatomy programs of the curated set: one ranked gene list per window."""
    present = {}
    for i, stage in enumerate(STAGES):
        het = set(tier.index[tier[stage] >= 1]) & measured[stage]
        for p in group_programs(at_stage(records, stage), mapping, het)["program"].unique():
            present.setdefault(p, []).append(i)
    rows = []
    for program, window in windows(present).items():
        first, last = zfin_check.ATLAS_HPF[window[0]], zfin_check.ATLAS_HPF[window[-1]]
        r = records[(records["start_hpf"] <= last) & (records["stop"] > first)
                    & (records["kind"] == "restricted")]
        het_any = set(tier.index[(tier[window] >= 1).any(axis=1)])
        ev = group_programs(r, mapping, het_any)
        ev = ev[ev["program"] == program].copy()
        if ev.empty:
            continue
        ev["tier"] = tier.loc[ev["gene"], window].mean(axis=1).to_numpy()
        ev = ev.sort_values(["tier", "pubs"], ascending=False)
        ev["rank"] = np.arange(1, len(ev) + 1)
        rows += [ev.assign(stage=stage)[ev["gene"].isin(measured[stage])] for stage in window]
    return pd.concat(rows, ignore_index=True)


def fixed_signalling_programs(tier: pd.DataFrame, measured: dict[str, set]) -> pd.DataFrame:
    t = pd.read_csv(SIGNALLING, sep="\t")
    t = t[t["gene"].isin(tier.index)]
    het = tier.loc[t["gene"]].to_numpy() >= 1
    present = {}
    for pathway in t["pathway"].unique():
        on = het[(t["pathway"] == pathway).to_numpy()]
        present[pathway] = [i for i in range(len(STAGES)) if on[:, i].sum() >= MIN_SIGNAL_GENES]
    rows = []
    for pathway, window in windows(present).items():
        genes = t.loc[t["pathway"] == pathway, "gene"]
        genes = [g for g in genes if (tier.loc[g, window] >= 1).any()]
        rows += [dict(program="signal:" + pathway, gene=g, stage=stage, rank=k + 1)
                 for stage in window for k, g in enumerate(genes) if g in measured[stage]]
    return pd.DataFrame(rows)


def signalling_programs(allowed: set) -> pd.DataFrame:
    t = pd.read_csv(SIGNALLING, sep="\t")
    t = t[t["gene"].isin(allowed)]
    sizes = t.groupby("pathway").size()
    t = t[t["pathway"].map(sizes) >= MIN_SIGNAL_GENES]
    return pd.DataFrame(dict(program="signal:" + t["pathway"], gene=t["gene"]))


def state_programs(allowed: set) -> pd.DataFrame:
    t = pd.read_csv(STATES_TSV, sep="\t").set_index("program")
    rows = [dict(program=f"state:{p}", gene=g) for p in STATES
            for g in t.loc[p, "genes"].split(",") if g in allowed]
    return pd.DataFrame(rows)


def farrell_programs(stage: str, modules: dict, allowed: set) -> pd.DataFrame:
    rows = [dict(program=f"farrell:{stage}:{i}", gene=g, weight=w)
            for i, weights in enumerate(modules.get(stage, [])) for g, w in weights.items()
            if g in allowed]
    return pd.DataFrame(rows).sort_values(["program", "weight"], ascending=[True, False])


def finish(t: pd.DataFrame, set_name: str, stage: str) -> pd.DataFrame:
    if t.empty:
        return t
    t = t.copy()
    t["set"], t["stage"] = set_name, stage
    t["rank"] = t.groupby("program").cumcount() + 1
    t["kept"] = t["rank"] <= CAP
    return t


def finish_fixed(t: pd.DataFrame) -> pd.DataFrame:
    """Like finish(), with the rank of the fixed list (over the window)."""
    t = t.copy()
    t["set"] = "curated"
    if "rank" not in t:
        t["rank"] = t.groupby(["stage", "program"]).cumcount() + 1
    t["kept"] = t["rank"] <= CAP
    return t


def latex_table(summary: pd.DataFrame) -> None:
    names = {"farrell": "Farrell et al.\\ modules", "zfin_pre": "ZFIN terms, pre-2018",
             "zfin_all": "ZFIN terms, all", "zfin_pre_het": "ZFIN terms, pre-2018, het.",
             "zfin_all_het": "ZFIN terms, all, het.", "signalling": "Signalling",
             "states": "Cell states", "curated_staged": "Curated, stage-specific genes",
             "curated": "Curated, fixed genes"}
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Set & Programs per stage & Genes per program & Genes per stage & Genes \\\\",
             "\\midrule"]
    for s in SETS:
        t = summary[summary["set"] == s]
        lines.append(f"{names[s]} & {t['programs'].min()}--{t['programs'].max()} & "
                     f"{t['median_genes'].median():.0f} & {t['genes'].min()}--{t['genes'].max()}"
                     f" & {int(t['all_genes'].iat[0]):,} \\\\".replace(",", "{,}"))
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPORT_TABLES / "programs.tex").write_text(
        "% Generated by scripts/16_program_sets.py; do not edit.\n" + "\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    adata = load_dropseq_counts()
    table = pd.read_csv(MAPPING, sep="\t", keep_default_na=False).set_index("term")
    mapping = table["program"]
    fixed_mapping = table["fixed_program"].where(table["fixed_program"] != "", mapping)
    het_table = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                            usecols=["stage", "gene", "heterogeneous"] + TIERS)
    tier = (het_table.assign(tier=het_table[TIERS].astype(bool).sum(axis=1))
            .pivot(index="gene", columns="stage", values="tier").reindex(columns=STAGES)
            .fillna(0))
    records = {"pre": zfin_check.zfin_genes(adata, zfin_check.MAX_YEAR)[0],
               "all": zfin_check.zfin_genes(adata, None)[0]}
    modules = farrell_modules()
    parts, measured = [], {}
    for stage in STAGES:
        sub = adata[adata.obs["stage"] == stage]
        total = np.asarray(sub.X.sum(axis=0)).ravel()
        measured[stage] = set(sub.var_names[total >= MIN_UMIS])
        het = set(het_table.loc[(het_table["stage"] == stage) & het_table["heterogeneous"],
                                "gene"]) & measured[stage]
        rec = {k: at_stage(r, stage) for k, r in records.items()}
        signalling = signalling_programs(het)
        states = state_programs(measured[stage])
        parts += [
            finish(farrell_programs(stage, modules, measured[stage]), "farrell", stage),
            finish(term_programs(rec["pre"], measured[stage]), "zfin_pre", stage),
            finish(term_programs(rec["all"], measured[stage]), "zfin_all", stage),
            finish(term_programs(rec["pre"], het), "zfin_pre_het", stage),
            finish(term_programs(rec["all"], het), "zfin_all_het", stage),
            finish(signalling, "signalling", stage),
            finish(states, "states", stage),
            finish(pd.concat([group_programs(rec["all"], mapping, het), signalling, states]),
                   "curated_staged", stage),
            finish_fixed(states.assign(stage=stage)),
        ]
    parts += [finish_fixed(fixed_group_programs(records["all"], fixed_mapping, tier, measured)),
              finish_fixed(fixed_signalling_programs(tier, measured))]
    long = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    long = long[["set", "stage", "program", "gene", "weight", "pubs", "tier", "rank", "kept"]]
    long.to_csv(TABLES / "programs.tsv", sep="\t", index=False, float_format="%.4g")
    kept = long[long["kept"]]
    summary = (kept.groupby(["set", "stage"])
               .agg(programs=("program", "nunique"), genes=("gene", "nunique"))
               .reset_index())
    median = kept.groupby(["set", "stage", "program"]).size().groupby(["set", "stage"]).median()
    summary["median_genes"] = summary.set_index(["set", "stage"]).index.map(median)
    summary["all_genes"] = summary["set"].map(kept.groupby("set")["gene"].nunique())
    summary["stage"] = pd.Categorical(summary["stage"], STAGES, ordered=True)
    summary = summary.sort_values(["set", "stage"])
    summary.to_csv(TABLES / "programs_summary.tsv", sep="\t", index=False)
    print(summary.pivot(index="stage", columns="set", values="programs").to_string())
    print(summary.pivot(index="stage", columns="set", values="genes").to_string())
    latex_table(summary)


if __name__ == "__main__":
    main()
