"""Sets of expression programs.

A program is a set of genes associated with one cell type, tissue, state or
signalling pathway. All sets except `curated` are stage-specific (a gene list
per stage). Sets:

  farrell          the NMF gene modules of Farrell et al. (Table S2): the
                   30 listed genes of each module of the stage (weights
                   kept). Found in these cells.
  zfin_all         one program per ZFIN anatomy term at the stage: genes with
                   an mRNA in situ record in the term whose stage interval
                   covers the atlas stage (01c_heterogeneity_zfin.zfin_genes,
                   all years). Broad terms (whole organism, unspecified, ...)
                   are left out; the yolk and YSL terms are kept (ZFIN_KINDS)
                   as a negative control, since the embryos were deyolked
                   (01c leaves them out); a gene needs MIN_PUBS publications
                   in the term; terms with fewer than MIN_GENES genes are
                   dropped. Nested terms (e.g. periderm in EVL) are kept as
                   programs of their own.
  zfin_all_het     the same, restricted to the genes heterogeneous at the
                   stage (script 01b, weak tier); MIN_GENES applies after the
                   restriction.
  zfin_pre_het     zfin_all_het with pre-2018 publications only (records
                   independent of the atlas); a sanity check, not reported.
  signalling       the signalling profiles of `curated` (see there), as a set
                   of their own: one gene list for all stages.
  states           maternal, blastomere, cell cycle S / G2M, heat shock and
                   p53 response (data/curated/archive/expression_programs.tsv;
                   not anatomy, so absent from ZFIN's anatomy terms). The same
                   genes at every stage; maternal and blastomere are expected
                   to act on whole libraries.
  curated          ZFIN terms (all years) grouped into programs by
                   data/curated/zfin_anatomy_programs.tsv (e.g. shield,
                   embryonic shield and organizer -> dorsal_organizer), with
                   one gene list for all stages, so that a program means the
                   same at every stage and the data, not ZFIN's stages,
                   show when it varies; plus the signalling profiles and
                   the state programs:
                   * at a stage, a gene belongs to a group if it has MIN_PUBS
                     publications in the group's terms and the group holds at
                     least MIN_PUB_SHARE of its publications in specific
                     terms (genes in data/curated/gene_overrides.tsv: in
                     the group given there); ZFIN's maternal and blastomere
                     groups, the ventral mesoderm, the floor plate, the
                     head mesoderm and the hemato-vascular group are left out (EXCLUDED_GROUPS, see there); the YSL group is built
                     the same way but becomes a control (see `controls`);
                   * groups whose identity changes are split (column
                     fixed_program of the mapping table: margin / mesoderm,
                     axial mesoderm / notochord, prechordal plate / hatching
                     gland);
                   * genes: the group's evidence pooled over all records
                     overlapping the atlas (high to 6-somite; MIN_PUBS,
                     MIN_PUB_SHARE as above), heterogeneous at >= 1 stage,
                     ranked by their mean heterogeneity tier over all stages
                     (0 = none, 1-3 = weak / moderate / strong; script 01b),
                     then by publications, and capped at CAP;
                   * ZFIN window (curated_zfin_windows.tsv, for comparison
                     only): the stages from the first to the last at which
                     the group has >= MIN_GENES genes recorded;
                   * disjoint: genes of the signalling, state and
                     pathway-control programs leave the anatomy programs,
                     and two anatomy programs share no gene unless one derives
                     from the other, directly or not (column derives_from of
                     data/curated/curated_programs.tsv: prior knowledge); a shared gene
                     stays in the program with more publications for it
                     (ties: where it ranks higher); see disjoint(); the
                     removed genes are listed in curated_removed_genes.tsv;
                   * signalling profiles (signal_profiles), assigned by the
                     column `program` of the evidence table: a general
                     profile (signal:<pathway>) where readouts and ligands
                     are expressed in the same cells (Nodal, BMP, Wnt, FGF),
                     otherwise profiles of the cells that receive
                     (receiving:), send (sending:; ligands, RA synthesis) or
                     block (blocking:; secreted antagonists, RA degradation)
                     the signal; readouts of several pathways are not used;
                   * states: as in `states` (already the same genes).
                   At every stage, genes not measured at the stage
                   (< MIN_UMIS) are left out; otherwise the list is fixed.
  controls         programs that should not describe cell identity:
                   whole_organism (genes ZFIN records only in the whole embryo
                   from high to shield; uniform expression expected), the
                   readouts of pathways
                   that should be inactive or uniform at these stages
                   (interferon, AhR, glucocorticoid, circadian clock) or that
                   flag dissociation stress, and YSL (the curated YSL group,
                   without the genes of curated programs and other controls;
                   the embryos were deyolked). See whole_organism_program,
                   and pathway_controls.
Every set is restricted to genes with at least MIN_UMIS UMIs in the stage and
capped at CAP genes per program (most publications first; Farrell et al.:
highest weight first; curated: see above).

Outputs:
  results/tables/programs.tsv           (set, stage, program, gene, weight,
                                         pubs, tier, rank, kept)
  results/tables/programs_summary.tsv   (per set and stage)
  results/tables/curated_zfin_windows.tsv (curated group, first and last
                                         stage recorded in ZFIN)
  results/tables/curated_removed_genes.tsv (genes removed from curated anatomy
                                         programs: program, gene, pubs, tier,
                                         rank_before, reason)
  results/tables/program_sets/<set>.tsv (readable gene lists: one row per
                                         program and stage; for curated, one
                                         row per program with its ZFIN window)
  report/tables/programs.tex

`16_program_sets.py tables` rewrites the readable tables from programs.tsv.
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd

from pipeline_utils import REPORT_TABLES, ROOT, STAGES, TABLES, ensure_dirs, load_dropseq_counts
from program_utils import farrell_modules

zfin_check = importlib.import_module("01c_heterogeneity_zfin")

MAPPING = ROOT / "data" / "curated" / "zfin_anatomy_programs.tsv"
# Pathway readouts and negative-control pathways, with literature evidence.
SIGNALLING = ROOT / "data" / "curated" / "signalling_evidence.tsv"
# Annotation of the curated programs; column `derives_from` (prior knowledge)
# lets a program share genes with the programs it derives from.
CURATED_PROGRAMS = ROOT / "data" / "curated" / "curated_programs.tsv"
# Genes assigned to a curated group by hand, against the ZFIN records (with
# reason and references); all their records count for that group.
GENE_OVERRIDES = ROOT / "data" / "curated" / "gene_overrides.tsv"
# Output: the stages at which ZFIN records each curated group (zfin_windows()).
ZFIN_WINDOWS = TABLES / "curated_zfin_windows.tsv"
STATES_TSV = ROOT / "data" / "curated" / "archive" / "expression_programs.tsv"
STATES = ["maternal", "blastomere", "cell_cycle_s", "cell_cycle_g2m", "heat_shock",
          "p53_response"]
MIN_UMIS = 10
MIN_PUBS = 1
MIN_GENES = 3
MIN_PUB_SHARE = 1 / 3
CAP = 30
# Groups left out of the curated set: broad terms and ZFIN's maternal /
# blastomere terms. The maternal records (egg, cleavage) end before the first
# atlas stage, and the blastomere / blastoderm / epiblast records collect
# ventral, non-neural and EVL genes even before shield (gata3, tfap2a, foxi1,
# vox, klf2a lead the group when capped at shield); the literature state
# programs describe these states instead. The yolk syncytial layer
# (CURATED_YOLK_GROUPS) is built like a curated program but becomes a control
# (control:YSL): the embryos were deyolked, so it should be invisible, or
# visible only in a few YSL-like cells. Two anatomy groups are dropped because
# the disjoint programs leave them without specific genes: ventral_bmp (ZFIN's
# "ventral mesoderm"; its markers are BMP readouts and ligands, i.e. the
# signal:BMP profile) and floor_plate (shhb and lft2 are in the Hedgehog and
# Nodal profiles, its other genes are shared with other programs). Two more
# have only three candidate genes, none of them a known marker:
# hemato_vascular (has2, ets2, ef1; ZFIN records it from 3-somite) and
# head_mesoderm (tpm4a, cdc42ep4a, prrx1a).
EXCLUDED_GROUPS = ["(broad)", "program", "maternal", "blastomere", "ventral_bmp", "floor_plate",
                   "hemato_vascular", "head_mesoderm"]
CURATED_YOLK_GROUPS = ["YSL"]
# Control programs (set `controls`): genes ZFIN records only in the whole
# organism (high to WHOLE_ORGANISM_END).
WHOLE_ORGANISM_END = "shield"
# Record kinds (01c) of the zfin_* term sets: the yolk and YSL terms are kept as
# a negative control.
ZFIN_KINDS = ("restricted", "yolk")
SETS = ["farrell", "zfin_all", "zfin_all_het", "zfin_pre_het", "signalling", "states",
        "curated", "controls"]
# How a set's programs relate across stages (column of the report table).
ACROSS_STAGES = {"farrell": "new per stage", "zfin_all": "per stage", "zfin_all_het": "per stage",
                 "zfin_pre_het": "per stage", "signalling": "fixed", "states": "fixed",
                 "curated": "fixed", "controls": "fixed"}
TIERS = ["weak", "moderate", "strong"]


def at_stage(records: pd.DataFrame, stage: str, kinds=("restricted",)) -> pd.DataFrame:
    hpf = zfin_check.ATLAS_HPF[stage]
    return records[(records["start_hpf"] <= hpf) & (records["stop"] > hpf)
                   & records["kind"].isin(kinds)]


def curated_records(records: pd.DataFrame, mapping: pd.Series) -> pd.DataFrame:
    """Records of the curated groups: 01c marks the YSL terms as "yolk"; here
    they count as restricted, so CURATED_YOLK_GROUPS can form programs."""
    yolk = records["kind"].eq("yolk") & records["term"].map(mapping).isin(CURATED_YOLK_GROUPS)
    return records.assign(kind=records["kind"].mask(yolk, "restricted"))


def term_programs(records: pd.DataFrame, allowed: set) -> pd.DataFrame:
    """One program per term with at least MIN_GENES genes."""
    ev = (records[records["gene"].isin(allowed)].groupby(["term", "gene"])
          .agg(pubs=("pub", "nunique")).reset_index())
    ev = ev[ev["pubs"] >= MIN_PUBS]
    sizes = ev.groupby("term").size()
    ev = ev[ev["term"].map(sizes) >= MIN_GENES].rename(columns={"term": "program"})
    return ev.sort_values(["program", "pubs"], ascending=[True, False])


def gene_overrides() -> pd.Series:
    t = pd.read_csv(GENE_OVERRIDES, sep="\t", keep_default_na=False)
    return t.set_index("gene")["program"]


def group_programs(records: pd.DataFrame, mapping: pd.Series, allowed: set,
                   overrides: pd.Series | None = None,
                   min_share: float = MIN_PUB_SHARE) -> pd.DataFrame:
    """Terms grouped by the mapping table; genes assigned to their main groups
    (or, for the genes in `overrides`, to the group given there)."""
    r = records.assign(program=records["term"].map(mapping))
    if overrides is not None:
        moved = r["gene"].isin(overrides.index) & r["program"].notna()
        r.loc[moved, "program"] = r.loc[moved, "gene"].map(overrides)
    r = r[r["program"].notna() & ~r["program"].isin(EXCLUDED_GROUPS)]
    ev = r.groupby(["program", "gene"]).agg(pubs=("pub", "nunique")).reset_index()
    share = ev["pubs"] / ev.groupby("gene")["pubs"].transform("sum")
    ev = ev[(ev["pubs"] >= MIN_PUBS) & (share >= min_share) & ev["gene"].isin(allowed)]
    sizes = ev.groupby("program").size()
    ev = ev[ev["program"].map(sizes) >= MIN_GENES]
    return ev.sort_values(["program", "pubs"], ascending=[True, False])


def zfin_windows(records: pd.DataFrame, mapping: pd.Series, groups) -> pd.DataFrame:
    """Where ZFIN expects each group: the stages from the first to the last at
    which the group has >= MIN_GENES genes recorded (MIN_PUBS, MIN_PUB_SHARE
    as in group_programs, any measured gene). Compared with the data in the
    timeline (script 24); it does not restrict the programs."""
    overrides = gene_overrides()
    present = {}
    for i, stage in enumerate(STAGES):
        r = at_stage(records, stage)
        for p in group_programs(r, mapping, set(r["gene"]), overrides)["program"].unique():
            present.setdefault(p, []).append(i)
    return pd.DataFrame([dict(program=p, first=STAGES[min(i)], last=STAGES[max(i)])
                         for p, i in present.items() if p in set(groups)])


def ranked_group_programs(records: pd.DataFrame, mapping: pd.Series,
                          tier: pd.DataFrame) -> pd.DataFrame:
    """Anatomy programs of the curated set: the ranked candidate genes of every
    group (before disjoint() and the cap), from all records that overlap the
    atlas (high to the last stage), heterogeneous at >= 1 stage, ranked by
    their mean tier over all stages, then by publications. A gene is a
    candidate of a group if it is one at >= 1 stage (group_programs on the
    records of the stage: the publication share is stage-local, otherwise
    genes recorded in several structures over time, e.g. ta, belong to none);
    its publications are counted over all records."""
    hpf = zfin_check.ATLAS_HPF
    r = records[(records["start_hpf"] <= hpf[STAGES[-1]]) & (records["stop"] > hpf[STAGES[0]])
                & (records["kind"] == "restricted")]
    het_any = set(tier.index[(tier >= 1).any(axis=1)])
    overrides = gene_overrides()
    pairs = pd.concat([group_programs(at_stage(r, s), mapping, het_any, overrides)
                       [["program", "gene"]] for s in STAGES]).drop_duplicates()
    pubs = group_programs(r, mapping, het_any, overrides, min_share=0)
    ev = pairs.merge(pubs, on=["program", "gene"])
    ev = ev[ev["program"].map(ev["program"].value_counts()) >= MIN_GENES].copy()
    ev["tier"] = tier.loc[ev["gene"]].mean(axis=1).to_numpy()
    return ev.sort_values(["program", "tier", "pubs"], ascending=[True, False, False],
                          ignore_index=True)


def ancestors(parents: pd.Series) -> dict[str, set]:
    """All ancestors of every program (column `derives_from` of
    curated_programs.tsv, followed transitively)."""
    up = {p: set(v.split(";")) - {""} for p, v in parents.items()}
    out = {}
    for p in up:
        seen, todo = set(), list(up[p])
        while todo:
            q = todo.pop()
            if q not in seen:
                seen.add(q)
                todo += up.get(q, [])
        out[p] = seen
    return out


def related() -> set[frozenset]:
    """Pairs of programs that may share genes: a program and the programs it
    derives from, directly or not (column `derives_from` of
    curated_programs.tsv; prior knowledge, e.g. hindbrain from neural plate)."""
    t = pd.read_csv(CURATED_PROGRAMS, sep="\t", keep_default_na=False)
    up = ancestors(t.set_index("program")["derives_from"])
    return {frozenset((p, q)) for p, qs in up.items() for q in qs}


def disjoint(ranked: pd.DataFrame, reserved: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make the anatomy programs disjoint: genes of the signalling and state
    programs (`reserved`: gene -> program) leave every anatomy program; a gene
    kept (within CAP) by two unrelated programs (see related()) stays only in
    the one with more ZFIN publications for it; a tie goes to the program
    where the gene ranks higher (then by name). Repeated until no kept gene is
    shared; removed genes are replaced by the next-ranked ones. Returns the
    remaining candidates and the removed genes with the reason."""
    ok = related()
    ranked = ranked.assign(rank_before=ranked.groupby("program").cumcount() + 1)
    removed = [ranked[ranked["gene"].isin(reserved)].assign(
        reason=lambda t: "in " + t["gene"].map(reserved))]
    ranked = ranked[~ranked["gene"].isin(reserved)]
    while True:
        kept = ranked[ranked.groupby("program").cumcount() < CAP]
        key = kept.assign(key=list(zip(-kept["pubs"], kept["rank_before"], kept["program"])))
        drop = {}
        for g, k in key.groupby("gene"):
            if len(k) < 2:
                continue
            k = k.sort_values("key")
            for a, b in ((a, b) for a in k["program"] for b in k["program"] if a != b):
                if frozenset((a, b)) in ok:
                    continue
                ka, kb = k.set_index("program").loc[[a, b], "key"]
                if kb < ka:  # b wins
                    tie = " (tie)" if ka[0] == kb[0] else ""
                    drop.setdefault((a, g), f"shared with {b}{tie}")
        if not drop:
            break
        hit = pd.Series([drop.get(k) for k in zip(ranked["program"], ranked["gene"])],
                        index=ranked.index)
        removed.append(ranked[hit.notna()].assign(reason=hit[hit.notna()]))
        ranked = ranked[hit.isna()]
    return ranked, pd.concat(removed, ignore_index=True)


def fixed_group_programs(ranked: pd.DataFrame, measured: dict[str, set]) -> pd.DataFrame:
    """One ranked gene list per program, repeated at every stage (genes
    measured at the stage)."""
    rows = []
    for program, ev in ranked.groupby("program", sort=False):
        ev = ev.assign(rank=np.arange(1, len(ev) + 1))
        rows += [ev.assign(stage=s)[ev["gene"].isin(measured[s])] for s in STAGES]
    return pd.concat(rows, ignore_index=True)


def signalling_table(kind: str = "readout") -> pd.DataFrame:
    """Rows of the evidence table: readouts with at least moderate evidence,
    or the genes of the negative-control pathways."""
    t = pd.read_csv(SIGNALLING, sep="\t", keep_default_na=False)
    t = t[t["kind"] == kind]
    return t[t["strength"] != "weak"] if kind == "readout" else t


def signal_profiles(tier: pd.DataFrame, measured: dict[str, set]) -> pd.DataFrame:
    """Signalling profiles of the curated set: the genes of the evidence table
    by its column `program` (general `signal:`, `receiving:`, `sending:`,
    `blocking:`; empty: not used, see `program_note`): the genes heterogeneous
    at >= 1 stage, in the table's order, the same list at every stage."""
    t = pd.read_csv(SIGNALLING, sep="\t", keep_default_na=False)
    t = t[(t["program"] != "") & t["gene"].isin(tier.index)].drop_duplicates(["program", "gene"])
    rows = []
    for name, g in t.groupby("program", sort=False):
        genes = [x for x in g["gene"] if (tier.loc[x] >= 1).any()]
        rows += [dict(program=name, gene=x, stage=s, rank=k + 1)
                 for s in STAGES for k, x in enumerate(genes) if x in measured[s]]
    return pd.DataFrame(rows)


def whole_organism_program(records: pd.DataFrame, measured: dict[str, set]) -> pd.DataFrame:
    """Control: genes recorded in the whole organism from high to
    WHOLE_ORGANISM_END and never in a localised structure (or the yolk) up to
    the last atlas stage; the CAP genes with the most publications, the same
    list at every stage."""
    window = STAGES[:STAGES.index(WHOLE_ORGANISM_END) + 1]
    first, last = zfin_check.ATLAS_HPF[window[0]], zfin_check.ATLAS_HPF[window[-1]]
    r = records[(records["start_hpf"] <= last) & (records["stop"] > first)]
    atlas = records[records["start_hpf"] <= zfin_check.ATLAS_HPF[STAGES[-1]]]
    localised = set(atlas.loc[atlas["kind"] != "broad", "gene"])
    whole = r[(r["term"] == "whole organism") & ~r["gene"].isin(localised)]
    pubs = whole.groupby("gene")["pub"].nunique().sort_values(ascending=False)
    ranked = pd.DataFrame(dict(gene=pubs.index, pubs=pubs.to_numpy(),
                               rank=np.arange(1, len(pubs) + 1)))
    return pd.concat([ranked[ranked["gene"].isin(measured[s])].assign(
        stage=s, program="control:whole_organism") for s in window], ignore_index=True)


def finish_controls(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["set"] = "controls"
    t["kept"] = t["rank"] <= CAP
    return t


def pathway_controls(measured: dict[str, set]) -> pd.DataFrame:
    """Control: readouts of pathways that should not vary between the cells
    of a library at these stages (signalling_evidence.tsv, negative_control),
    all their measured genes at every stage; the dissociation-stress genes
    flag stressed cells rather than a uniform response."""
    t = signalling_table("negative_control")
    return pd.DataFrame([dict(program="control:" + p, gene=g, stage=s, rank=k + 1)
                         for p, g_ in t.groupby("pathway", sort=False)
                         for k, g in enumerate(g_["gene"]) for s in STAGES
                         if g in measured[s]])


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
    """Like finish(), with the rank of the fixed list."""
    t = t.copy()
    t["set"] = "curated"
    if "rank" not in t:
        t["rank"] = t.groupby(["stage", "program"]).cumcount() + 1
    t["kept"] = t["rank"] <= CAP
    return t


def latex_table(summary: pd.DataFrame) -> None:
    names = {"farrell": "Farrell et al.\\ modules", "zfin_all": "ZFIN terms",
             "zfin_all_het": "ZFIN terms, het.", "zfin_pre_het": "ZFIN, het., pre-2018",
             "signalling": "Signalling", "states": "Cell states", "curated": "Curated",
             "controls": "Controls"}
    lines = ["\\begin{tabular}{llrrrr}", "\\toprule",
             "Set & Across stages & Prog./stage & Genes/prog. & Genes/stage & Genes "
             "\\\\",
             "\\midrule"]
    for s in SETS:
        t = summary[summary["set"] == s]
        lines.append(f"{names[s]} & {ACROSS_STAGES[s]} & {t['programs'].min()}--{t['programs'].max()} & "
                     f"{t['median_genes'].median():.0f} & {t['genes'].min()}--{t['genes'].max()}"
                     f" & {int(t['all_genes'].iat[0]):,} \\\\".replace(",", "{,}"))
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPORT_TABLES / "programs.tex").write_text(
        "% Generated by scripts/16_program_sets.py; do not edit.\n" + "\n".join(lines) + "\n")


def program_tables(long: pd.DataFrame) -> None:
    """Readable gene lists of every set (kept genes, in rank order; curated:
    with the ZFIN window)."""
    out = TABLES / "program_sets"
    out.mkdir(exist_ok=True)
    kept = long[long["kept"]]
    for s in SETS:
        t = kept[kept["set"] == s]
        if s == "curated":  # one fixed list per program
            zw = pd.read_csv(ZFIN_WINDOWS, sep="\t").set_index("program")
            rows = [dict(program=p, zfin_window=(f"{zw.at[p, 'first']} -- {zw.at[p, 'last']}"
                                                 if p in zw.index else ""),
                         genes=g["gene"].nunique(),
                         list=",".join(g.sort_values("rank").drop_duplicates("gene")["gene"]))
                    for p, g in t.groupby("program", sort=True)]
        else:
            rows = [dict(program=p, stage=st, genes=len(g),
                         list=",".join(g.sort_values("rank")["gene"]))
                    for (p, st), g in t.groupby(["program", "stage"], sort=False)]
        table = pd.DataFrame(rows)
        if "stage" in table:
            table["stage"] = pd.Categorical(table["stage"], STAGES, ordered=True)
            table = table.sort_values(["stage", "program"])
        table.to_csv(out / f"{s}.tsv", sep="\t", index=False)


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
    records["curated"] = curated_records(records["all"], mapping)
    modules = farrell_modules()
    parts, measured = [], {}
    for stage in STAGES:
        sub = adata[adata.obs["stage"] == stage]
        total = np.asarray(sub.X.sum(axis=0)).ravel()
        measured[stage] = set(sub.var_names[total >= MIN_UMIS])
        het = set(het_table.loc[(het_table["stage"] == stage) & het_table["heterogeneous"],
                                "gene"]) & measured[stage]
        rec = {k: at_stage(r, stage, ZFIN_KINDS) for k, r in records.items()
               if k in ("pre", "all")}
        states = state_programs(measured[stage])
        parts += [
            finish(farrell_programs(stage, modules, measured[stage]), "farrell", stage),
            finish(term_programs(rec["all"], measured[stage]), "zfin_all", stage),
            finish(term_programs(rec["all"], het), "zfin_all_het", stage),
            finish(term_programs(rec["pre"], het), "zfin_pre_het", stage),
            finish(states, "states", stage),
            finish_fixed(states.assign(stage=stage)),
        ]
    signals = signal_profiles(tier, measured)
    states_all = state_programs(set(adata.var_names))
    negative = signalling_table("negative_control")
    reserved = dict(zip(negative["gene"], "control:" + negative["pathway"]))
    reserved.update(zip(states_all["gene"], states_all["program"]))
    evidence = pd.read_csv(SIGNALLING, sep="\t", keep_default_na=False)
    evidence = evidence[evidence["program"] != ""]
    reserved.update(zip(evidence["gene"], evidence["program"]))
    ranked = ranked_group_programs(records["curated"], fixed_mapping, tier)
    zw = zfin_windows(records["curated"], fixed_mapping, ranked["program"])
    zw["program"] = zw["program"].mask(zw["program"].isin(CURATED_YOLK_GROUPS),
                                       "control:" + zw["program"])
    zw.to_csv(ZFIN_WINDOWS, sep="\t", index=False)
    yolk = ranked["program"].isin(CURATED_YOLK_GROUPS)
    ranked, ysl = ranked[~yolk], ranked[yolk]
    ranked, removed = disjoint(ranked, reserved)
    removed.to_csv(TABLES / "curated_removed_genes.tsv", sep="\t", index=False,
                   float_format="%.4g")
    print(f"curated: {(removed['rank_before'] <= CAP).sum()} kept genes removed "
          f"({len(removed)} candidates)")
    parts += [finish_fixed(signals).assign(set="signalling"),
              finish_fixed(fixed_group_programs(ranked, measured)),
              finish_fixed(signals)]
    # curated anatomy, signalling and state genes
    identity = set(pd.concat(parts[-2:])["gene"]) | set(states_all["gene"])
    # the YSL control without any gene of the curated programs or other controls
    ysl = ysl[~ysl["gene"].isin(identity | set(reserved))]
    ysl = fixed_group_programs(ysl, measured)
    ysl["program"] = "control:" + ysl["program"]
    parts += [finish_controls(pd.concat([whole_organism_program(records["all"], measured),
                                         ysl, pathway_controls(measured)]))]
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
    program_tables(long)


if __name__ == "__main__":
    if sys.argv[1:] == ["tables"]:
        program_tables(pd.read_csv(TABLES / "programs.tsv", sep="\t"))
    else:
        main()
