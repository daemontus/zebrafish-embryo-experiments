"""Overview figures of the program sets (report section "program sets").

1. Timeline (report figure `program_timeline`): the curated programs and
   controls (rows; names on the right, groups on the left) at every stage
   (columns). Rows keep the groups of
   data/curated/curated_programs.tsv; within a group they follow a tree:
   a program is indented under the program of the same group it derives
   from (column derives_from). The curated programs are defined at every
   stage; a marker's area is the number of the program's genes heterogeneous
   at the stage (weak tier, results/tables/heterogeneous_genes.tsv.gz; no
   marker: none), its fill opacity their mean tier (1 weak to 3 strong);
   open below MIN_FILLED genes, where the mean says little. Box: the stages
   at which ZFIN records the program (results/tables/curated_zfin_windows.tsv,
   script 16), to compare the literature with the data. Elbow arrows: the
   presumed derivation of a program from another (derives_from), from the
   source at the stage before the program's ZFIN window starts (a dot where
   it leaves the source's row) to the window's first stage.
   Coherence map (report figure `program_coherence`): in the same rows, the
   relative coherence of every program at every stage (script 17, median
   over the stage's libraries; 0 = as coherent as random heterogeneous
   genes, 1 = the threshold of support), a dot where it is supported.
   Programs per stage of the Farrell et al. modules, the ZFIN terms (all
   years) and the curated set (programs with >= 1 measured gene, and with
   >= 1 heterogeneous gene at the stage): report figure `program_set_sizes`.

2. Overlap (report figure `program_overlap`): at the stages in OVERLAP_STAGES,
   the Farrell et al. modules (rows) against the curated programs (columns;
   their genes measured at the stage): their shared genes, as a share of the
   smaller of the two (so that small programs, e.g. PGC or p53, can match),
   counted only if they share at least MIN_SHARED genes. A module whose best
   share is below NAMED_SHARE is unnamed; modules are ordered by their best
   program. As a check, the number of modules named by the ZFIN groups (all
   genes of the stage's terms that the mapping table groups into one curated
   program, plus the signalling and state programs) is printed.

3. Curated programs table (report table `curated_programs`): the
   annotation of every curated program (data/curated/curated_programs.tsv:
   the program of Table 1 it corresponds to, what it derives from, location
   and signals, its role for the model, interpretation and references) with
   its source (ZFIN terms of the mapping table, or the signalling and state
   tables), stages and first genes.

4. Signalling evidence table (report table `signalling_evidence`, appendix):
   every gene of data/curated/signalling_evidence.tsv with its role, type of
   evidence, strength, curated signalling profile and references.

5. Removed genes (report table `curated_removed`, appendix): the genes that
   script 16 removed from the curated anatomy programs to make them disjoint
   (results/tables/curated_removed_genes.tsv).

Reads results/tables/programs.tsv and curated_zfin_windows.tsv (16),
heterogeneous_genes.tsv.gz (01b) and program_support.tsv (17);
takes a few seconds.

Outputs:
  report/figures/program_timeline.pdf, report/figures/program_coherence.pdf,
  report/figures/program_set_sizes.pdf,
  report/figures/program_overlap.pdf
  report/tables/curated_programs.tex, report/tables/signalling_evidence.tex,
  report/tables/curated_removed.tex
  results/tables/program_overlap.tsv   (stage, module, best program, share,
                                         second program, share)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

import importlib

from pipeline_utils import (
    REPORT_TABLES,
    REPORT_WIDTH_IN,
    ROOT,
    STAGES,
    TABLES,
    ensure_dirs,
    report_style,
    save_report_figure,
)

program_sets = importlib.import_module("16_program_sets")

ANNOTATION = ROOT / "data" / "curated" / "curated_programs.tsv"
GROUP_COLOR = {"controls": "#A0A0A0", "early states": "#3C3C3C", "blastoderm and margin": "#E69F00", "endoderm and mesoderm": "#D55E00",
               "axial": "#CC79A7", "ectoderm": "#0072B2", "signalling profiles": "#009E73",
               "cell states": "#3C3C3C", "other": "#A0A0A0"}
SET_STYLE = {"farrell": dict(color="#E69F00", marker="s", label="Farrell et al. modules"),
             "zfin_all": dict(color="#0072B2", marker="o", label="ZFIN terms"),
             "curated": dict(color="#009E73", marker="^", label="curated (measured)")}
HET_STYLE = dict(color="#009E73", marker="^", mfc="white", ls="--",
                 label="curated (heterogeneous)")
MIN_FILLED = 3  # timeline: fewer heterogeneous genes are drawn as an open ring
COHERENCE_RANGE = (-1.0, 3.0)  # colour scale of the relative coherence (values clipped)
OVERLAP_STAGES = ["shield", "75% epiboly"]
NAMED_SHARE = 0.2
MIN_SHARED = 3


def label(program: str) -> str:
    for kind in ("receiving", "sending", "blocking"):
        if program.startswith(kind + ":"):
            program = f"{program.removeprefix(kind + ':')} ({kind})"
    return (program.replace("_", " ").replace("signal:", "").replace("state:", "")
            .replace("control:", "").replace(" zfin", " (ZFIN)"))


def heterogeneity(cur: pd.DataFrame) -> pd.DataFrame:
    """Per program and stage: its heterogeneous genes (weak tier
    at the stage) and their mean tier (1 weak, 2 moderate, 3 strong)."""
    het = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                      usecols=["stage", "gene", "weak", "moderate", "strong"])
    het["tier"] = het[["weak", "moderate", "strong"]].sum(axis=1)
    g = cur[["program", "stage", "gene"]].merge(het[het["weak"]][["stage", "gene", "tier"]], on=["stage", "gene"], how="left")
    out = g.groupby(["program", "stage"]).agg(n=("tier", "count"), tier=("tier", "mean"))
    return out


def row_order(notes: pd.DataFrame, present: set) -> list[tuple[str, str, int]]:
    """(program, group, depth) in group order. Within a group, programs follow
    a tree: a program under the last program of the same group it derives
    from (depth = length of that chain); siblings in file order."""
    notes = notes[notes["program"].isin(present)].reset_index(drop=True)
    pos = dict(zip(notes["program"], notes.index))
    rows = []
    for group, g in notes.groupby("group", sort=False):
        members = set(g["program"])
        attach = {}
        for p, d in zip(g["program"], g["derives_from"]):
            within = [q for q in str(d).split(";") if q in members]
            if within:
                attach[p] = within[-1]

        def depth(p):
            return 0 if p not in attach else 1 + depth(attach[p])

        def visit(p):
            rows.append((p, group, depth(p)))
            for c in sorted((c for c, a in attach.items() if a == p), key=pos.get):
                visit(c)
        for p in g["program"]:
            if p not in attach:
                visit(p)
    listed = {p for p, _, _ in rows}
    rows += [(p, "other", 0) for p in sorted(present - listed)]
    return rows


def rows_axis(ax, rows: list[tuple[str, str, int]]) -> None:
    """Program names on the right, next to the ends of the rows (derived
    programs indented), group separators, group names on the left, stages
    below."""
    labels = [("    " * (d - 1) + "  └ " if d else "") + label(p) for p, _, d in rows]
    ax.yaxis.tick_right()
    ax.set_yticks(range(len(rows)), labels, fontsize=6)
    for tick, (_, group, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(GROUP_COLOR[group])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    x = np.arange(len(STAGES))
    ax.set_xticks(x, STAGES, rotation=45, ha="right")
    ax.set_xlim(-0.5, len(STAGES) - 0.5)
    # group separators, and group names on the left
    starts = [i for i, r in enumerate(rows) if i == 0 or rows[i - 1][1] != r[1]]
    for i in starts[1:]:
        ax.axhline(i - 0.5, color="#DDDDDD", lw=0.5, zorder=0)
    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(rows)
        ax.text(-0.7, (i + end - 1) / 2, rows[i][1].replace(" and ", "\nand "), fontsize=6,
                va="center", ha="right", color=GROUP_COLOR[rows[i][1]], clip_on=False)
    ax.spines["left"].set_visible(False)


def area(n: float) -> float:
    return 3 + 2.2 * n


def zfin_spans() -> dict[str, list[int]]:
    """Stage indices of the ZFIN window of every curated group (script 16)."""
    t = pd.read_csv(program_sets.ZFIN_WINDOWS, sep="\t")
    return {p: list(range(STAGES.index(a), STAGES.index(b) + 1))
            for p, a, b in zip(t["program"], t["first"], t["last"])}


def zfin_box(ax, row: int, span: list[int], color: str) -> None:
    """Box around the stages at which ZFIN records a program."""
    a, b = min(span), max(span)
    ax.add_patch(FancyBboxPatch((a - 0.32, row - 0.3), b - a + 0.64, 0.6,
                                boxstyle="round,pad=0,rounding_size=0.25",
                                fc=(*to_rgb(color), 0.13), ec=(*to_rgb(color), 0.6), lw=0.4,
                                zorder=0.5))


def zfin_legend() -> Patch:
    grey = to_rgb("#555555")
    return Patch(fc=(*grey, 0.13), ec=(*grey, 0.6), lw=0.4,
                 label="stages at which ZFIN records the program")


def timeline(programs: pd.DataFrame, notes: pd.DataFrame) -> None:
    kept = programs[programs["kept"]]
    cur = kept[kept["set"].isin(["curated", "controls"])]
    n_genes = cur.groupby(["program", "stage"]).size()
    het = heterogeneity(cur)
    rows = row_order(notes, set(cur["program"]))
    zfin = zfin_spans()

    fig, ax = plt.subplots(figsize=(REPORT_WIDTH_IN, 0.155 * len(rows) + 1.0))
    for i, (p, group, _) in enumerate(rows):
        color = GROUP_COLOR[group]
        defined = window(p, n_genes)
        ax.plot([min(defined), max(defined)], [i, i], color=color, lw=0.5, alpha=0.5, zorder=1)
        if max(defined) < len(STAGES) - 1:  # faint guide from an early end to the name
            ax.plot([max(defined) + 0.2, len(STAGES) - 0.5], [i, i], color=color, lw=0.4,
                     alpha=0.35, ls=(0, (1, 2)), zorder=0)
        if p in zfin:
            zfin_box(ax, i, zfin[p], color)
        for j in defined:
            n, tier = het.loc[(p, STAGES[j])] if (p, STAGES[j]) in het.index else (0, np.nan)
            if n == 0:
                continue
            face = (*to_rgb(color), tier_alpha(tier)) if n >= MIN_FILLED else "white"
            ax.scatter(j, i, s=area(n), color=face, edgecolors=color, linewidths=0.6, zorder=3)
    lineage_arrows(ax, rows, n_genes, notes, zfin)
    rows_axis(ax, rows)
    grey = to_rgb("#555555")
    handles = [Line2D([], [], ls="", marker="o", ms=np.sqrt(area(0)), mfc="white", mec="#555555",
                      label=f"< {MIN_FILLED}")]
    handles += [Line2D([], [], ls="", marker="o", ms=np.sqrt(area(n)), mfc=(*grey, 0.6),
                       mec="#555555", label=str(n)) for n in (5, 15, 30)]
    handles[0].set_markersize(np.sqrt(area(1)))
    size_legend = ax.legend(handles=handles, frameon=False, ncol=4, fontsize=6,
                            title="heterogeneous genes", title_fontsize=6, loc="upper left",
                            bbox_to_anchor=(0.0, -0.09))
    ax.add_artist(size_legend)
    handles = [Line2D([], [], ls="", marker="o", ms=5, mfc=(*grey, tier_alpha(t)), mec="#555555",
                      label=name) for t, name in ((1, "weak"), (2, "moderate"), (3, "strong"))]
    tier_legend = ax.legend(handles=handles, frameon=False, ncol=3, fontsize=6, title="mean tier",
                            title_fontsize=6, loc="upper right", bbox_to_anchor=(1.0, -0.09))
    ax.add_artist(tier_legend)
    ax.legend(handles=[zfin_legend()], frameon=False, fontsize=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    save_report_figure(fig, "program_timeline")
    plt.close(fig)


def coherence_map(notes: pd.DataFrame) -> None:
    """Relative coherence of the curated programs and controls per stage
    (script 17: median over the stage's libraries; 0 = a typical random set of
    heterogeneous genes, 1 = the threshold), in the rows of the timeline; a
    dot marks the supported programs."""
    support = pd.read_csv(TABLES / "program_support.tsv", sep="\t")
    support = support[support["set"].isin(["curated", "controls"])
                      & support["relative_coherence"].notna()]
    table = support.set_index(["program", "stage"])
    rows = row_order(notes, set(support["program"]))
    zfin = zfin_spans()
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-COHERENCE_RANGE[1] / 3, vcenter=0, vmax=COHERENCE_RANGE[1])
    fig, ax = plt.subplots(figsize=(REPORT_WIDTH_IN, 0.155 * len(rows) + 1.0))
    for i, (p, group, _) in enumerate(rows):
        if p in zfin:  # outline around the squares, drawn on top
            a, b = min(zfin[p]), max(zfin[p])
            ax.add_patch(FancyBboxPatch((a - 0.4, i - 0.44), b - a + 0.8, 0.88,
                                        boxstyle="round,pad=0,rounding_size=0.3", fc="none",
                                        ec=GROUP_COLOR[group], lw=0.9, zorder=4))
        for j, stage in enumerate(STAGES):
            if (p, stage) not in table.index:
                continue
            row = table.loc[(p, stage)]
            ax.add_patch(Rectangle((j - 0.27, i - 0.3), 0.54, 0.6, zorder=2, lw=0,
                                   color=cmap(norm(np.clip(row["relative_coherence"],
                                                           *COHERENCE_RANGE)))))
            if row["supported"]:
                ax.scatter(j, i, s=8, color="white", edgecolors="#222222", linewidths=0.5,
                           zorder=3)
    rows_axis(ax, rows)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = ax.inset_axes([0.0, -0.13, 0.4, 0.012])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([-1, 0, 1, 2, 3])
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.set_label("relative coherence (0: random genes, 1: threshold; median over libraries)",
                   fontsize=6)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=2.8, mfc="white", mec="#222222",
                              mew=0.5, label="supported"),
                       Patch(fc="none", ec="#555555", lw=0.9,
                             label="stages at which ZFIN records the program")],
              frameon=False, fontsize=6, ncol=2, loc="upper right", bbox_to_anchor=(1.0, -0.09))
    save_report_figure(fig, "program_coherence")
    plt.close(fig)


def tier_alpha(tier: float) -> float:
    """Fill opacity for a mean tier from 1 (weak) to 3 (strong)."""
    return 0.0 if np.isnan(tier) else 0.22 + 0.78 * (tier - 1) / 2


def set_sizes(programs: pd.DataFrame) -> None:
    """Programs per stage of the Farrell et al. modules, the ZFIN terms and the
    curated set (report figure `program_set_sizes`); curated programs both
    with >= 1 measured gene and with >= 1 heterogeneous gene at the stage."""
    kept = programs[programs["kept"]]
    counts = kept.groupby(["set", "stage"])["program"].nunique()
    het = heterogeneity(kept[kept["set"] == "curated"])
    het = het[het["n"] > 0].reset_index().groupby("stage")["program"].nunique()
    fig, ax = plt.subplots(figsize=(REPORT_WIDTH_IN * 0.6, 1.9))
    x = np.arange(len(STAGES))
    for s, style in SET_STYLE.items():
        ax.plot(x, counts[s].reindex(STAGES).fillna(0).to_numpy(), lw=1, ms=3, **style)
    ax.plot(x, het.reindex(STAGES).fillna(0).to_numpy(), lw=1, ms=3, **HET_STYLE)
    print("curated programs with a heterogeneous gene:", het.reindex(STAGES).fillna(0).astype(int).tolist())
    ax.set_ylabel("programs per stage")
    ax.set_xticks(x, STAGES, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=6)
    ax.set_ylim(0, None)
    save_report_figure(fig, "program_set_sizes")
    plt.close(fig)


def window(program: str, n_genes: pd.Series) -> list[int]:
    return [j for j, st in enumerate(STAGES) if (program, st) in n_genes.index]


def lineage_arrows(ax, rows: list, n_genes: pd.Series, notes: pd.DataFrame,
                   zfin: dict[str, list[int]]) -> None:
    """Elbow arrow from each parent (derives_from) to the first stage of the
    child's ZFIN window (programs without one: their first stage): along the
    parent's row from the stage before (or the last stage of the parent's
    ZFIN window), down or up between the columns (a dot marks where it leaves
    the parent's row), into the child's first stage. Arrows from one parent
    into one column share the
    vertical line; parents sharing a column are offset. None if the parent is
    not defined before the child."""
    row_of = {p: i for i, (p, _, _) in enumerate(rows)}
    parents = notes.set_index("program")["derives_from"]
    links = []

    def span(p):
        return zfin.get(p) or window(p, n_genes)

    for child in row_of:
        first = min(span(child))
        for parent in filter(None, str(parents.get(child, "")).split(";")):
            if parent not in row_of:
                continue
            w = span(parent)
            if min(w) >= first:  # not defined before the child: no direction to draw
                continue
            links.append((parent, child, min(first - 1, max(w)), first))
    style = dict(color="#777777", lw=0.5, alpha=0.8, zorder=2)
    columns = {}
    for parent, _, _, first in links:
        columns.setdefault(first, [])
        if parent not in columns[first]:
            columns[first].append(parent)
    for parent, child, start, first in links:
        same = columns[first]
        x = first - 0.5 + 0.1 * (same.index(parent) - (len(same) - 1) / 2)
        yp, yc = row_of[parent], row_of[child]
        ax.plot([start + 0.15, x, x], [yp, yp, yc], solid_joinstyle="round", **style)
        ax.plot(x, yp, marker="o", ms=1.6, ls="", **style)  # where it leaves the parent
        ax.annotate("", xy=(first - 0.14, yc), xytext=(x, yc),
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color=style["color"],
                                    alpha=0.8, shrinkA=0, shrinkB=0, mutation_scale=5),
                    zorder=2)


def tex(s: str) -> str:
    return str(s).replace("%", "\\%").replace("_", "\\_").replace("&", "\\&")


def annotation_table(programs: pd.DataFrame, notes: pd.DataFrame) -> None:
    """Report table of the curated programs: source, stages, Table 1 and
    derivation, location and signals, role and interpretation, first genes."""
    table = pd.read_csv(program_sets.MAPPING, sep="\t", keep_default_na=False)
    table["group"] = table["fixed_program"].where(table["fixed_program"] != "", table["program"])
    terms = table.groupby("group")["term"].agg(list)
    cur = programs[programs["set"].isin(["curated", "controls"]) & programs["kept"]]
    zfin = zfin_spans()
    lines = ["\\begin{longtable}{@{}" + "".join(
                 f">{{\\raggedright\\arraybackslash}}p{{{w}cm}}"
                 for w in (2.5, 1.9, 3.4, 3.2, 3.6, 6.6, 2.6)) + "@{}}",
             "\\caption{Curated programs: source (ZFIN anatomy terms grouped by "
             "\\protect\\nolinkurl{data/curated/zfin_anatomy_programs.tsv}, or the signalling and "
             "state tables), the stages at which ZFIN records it (anatomy programs; all "
             "curated programs are defined at every stage), the program of "
             "Table~\\ref{tab:programs} it corresponds to and the programs it is presumed to "
             "derive from (the arrows of Figure~\\ref{fig:program-timeline}), where and under "
             "which signals it arises, its role for the model and interpretation, and its "
             "first genes (by mean heterogeneity over the stages, then records). Annotation: "
             "\\protect\\nolinkurl{data/curated/curated_programs.tsv}.}"
             "\\label{tab:curated-programs}\\\\",
             "\\toprule",
             "Program & ZFIN stages & Source & Table~\\ref{tab:programs} / from & Location, signals "
             "& Role and interpretation & First genes \\\\",
             "\\midrule", "\\endfirsthead",
             "\\toprule",
             "Program & ZFIN stages & Source & Table~\\ref{tab:programs} / from & Location, signals "
             "& Role and interpretation & First genes \\\\",
             "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot"]
    previous = None
    for r in notes.itertuples():
        c = cur[cur["program"] == r.program]
        if c.empty:
            continue
        if r.group != previous:
            if previous is not None:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{tex(r.group)}}}}} \\\\")
            previous, row = r.group, 0
        row += 1
        stages = [STAGES[j] for j in zfin.get(r.program, [])]
        span = ("" if not stages else stages[0] if len(stages) == 1
                else f"{stages[0]} -- {stages[-1]}")
        if r.program.startswith("signal:"):
            source = "readouts, ligands and antagonists (signalling table)"
        elif r.program.startswith("receiving:"):
            source = "readouts (signalling table)"
        elif r.program.startswith("sending:"):
            source = "ligands, synthesis (signalling table)"
        elif r.program.startswith("blocking:"):
            source = "secreted antagonists, degradation (signalling table)"
        elif r.program == "control:whole_organism":
            source = "ZFIN: whole organism only"
        elif r.program == "control:YSL":
            t = terms.get("YSL", [])
            source = "ZFIN: " + "; ".join(t[:4]) + " (genes of curated programs left out)"
        elif r.program.startswith("control:"):
            source = "readouts (signalling table)"
        elif r.program.startswith("state:"):
            source = "literature table"
        else:
            t = terms.get(r.program, [])
            source = "ZFIN: " + "; ".join(t[:4]) + (f" (+{len(t) - 4} more)" if len(t) > 4 else "")
        genes = ", ".join(c.sort_values("rank").drop_duplicates("gene")["gene"].str.lower()[:6])
        link = "; ".join(x for x in [
            r.table1,
            "from " + ", ".join(label(p) for p in r.derives_from.split(";")) if r.derives_from
            else ""] if x)
        cite = f"~\\cite{{{r.references.replace(';', ',')}}}" if r.references else ""
        lines.append((STRIPE if row % 2 == 0 else "") + " & ".join([
            tex(label(r.program)), tex(span), tex(source), tex(link), tex(r.location_signals),
            tex(r.role[:1].upper() + r.role[1:]) + ". " + tex(r.interpretation[:1].upper()
                                                             + r.interpretation[1:]) + cite + ".",
            f"\\emph{{{tex(genes)}}}"]) + " \\\\")
    lines.append("\\end{longtable}")
    (REPORT_TABLES / "curated_programs.tex").write_text(
        "% Generated by scripts/24_program_overview.py; do not edit.\n" + "\n".join(lines) + "\n")


# Shorter labels of the signalling evidence table (narrow columns).
SHORT_NOTES = {"readout of several pathways (also a Wnt8 target)": "multi-readout and Wnt8 target",
               "readout of several pathways (also a BMP target)": "multi-readout and BMP target",
               "readout of several pathways (also acts in Wnt/Lef1 signalling)":
                   "multi-readout, also acts in Wnt/Lef1"}
SHORT_LABELS = {"dissociation": "dissoc.", "degradation": "de\\-gra\\-da\\-tion"}
STRIPE = "\\rowcolor{gray!10}"  # every second row of a group (as Table 1)


def evidence_table() -> None:
    """Appendix table of the signalling readouts and negative controls; every
    second row of a group is shaded."""
    e = pd.read_csv(program_sets.SIGNALLING, sep="\t", keep_default_na=False)
    lines = ["\\begin{longtable}{@{}" + "".join(
                 f">{{\\raggedright\\arraybackslash}}p{{{w}cm}}"
                 for w in (1.3, 1.3, 1.4, 5.5, 1.3, 2.0, 1.7)) + "@{}}",
             "\\caption{Signalling readouts and negative-control pathways "
             "(\\protect\\nolinkurl{data/curated/signalling_evidence.tsv}): role, type of "
             "evidence and its strength (strong: zebrafish, direct or loss/gain of the pathway; "
             "moderate: indirect zebrafish or other vertebrate evidence). "
             "Readouts mark cells that receive a signal; ligands and secreted antagonists mark "
             "cells that send or block one. Profile: the curated signalling profile of the gene "
             "(or why it is not used). Negative controls are pathways that should be inactive or "
             "uniform at 3--12\\,hpf, and dissociation-induced genes.}\\label{tab:signalling-evidence}\\\\",
             "\\toprule", "Pathway & Gene & Role & Evidence & Strength & Profile & References \\\\",
             "\\midrule", "\\endfirsthead",
             "\\toprule", "Pathway & Gene & Role & Evidence & Strength & Profile & References \\\\",
             "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot"]
    previous = None
    for r in e.itertuples():
        if r.kind != previous:
            if previous is not None:
                lines.append("\\midrule")
            name = {"readout": "readouts (receiving)", "negative_control": "negative controls",
                    "ligand": "ligands (sending)",
                    "antagonist": "secreted antagonists and degradation (blocking)"}[r.kind]
            lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{name}}}}} \\\\")
            previous, row = r.kind, 0
        role = ("control" if r.kind == "negative_control"
                else r.role.replace("autoregulation", "auto-regulation"))
        cite = f"\\cite{{{r.references.replace(';', ',')}}}" if r.references else ""
        note = SHORT_NOTES.get(r.program_note, r.program_note)
        profile = (label(r.program) if r.program else
                   "not used: " + note if note else "--")
        row += 1
        lines.append((STRIPE if row % 2 == 0 else "")
                     + " & ".join([SHORT_LABELS.get(r.pathway, tex(r.pathway)),
                                   f"\\emph{{{tex(r.gene.lower())}}}",
                                   SHORT_LABELS.get(role, tex(role)),
                                   tex(r.evidence_type), tex(r.strength), tex(profile), cite])
                     + " \\\\")
    lines.append("\\end{longtable}")
    (REPORT_TABLES / "signalling_evidence.tex").write_text(
        "% Generated by scripts/24_program_overview.py; do not edit.\n" + "\n".join(lines) + "\n")


def removed_table(notes: pd.DataFrame) -> None:
    """Appendix table of the genes that script 16 removed from the curated
    anatomy programs (only genes that were within the cap)."""
    r = pd.read_csv(TABLES / "curated_removed_genes.tsv", sep="\t", keep_default_na=False)
    r = r[r["rank_before"] <= program_sets.CAP]
    order = {p: i for i, p in enumerate(notes["program"])}
    lines = ["\\begin{longtable}{@{}" + "".join(
                 f">{{\\raggedright\\arraybackslash}}p{{{w}cm}}" for w in (2.4, 5.0, 7.7))
             + "@{}}",
             "\\caption{Genes removed from the curated anatomy programs to make them disjoint "
             "(script 16; only genes that ranked within the cap of " + str(program_sets.CAP)
             + "). Left: genes of a signalling profile or a cell-state program. Right: genes "
             "shared with another anatomy program that it neither derives from nor gives rise to "
             "(column derives\\_from of the annotation); the gene stays in the program with more "
             "ZFIN publications for it, or, if they have as many (marked $^{=}$), in the one "
             "where it ranks higher.}"
             "\\label{tab:curated-removed}\\\\",
             "\\toprule", "Program & In a signalling or state program & Shared with \\\\",
             "\\midrule", "\\endfirsthead",
             "\\toprule", "Program & In a signalling or state program & Shared with \\\\",
             "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot"]
    for i, (program, t) in enumerate(sorted(r.groupby("program"), key=lambda x: order.get(x[0], 99))):
        reserved = t[t["reason"].str.startswith("in ")]
        moved = ", ".join(f"\\emph{{{tex(g.lower())}}} ({tex(label(x.removeprefix('in ')))})"
                          for g, x in zip(reserved["gene"], reserved["reason"]))
        shared = t[t["reason"].str.startswith("shared")].assign(
            other=lambda x: x["reason"].str.extract(r"shared with (\S+)")[0],
            tie=lambda x: x["reason"].str.contains("tie"))
        parts = [f"{tex(label(o))}: " + ", ".join(
                     f"\\emph{{{tex(g.lower())}}}" + ("$^{=}$" if tie else "")
                     for g, tie in zip(u["gene"], u["tie"]))
                 for o, u in shared.groupby("other", sort=False)]
        lines.append((STRIPE if i % 2 else "") + f"{tex(label(program))} & {moved} & {'; '.join(parts)} \\\\")
    lines.append("\\end{longtable}")
    (REPORT_TABLES / "curated_removed.tex").write_text(
        "% Generated by scripts/24_program_overview.py; do not edit.\n" + "\n".join(lines) + "\n")


def curated_lists(programs: pd.DataFrame, stage: str) -> pd.Series:
    """The curated programs at the stage (their measured genes)."""
    cur = programs[(programs["set"] == "curated") & (programs["stage"] == stage)
                   & programs["kept"]]
    return cur.groupby("program")["gene"].agg(set)


def zfin_groups(programs: pd.DataFrame, stage: str) -> pd.Series:
    """ZFIN terms of the stage (all genes) grouped as the curated programs,
    plus the curated signalling and state programs (a check only)."""
    table = pd.read_csv(program_sets.MAPPING, sep="\t", keep_default_na=False).set_index("term")
    group = table["fixed_program"].where(table["fixed_program"] != "", table["program"])
    z = programs[(programs["set"] == "zfin_all") & (programs["stage"] == stage)]
    z = z.assign(group=z["program"].map(group))
    z = z[z["group"].notna() & ~z["group"].isin(program_sets.EXCLUDED_GROUPS)]
    cur = programs[(programs["set"] == "curated") & (programs["stage"] == stage)
                   & programs["program"].str.contains(":") & programs["kept"]]
    return pd.concat([z.groupby("group")["gene"].agg(set),
                      cur.groupby("program")["gene"].agg(set)])


def shares(programs: pd.DataFrame, stage: str, columns) -> pd.DataFrame:
    """Shared genes of every Farrell et al. module of the stage (rows) and
    every program of `columns(programs, stage)`, as a share of the smaller
    set; 0 below MIN_SHARED genes."""
    farrell = programs[(programs["set"] == "farrell") & programs["kept"]
                       & (programs["stage"] == stage)]
    modules = farrell.groupby("program", sort=False)["gene"].agg(set)
    return pd.DataFrame({p: [len(g & genes) / min(len(g), len(genes))
                             if len(g & genes) >= MIN_SHARED else 0.0 for g in modules]
                         for p, genes in columns(programs, stage).items()},
                        index=modules.index)


def overlap(programs: pd.DataFrame) -> pd.DataFrame:
    """Farrell et al. modules against the curated programs (report figure
    `program_overlap`); the ZFIN-group version is printed as a check."""
    rows, tables = [], {}
    for stage in OVERLAP_STAGES:
        z = shares(programs, stage, zfin_groups)
        print(f"{stage}: {(z.max(axis=1) >= NAMED_SHARE).sum()} of {len(z)} modules named "
              "by the ZFIN groups (all genes of the stage's terms)")
        modules = programs[(programs["set"] == "farrell") & programs["kept"]
                           & (programs["stage"] == stage)].groupby("program")["gene"].agg(set)
        share = shares(programs, stage, curated_lists)
        tables[stage] = share
        for m, r in share.iterrows():
            top2 = r.sort_values(ascending=False)
            rows.append(dict(stage=stage, module=m, genes=len(modules[m]),
                             best=top2.index[0] if top2.iat[0] > 0 else "", share=top2.iat[0],
                             second=top2.index[1] if top2.iat[1] > 0 else "",
                             second_share=top2.iat[1],
                             named=top2.iat[0] >= NAMED_SHARE))
    out = pd.DataFrame(rows)

    def columns(share):  # programs that name a module or share >= NAMED_SHARE with one
        return [c for c in share.columns if share[c].max() >= NAMED_SHARE]
    fig, axes = plt.subplots(1, len(OVERLAP_STAGES), figsize=(REPORT_WIDTH_IN, 4.6),
                             gridspec_kw=dict(wspace=0.45, width_ratios=[
                                 len(columns(tables[s])) for s in OVERLAP_STAGES]))
    for ax, stage in zip(axes, OVERLAP_STAGES):
        share = tables[stage]
        best = share.idxmax(axis=1).where(share.max(axis=1) >= NAMED_SHARE)
        cols = columns(share)
        cols = sorted(cols, key=lambda c: (":" in c,
                                           -(best == c).sum(), c))
        order = sorted(share.index, key=lambda m: (cols.index(best[m]) if best[m] in cols
                                                   else len(cols), -share.loc[m].max()))
        im = ax.imshow(share.loc[order, cols].to_numpy(), cmap="Blues", vmin=0, vmax=0.6,
                       aspect="auto")
        ax.set_xticks(range(len(cols)), [label(c) for c in cols], rotation=90, fontsize=6)
        ax.set_yticks(range(len(order)),
                      [f"{m.rsplit(':', 1)[1]}" + ("" if pd.notna(best[m]) else " *")
                       for m in order], fontsize=5.5)
        ax.set_ylabel("Farrell et al. module (* unnamed)")
        ax.set_title(f"{stage}: {best.notna().sum()} of {len(order)} modules named",
                     fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02)
    cbar.set_label("shared genes / smaller set", fontsize=7)
    save_report_figure(fig, "program_overlap")
    plt.close(fig)
    return out


def main() -> None:
    ensure_dirs()
    report_style()
    programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    notes = pd.read_csv(ANNOTATION, sep="\t", keep_default_na=False)
    timeline(programs, notes)
    coherence_map(notes)
    set_sizes(programs)
    annotation_table(programs, notes)
    evidence_table()
    removed_table(notes)
    out = overlap(programs)
    out.to_csv(TABLES / "program_overlap.tsv", sep="\t", index=False, float_format="%.3f")
    for stage, t in out.groupby("stage", sort=False):
        print(stage, f"{t['named'].sum()} of {len(t)} named")
        print(t.sort_values("share", ascending=False)[
            ["module", "best", "share", "second", "second_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
