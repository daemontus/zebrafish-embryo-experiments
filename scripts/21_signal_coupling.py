"""Which programs are co-active with which signalling profiles, at which stage?

For every stage, the smoothed scores of the curated programs (as in scripts
18 and 19) are correlated over the stage's cells (scores are centred per
library, so this is co-variation within libraries). The curated programs are
disjoint (script 16), so no correlation comes from shared genes. First, the
profiles that read a signal (general `signal:` and `receiving:`) against
the identity and state programs: a correlation means that the pathway's genes
are expressed in the cells of a program at that stage; it does not show that
the signal causes the program. Smoothing over neighbouring cells inflates
absolute values; changes over time and signs are more informative.

Second, HYPOTHESES lists the expected couplings of the signalling profiles
(which tissue sends, receives or blocks which signal, and whether senders
also receive it), correlated in the same way with all other curated
programs. A correlation over all cells misses small source populations (e.g.
the organizer among all gastrula cells) and pathways with several source
tissues (FGF at the margin, tailbud and MHB), so each hypothesis is also
tested on the cells: the AUROC of the profile's score for the cells in which
the partner program is active (identity partners), or of the partner
profile's score for the cells in which the profile is active (signalling
partners); active calls as in scripts 18-19. A hypothesis is consistent if
the AUROC is on the expected side of 0.5 by >= MIN_AUC_GAP at one or more
stages and never by as much on the other side; correlations are reported
alongside.

Outputs:
  results/tables/signal_coupling.tsv          (stage, signal, program, r; reading profiles)
  results/tables/signal_outputs.tsv           (stage, signal, program, r; all profiles)
  results/tables/signal_output_auroc.tsv      (stage, output, partner, AUROC)
  results/tables/signal_output_hypotheses.tsv (one row per hypothesis)
  report/figures/signal_coupling.pdf, report/figures/signal_outputs.pdf
  report/tables/signal_output_hypotheses.tex
"""
from __future__ import annotations

import importlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import roc_auc_score

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

program_populations = importlib.import_module("18_program_populations")

MIN_R = 0.3  # programs shown: |r| >= MIN_R at some stage of their ZFIN window (or any stage)
MIN_R_OUTPUTS = 0.4  # the same for the sending and blocking profiles (more of them; one page)
MIN_AUC_GAP = 0.15  # hypothesis verdicts: AUROC >= 0.65 or <= 0.35
MIN_ACTIVE = 10  # cells with the program active, at least, for an AUROC
SIGNAL_PREFIXES = ("signal:", "receiving:", "sending:", "blocking:")  # signalling profiles
READING = ("signal:", "receiving:")  # profiles that contain readouts
SIGNALS = ["signal:Nodal", "signal:BMP", "signal:Wnt", "signal:FGF", "receiving:Notch",
           "receiving:RA"]
OUTPUTS = ["sending:BMP_organizer", "blocking:BMP", "blocking:Wnt", "sending:Wnt/PCP",
           "sending:RA", "blocking:RA", "sending:Notch", "sending:Hedgehog"]
# (signalling profile, partner program, expected sign, reason)
HYPOTHESES = [
    ("signal:Nodal", "margin", 1, "Nodal is made and received at the margin"),
    ("signal:Nodal", "endoderm", 1, "and in the endoderm precursors"),
    ("signal:BMP", "non_neural_ectoderm", 1, "BMP is received ventrally, in the non-neural ectoderm"),
    ("sending:BMP_organizer", "dorsal_organizer", 1, "admp is a BMP ligand of the organizer"),
    ("sending:BMP_organizer", "signal:BMP", -1, "and the organizer has low BMP"),
    ("blocking:BMP", "dorsal_organizer", 1, "chordin and noggin come from the organizer"),
    ("blocking:BMP", "signal:BMP", -1, "cells that secrete BMP antagonists have low BMP"),
    ("signal:Wnt", "margin", 1, "Wnt8 is made and received at the ventrolateral margin"),
    ("signal:Wnt", "tailbud", 1, "and later in the tailbud"),
    ("blocking:Wnt", "dorsal_organizer", 1, "Dkk1 and Frzb come from the organizer"),
    ("blocking:Wnt", "prechordal_plate", 1, "and from the prechordal plate"),
    ("blocking:Wnt", "signal:Wnt", -1, "cells that secrete Wnt antagonists have low Wnt"),
    ("signal:FGF", "margin", 1, "Fgf8 and Fgf24 are made and received at the margin"),
    ("signal:FGF", "tailbud", 1, "and in the tailbud"),
    ("signal:FGF", "midbrain_hindbrain_boundary", 1, "Fgf8 at the MHB organizer"),
    ("sending:RA", "presomitic_mesoderm", 1, "aldh1a2 in the paraxial mesoderm"),
    ("sending:RA", "blocking:RA", -1, "RA is made and degraded in different tissues"),
    ("blocking:RA", "tailbud", 1, "cyp26a1 in the tailbud"),
    ("blocking:RA", "forebrain_eye", 1, "and in the anterior neural plate"),
    ("sending:Hedgehog", "axial_mesoderm", 1, "shha in the axial mesoderm"),
    ("sending:Hedgehog", "notochord", 1, "and the notochord"),
    ("sending:Notch", "presomitic_mesoderm", 1, "deltaC/D in the presomitic mesoderm"),
    ("sending:Notch", "primary_neurons", 1, "deltaA/B in the primary neurons"),
    ("sending:Notch", "receiving:Notch", 1, "Delta-expressing cells and their neighbours respond"),
]


def coupling(adata, programs: pd.DataFrame, support: pd.DataFrame,
             genes_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlation of every signalling profile with every other curated program
    per stage, and the AUROCs of HYPOTHESES."""
    space = program_populations.ProgramSpace(programs, support, genes_table,
                                             supported_only=False)
    rows, aucs = [], []
    for stage in STAGES:
        sub = adata[adata.obs["stage"] == stage]
        libs = sorted(sub.obs["library"].unique())
        raw, _, _ = space(sub, libs, stage)
        names = list(space.stage_programs(stage).index)
        scores = pd.DataFrame(np.vstack([raw[lib] for lib in libs]), columns=names)
        active = pd.DataFrame(np.vstack([space.active[lib] for lib in libs]), columns=names)
        corr = scores.corr()
        for o in [n for n in names if n.startswith(SIGNAL_PREFIXES)]:
            rows += [dict(stage=stage, signal=o, program=p.removeprefix("state:"),
                          r=corr.loc[o, p]) for p in names if p != o]
        for o, partner, _, _ in HYPOTHESES:
            if o not in names or partner not in names:
                continue
            # cells active in the identity partner (or in the profile, for signalling partners)
            group, score = ((o, partner) if partner.startswith(SIGNAL_PREFIXES)
                            else (partner, o))
            on = active[group].to_numpy()
            auc = (roc_auc_score(on, scores[score]) if MIN_ACTIVE <= on.sum() < len(on)
                   else np.nan)
            aucs.append(dict(stage=stage, output=o, partner=partner, active_cells=int(on.sum()),
                             auc=auc))
        print(stage, flush=True)
    return pd.DataFrame(rows), pd.DataFrame(aucs)


def reading(table: pd.DataFrame) -> pd.DataFrame:
    """Rows of the reading profiles against identity and state programs."""
    return table[table["signal"].str.startswith(READING)
                 & ~table["program"].str.startswith(SIGNAL_PREFIXES)]


def plot(table: pd.DataFrame, signals=SIGNALS, stem="signal_coupling",
         label="correlation with the profile (within libraries)", min_r: float = MIN_R) -> None:
    w = pd.read_csv(TABLES / "curated_zfin_windows.tsv", sep="\t")
    zfin = {p: STAGES[STAGES.index(a):STAGES.index(b) + 1]
            for p, a, b in zip(w["program"], w["first"], w["last"])}
    panels = []
    for signal in signals:
        if signal not in set(table["signal"]):
            continue
        t = table[table["signal"] == signal].pivot(index="program", columns="stage", values="r")
        t = t.reindex(columns=STAGES)
        recorded = pd.DataFrame([[s in zfin.get(p, STAGES) for s in STAGES] for p in t.index],
                                index=t.index, columns=STAGES)
        t = t[(t.abs().where(recorded) >= min_r).any(axis=1)]
        t = t.loc[t.abs().max(axis=1).sort_values(ascending=False).index]
        panels.append((signal, t))
    left, right = panels[0::2], panels[1::2]
    heights = [max(sum(len(t) for _, t in col), 1) for col in (left, right)]
    fig = plt.figure(figsize=(REPORT_WIDTH_IN, 0.13 * max(heights) + 1.6))
    for c, col in enumerate((left, right)):
        grid = fig.add_gridspec(len(col), 1, height_ratios=[max(len(t), 1) for _, t in col],
                                left=0.17 + 0.5 * c, right=0.47 + 0.5 * c, top=0.96,
                                bottom=0.15, hspace=0.45)
        for k, (signal, t) in enumerate(col):
            ax = fig.add_subplot(grid[k])
            ax.set_facecolor("#F2F2F2")
            im = ax.imshow(t.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r",
                           norm=TwoSlopeNorm(0, -1, 1))
            ax.set_yticks(range(len(t)), [p.replace("_", " ") for p in t.index], fontsize=5)
            last = k == len(col) - 1
            ax.set_xticks(range(len(STAGES)), STAGES if last else [], rotation=60, ha="right",
                          fontsize=5)
            ax.set_title(signal, fontsize=7, loc="left")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    cax = fig.add_axes([0.35, 0.025, 0.3, 0.01])
    fig.colorbar(im, cax=cax, orientation="horizontal").ax.tick_params(labelsize=5)
    cax.set_title(label, fontsize=6)
    save_report_figure(fig, stem)
    plt.close(fig)


def hypotheses(table: pd.DataFrame, aucs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for output, partner, sign, reason in HYPOTHESES:
        t = table[(table["signal"] == output) & (table["program"] == partner)].set_index("stage")
        r = t["r"].reindex([s for s in STAGES if s in t.index]).dropna()
        a = aucs[(aucs["output"] == output) & (aucs["partner"] == partner)].set_index("stage")
        a = a["auc"].reindex([s for s in STAGES if s in a.index]).dropna()
        gap = sign * (a - 0.5)
        pro, con = gap >= MIN_AUC_GAP, gap <= -MIN_AUC_GAP
        best = a.index[np.argmax(gap.to_numpy())] if len(a) else ""
        rows.append(dict(output=output, partner=partner, expected="+" if sign > 0 else "-",
                         reason=reason, stages=len(a),
                         first=a.index[0] if len(a) else "", last=a.index[-1] if len(a) else "",
                         best_stage=best, best_auc=a.get(best, np.nan),
                         median_auc=a.median() if len(a) else np.nan,
                         consistent_stages=",".join(a.index[pro]),
                         opposite_stages=",".join(a.index[con]),
                         median_r=r.median() if len(r) else np.nan,
                         verdict=("not testable" if not len(a) else
                                  "mixed" if pro.any() and con.any() else
                                  "consistent" if pro.any() else
                                  "opposite" if con.any() else "weak")))
    return pd.DataFrame(rows)


def latex_hypotheses(h: pd.DataFrame) -> None:
    """Report table of the hypotheses; every second row shaded."""
    def tex(s):
        return str(s).replace("%", "\\%").replace("_", " ")
    lines = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{2.2cm}"
             ">{\\raggedright\\arraybackslash}p{2.3cm}c>{\\raggedright\\arraybackslash}p{2.5cm}"
             "rrr>{\\raggedright\\arraybackslash}p{1.5cm}@{}}",
             "\\toprule",
             "Profile & Partner & Exp. & Stages & Best AUROC & Median & Median $r$ & Verdict \\\\",
             "\\midrule"]
    for i, r in enumerate(h.itertuples()):
        span = "--" if not r.stages else (r.first if r.first == r.last else f"{r.first} -- {r.last}")
        best = "--" if pd.isna(r.best_auc) else f"{r.best_auc:.2f} ({tex(r.best_stage)})"
        med = "--" if pd.isna(r.median_auc) else f"{r.median_auc:.2f}"
        mr = "--" if pd.isna(r.median_r) else f"{r.median_r:.2f}"
        name = lambda s: tex(s).replace(":", ":\\allowbreak ")  # e.g. sending:Hedgehog
        lines.append(("\\rowcolor{gray!10}" if i % 2 else "") + f"{name(r.output)} & {name(r.partner)} & {r.expected} & {tex(span)} & "
                     f"{best} & {med} & {mr} & {r.verdict} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPORT_TABLES / "signal_output_hypotheses.tex").write_text(
        "% Generated by scripts/21_signal_coupling.py; do not edit.\n" + "\n".join(lines) + "\n")


def plot_outputs(table: pd.DataFrame) -> None:
    plot(table, OUTPUTS, "signal_outputs", "correlation with the profile (within libraries)",
         MIN_R_OUTPUTS)


def main() -> None:
    ensure_dirs()
    report_style()
    adata = load_dropseq_counts()
    programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    curated = programs[(programs["set"] == "curated") & programs["kept"]]
    support = pd.read_csv(TABLES / "program_support.tsv", sep="\t")
    genes_table = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                              usecols=["stage", "gene", "heterogeneous"])
    table, aucs = coupling(adata, curated, support, genes_table)
    reading(table).to_csv(TABLES / "signal_coupling.tsv", sep="\t", index=False,
                          float_format="%.3g")
    plot(reading(table))
    table.to_csv(TABLES / "signal_outputs.tsv", sep="\t", index=False, float_format="%.3g")
    aucs.to_csv(TABLES / "signal_output_auroc.tsv", sep="\t", index=False, float_format="%.3g")
    h = hypotheses(table, aucs)
    h.to_csv(TABLES / "signal_output_hypotheses.tsv", sep="\t", index=False, float_format="%.3g")
    print(h[["output", "partner", "expected", "stages", "best_auc", "median_auc", "median_r",
             "consistent_stages", "opposite_stages", "verdict"]].to_string(index=False))
    latex_hypotheses(h)
    plot_outputs(table)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["plot"]:
        report_style()
        plot(pd.read_csv(TABLES / "signal_coupling.tsv", sep="\t"))
        outputs = pd.read_csv(TABLES / "signal_outputs.tsv", sep="\t")
        aucs = pd.read_csv(TABLES / "signal_output_auroc.tsv", sep="\t")
        latex_hypotheses(hypotheses(outputs, aucs))
        plot_outputs(outputs)
    else:
        main()
