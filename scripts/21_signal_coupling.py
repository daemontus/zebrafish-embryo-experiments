"""Which programs are co-active with which signalling profiles, at which stage?

Per stage and library, the smoothed scores of the curated programs (as
program_utils.ProgramSpace; the curated programs are disjoint, script 16, so no correlation
comes from shared genes) are correlated over the library's cells: every
signalling profile with every other program. Smoothing and the few shared
axes of variation of a stage make any two sets of heterogeneous genes
correlate, so each correlation is compared with N_RANDOM random sets of
background genes (heterogeneous at the stage, in no curated program) of the
profile's size, scored and smoothed in the same way: relative coupling 0 =
as coupled as random genes, +1 / -1 = at the 95% / 5% quantile of the random
sets. Per stage: the median over the libraries, the range, and whether every
library is beyond the null on the same side. A coupling means that a
pathway's genes are expressed in the cells of a program, not that the
signal causes the program.

HYPOTHESES lists expected couplings (which tissue sends, receives or blocks
which signal). Each is tested on the cells: the AUROC of the profile's score
for the cells in which the identity partner is active (signalling partners:
of the partner's score for the cells in which the profile is active; active
calls of program_utils.ProgramSpace), against the cells active in the partner's parent
programs (derives_from of curated_programs.tsv) where it has any, otherwise
against all other cells, and against random sets as above (relative AUROC).
Tested at the stages where the identity partner is supported (script 17).

Transitions: for every derivation (parent -> program), every signalling
profile is compared in the same way between the program's active cells and
the parent's other active cells: a profile higher in the program's cells is a
candidate input of the transition (e.g. blastomere + Nodal -> margin), a
lower one a candidate inhibitor. Summarised at the program's emergence: the
first TRANSITION_STAGES stages at which it is supported, at >=
TRANSITION_HITS of which the profile must be beyond the null, never opposite.
A hypothesis is consistent if the relative AUROC (median over the libraries)
is beyond the null on the expected side at one or more stages and never on
the other side.

Outputs:
  results/tables/signal_outputs_libraries.tsv (stage, library, signal, program, r, relative)
  results/tables/signal_outputs.tsv           (per stage: median r and relative, range)
  results/tables/signal_coupling.tsv          (the same, reading profiles only)
  results/tables/signal_output_auroc.tsv      (stage, library, output, partner, contrast, AUROC)
  results/tables/signal_transitions_auroc.tsv (the same for every derivation and profile)
  results/tables/signal_transitions.tsv       (per derivation: higher / lower profiles)
  report/tables/signal_transitions.tex
  results/tables/signal_output_hypotheses.tsv (one row per hypothesis)
  report/figures/signal_coupling.pdf, report/figures/signal_outputs.pdf
  report/tables/signal_output_hypotheses.tex
"""
from __future__ import annotations


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from pipeline_utils import (
    REPORT_TABLES,
    ROOT,
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


ANNOTATION = ROOT / "data" / "curated" / "curated_programs.tsv"  # derives_from
N_RANDOM = 100  # random sets of background genes per library and program size
NULL_QUANTILE = 0.95  # relative = +-1 at the 95% / 5% quantile of the random sets
REL_LIMIT = 3.0  # colour scale of the figures (clipped)
N_COLUMNS = 3  # columns of panels in the coupling figures
ROW_HEIGHT = 0.105  # inches per program row
MIN_ACTIVE = 10  # active cells (and other cells), at least, for an AUROC
TRANSITION_STAGES = 3  # transitions: the first stages at which the program is supported
TRANSITION_HITS = 2  # ... at which a profile must be beyond the null (and never opposite)
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


def relative(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Position of obs (one value per column) against a random null (rows:
    random sets): 0 = the median, +1 = the NULL_QUANTILE quantile, -1 = the
    1 - NULL_QUANTILE quantile."""
    med = np.nanmedian(null, axis=0)
    hi = np.nanquantile(null, NULL_QUANTILE, axis=0)
    lo = np.nanquantile(null, 1 - NULL_QUANTILE, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(obs >= med, (obs - med) / (hi - med), (obs - med) / (med - lo))


def standardise(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = np.nan
    return x / sd


def library_scores(counts, het_idx: np.ndarray, prog_idx: dict, background: np.ndarray,
                   sizes: set, rng) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Smoothed scores and active calls of the programs of one library (as
    program_utils.ProgramSpace), and the smoothed scores of N_RANDOM random sets of background
    genes for every program size."""
    seed = prediction.SEED
    n = counts.shape[0]
    coords = PCA(min(prediction.N_PCS, n - 1), random_state=seed).fit_transform(
        pearson_residuals(counts, het_idx))
    library = np.zeros(n, dtype=int)
    genes = sorted({g for gs in prog_idx.values() for g in gs})
    col = {g: k for k, g in enumerate(genes)}
    res = pearson_residuals(counts, np.array(genes))
    names = list(prog_idx)
    score = prediction.smooth(np.column_stack([res[:, [col[g] for g in prog_idx[p]]].mean(axis=1)
                                               for p in names]), library, coords)
    threshold = []
    for p in names:
        r = res[:, [col[g] for g in prog_idx[p]]]
        null = np.concatenate([prediction.smooth(rng.permuted(r, axis=0).mean(axis=1)[:, None],
                                                 library, coords)[:, 0]
                               for _ in range(prediction.ACTIVE_N_PERM)])
        threshold.append(np.quantile(null, prediction.ACTIVE_QUANTILE))
    active = score > np.array(threshold)[None, :]
    bres = pearson_residuals(counts, background)
    random = {}
    for size in sizes:
        k = min(size, len(background))
        sets = [rng.choice(len(background), k, replace=False) for _ in range(N_RANDOM)]
        random[size] = prediction.smooth(np.column_stack([bres[:, s].mean(axis=1) for s in sets]),
                                         library, coords)
    return (pd.DataFrame(score, columns=names), pd.DataFrame(active, columns=names), random)


def auroc_rows(stage, lib, output, partner, contrast, on, base, score, random_scores):
    """AUROC of `score` for the `on` cells among the `on | base` cells, and the
    same for random sets of the same size (null)."""
    use = on | base
    if on.sum() < MIN_ACTIVE or (base & ~on).sum() < MIN_ACTIVE:
        return []
    y = on[use]
    auc = roc_auc_score(y, score[use])
    null = np.array([roc_auc_score(y, random_scores[use, k])
                     for k in range(random_scores.shape[1])])
    return [dict(stage=stage, library=lib, output=output, partner=partner, contrast=contrast,
                 active_cells=int(y.sum()), auc=auc, null_median=np.median(null),
                 relative=float(relative(np.array([auc]), null[:, None])[0]))]


def coupling(adata, programs: pd.DataFrame, het_table: pd.DataFrame,
             parents: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per stage and library: the correlation of every signalling profile with
    every other curated program, against random sets of background genes of
    the profile's size; and the AUROCs of HYPOTHESES (against all other cells
    and against the cells of the partner's parent programs), with the same
    null; and, for every derivation (parent -> program, derives_from), the
    AUROC of every signalling profile for the program's active cells against
    the parent's other active cells (transitions)."""
    rows, aucs, trans = [], [], []
    for stage in STAGES:
        sub = adata[adata.obs["stage"] == stage]
        p = programs[programs["stage"] == stage]
        prog_genes = p.groupby("program", sort=False)["gene"].agg(list)
        het = set(het_table.loc[(het_table["stage"] == stage) & het_table["heterogeneous"],
                                "gene"])
        het_idx = np.flatnonzero(sub.var_names.isin(het))
        background = np.flatnonzero(sub.var_names.isin(het - set(p["gene"])))
        prog_idx = {k: list(sub.var_names.get_indexer(v)) for k, v in prog_genes.items()}
        signals = [k for k in prog_idx if k.startswith(SIGNAL_PREFIXES)]
        sizes = {len(prog_idx[k]) for k in prog_idx}
        X = sub.X.tocsr()
        lib_of = sub.obs["library"].astype(str).to_numpy()
        for lib in sorted(set(lib_of)):
            rng = np.random.default_rng([prediction.SEED, STAGES.index(stage),
                                         sorted(set(lib_of)).index(lib)])
            score, active, random = library_scores(X[lib_of == lib], het_idx, prog_idx,
                                                   background, sizes, rng)
            z = standardise(score.to_numpy())
            names = list(score.columns)
            for o in signals:
                n_cells = len(score)
                r_obs = z[:, names.index(o)] @ z / n_cells
                null = standardise(random[len(prog_idx[o])]).T @ z / n_cells
                rel = relative(r_obs, null)
                rows += [dict(stage=stage, library=lib, signal=o,
                              program=q.removeprefix("state:"), r=r_obs[k],
                              null_median=np.nanmedian(null[:, k]), relative=rel[k])
                         for k, q in enumerate(names) if q != o]
            for o, partner, _, _ in HYPOTHESES:
                if o not in names or partner not in names:
                    continue
                # cells active in the identity partner (or in the profile, for signalling
                # partners), scored by the other one
                group, scored = ((o, partner) if partner.startswith(SIGNAL_PREFIXES)
                                 else (partner, o))
                on = active[group].to_numpy()
                s, null = score[scored].to_numpy(), random[len(prog_idx[scored])]
                aucs += auroc_rows(stage, lib, o, partner, "all", on, ~on, s, null)
                ups = [q for q in parents.get(partner, []) if q in names]
                if ups:
                    base = active[ups].to_numpy().any(axis=1)
                    aucs += auroc_rows(stage, lib, o, partner, "parents", on, base & ~on, s, null)
            for child, ups in parents.items():
                for parent in [q for q in ups if q in names and child in names]:
                    on = active[child].to_numpy()
                    base = active[parent].to_numpy() & ~on
                    for o in signals:
                        if o in (child, parent):
                            continue
                        trans += auroc_rows(stage, lib, o, f"{parent}>{child}", "transition", on,
                                            base, score[o].to_numpy(), random[len(prog_idx[o])])
        print(stage, flush=True)
    return pd.DataFrame(rows), pd.DataFrame(aucs), pd.DataFrame(trans)


def transitions(trans: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    """Per derivation: the signalling profiles that are higher (relative AUROC
    >= 1, median over the libraries) or lower (<= -1) in the program's active
    cells than in the parent's other active cells at >= TRANSITION_HITS of
    the first TRANSITION_STAGES stages at which the program is supported (its
    emergence), and never on the other side there."""
    c = support[(support["set"] == "curated") & support["supported"]]
    first = {p: [s for s in STAGES if s in set(g["stage"])][:TRANSITION_STAGES]
             for p, g in c.groupby("program")}
    a = trans.groupby(["stage", "output", "partner"], sort=False).agg(
        relative=("relative", "median"), auc=("auc", "median")).reset_index()
    rows = []
    for edge in dict.fromkeys(trans["partner"]):
        parent, child = edge.split(">")
        window = first.get(child, [])
        t = a[(a["partner"] == edge) & a["stage"].isin(window)]
        higher, lower = [], []
        for signal, g in t.groupby("output", sort=False):
            up, down = (g["relative"] >= 1).sum(), (g["relative"] <= -1).sum()
            if up >= TRANSITION_HITS and not down:
                higher.append(signal)
            elif down >= TRANSITION_HITS and not up:
                lower.append(signal)
        rows.append(dict(parent=parent, program=child, stages=",".join(window),
                         higher=";".join(higher), lower=";".join(lower)))
    return pd.DataFrame(rows)


def latex_transitions(tr: pd.DataFrame, notes: pd.DataFrame) -> None:
    """Report table of the transitions, with the signals expected from the
    annotation (curated_programs.tsv, column location_signals)."""
    expected = notes.set_index("program")["location_signals"]
    order = {p: k for k, p in enumerate(notes["program"])}
    def tex(s):
        return str(s).replace("%", "\\%").replace("_", " ")
    def short(names):
        return ", ".join(tex(n.split(":")[1] + ("" if n.startswith("signal:") else
                                               " (" + n.split(":")[0] + ")"))
                         for n in names.split(";") if n) or "--"
    lines = ["\\begin{longtable}{@{}" + "".join(
                 f">{{\\raggedright\\arraybackslash}}p{{{w}cm}}" for w in (3.3, 2.0, 4.0, 3.4, 3.3))
             + "@{}}",
             "\\caption{Phenotype transitions (\\protect\\nolinkurl{scripts/21_signal_coupling.py}): "
             "for every presumed derivation (parent $\\rightarrow$ program), the signalling "
             "profiles that are higher or lower in the program's active cells than in the parent's "
             "other active cells, beyond random genes, at two or more of the first three stages at "
             "which the program is supported (Stages), and never on the other side there. "
             "Expected: where and under which signals the program arises, from the annotation "
             "(Table~\\ref{tab:curated-programs}). Profiles: general unless marked (sending, "
             "blocking, receiving).}\\label{tab:transitions}\\\\",
             "\\toprule", "Transition & Stages & Expected & Higher & Lower \\\\", "\\midrule",
             "\\endfirsthead", "\\toprule",
             "Transition & Stages & Expected & Higher & Lower \\\\", "\\midrule", "\\endhead",
             "\\bottomrule", "\\endlastfoot"]
    tr = tr.assign(_o=tr["program"].map(order)).sort_values("_o", kind="stable")
    for i, r in enumerate(tr.itertuples()):
        st = r.stages.split(",") if r.stages else []
        span = "not supported" if not st else st[0] if len(st) == 1 else f"{st[0]} -- {st[-1]}"
        lines.append(("\\rowcolor{gray!10}" if i % 2 else "")
                     + f"{tex(r.parent.removeprefix('state:'))} $\\rightarrow$ "
                       f"{tex(r.program.removeprefix('state:'))} & {tex(span)} & "
                       f"{tex(expected.get(r.program, ''))} & {short(r.higher)} & "
                       f"{short(r.lower)} \\\\")
    lines.append("\\end{longtable}")
    (REPORT_TABLES / "signal_transitions.tex").write_text(
        "% Generated by scripts/21_signal_coupling.py; do not edit.\n" + "\n".join(lines) + "\n")


def per_stage(table: pd.DataFrame) -> pd.DataFrame:
    """Median over the libraries of a stage, the range, and whether every
    library is beyond the null on the same side."""
    g = table.groupby(["stage", "signal", "program"], sort=False)
    out = g.agg(r=("r", "median"), relative=("relative", "median"),
                relative_min=("relative", "min"), relative_max=("relative", "max"),
                libraries=("library", "size")).reset_index()
    out["every_library"] = (out["relative_min"] >= 1) | (out["relative_max"] <= -1)
    return out


def reading(table: pd.DataFrame) -> pd.DataFrame:
    """Rows of the reading profiles against identity and state programs."""
    return table[table["signal"].str.startswith(READING)
                 & ~table["program"].str.startswith(SIGNAL_PREFIXES)]


def plot(table: pd.DataFrame, signals=SIGNALS, stem="signal_coupling",
         n_columns: int = N_COLUMNS, row_height: float = ROW_HEIGHT) -> None:
    """Relative coupling (median over the libraries) of each profile with the
    programs that are beyond the null at some stage; circle: in every library."""
    panels = []
    for signal in signals:
        t = table[table["signal"] == signal]
        if t.empty:
            continue
        rel = t.pivot(index="program", columns="stage", values="relative").reindex(columns=STAGES)
        every = t.pivot(index="program", columns="stage", values="every_library") \
            .reindex(columns=STAGES).fillna(False).astype(bool)
        keep = rel.index[(rel.abs() >= 1).any(axis=1)]
        rel = rel.loc[rel.loc[keep].abs().max(axis=1).sort_values(ascending=False).index]
        panels.append((signal, rel, every.loc[rel.index]))
    columns = [[] for _ in range(n_columns)]  # balanced: each panel to the shortest column
    for panel in sorted(panels, key=lambda x: -len(x[1])):
        min(columns, key=lambda col: sum(len(t) for _, t, _ in col)).append(panel)
    columns = [sorted(col, key=lambda x: signals.index(x[0])) for col in columns if col]
    heights = [max(sum(len(t) for _, t, _ in col), 1) for col in columns]
    fig = plt.figure(figsize=(REPORT_WIDTH_IN, row_height * max(heights) + 1.4))
    norm = TwoSlopeNorm(0, -REL_LIMIT, REL_LIMIT)
    width = 1 / len(columns)
    for c, col in enumerate(columns):
        grid = fig.add_gridspec(len(col), 1, height_ratios=[max(len(t), 1) for _, t, _ in col],
                                left=width * c + 0.39 * width, right=width * (c + 1) - 0.01,
                                top=0.97,
                                bottom=0.13, hspace=0.35)
        for k, (signal, rel, every) in enumerate(col):
            ax = fig.add_subplot(grid[k])
            ax.set_facecolor("#F2F2F2")
            im = ax.imshow(rel.to_numpy(dtype=float).clip(-REL_LIMIT, REL_LIMIT), aspect="auto",
                           cmap="RdBu_r", norm=norm)
            yy, xx = np.nonzero(every.to_numpy())
            ax.scatter(xx, yy, s=5, color="white", edgecolors="#222222", linewidths=0.4)
            ax.set_yticks(range(len(rel)), [p.replace("_", " ") for p in rel.index],
                          fontsize=5 if row_height >= 0.1 else 4)
            last = k == len(col) - 1
            ax.set_xticks(range(len(STAGES)), STAGES if last else [], rotation=60, ha="right",
                          fontsize=5)
            ax.set_title(signal, fontsize=7, loc="left")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    cax = fig.add_axes([0.35, 0.025, 0.3, 0.01])
    fig.colorbar(im, cax=cax, orientation="horizontal").ax.tick_params(labelsize=5)
    cax.set_title("relative coupling (0: random genes, ±1: 95% beyond; median over "
                  "libraries; circle: every library)", fontsize=6)
    save_report_figure(fig, stem)
    plt.close(fig)


def hypotheses(table: pd.DataFrame, aucs: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    """Verdict per hypothesis from the AUROCs relative to random sets (median
    over the libraries; against the partner's parent programs where it has
    any, otherwise against all other cells), at the stages where the identity
    partner is supported (signalling partners: every stage with an AUROC)."""
    supported = set(zip(*support.loc[(support["set"] == "curated") & support["supported"],
                                     ["program", "stage"]].T.to_numpy()))
    a = aucs.groupby(["stage", "output", "partner", "contrast"], sort=False).agg(
        auc=("auc", "median"), relative=("relative", "median"),
        libraries=("library", "size")).reset_index()
    rows = []
    for output, partner, sign, reason in HYPOTHESES:
        t = a[(a["output"] == output) & (a["partner"] == partner)]
        contrast = "parents" if (t["contrast"] == "parents").any() else "all"
        t = t[t["contrast"] == contrast].set_index("stage")
        stages = [s for s in STAGES if s in t.index and (partner.startswith(SIGNAL_PREFIXES)
                                                          or (partner, s) in supported)]
        t = t.reindex(stages)
        r = table[(table["signal"] == output) & (table["program"] == partner)] \
            .set_index("stage")["r"]
        gap = sign * t["relative"]
        pro, con = gap >= 1, gap <= -1
        best = t.index[np.argmax(gap.to_numpy())] if len(t) else ""
        rows.append(dict(output=output, partner=partner, expected="+" if sign > 0 else "-",
                         reason=reason, contrast=contrast, stages=len(t),
                         first=stages[0] if stages else "", last=stages[-1] if stages else "",
                         best_stage=best, best_auc=t["auc"].get(best, np.nan),
                         median_auc=t["auc"].median() if len(t) else np.nan,
                         consistent_stages=",".join(t.index[pro]),
                         opposite_stages=",".join(t.index[con]),
                         median_r=r.reindex(stages).median() if stages else np.nan,
                         verdict=("not testable" if not len(t) else
                                  "mixed" if pro.any() and con.any() else
                                  "consistent" if pro.any() else
                                  "opposite" if con.any() else "weak")))
    return pd.DataFrame(rows)


def latex_hypotheses(h: pd.DataFrame) -> None:
    """Report table of the hypotheses; every second row shaded."""
    def tex(s):
        return str(s).replace("%", "\\%").replace("_", " ")
    lines = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{2.2cm}"
             ">{\\raggedright\\arraybackslash}p{2.3cm}cl>{\\raggedright\\arraybackslash}p{2.3cm}"
             "rrr>{\\raggedright\\arraybackslash}p{1.4cm}@{}}",
             "\\toprule",
             "Profile & Partner & Exp. & Against & Stages & Best AUROC & Median & Median $r$ & "
             "Verdict \\\\",
             "\\midrule"]
    for i, r in enumerate(h.itertuples()):
        span = "--" if not r.stages else (r.first if r.first == r.last else f"{r.first} -- {r.last}")
        best = "--" if pd.isna(r.best_auc) else f"{r.best_auc:.2f} ({tex(r.best_stage)})"
        med = "--" if pd.isna(r.median_auc) else f"{r.median_auc:.2f}"
        mr = "--" if pd.isna(r.median_r) else f"{r.median_r:.2f}"
        name = lambda s: tex(s).replace(":", ":\\allowbreak ")  # e.g. sending:Hedgehog
        lines.append(("\\rowcolor{gray!10}" if i % 2 else "") + f"{name(r.output)} & {name(r.partner)} & {r.expected} & "
                     f"{'parents' if r.contrast == 'parents' else 'all'} & {tex(span)} & "
                     f"{best} & {med} & {mr} & {r.verdict} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPORT_TABLES / "signal_output_hypotheses.tex").write_text(
        "% Generated by scripts/21_signal_coupling.py; do not edit.\n" + "\n".join(lines) + "\n")


def plot_outputs(table: pd.DataFrame) -> None:
    plot(table, OUTPUTS, "signal_outputs", row_height=0.075)  # one page


def main() -> None:
    ensure_dirs()
    report_style()
    adata = load_dropseq_counts()
    programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    curated = programs[(programs["set"] == "curated") & programs["kept"]]
    support = pd.read_csv(TABLES / "program_support.tsv", sep="\t")
    het = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                      usecols=["stage", "gene", "heterogeneous"])
    notes = pd.read_csv(ANNOTATION, sep="\t", keep_default_na=False)
    parents = {p: [q for q in d.split(";") if q] for p, d in zip(notes["program"],
                                                                 notes["derives_from"])}
    libraries, aucs, trans = coupling(adata, curated, het, parents)
    trans.to_csv(TABLES / "signal_transitions_auroc.tsv", sep="\t", index=False,
                 float_format="%.3g")
    tr = transitions(trans, support)
    tr.to_csv(TABLES / "signal_transitions.tsv", sep="\t", index=False)
    print(tr.to_string(index=False))
    latex_transitions(tr, notes)
    libraries.to_csv(TABLES / "signal_outputs_libraries.tsv", sep="\t", index=False,
                     float_format="%.3g")
    table = per_stage(libraries)
    table.to_csv(TABLES / "signal_outputs.tsv", sep="\t", index=False, float_format="%.3g")
    reading(table).to_csv(TABLES / "signal_coupling.tsv", sep="\t", index=False,
                          float_format="%.3g")
    aucs.to_csv(TABLES / "signal_output_auroc.tsv", sep="\t", index=False, float_format="%.3g")
    h = hypotheses(table, aucs, support)
    h.to_csv(TABLES / "signal_output_hypotheses.tsv", sep="\t", index=False, float_format="%.3g")
    print(h[["output", "partner", "expected", "contrast", "stages", "best_auc", "median_auc",
             "consistent_stages", "opposite_stages", "verdict"]].to_string(index=False))
    print(h["verdict"].value_counts().to_string())
    latex_hypotheses(h)
    plot(reading(table))
    plot_outputs(table)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["plot"]:
        report_style()
        table = pd.read_csv(TABLES / "signal_outputs.tsv", sep="\t")
        plot(reading(table))
        latex_hypotheses(hypotheses(table, pd.read_csv(TABLES / "signal_output_auroc.tsv",
                                                       sep="\t"),
                                    pd.read_csv(TABLES / "program_support.tsv", sep="\t")))
        plot_outputs(table)
        notes = pd.read_csv(ANNOTATION, sep="\t", keep_default_na=False)
        latex_transitions(transitions(pd.read_csv(TABLES / "signal_transitions_auroc.tsv",
                                                  sep="\t"),
                                      pd.read_csv(TABLES / "program_support.tsv", sep="\t")),
                          notes)
    else:
        main()
