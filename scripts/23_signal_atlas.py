"""Do the signalling readouts agree with the programs they are known to drive?

An exploration figure (not in the report). For every pathway, two curated
programs: one expected where the pathway is active (A) and one where it is
not (B), at stages where both and the readouts are defined (PAIRS). Cells in
which exactly one of A and B is active (smoothed score above the 99.9%
quantile of a gene-wise permutation within the library, as in scripts 18 and
19) are coloured by the pathway's readout score:
  A active, readouts high   expected
  B active, readouts low    expected
  A active, readouts low    against expectation
  B active, readouts high   against expectation
"High" and "low" are above and below the library average (the smoothed
readout score is centred per library, so the split is at 0). As in script
21, the readout genes are removed from the other programs first, so A and B
cannot agree with the readouts through shared genes. Shown on the aligned
and unaligned atlas UMAPs of script 15, only for the stages where all three
programs are defined.

`23_signal_atlas.py plot` redraws from the cache.

Outputs:
  results/figures/signal_atlas.pdf
  results/tables/signal_atlas.tsv                (stage x pathway: cells per category)
  data/processed/signal_atlas_{scores,active}.tsv.gz
"""
from __future__ import annotations

import importlib
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from pipeline_utils import PROCESSED, ROOT, STAGES, TABLES, ensure_dirs, load_dropseq_counts

warnings.filterwarnings("ignore", category=FutureWarning)
program_populations = importlib.import_module("18_program_populations")
coupling = importlib.import_module("21_signal_coupling")
atlas_figures = importlib.import_module("19_program_atlas")

FIGURES = ROOT / "results" / "figures"
SCORES = PROCESSED / "signal_atlas_scores.tsv.gz"
ACTIVE = PROCESSED / "signal_atlas_active.tsv.gz"
# pathway: (program expected with high readouts, program expected with low readouts)
PAIRS = {
    "BMP": ("non_neural_ectoderm", "neural_plate"),
    "Nodal": ("margin", "presumptive_neural"),
    "Wnt": ("tailbud", "forebrain_eye"),
    "FGF": ("mesoderm", "non_neural_ectoderm"),
    "Notch": ("presomitic_mesoderm", "lateral_plate"),
}
CATEGORIES = [("A", "high", "#0072B2"), ("B", "low", "#009E73"),
              ("A", "low", "#D55E00"), ("B", "high", "#E69F00")]


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    adata = load_dropseq_counts()
    programs = pd.read_csv(TABLES / "programs.tsv", sep="\t")
    curated = programs[(programs["set"] == "curated") & programs["kept"]]
    stripped, _ = coupling.without_readouts(curated)
    stripped = pd.concat([stripped, programs[programs["set"] != "curated"]])
    support = pd.read_csv(TABLES / "program_support.tsv", sep="\t")
    genes_table = pd.read_csv(TABLES / "heterogeneous_genes.tsv.gz", sep="\t",
                              usecols=["stage", "gene", "heterogeneous"])
    space = program_populations.ProgramSpace(stripped, support, genes_table,
                                             supported_only=False)
    wanted = {f"signal:{s}" for s in PAIRS} | {p for pair in PAIRS.values() for p in pair}
    scores, active = {}, {}
    for stage in STAGES:
        sub = adata[adata.obs["stage"] == stage]
        libs = sorted(sub.obs["library"].unique())
        names = list(space.stage_programs(stage).index)
        if not wanted & set(names):
            continue
        raw, _, _ = space(sub, libs, stage)
        lib_of = sub.obs["library"].to_numpy()
        for j, name in enumerate(names):
            if name not in wanted:
                continue
            col = scores.setdefault(name, pd.Series(np.nan, index=adata.obs_names))
            act = active.setdefault(name, pd.Series(np.nan, index=adata.obs_names))
            for lib in libs:
                cells = sub.obs_names[lib_of == lib]
                col.loc[cells] = raw[lib][:, j]
                act.loc[cells] = space.active[lib][:, j].astype(float)
        print(stage, flush=True)
    scores, active = pd.DataFrame(scores), pd.DataFrame(active)
    scores.to_csv(SCORES, sep="\t", float_format="%.4g")
    active.to_csv(ACTIVE, sep="\t", float_format="%.0f")
    return scores, active


def categorise(signal: str, a: str, b: str, scores: pd.DataFrame,
               active: pd.DataFrame) -> pd.Series:
    """Category index (0-3, see CATEGORIES) per cell; -1 none, NaN not defined."""
    cols = [f"signal:{signal}", a, b]
    if any(c not in scores for c in cols):
        return pd.Series(np.nan, index=scores.index)
    defined = scores[cols].notna().all(axis=1)
    on_a, on_b = active[a].fillna(0) > 0, active[b].fillna(0) > 0
    high = scores[f"signal:{signal}"] > 0
    cat = pd.Series(-1.0, index=scores.index)
    cat[on_a & ~on_b & high] = 0
    cat[on_b & ~on_a & ~high] = 1
    cat[on_a & ~on_b & ~high] = 2
    cat[on_b & ~on_a & high] = 3
    return cat.where(defined)


def plot(scores: pd.DataFrame, active: pd.DataFrame) -> None:
    atlas = atlas_figures.atlas_coordinates(None, None)
    scores, active = scores.reindex(atlas.index), active.reindex(atlas.index)
    rows = []
    with PdfPages(FIGURES / "signal_atlas.pdf") as pdf:
        for signal, (a, b) in PAIRS.items():
            cat = categorise(signal, a, b, scores, active)
            defined = cat.notna().to_numpy()
            stages = [s for s in STAGES if defined[(atlas["stage"] == s).to_numpy()].any()]
            if not stages:
                print(f"{signal}: no stage with all three programs")
                continue
            for stage in stages:
                c = cat[atlas["stage"] == stage].dropna()
                rows.append(dict(stage=stage, signal=signal, A=a, B=b,
                                 **{f"{p}_{lvl}": int((c == k).sum())
                                    for k, (p, lvl, _) in enumerate(CATEGORIES)}))
            v = cat.to_numpy()
            colour = np.full(len(v), "#F2F2F2", dtype=object)
            colour[v == -1] = "#D9D9D9"
            for k, (_, _, hex_) in enumerate(CATEGORIES):
                colour[v == k] = hex_
            order = np.argsort(np.nan_to_num(v, nan=-2) >= 0, kind="stable")
            fig, axes = plt.subplots(1, 2, figsize=(16, 7.8))
            for ax, prefix in zip(axes, ["aligned", "unaligned"]):
                xy = atlas[[f"{prefix}_1", f"{prefix}_2"]].to_numpy()
                ax.scatter(xy[order, 0], xy[order, 1],
                           s=np.where(np.nan_to_num(v[order], nan=-2) >= 0, 2.0, 0.8),
                           c=colour[order], linewidths=0, rasterized=True)
                ax.set_title(f"atlas, {prefix}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            name = {"A": a, "B": b}
            n = {k: int((v == k).sum()) for k in range(4)}
            agree_a = n[0] / max(n[0] + n[2], 1)
            agree_b = n[1] / max(n[1] + n[3], 1)
            labels = [f"{name['A']} active, {signal} high (expected; {n[0]} cells)",
                      f"{name['B']} active, {signal} low (expected; {n[1]} cells)",
                      f"{name['A']} active, {signal} low (against; {n[2]} cells)",
                      f"{name['B']} active, {signal} high (against; {n[3]} cells)"]
            handles = [plt.Line2D([], [], ls="", marker="o", ms=6, color=hex_, label=label)
                       for (_, _, hex_), label in zip(CATEGORIES, labels)]
            handles += [plt.Line2D([], [], ls="", marker="o", ms=6, color="#D9D9D9",
                                   label="neither (or both) active"),
                        plt.Line2D([], [], ls="", marker="o", ms=6, color="#F2F2F2",
                                   markeredgecolor="#BBBBBB", label="not defined at the stage")]
            fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
            fig.suptitle(f"{signal} readouts vs {a} (expected high) and {b} (expected low), "
                         f"{stages[0]} - {stages[-1]}: {agree_a:.0%} of {a} cells and "
                         f"{agree_b:.0%} of {b} cells as expected\n(high / low: smoothed "
                         "readout score above / below the library average; readout genes "
                         "removed from the programs)", fontsize=11)
            fig.tight_layout(rect=(0, 0.08, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)
            print(f"{signal} ({stages[0]} - {stages[-1]}): {a} {agree_a:.0%} expected "
                  f"({n[0] + n[2]} cells), {b} {agree_b:.0%} ({n[1] + n[3]} cells)")
    pd.DataFrame(rows).to_csv(TABLES / "signal_atlas.tsv", sep="\t", index=False)


def main() -> None:
    ensure_dirs()
    plot(*compute())


if __name__ == "__main__":
    if sys.argv[1:] == ["plot"]:
        plot(pd.read_csv(SCORES, sep="\t", index_col=0),
             pd.read_csv(ACTIVE, sep="\t", index_col=0))
    else:
        main()
