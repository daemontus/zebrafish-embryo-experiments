#!/usr/bin/env python3
"""Stage 09: consensus NMF gene programs, run separately in each 50% epiboly library.

Input : data/processed/07_50pct_joint.h5ad          (raw counts in layers['counts'])
Output: data/processed/09_cnmf/<library>[_ext]/...    (cNMF working directories)
        results/tables/09_k_selection.tsv             stability / error per library x K
        results/tables/09_selected_k.tsv              common K and sensitivity Ks
        data/processed/09_programs/09_programs_<library>_k<K>_{spectra_score,usage}.tsv
                                   genes x programs / cells x programs (usage sums to 1), every K
        results/tables/09_programs_<library>_k<main K>_*.tsv   copy of the main-K tables
        results/figures/09_k_selection.png/.pdf

Why per library: each library is an independent sample, so a program that is
found again in other libraries (stage 10) is replicated, not an artefact of one
library or of a batch-correction step. No batch correction is used.

cNMF (Kotliar et al., eLife 2019) runs NMF N_ITER times per K on the top
variable genes, clusters the resulting components, drops outlier components
(local density filter) and takes the median of each cluster as the consensus
program. Spectra are then refit to all genes: 'spectra_score' is a z-scored
regression coefficient per gene and program (used for ranking genes).

K selection (explicit rule): one COMMON K for all libraries, so that programs
are comparable across them. It is the K >= MIN_K that maximises the worst
library's consensus stability (silhouette of component clusters). MIN_K = 7
because stage 07 already resolved 7 populations at 50% epiboly (EVL, p53,
endoderm, margin, prechordal, organizer, bulk), so fewer programs could not
represent them. Consensus results are also written for SENSITIVITY_OFFSETS
around the chosen K; stage 10 reports whether each replicated program is
recovered at those K as well.

K is scanned over 5-15 (seed 14) plus an extension 16-20 run in a separate
cNMF directory with its own seed (15), so adding the extension did not alter
the 5-15 factorisations. Consensus programs are written for every K so that
stage 11 can track replication across the whole range.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

from cnmf_tools import MIN_K, choose_common_k, consensus, factorize_counts, k_selection
from pipeline_utils import PROCESSED, TABLES, ensure_dirs, save_figure

INPUT_H5AD = PROCESSED / "07_50pct_joint.h5ad"
CNMF_DIR = PROCESSED / "09_cnmf"
PROGRAMS_DIR = PROCESSED / "09_programs"

# (directory suffix, K values, seed); the extension was added after the first run.
K_RUNS = [("", list(range(5, 16)), 14), ("_ext", list(range(16, 21)), 15)]
K_ALL = [k for _, ks, _ in K_RUNS for k in ks]
SENSITIVITY_OFFSETS = (-2, 3)

plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-run the NMF factorisations even if they exist")
    return parser.parse_args()


def lib_name(lib: str) -> str:
    """'50% DS4b' -> 'ZF50_DS4b' (filesystem-safe)."""
    return "ZF50_" + lib.split()[-1]


def plot_k_selection(ks: pd.DataFrame, k: int, sens: list[int]) -> None:
    libs = list(dict.fromkeys(ks["library"]))
    fig, axes = plt.subplots(1, len(libs), figsize=(4 * len(libs), 3.4), squeeze=False)
    for ax, lib in zip(axes[0], libs):
        d = ks[ks["library"] == lib]
        ax.plot(d["k"], d["stability"], "o-", color="#4c72b0", ms=4)
        ax.axvline(k, color="0.3", ls="--", lw=0.8)
        for s in sens:
            ax.axvline(s, color="0.6", ls=":", lw=0.8)
        ax.set_xlabel("K")
        ax.set_ylabel("stability (silhouette)", color="#4c72b0")
        ax2 = ax.twinx()
        ax2.plot(d["k"], d["prediction_error"], "s-", color="#c44e52", ms=3)
        ax2.set_ylabel("prediction error", color="#c44e52")
        ax2.spines["top"].set_visible(False)
        ax.set_title(lib)
    fig.suptitle(f"cNMF K selection: common K={k} (dashed; maximises worst-library stability, "
                 f"K>={MIN_K}); sensitivity K={sens} (dotted)", y=1.03)
    fig.tight_layout()
    save_figure(fig, "09_k_selection")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    CNMF_DIR.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(INPUT_H5AD)
    libs = list(adata.obs["library_short"].cat.categories)

    objs, all_ks = {}, []
    for lib in libs:
        for suffix, run_ks, seed in K_RUNS:
            sub = adata[adata.obs["library_short"] == lib]
            obj = factorize_counts(sub, lib_name(lib) + suffix, CNMF_DIR, run_ks, seed,
                                   args.force, log_prefix="[09]")
            for kk in run_ks:
                objs[(lib, kk)] = obj
            all_ks.append(k_selection(obj, lib, run_ks))
    ks = pd.concat(all_ks, ignore_index=True)
    k = choose_common_k(ks)
    sens = [k + o for o in SENSITIVITY_OFFSETS if K_ALL[0] <= k + o <= K_ALL[-1]]
    print(f"[09] common K = {k}; sensitivity K = {sens}")

    PROGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    for lib in libs:
        for kk in K_ALL:
            usage, scores = consensus(objs[(lib, kk)], lib_name(lib), kk)
            stem = f"09_programs_{lib_name(lib)}_k{kk}"
            for out_dir in [PROGRAMS_DIR] + ([TABLES] if kk == k else []):
                scores.to_csv(out_dir / f"{stem}_spectra_score.tsv", sep="\t", float_format="%.4f")
                usage.to_csv(out_dir / f"{stem}_usage.tsv", sep="\t", float_format="%.4f")

    ks.to_csv(TABLES / "09_k_selection.tsv", sep="\t", index=False, float_format="%.4f")
    pd.DataFrame({"role": ["main"] + ["sensitivity"] * len(sens), "k": [k] + sens}).to_csv(
        TABLES / "09_selected_k.tsv", sep="\t", index=False)
    plot_k_selection(ks, k, sens)
    print(ks.to_string(index=False))
    print("[09] done")


if __name__ == "__main__":
    main()
