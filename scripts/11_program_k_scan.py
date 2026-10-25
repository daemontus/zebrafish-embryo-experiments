#!/usr/bin/env python3
"""Stage 11: how many replicated gene programs are there? Scan cNMF K = 5-20.

Input : data/processed/09_programs/09_programs_<library>_k<K>_*.tsv  (every K, stage 09)
        data/reference/*                                  (stage-10 annotation sources)
Output: results/tables/11_k_scan_programs.tsv     every meta-program at every K
        results/tables/11_k_scan_identities.tsv   program identities tracked across K
        results/tables/11_k_scan_summary.tsv      counts per K
        results/figures/11_k_scan_counts.png/.pdf
        results/figures/11_k_scan_identities.png/.pdf

The number of replicated programs is not chosen directly; it follows from K.
At every K the stage-10 procedure is repeated unchanged (mutual best match +
top-50 overlap q < 0.001 -> meta-programs; replicated = >= 3 of 4 libraries;
ZFIN in-situ and GO BP enrichment of the 50-gene consensus signature).

Identities across K: every meta-program at every K (replicated or not) is
represented by its mean spectra score over the shared genes. Average-linkage
clustering on 1 - Pearson r, cut at IDENTITY_R, groups them into identities
(e.g. "EVL" found at K=7, 9, 12, ...). For each identity and K the table
records the largest number of libraries it spans, so one can read off at
which K a program first replicates and whether it persists.

Labels combine the representative meta-program's best ZFIN term and best GO
BP term ("ZFIN | GO: ..."), using specific terms only (q < 0.05, <= 5% of the
universe):
  - ZFIN: highest fold enrichment among terms with >= MIN_LABEL_GENES genes.
    Anatomy terms nest (margin > presumptive endoderm), and fold picks the
    most specific supported one.
  - GO: smallest p. Ranking GO by fold favours small pleiotropic terms (e.g.
    "embryonic retina morphogenesis" for the endoderm program, carried by
    developmental transcription factors).
Showing both avoids presenting a weak ZFIN hit (p53 -> yolk syncytial layer,
4 genes) as the identity; such hits fall below MIN_LABEL_GENES.
The representative is the stage-10 (main K) meta-program if the identity is
replicated there; otherwise the member with the most libraries, then the
smallest K. Low-K members can be merged programs (e.g. ventral margin +
endoderm at K=5) and give misleading labels.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from pipeline_utils import TABLES, ensure_dirs, save_figure
from program_tools import (
    MAX_GO_GENES,
    MIN_LIBRARIES,
    enrich,
    go_annotations,
    load_programs,
    match_programs,
    meta_programs,
    signatures,
    zfin_annotations,
)

K_SCAN = list(range(5, 21))
IDENTITY_R = 0.6
ENRICH_Q = 0.05
SPECIFIC_TERM_FRACTION = 0.05
MIN_LABEL_GENES = 5

plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def specific_label(df: pd.DataFrame, universe: int) -> pd.Series:
    d = df[(df["q"] < ENRICH_Q) & (df["term_size"] <= SPECIFIC_TERM_FRACTION * universe)]
    best_p = d.sort_values(["mp", "p", "term"]).groupby("mp")["name"].first()
    d = d[d["overlap"] >= MIN_LABEL_GENES].sort_values(["mp", "fold", "term"],
                                                      ascending=[True, False, True])
    out = d.groupby("mp").agg(
        label=("name", "first"), label_q=("q", "first"),
        label_genes=("genes", "first"), n_specific=("name", "size"))
    out["best_p_term"] = best_p
    return out


def scan(zfin_terms, zfin_names, go_terms, go_names):
    rows, spectra = [], {}
    for k in K_SCAN:
        S, usages, lib_of = load_programs(k)
        pairs, _ = match_programs(S, lib_of)
        mp = meta_programs(S, pairs, lib_of)
        mean_scores, sig, _ = signatures(S, mp)
        zf, zm = enrich(sig, list(S.index), zfin_terms, zfin_names)
        go, gm = enrich(sig, list(S.index), go_terms, go_names, max_genes=MAX_GO_GENES)
        zl = specific_label(zf, zm)
        gl = specific_label(go, gm)
        for r in mp.itertuples():
            members = r.members.split(",")
            frac = np.mean([
                float((usages[lib_of[m]][m] > 0.2).mean()) for m in members])
            key = f"K{k}:{r.mp}"
            spectra[key] = mean_scores[r.mp]
            rows.append({
                "k": k, "mp": r.mp, "key": key, "n_libraries": r.n_libraries,
                "replicated": r.replicated, "members": r.members,
                "mean_edge_r": r.mean_edge_r,
                "frac_cells_usage_gt_0.2": frac,
                "zfin_label": zl["label"].get(r.mp, ""),
                "zfin_q": zl["label_q"].get(r.mp, np.nan),
                "zfin_genes": zl["label_genes"].get(r.mp, ""),
                "zfin_best_p_term": zl["best_p_term"].get(r.mp, ""),
                "n_specific_zfin": int(zl["n_specific"].get(r.mp, 0)),
                "go_label": gl["best_p_term"].get(r.mp, ""),
                "n_specific_go": int(gl["n_specific"].get(r.mp, 0)),
                "top10": ",".join(sig[r.mp][:10]),
            })
        print(f"[11] K={k}: {len(mp)} meta-programs, {int(mp['replicated'].sum())} replicated")
    return pd.DataFrame(rows), pd.DataFrame(spectra)


def identities(progs: pd.DataFrame, spectra: pd.DataFrame, main_k: int) -> pd.DataFrame:
    R = np.corrcoef(spectra.to_numpy().T)
    D = np.clip(1 - R, 0, None)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), "average")
    progs = progs.copy()
    progs["identity"] = fcluster(Z, t=1 - IDENTITY_R, criterion="distance")

    rows = []
    for ident, d in progs.groupby("identity"):
        d = d.assign(_main=((d["k"] == main_k) & d["replicated"]).astype(int))
        rep = d.sort_values(["_main", "n_libraries", "k"], ascending=[False, False, True]).iloc[0]
        parts = [rep["zfin_label"], f"GO: {rep['go_label']}" if rep["go_label"] else ""]
        label = " | ".join(x for x in parts if x) or "unannotated"
        per_k = d.groupby("k")["n_libraries"].max()
        rep_k = d[d["replicated"]]["k"]
        rows.append({
            "identity": ident, "label": label,
            "label_source": "ZFIN" if rep["zfin_label"] else ("GO" if rep["go_label"] else ""),
            "representative": rep["key"], "rep_top10": rep["top10"],
            "rep_zfin_q": rep["zfin_q"], "rep_zfin_genes": rep["zfin_genes"],
            "rep_zfin_best_p_term": rep["zfin_best_p_term"], "rep_go_label": rep["go_label"],
            "max_libraries": int(d["n_libraries"].max()),
            "first_replicated_k": int(rep_k.min()) if len(rep_k) else np.nan,
            "n_k_replicated": int(rep_k.nunique()),
            "replicated_at_main_k": bool(((d["k"] == main_k) & d["replicated"]).any()),
            "frac_cells_usage_gt_0.2": float(d.loc[d["replicated"], "frac_cells_usage_gt_0.2"].mean())
            if len(rep_k) else float(d["frac_cells_usage_gt_0.2"].mean()),
            **{f"libs_k{k}": int(per_k.get(k, 0)) for k in K_SCAN},
        })
    ids = pd.DataFrame(rows)
    ids = ids.sort_values(["n_k_replicated", "max_libraries", "first_replicated_k"],
                          ascending=[False, False, True]).reset_index(drop=True)
    ids.insert(0, "id", [f"ID{i + 1}" for i in range(len(ids))])
    progs["identity"] = progs["identity"].map(dict(zip(ids["identity"], ids["id"])))
    ids = ids.drop(columns="identity")
    return ids, progs


def plot_counts(summary: pd.DataFrame, main_k: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(summary["k"], summary["n_replicated"], "o-", color="#4c72b0",
            label="replicated programs (>=3 libraries)")
    ax.plot(summary["k"], summary["n_replicated_annotated"], "s--", color="#55a868",
            label="... with a specific ZFIN or GO term (q<0.05)")
    ax.plot(summary["k"], summary["n_replicated_identities_cum"], ":", color="0.4",
            label="distinct identities replicated at any K <= this K")
    ax.axvline(main_k, color="0.3", ls="--", lw=0.8)
    ax.set_xlabel("cNMF K (programs per library)")
    ax.set_ylabel("count")
    ax.set_xticks(summary["k"])
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Replicated programs vs K (dashed: stage-10 K={main_k})")
    fig.tight_layout()
    save_figure(fig, "11_k_scan_counts")
    plt.close(fig)


def plot_identities(ids: pd.DataFrame, main_k: int) -> None:
    show = ids[(ids["n_k_replicated"] > 0) | (ids["max_libraries"] >= 2)]
    mat = show[[f"libs_k{k}" for k in K_SCAN]].to_numpy()
    fig, ax = plt.subplots(figsize=(0.45 * len(K_SCAN) + 6, 0.3 * len(show) + 1.6))
    cmap = plt.get_cmap("Blues", 5)
    im = ax.imshow(mat, cmap=cmap, vmin=-0.5, vmax=4.5, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] >= MIN_LIBRARIES:
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=6, color="white")
    ax.set_xticks(range(len(K_SCAN)), K_SCAN)
    ax.set_yticks(range(len(show)),
                  [f"{r.id} {r.label[:70]}" + (f"  [{r.frac_cells_usage_gt_0_2:.1%}]")
                   for r in show.rename(columns={"frac_cells_usage_gt_0.2": "frac_cells_usage_gt_0_2"})
                   .itertuples()], fontsize=7)
    ax.axvline(K_SCAN.index(main_k), color="red", lw=0.8, ls="--")
    ax.set_xlabel("cNMF K")
    cb = fig.colorbar(im, ax=ax, ticks=range(5), shrink=0.7)
    cb.set_label("libraries spanned (numbers: replicated)")
    ax.set_title("Program identities across K (label from database enrichment; "
                 "[share of cells with usage > 0.2])", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "11_k_scan_identities")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    main_k = int(pd.read_csv(TABLES / "09_selected_k.tsv", sep="\t").query("role == 'main'")["k"].iloc[0])
    zfin_terms, zfin_names = zfin_annotations()
    go_terms, go_names = go_annotations()
    progs, spectra = scan(zfin_terms, zfin_names, go_terms, go_names)
    ids, progs = identities(progs, spectra, main_k)

    summary = progs.groupby("k").agg(
        n_meta_programs=("mp", "size"),
        n_replicated=("replicated", "sum"),
        n_replicated_annotated=("replicated", lambda s: int(
            (s & ((progs.loc[s.index, "n_specific_zfin"] > 0)
                  | (progs.loc[s.index, "n_specific_go"] > 0))).sum())),
    ).reset_index()
    seen, cum = set(), []
    for k in K_SCAN:
        seen |= set(progs.loc[(progs["k"] == k) & progs["replicated"], "identity"])
        cum.append(len(seen))
    summary["n_replicated_identities_cum"] = cum

    fmt = dict(sep="\t", index=False, float_format="%.4g")
    progs.to_csv(TABLES / "11_k_scan_programs.tsv", **fmt)
    ids.to_csv(TABLES / "11_k_scan_identities.tsv", **fmt)
    summary.to_csv(TABLES / "11_k_scan_summary.tsv", **fmt)
    plot_counts(summary, main_k)
    plot_identities(ids, main_k)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(summary.to_string(index=False))
    cols = ["id", "label", "max_libraries", "first_replicated_k", "n_k_replicated",
            "replicated_at_main_k", "frac_cells_usage_gt_0.2"]
    print(ids[ids["n_k_replicated"] > 0][cols].to_string(index=False))
    print("[11] done")


if __name__ == "__main__":
    main()
