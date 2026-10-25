#!/usr/bin/env python3
"""Stage 12: gene programs in the DS5 libraries (3.3-6.0 hpf) and their match to 50% epiboly.

Input : data/processed/07_ds5_joint.h5ad              (DS5 merged object, raw counts)
        data/processed/09_programs/                    (50% epiboly programs, every K)
        results/tables/11_k_scan_programs.tsv          (50% program -> identity)
        data/reference/*                               (ZFIN / GO)
Output: data/processed/12_cnmf/, data/processed/12_programs/
        results/tables/12_*.tsv, results/figures/12_*.png/.pdf

Units. cNMF needs a few hundred cells, so the four tiny high/oblong libraries
(48-263 cells) are pooled into one unit (511 cells; stage 07 showed they are
transcriptionally inseparable and co-embed completely). The other units are
single libraries: dome, 30% epiboly a, 30% epiboly b, shield. Each unit is
factorised separately on raw counts (no batch correction), K = 5-15, with the
stage-09 settings and common-K rule.

Recurrence. Programs are linked across units exactly as in stage 10 (mutual
best match + top-50 overlap q < 0.001). Because the units are different
stages, a stage-specific program can recur in at most 2 units (30% a/b), so a
DS5 meta-program is called "recurrent" if it spans >= MIN_UNITS = 2 units.
Signatures are annotated with ZFIN in situ (3.33-6.0 hpf window) and GO BP.

Match to 50% epiboly. Every DS5 program (main K) is compared with every
50%-epiboly library program at every K (5-20) over the genes shared by all
units and libraries: Pearson r of spectra scores and top-50 overlap
(hypergeometric, BH over all pairs). Each DS5 program's best match is mapped
to its stage-11 identity, so a DS5 program can confirm a 50% program that
replicated, or one that was seen in a single 50% library. A match requires a
top-50 overlap of >= MATCH_MIN_OVERLAP genes with q < MATCH_Q; the best
partner is the one with the largest overlap (then the highest r). Pearson r
is reported but not thresholded: across batch and stage, r over all ~12k
genes is depressed by gene-wide noise (best cross-context r 0.3-0.9 for
programs whose top-50 overlap has q < 1e-20), whereas top-gene overlap is
robust.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from cnmf_tools import choose_common_k, consensus, factorize_counts, k_selection
from pipeline_utils import PROCESSED, TABLES, ensure_dirs, save_figure
from program_tools import (
    MAX_GO_GENES,
    N_TOP,
    enrich,
    go_annotations,
    load_programs,
    match_programs,
    meta_programs,
    signatures,
    zfin_annotations,
)

INPUT_H5AD = PROCESSED / "07_ds5_joint.h5ad"
CNMF_DIR = PROCESSED / "12_cnmf"
PROGRAMS_DIR = PROCESSED / "12_programs"
UNITS = {
    "DS5_highoblong": ["high DS5", "high DS5b", "oblong DS5", "oblong DS5b"],
    "DS5_dome": ["dome DS5"],
    "DS5_30a": ["30% DS5"],
    "DS5_30b": ["30% DS5b"],
    "DS5_shield": ["shield DS5"],
}
UNIT_LABELS = {"DS5_highoblong": "high+oblong (pooled)", "DS5_dome": "dome",
               "DS5_30a": "30% a", "DS5_30b": "30% b", "DS5_shield": "shield"}
K_RANGE = list(range(5, 16))
SEED = 14
MIN_UNITS = 2
ZFIN_WINDOW_DS5 = (3.33, 6.0)
K_50 = list(range(5, 21))
MATCH_Q = 1e-3
MATCH_MIN_OVERLAP = 10
ENRICH_Q = 0.05
SPECIFIC_TERM_FRACTION = 0.05
MIN_LABEL_GENES = 5

plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-run the NMF factorisations even if they exist")
    return parser.parse_args()


def unit_tag(unit: str) -> str:
    return unit


def factorize_units(force: bool) -> tuple[dict, pd.DataFrame, int]:
    adata = sc.read_h5ad(INPUT_H5AD)
    objs, all_ks = {}, []
    for unit, libs in UNITS.items():
        sub = adata[adata.obs["library_short"].isin(libs)]
        objs[unit] = factorize_counts(sub, unit, CNMF_DIR, K_RANGE, SEED, force,
                                      log_prefix="[12]")
        all_ks.append(k_selection(objs[unit], unit, K_RANGE))
    ks = pd.concat(all_ks, ignore_index=True)
    k = choose_common_k(ks)
    PROGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    for unit, obj in objs.items():
        for kk in K_RANGE:
            usage, scores = consensus(obj, unit, kk)
            stem = f"12_programs_{unit}_k{kk}"
            for out_dir in [PROGRAMS_DIR] + ([TABLES] if kk == k else []):
                scores.to_csv(out_dir / f"{stem}_spectra_score.tsv", sep="\t", float_format="%.4f")
                usage.to_csv(out_dir / f"{stem}_usage.tsv", sep="\t", float_format="%.4f")
    return objs, ks, k


def specific_labels(zf, zm, go, gm) -> pd.DataFrame:
    z = zf[(zf["q"] < ENRICH_Q) & (zf["term_size"] <= SPECIFIC_TERM_FRACTION * zm)
           & (zf["overlap"] >= MIN_LABEL_GENES)]
    z = z.sort_values(["mp", "fold", "term"], ascending=[True, False, True]).groupby("mp").first()
    g = go[(go["q"] < ENRICH_Q) & (go["term_size"] <= SPECIFIC_TERM_FRACTION * gm)]
    g = g.sort_values(["mp", "p", "term"]).groupby("mp").first()
    out = pd.DataFrame({"zfin_label": z["name"], "zfin_q": z["q"], "zfin_genes": z["genes"],
                        "go_label": g["name"], "go_q": g["q"]})
    return out


def load_50pct_all_k(shared_genes) -> tuple[pd.DataFrame, dict]:
    cols, meta = [], {}
    for k in K_50:
        S, _, lib_of = load_programs(k)
        S = S.reindex(shared_genes)
        S.columns = [f"K{k}:{c}" for c in S.columns]
        cols.append(S)
        for c in S.columns:
            meta[c] = {"k": k, "program": c.split(":", 1)[1], "library": lib_of[c.split(":", 1)[1]]}
    return pd.concat(cols, axis=1), meta


def program_identity_map() -> dict:
    progs = pd.read_csv(TABLES / "11_k_scan_programs.tsv", sep="\t")
    out = {}
    for r in progs.itertuples():
        for m in r.members.split(","):
            out[f"K{r.k}:{m}"] = (r.identity, r.mp, r.n_libraries)
    return out


def cross_match(S_ds5: pd.DataFrame, S_50: pd.DataFrame, meta50: dict) -> pd.DataFrame:
    tops_a = {c: set(S_ds5[c].nlargest(N_TOP).index) for c in S_ds5.columns}
    tops_b = {c: set(S_50[c].nlargest(N_TOP).index) for c in S_50.columns}
    A = (S_ds5 - S_ds5.mean()) / S_ds5.std(ddof=0)
    B = (S_50 - S_50.mean()) / S_50.std(ddof=0)
    R = (A.T @ B) / len(A)
    M = len(S_ds5)
    rows = []
    for a in S_ds5.columns:
        for b in S_50.columns:
            k = len(tops_a[a] & tops_b[b])
            rows.append({"ds5_program": a, "pct50_program": b, "pearson_r": R.at[a, b],
                         "top50_overlap": k,
                         "overlap_p": hypergeom.sf(k - 1, M, N_TOP, N_TOP) if k else 1.0})
    df = pd.DataFrame(rows)
    df["overlap_q"] = multipletests(df["overlap_p"], method="fdr_bh")[1]
    df["k50"] = df["pct50_program"].map(lambda c: meta50[c]["k"])
    df["library50"] = df["pct50_program"].map(lambda c: meta50[c]["library"])
    return df


def plot_similarity(R, mp) -> None:
    order = [p for m in mp["members"] for p in m.split(",")]
    labels = []
    for m, members in zip(mp["mp"], mp["members"]):
        labels += [f"{m} {p.replace('DS5_', '')}" for p in members.split(",")]
    M = R.loc[order, order].to_numpy()
    n = len(order)
    fig, ax = plt.subplots(figsize=(0.2 * n + 3, 0.2 * n + 2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=5)
    ax.set_yticks(range(n), labels, fontsize=5)
    pos = 0
    for members in mp["members"]:
        k = len(members.split(","))
        ax.add_patch(plt.Rectangle((pos - 0.5, pos - 0.5), k, k, fill=False, lw=0.8, ec="black"))
        pos += k
    fig.colorbar(im, ax=ax, shrink=0.6, label="Pearson r of spectra scores")
    ax.set_title("cNMF programs of the five DS5 units, grouped into meta-programs")
    fig.tight_layout()
    save_figure(fig, "12_ds5_program_similarity")
    plt.close(fig)


def plot_identity_matrix(ident_by_unit: pd.DataFrame) -> None:
    mat = ident_by_unit.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(0.9 * mat.shape[1] + 6, 0.32 * mat.shape[0] + 1.8))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=50, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=6,
                        color="white" if v < 30 else "black")
    ax.set_xticks(range(mat.shape[1]), [UNIT_LABELS[u] for u in ident_by_unit.columns],
                  rotation=30, ha="right")
    ax.set_yticks(range(mat.shape[0]), ident_by_unit.index, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="top-50 genes shared with the best DS5 program")
    ax.set_title("50% epiboly program identities (stage 11) found in DS5 units\n"
                 f"(shown when >= {MATCH_MIN_OVERLAP} of the top 50 genes are shared, q < {MATCH_Q:g}; "
                 "blank = no match)", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "12_ds5_vs_50pct_identities")
    plt.close(fig)


def plot_usage_umap(adata, mp_rec, usages, title, stem, ncol=5) -> None:
    progs = list(mp_rec["mp"])
    nrow = int(np.ceil(len(progs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow), squeeze=False)
    for ax, r in zip(axes.ravel(), mp_rec.itertuples()):
        vals = pd.Series(np.nan, index=adata.obs_names)
        for m in r.members.split(","):
            unit = m.rsplit("_P", 1)[0]
            u = usages[unit][m]
            vals.loc[u.index] = u
        adata.obs["_u"] = vals
        missing = ", ".join(UNIT_LABELS[u] for u in UNITS if u not in r.units.split(","))
        sc.pl.umap(adata, color="_u", ax=ax, show=False, frameon=False, size=6, cmap="viridis",
                   vmin=0, vmax="p99", na_color="#e6e6e6", colorbar_loc=None,
                   title=f"{r.mp}: {r.label[:28]}" + (f"\n(absent: {missing})" if missing else ""))
    for ax in axes.ravel()[len(progs):]:
        ax.set_axis_off()
    adata.obs.drop(columns="_u", inplace=True)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    _, ks, k = factorize_units(args.force)
    print(f"[12] common K = {k}")
    ks.to_csv(TABLES / "12_k_selection.tsv", sep="\t", index=False, float_format="%.4f")
    units = list(UNITS)
    load = dict(libs=units, tag_fn=unit_tag, programs_dir=PROGRAMS_DIR, prefix="12_programs")

    # --- recurrence across K (counts only) and at the main K -------------------
    scan = []
    for kk in K_RANGE:
        S_k, _, lib_k = load_programs(kk, **load)
        pairs_k, _ = match_programs(S_k, lib_k, libs=units)
        mp_k = meta_programs(S_k, pairs_k, lib_k, min_libraries=MIN_UNITS)
        scan.append({"k": kk, "n_meta_programs": len(mp_k),
                     "n_recurrent_ge2": int((mp_k["n_libraries"] >= 2).sum()),
                     "n_recurrent_ge3": int((mp_k["n_libraries"] >= 3).sum())})
    scan = pd.DataFrame(scan)

    S, usages, lib_of = load_programs(k, **load)
    pairs, R = match_programs(S, lib_of, libs=units)
    mp = meta_programs(S, pairs, lib_of, min_libraries=MIN_UNITS)
    mp = mp.rename(columns={"n_libraries": "n_units", "libraries": "units",
                            "replicated": "recurrent"})
    mean_scores, sig, _ = signatures(S, mp.rename(columns={"n_units": "n_libraries"}),
                                     min_libraries=MIN_UNITS)

    print("[12] annotation")
    zfin_terms, zfin_names = zfin_annotations(window=ZFIN_WINDOW_DS5)
    go_terms, go_names = go_annotations()
    zf, zm = enrich(sig, list(S.index), zfin_terms, zfin_names)
    go, gm = enrich(sig, list(S.index), go_terms, go_names, max_genes=MAX_GO_GENES)
    lab = specific_labels(zf, zm, go, gm)
    mp = mp.merge(lab, left_on="mp", right_index=True, how="left")
    mp["label"] = [" | ".join(x for x in (z, f"GO: {g}" if isinstance(g, str) else "")
                              if isinstance(x, str) and x) or "unannotated"
                   for z, g in zip(mp["zfin_label"], mp["go_label"])]
    mp["top10"] = mp["mp"].map(lambda m: ",".join(sig[m][:10]))
    mp["frac_cells_usage_gt_0.2"] = [
        np.mean([float((usages[lib_of[m]][m] > 0.2).mean()) for m in r.members.split(",")])
        for r in mp.itertuples()]

    print("[12] matching to 50% epiboly programs (all K)")
    S50_k9, _, _ = load_programs(9)
    shared = sorted(set(S.index) & set(S50_k9.index))
    S_50, meta50 = load_50pct_all_k(shared)
    ident = program_identity_map()
    cm = cross_match(S.loc[shared], S_50, meta50)
    cm["identity"] = cm["pct50_program"].map(lambda c: ident.get(c, (None,))[0])
    cm["match"] = (cm["overlap_q"] < MATCH_Q) & (cm["top50_overlap"] >= MATCH_MIN_OVERLAP)
    best = (cm.sort_values(["top50_overlap", "pearson_r"], ascending=False)
            .groupby("ds5_program").head(1).set_index("ds5_program"))
    mp_of = {m: r.mp for r in mp.itertuples() for m in r.members.split(",")}
    best["ds5_mp"] = best.index.map(mp_of)
    best["ds5_unit"] = best.index.map(lib_of)

    ids = pd.read_csv(TABLES / "11_k_scan_identities.tsv", sep="\t").set_index("id")
    best["identity_label"] = best["identity"].map(ids["label"])
    best["identity_max_libraries_50"] = best["identity"].map(ids["max_libraries"])
    best["identity_replicated_50"] = best["identity"].map(ids["n_k_replicated"] > 0)

    # per DS5 MP: the 50% identity its members match
    def mp_match(r):
        b = best.loc[r.members.split(",")]
        b = b[b["match"]]
        if not len(b):
            return pd.Series({"match_50_identity": "", "match_50_label": "", "match_50_r": np.nan,
                              "match_50_overlap": np.nan, "match_50_libraries": np.nan,
                              "match_50_units": 0})
        top = b["identity"].value_counts().idxmax()
        return pd.Series({"match_50_identity": top,
                          "match_50_label": ids.loc[top, "label"],
                          "match_50_r": b.loc[b["identity"] == top, "pearson_r"].max(),
                          "match_50_overlap": b.loc[b["identity"] == top, "top50_overlap"].max(),
                          "match_50_libraries": ids.loc[top, "max_libraries"],
                          "match_50_units": int((b["identity"] == top).sum())})
    mp = pd.concat([mp, mp.apply(mp_match, axis=1)], axis=1)

    # identity x unit: top-50 overlap of DS5 programs whose BEST 50% partner belongs to the
    # identity. Counting every pair instead would let one broad DS5 program "match" each
    # 50% sub-split of the same broad program at high K.
    m = best[best["match"] & best["identity"].notna()].copy()
    m["unit"] = m["ds5_unit"]
    ident_by_unit = m.groupby(["identity", "unit"])["top50_overlap"].max().unstack().reindex(columns=units)
    ident_by_unit.index = [f"{i} {ids.loc[i, 'label'][:45]} [{int(ids.loc[i, 'max_libraries'])} lib]"
                           for i in ident_by_unit.index]
    order = sorted(ident_by_unit.index, key=lambda s: int(s.split()[0][2:]))
    ident_by_unit = ident_by_unit.loc[order]

    fmt = dict(sep="\t", index=False, float_format="%.4g")
    scan.to_csv(TABLES / "12_ds5_recurrence_by_k.tsv", **fmt)
    mp.to_csv(TABLES / "12_ds5_meta_programs.tsv", **fmt)
    zf.to_csv(TABLES / "12_ds5_zfin_enrichment.tsv", **fmt)
    go.to_csv(TABLES / "12_ds5_go_enrichment.tsv", **fmt)
    best.reset_index().to_csv(TABLES / "12_ds5_best_50pct_match.tsv", **fmt)
    cm[cm["match"]].to_csv(TABLES / "12_ds5_vs_50pct_matches.tsv", **fmt)
    ident_by_unit.to_csv(TABLES / "12_ds5_vs_50pct_identity_by_unit.tsv", sep="\t",
                         float_format="%.3f")

    plot_similarity(R, mp)
    plot_identity_matrix(ident_by_unit)
    ad = sc.read_h5ad(INPUT_H5AD)
    plot_usage_umap(ad, mp[mp["recurrent"]], usages,
                    f"Recurrent DS5 programs (K={k}): usage of each unit's member program",
                    "12_ds5_umap_recurrent")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(scan.to_string(index=False))
    print(mp[["mp", "n_units", "units", "label", "frac_cells_usage_gt_0.2", "match_50_identity",
              "match_50_label", "match_50_overlap", "match_50_r", "match_50_libraries",
              "match_50_units"]].to_string(index=False))
    print("[12] done")


if __name__ == "__main__":
    main()
