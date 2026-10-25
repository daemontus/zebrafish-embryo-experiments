#!/usr/bin/env python3
"""Stage 10: replicate, annotate and project the stage-09 gene programs.

Input : data/processed/09_programs/09_programs_<library>_k<K>_*.tsv, results/tables/09_selected_k.tsv
        data/processed/07_50pct_joint.h5ad, 08_50pct_corrected.h5ad, 07_ds5_joint.h5ad
        results/tables/08_50pct_population_labels.tsv
        data/reference/*  (scripts/download_references.sh)
Output: results/tables/10_*.tsv, results/figures/10_*.png/.pdf

1. Replication. Programs from different libraries are compared by Pearson r
   of their spectra scores over genes present in all four libraries, and by
   the overlap of their top-50 genes (hypergeometric, BH across all pairs).
   An edge links two programs when they are MUTUAL best matches by r AND
   their top-50 overlap has q < OVERLAP_Q. Connected components are meta-
   programs (MP); an MP is "replicated" if it spans >= MIN_LIBRARIES libraries.

2. Consensus signature. Genes are ranked by the mean spectra score over the
   MP's member programs; the top N_SIGNATURE genes form the signature. Core
   genes are those in the top-50 of >= MIN_LIBRARIES member programs.

3. Annotation, from databases only (no hand-picked markers):
   - ZFIN curated wild-type expression, mRNA in situ only, restricted to
     annotations whose stage range overlaps 4.66-6.0 hpf (30% epiboly to
     shield); anatomy terms are propagated to ancestors through ZFA is_a and
     part_of. Hypergeometric test, universe = shared genes with any in-window
     ZFIN annotation; terms need >= MIN_TERM_GENES universe genes.
   - GO biological process (GO Consortium zebrafish GAF, NOT-qualified
     annotations removed, propagated through is_a/part_of); universe = shared
     genes with any BP annotation; terms with MIN_TERM_GENES..MAX_GO_GENES
     universe genes.
   BH correction is applied across all (MP x term) tests of each source.
   All MPs are annotated (including single-library ones); summary columns use
   only "specific" terms (<= SPECIFIC_TERM_FRACTION of the universe), so broad
   ontology parents such as "embryonic structure" do not crowd them out.

4. Context. Per library: Spearman correlation of program usage with depth
   (log10 UMIs) and ribosomal fraction (technical association), and mean
   usage per stage-07/08 population plus the AUROC of usage for the most
   associated population (discrete vs graded activity).

5. K sensitivity. For each replicated MP and library, the best Pearson r
   between the MP's member program and any program of the same library at
   the sensitivity Ks (stage 09) is reported.

6. Projection. Signature scores (scanpy score_genes, expression-matched
   control genes) are computed in the 50% epiboly object and in the DS5
   merged object (3.3-6.0 hpf), to follow each replicated program over time
   within DS5. Absolute scores are not compared between DS5 and DS2-DS4.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from pipeline_utils import PROCESSED, TABLES, ensure_dirs, save_figure
from program_tools import (
    LIBS,
    MAX_GO_GENES,
    ZFIN_WINDOW_HPF,
    enrich,
    go_annotations,
    load_programs,
    match_programs,
    meta_programs,
    missing_library_matches,
    signatures,
    tag,
    zfin_annotations,
)

DS5_ORDER = ["high DS5", "high DS5b", "oblong DS5", "oblong DS5b", "dome DS5",
             "30% DS5", "30% DS5b", "shield DS5"]

ENRICH_Q = 0.05
SPECIFIC_TERM_FRACTION = 0.05
RANDOM_STATE = 0

plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


# --- 1. load and match programs -----------------------------------------------

def selected_k() -> tuple[int, list[int]]:
    ks = pd.read_csv(TABLES / "09_selected_k.tsv", sep="\t")
    return (int(ks.loc[ks["role"] == "main", "k"].iloc[0]),
            ks.loc[ks["role"] == "sensitivity", "k"].astype(int).tolist())


# --- 3. annotation sources --------------------------------------------------------


# --- 4. context -------------------------------------------------------------------

def program_context(mp, usages, obs) -> tuple[pd.DataFrame, pd.DataFrame]:
    tech, pops = [], []
    for row in mp[mp["replicated"]].itertuples():
        for prog in row.members.split(","):
            lib = next(l for l in LIBS if prog.startswith(tag(l) + "_"))
            u = usages[lib][prog]
            o = obs.loc[u.index]
            tech.append({
                "mp": row.mp, "program": prog, "library": lib,
                "rho_log_umis": spearmanr(u, np.log10(o["total_counts"]))[0],
                "rho_pct_ribo": spearmanr(u, o["pct_counts_ribo"])[0],
                "frac_cells_usage_gt_0.2": float((u > 0.2).mean()),
            })
            means = u.groupby(o["population"], observed=True).mean()
            best = means.idxmax()
            auc = roc_auc_score(o["population"] == best, u)
            for p, v in means.items():
                pops.append({"mp": row.mp, "program": prog, "library": lib,
                             "population": p, "mean_usage": v,
                             "is_top": p == best, "auroc_top": auc if p == best else np.nan})
    return pd.DataFrame(tech), pd.DataFrame(pops)


def k_sensitivity(S, mp, sens_k) -> pd.DataFrame:
    rows = []
    for k in sens_k:
        alt, _, _ = load_programs(k)
        alt = alt.reindex(S.index).fillna(0)
        for row in mp[mp["replicated"]].itertuples():
            for prog in row.members.split(","):
                lib = next(l for l in LIBS if prog.startswith(tag(l) + "_"))
                cands = [c for c in alt.columns if c.startswith(tag(lib) + "_")]
                r = [pearsonr(S[prog], alt[c])[0] for c in cands]
                rows.append({"mp": row.mp, "program": prog, "library": lib, "k": k,
                             "best_match": cands[int(np.argmax(r))], "best_r": max(r)})
    return pd.DataFrame(rows)


# --- figures ------------------------------------------------------------------------

def plot_similarity(R, mp, lib_of) -> None:
    order = [p for m in mp["members"] for p in m.split(",")]
    labels = []
    for m, members in zip(mp["mp"], mp["members"]):
        labels += [f"{m} {p.replace('ZF50_', '')}" for p in members.split(",")]
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
    ax.set_title("cNMF programs of the four 50% epiboly libraries, grouped into meta-programs")
    fig.tight_layout()
    save_figure(fig, "10_program_similarity")
    plt.close(fig)


def plot_population_usage(pops, mp_rep) -> None:
    mat = (pops.groupby(["mp", "population"])["mean_usage"].mean().unstack()
           .reindex(index=mp_rep))
    cols = [c for c in ["EVL", "p53", "endoderm", "margin", "prechordal", "organizer",
                        "bulk", "low_quality", "heat_shock"] if c in mat.columns]
    mat = mat[cols]
    fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 3, 0.35 * len(mat) + 1.5))
    im = ax.imshow(mat.to_numpy(), cmap="magma_r", aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(mat)), mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if mat.iat[i, j] > mat.to_numpy().max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean usage (avg. over libraries)")
    ax.set_title("Replicated programs: usage by stage-07 population")
    fig.tight_layout()
    save_figure(fig, "10_program_population_usage")
    plt.close(fig)


def plot_umaps(adata, key_umap, programs, title, stem, ncol=5) -> None:
    adata.obsm["X_umap"] = adata.obsm[key_umap]
    nrow = int(np.ceil(len(programs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow), squeeze=False)
    for ax, p in zip(axes.ravel(), programs):
        sc.pl.umap(adata, color=f"score_{p}", ax=ax, show=False, frameon=False, size=6,
                   cmap="viridis", title=p, vmin="p1", vmax="p99", colorbar_loc=None)
    for ax in axes.ravel()[len(programs):]:
        ax.set_axis_off()
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def plot_ds5_trend(scores: pd.DataFrame, programs) -> None:
    mat = scores.groupby("library_short", observed=True)[[f"score_{p}" for p in programs]].mean()
    mat = mat.reindex([l for l in DS5_ORDER if l in mat.index])
    z = (mat - mat.mean()) / mat.std(ddof=0)
    fig, ax = plt.subplots(figsize=(0.5 * len(programs) + 3, 0.4 * len(mat) + 1.8))
    im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(programs)), programs, rotation=45, ha="right")
    ax.set_yticks(range(len(mat)), mat.index)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean signature score, z-scored across libraries")
    ax.set_title("Replicated 50%-epiboly programs projected onto DS5 libraries (3.3–6.0 hpf)")
    fig.tight_layout()
    save_figure(fig, "10_ds5_program_trend")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    k_main, sens_k = selected_k()
    S, usages, lib_of = load_programs(k_main)
    print(f"[10] K={k_main}: {S.shape[1]} programs over {S.shape[0]} shared genes")
    pairs, R = match_programs(S, lib_of)
    mp = meta_programs(S, pairs, lib_of)
    mean_scores, sig, core = signatures(S, mp)
    mp_rep = list(mp.loc[mp["replicated"], "mp"])
    mp["signature_top20"] = mp["mp"].map(lambda m: ",".join(sig.get(m, [])[:20]))
    mp["core_genes"] = mp["mp"].map(lambda m: ",".join(core.get(m, [])))
    mp["n_core_genes"] = mp["mp"].map(lambda m: len(core.get(m, [])))

    print("[10] annotation: ZFIN anatomy and GO BP")
    zfin_terms, zfin_names = zfin_annotations()
    go_terms, go_names = go_annotations()
    universe = list(S.index)
    zfin, zfin_m = enrich(sig, universe, zfin_terms, zfin_names)
    go, go_m = enrich(sig, universe, go_terms, go_names, max_genes=MAX_GO_GENES)
    mp["best_partner_in_missing_library"] = mp["mp"].map(
        missing_library_matches(mp, pairs, lib_of)).fillna("")
    for name, df, m_univ in (("zfin", zfin, zfin_m), ("go", go, go_m)):
        specific = df[(df["q"] < ENRICH_Q) & (df["term_size"] <= SPECIFIC_TERM_FRACTION * m_univ)]
        best = (specific.sort_values("p").groupby("mp").head(3)
                .groupby("mp")["name"].apply(lambda s: "; ".join(s)))
        mp[f"top_{name}_terms"] = mp["mp"].map(best).fillna("")
        sig_n = df[df["q"] < ENRICH_Q].groupby("mp").size()
        mp[f"n_{name}_terms_q05"] = mp["mp"].map(sig_n).fillna(0).astype(int)
    mp["zfin_annotated_in_signature"] = mp["mp"].map(
        lambda m: sum(g in zfin_terms for g in sig.get(m, [])) if m in sig else np.nan)

    print("[10] context: technical association, populations")
    a50 = sc.read_h5ad(PROCESSED / "08_50pct_corrected.h5ad")
    obs = a50.obs
    tech, pops = program_context(mp, usages, obs)
    tsum = tech.groupby("mp")[["rho_log_umis", "rho_pct_ribo", "frac_cells_usage_gt_0.2"]].mean()
    top_pop = pops[pops["is_top"]].groupby("mp").agg(
        top_population=("population", lambda s: s.value_counts().idxmax()),
        top_population_agreement=("population", lambda s: s.value_counts().max() / len(s)),
        auroc_top=("auroc_top", "mean"))
    mp = mp.merge(tsum, left_on="mp", right_index=True, how="left")
    mp = mp.merge(top_pop, left_on="mp", right_index=True, how="left")

    sens = k_sensitivity(S, mp, sens_k)
    ssum = sens.groupby(["mp", "k"])["best_r"].min().unstack()
    ssum.columns = [f"min_r_at_k{c}" for c in ssum.columns]
    mp = mp.merge(ssum, left_on="mp", right_index=True, how="left")

    print("[10] projection: signature scores")
    for p in mp_rep:
        sc.tl.score_genes(a50, sig[p], score_name=f"score_{p}", random_state=RANDOM_STATE)
    ds5 = sc.read_h5ad(PROCESSED / "07_ds5_joint.h5ad")
    for p in mp_rep:
        genes = [g for g in sig[p] if g in ds5.var_names]
        sc.tl.score_genes(ds5, genes, score_name=f"score_{p}", random_state=RANDOM_STATE)
    ds5_scores = ds5.obs[["library_short"] + [f"score_{p}" for p in mp_rep]]

    # --- write tables ---
    fmt = dict(sep="\t", index=False, float_format="%.4g")
    pairs.to_csv(TABLES / "10_program_pairs.tsv", **fmt)
    mp.to_csv(TABLES / "10_meta_programs.tsv", **fmt)
    mean_scores.to_csv(TABLES / "10_meta_program_gene_scores.tsv", sep="\t", float_format="%.4f")
    zfin.to_csv(TABLES / "10_zfin_anatomy_enrichment.tsv", **fmt)
    go.to_csv(TABLES / "10_go_bp_enrichment.tsv", **fmt)
    tech.to_csv(TABLES / "10_program_technical.tsv", **fmt)
    sens.to_csv(TABLES / "10_k_sensitivity.tsv", **fmt)
    pops.to_csv(TABLES / "10_program_populations.tsv", **fmt)
    (ds5_scores.groupby("library_short", observed=True).mean()
     .reindex([l for l in DS5_ORDER]).to_csv(TABLES / "10_ds5_program_scores.tsv", sep="\t",
                                             float_format="%.4f"))
    with open(TABLES / "10_annotation_universe.txt", "w") as fh:
        fh.write(f"shared genes: {len(universe)}\nZFIN universe (in situ, "
                 f"{ZFIN_WINDOW_HPF[0]}-{ZFIN_WINDOW_HPF[1]} hpf): {zfin_m}\n"
                 f"GO BP universe: {go_m}\n")

    # --- figures ---
    plot_similarity(R, mp, lib_of)
    plot_population_usage(pops, mp_rep)
    plot_umaps(a50, "X_umap_harmony", mp_rep,
               "Replicated programs on the 50% epiboly Harmony UMAP (signature scores)",
               "10_50pct_umap_programs")
    plot_umaps(ds5, "X_umap", mp_rep,
               "Replicated programs projected onto DS5 (3.3–6.0 hpf; signature scores)",
               "10_ds5_umap_programs")
    plot_ds5_trend(ds5_scores, mp_rep)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(mp.drop(columns=["members", "signature_top20", "core_genes"]).to_string(index=False))
    print("[10] done")


if __name__ == "__main__":
    main()
