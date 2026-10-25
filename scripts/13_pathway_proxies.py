#!/usr/bin/env python3
"""Stage 13: transcriptional proxies of signalling-pathway activity across libraries.

Input : scripts/pathway_proxies.tsv            (resource gene lists, transcribed verbatim)
        data/reference/zfin_gene.txt, zfin_aliases.txt, zfin.gaf.gz, go-basic.obo,
        zfin_wildtype_expression.txt, zfa.obo  (scripts/download_references.sh)
        data/processed/dropseq_3.3-6.0hpf_qc.h5ad, data/processed/libraries/*.h5ad
        results/tables/09_programs_*_k9_*, 12_programs_*_k13_*  (program usage/spectra)
        results/tables/11_k_scan_programs.tsv, 11_k_scan_identities.tsv,
        12_ds5_best_50pct_match.tsv                             (program identities)
Output: results/tables/13_*.tsv, results/figures/13_*.png/.pdf

RNA does not show signalling directly; the readout tier (direct feedback targets)
is used as a proxy, and every step is designed so the proxy can fail.

0. Gene sets (fixed before any scoring). Resource symbols are resolved to ZFIN genes
   and matrix names (scripts/pathway_tools.py). Genes are used if detected (>= 1 UMI)
   in >= 1% of cells in >= 2 of the 12 libraries. Per pathway:
     readout     resource "readout" tier                         (primary activity proxy)
     secondary   resource "secondary" tier minus ligands/receptors (lineage-confounded;
                 sensitivity only)
     sender      ligands (GO MF receptor ligand activity) from the non-readout tiers
     receiver    receptors (GO MF signaling receptor activity) from the non-readout tiers
     antagonist  BMP secreted antagonists
   Sets with < 2 usable genes are "not scorable".

1. Null. For each set, 1,000 random gene sets are drawn gene-by-gene from genes in the
   same expression bin (10 mean x 5 detection quantile bins over all 10,101 cells),
   excluding all pathway genes. The same random sets are used in every library.

2. Proxy validation.
   a. Coherence (data): within each library, expression is kNN-smoothed on the stage-06
      graph (connectivities + self, row-normalised) and z-scored per gene; the mean
      pairwise Pearson r of the set's genes is compared with the 1,000 random sets
      (one-sided empirical p). A pathway's readout passes if p < 0.05 in at least half of
      the libraries with >= MIN_CELLS_COHERENCE cells.
   b. Spatial agreement (ZFIN): mean pairwise Jaccard similarity of the readout genes'
      ZFIN in-situ anatomy terms at 3.33-6.0 hpf (propagated through ZFA; terms carried
      by > 20% of annotated genes removed) vs 10,000 random annotated gene sets.
   Hh and VEGF are expected to fail (resource: Hh inactive before ~6 hpf, VEGF from
   ~8 hpf) and act as negative controls.

3. Cell scores (within library only): mean smoothed z of the set's genes; the null
   scores of the random sets give a per-cell z (score vs its own matched null).

4. Program association. Within each unit (50% library; DS5 unit as in stage 12) the
   Spearman correlation between a set score and each program's usage is compared with
   the correlations of the 1,000 random-set scores: z, two-sided normal p, BH per tier.
   Circularity: readout genes among a program's top-50 spectra genes are listed, and the
   test is repeated without them ("leave-out z"; a single remaining gene is tested
   against single-gene matched nulls).
   Replication (50%): an identity x pathway link is "high"/"low" if it has the same sign
   and q < 0.05 in >= 3 libraries and the opposite sign in none. For readout links with
   circularity, the call must also replicate with the leave-out test; otherwise it is
   "not separable from program genes".
   50% epiboly programs are tested at K = 9 (main) and K = 13 (stage 11: the K at which
   the proliferation and dorsal-organizer programs replicate).

4b. Calibration: N_CAL random "pseudo-pathways" (4 genes each, drawn from the null pool)
   are run through steps 1-4 unchanged; their coherence pass rate and replicated-call rate
   estimate the false-positive rates of the procedure.

5. DS5 time course: per library pseudobulk log1p(CP10k) of the readout genes minus the
   same for the matched random sets (z vs the null). Comparable within DS5 (one batch);
   50% libraries are shown for reference only.

6. BMP antagonist test: across programs within a unit, Spearman between the BMP readout
   z and the antagonist z (prediction: negative). Each antagonist gene is also tested on
   its own (single-gene set vs matched single-gene nulls).

7. State table: per replicated identity and pathway, "high" / "low" (relative to other
   cells of the same library) / "no consistent difference" / "unknown: <reason>".
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import norm, rankdata
from statsmodels.stats.multitest import multipletests

from pathway_tools import resolve
from pipeline_utils import PROCESSED, TABLES, ensure_dirs, library_of, save_figure
from program_tools import zfin_annotations

LIBRARY_DIR = PROCESSED / "libraries"
PROGRAMS_ALL_K = PROCESSED / "09_programs"
LIBS = {  # short name -> stage-06 library
    "high DS5": "ZFHIGH_WT_DS5", "high DS5b": "ZFHIGH_WT_DS5b",
    "oblong DS5": "ZFOBLONG_WT_DS5", "oblong DS5b": "ZFOBLONG_WT_DS5b",
    "dome DS5": "ZFDOME_WT_DS5", "30% DS5": "ZF30_WT_DS5", "30% DS5b": "ZF30_WT_DS5b",
    "shield DS5": "ZFS_WT_DS5",
    "50% DS2": "ZF50_WT_DS2", "50% DS3": "ZF50_WT_DS3", "50% DS4": "ZF50_WT_DS4",
    "50% DS4b": "ZF50_WT_DS4b",
}
DS5_LIBS = [l for l in LIBS if "DS5" in l]
UNITS = {  # program unit -> (context, libraries, usage/spectra stem)
    "50% DS2": ("50pct", ["50% DS2"], "09_programs_ZF50_DS2_k9"),
    "50% DS3": ("50pct", ["50% DS3"], "09_programs_ZF50_DS3_k9"),
    "50% DS4": ("50pct", ["50% DS4"], "09_programs_ZF50_DS4_k9"),
    "50% DS4b": ("50pct", ["50% DS4b"], "09_programs_ZF50_DS4b_k9"),
    "50% DS2 K13": ("50pct_k13", ["50% DS2"], "09_programs_ZF50_DS2_k13"),
    "50% DS3 K13": ("50pct_k13", ["50% DS3"], "09_programs_ZF50_DS3_k13"),
    "50% DS4 K13": ("50pct_k13", ["50% DS4"], "09_programs_ZF50_DS4_k13"),
    "50% DS4b K13": ("50pct_k13", ["50% DS4b"], "09_programs_ZF50_DS4b_k13"),
    "DS5_highoblong": ("ds5", ["high DS5", "high DS5b", "oblong DS5", "oblong DS5b"],
                       "12_programs_DS5_highoblong_k13"),
    "DS5_dome": ("ds5", ["dome DS5"], "12_programs_DS5_dome_k13"),
    "DS5_30a": ("ds5", ["30% DS5"], "12_programs_DS5_30a_k13"),
    "DS5_30b": ("ds5", ["30% DS5b"], "12_programs_DS5_30b_k13"),
    "DS5_shield": ("ds5", ["shield DS5"], "12_programs_DS5_shield_k13"),
}
PATHWAYS = ["Nodal", "BMP", "Wnt", "FGF", "Hh", "Notch", "VEGF"]
TIERS = ["readout", "secondary", "sender", "receiver", "antagonist"]
MIN_DETECT = 0.01
MIN_DETECT_LIBS = 2
MIN_SET_GENES = 2
MIN_LEAVE_OUT_GENES = 1
N_CAL = 50
CAL_SET_SIZE = 4
N_NULL = 1000
N_NULL_ZFIN = 10000
MIN_CELLS_COHERENCE = 200
COHERENCE_P = 0.05
ZFIN_WINDOW = (3.33, 6.0)
ZFIN_GENERIC_FRACTION = 0.2
Q = 0.05
MIN_REPLICATE_LIBS = 3
N_TOP = 50
SEED = 0

plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


# --- 0. gene sets -----------------------------------------------------------------

def detection_by_library(qc) -> pd.DataFrame:
    X = qc.X.tocsc() if sp.issparse(qc.X) else sp.csc_matrix(qc.X)
    lib = np.asarray(library_of(qc.obs_names))
    det = {}
    for short, name in LIBS.items():
        m = lib == name
        det[short] = np.asarray((X[m] > 0).mean(axis=0)).ravel()
    return pd.DataFrame(det, index=qc.var_names)


def build_gene_sets(res: pd.DataFrame, det: pd.DataFrame):
    res = res.copy()
    ok = res["matrix_gene"].astype(bool)
    res["n_libs_detected"] = 0
    res.loc[ok, "n_libs_detected"] = [
        int((det.loc[g] >= MIN_DETECT).sum()) for g in res.loc[ok, "matrix_gene"]]
    res["max_detection"] = np.nan
    res.loc[ok, "max_detection"] = [float(det.loc[g].max()) for g in res.loc[ok, "matrix_gene"]]
    res["used"] = ok & (res["n_libs_detected"] >= MIN_DETECT_LIBS)

    sets, rows = {}, []
    for pw in PATHWAYS:
        d = res[(res["pathway"] == pw) & res["used"]]
        readout = sorted(set(d.loc[d["tier"] == "readout", "matrix_gene"]))
        nonread = d[(d["tier"] != "readout") & (d["tier"] != "antagonist")
                    & ~d["matrix_gene"].isin(readout)]
        role = nonread["role"].fillna("")
        tiers = {
            "readout": readout,
            "secondary": sorted(set(nonread.loc[(nonread["tier"] == "secondary")
                                                & (role == ""), "matrix_gene"])),
            "sender": sorted(set(nonread.loc[role.str.contains("ligand"), "matrix_gene"])),
            "receiver": sorted(set(nonread.loc[role.str.contains("receptor"), "matrix_gene"])),
            "antagonist": sorted(set(d.loc[d["tier"] == "antagonist", "matrix_gene"])),
        }
        for t, genes in tiers.items():
            if t == "antagonist" and pw != "BMP":
                continue
            scorable = len(genes) >= MIN_SET_GENES
            rows.append({"pathway": pw, "tier": t, "n_genes": len(genes),
                         "genes": ",".join(genes), "scorable": scorable})
            if scorable:
                sets[(pw, t)] = genes
    return res, sets, pd.DataFrame(rows)


# --- 1. matched null -----------------------------------------------------------------

def gene_bins(qc) -> pd.DataFrame:
    a = qc.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    mean = np.asarray(a.X.mean(axis=0)).ravel()
    det = np.asarray((qc.X > 0).mean(axis=0)).ravel()
    st = pd.DataFrame({"mean": mean, "det": det}, index=qc.var_names)
    st = st[st["det"] >= MIN_DETECT].copy()
    st["mbin"] = pd.qcut(st["mean"].rank(method="first"), 10, labels=False)
    st["dbin"] = pd.qcut(st["det"].rank(method="first"), 5, labels=False)
    st["bin"] = st["mbin"] * 5 + st["dbin"]
    return st


def null_sets(sets, bins, exclude, rng) -> dict:
    pool = bins[~bins.index.isin(exclude)]
    by_bin = {b: g.index.to_numpy() for b, g in pool.groupby("bin")}
    by_mbin = {b: g.index.to_numpy() for b, g in pool.groupby("mbin")}
    out = {}
    for key, genes in sets.items():
        cols = []
        for g in genes:
            if g in bins.index:
                cand = by_bin.get(bins.at[g, "bin"], np.array([]))
                if len(cand) < 20:
                    cand = by_mbin[bins.at[g, "mbin"]]
            else:  # below the global detection floor: use the lowest mean bin
                cand = by_mbin[0]
            cols.append(rng.choice(cand, size=N_NULL, replace=True))
        out[key] = np.stack(cols, axis=1)  # N_NULL x n_genes
    return out


# --- 2-3. per-library smoothed z, coherence, cell scores -------------------------------

def smoothed_z(lib_short: str, genes_needed: set):
    ad = sc.read_h5ad(LIBRARY_DIR / f"{LIBS[lib_short]}.h5ad")
    keep = [g for g in ad.var_names if g in genes_needed]
    W = ad.obsp["connectivities"] + sp.identity(ad.n_obs, format="csr")
    W = sp.diags(1 / np.asarray(W.sum(axis=1)).ravel()) @ W
    X = ad[:, keep].X
    Xs = np.asarray((W @ X).todense() if sp.issparse(X) else W @ X, dtype=np.float32)
    sd = Xs.std(axis=0)
    good = sd > 0
    Z = (Xs[:, good] - Xs[:, good].mean(axis=0)) / sd[good]
    cols = {g: i for i, g in enumerate(np.array(keep)[good])}
    raw = ad[:, keep].X
    raw = np.asarray(raw.mean(axis=0)).ravel()  # pseudobulk mean log1p(CP10k)
    pseudo = pd.Series(raw, index=keep)
    return Z, cols, list(ad.obs_names), pseudo


def gather(Z, cols, names: np.ndarray):
    """names: (..., n) gene names -> (cells, ..., n) values, NaN where missing."""
    idx = np.vectorize(lambda g: cols.get(g, -1))(names)
    out = Z[:, np.where(idx >= 0, idx, 0)]
    out[:, idx < 0] = np.nan
    return out


def mean_pairwise_r(M):
    """M: cells x ... x n (z-scored columns, NaN = missing gene) -> mean off-diagonal r."""
    n_cells = M.shape[0]
    present = ~np.isnan(M[0])
    Mz = np.nan_to_num(M)
    C = np.einsum("c...i,c...j->...ij", Mz, Mz) / n_cells
    n = present.sum(axis=-1)
    tot = C.sum(axis=(-1, -2)) - np.trace(C, axis1=-2, axis2=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= 2, tot / (n * (n - 1)), np.nan)


def set_scores(Z, cols, genes, null):
    obs = gather(Z, cols, np.array(genes))            # cells x n
    nul = gather(Z, cols, null)                       # cells x N x n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s = np.nanmean(obs, axis=-1)
        N = np.nanmean(nul, axis=-1)
    return obs, nul, s, N


# --- 2b. ZFIN spatial agreement -------------------------------------------------------

def zfin_agreement(sets, universe, rng) -> pd.DataFrame:
    terms, names = zfin_annotations(window=ZFIN_WINDOW)
    annotated = [g for g in universe if g in terms and terms[g]]
    freq = pd.Series([t for g in annotated for t in terms[g]]).value_counts()
    generic = set(freq[freq > ZFIN_GENERIC_FRACTION * len(annotated)].index)
    T = {g: terms[g] - generic for g in annotated}
    T = {g: t for g, t in T.items() if t}
    pool = np.array(list(T))

    def mean_jaccard(gs):
        js = [len(T[a] & T[b]) / len(T[a] | T[b]) for i, a in enumerate(gs) for b in gs[i + 1:]]
        return float(np.mean(js))

    rows = []
    for (pw, tier), genes in sets.items():
        if tier != "readout":
            continue
        gs = [g for g in genes if g in T]
        if len(gs) < 2:
            rows.append({"pathway": pw, "n_annotated": len(gs), "mean_jaccard": np.nan,
                         "null_mean": np.nan, "p": np.nan, "shared_terms": ""})
            continue
        obs = mean_jaccard(gs)
        null = np.array([mean_jaccard(list(rng.choice(pool, len(gs), replace=False)))
                         for _ in range(N_NULL_ZFIN)])
        shared = pd.Series([t for g in gs for t in T[g]]).value_counts()
        shared = shared[shared >= 2].head(5)
        rows.append({"pathway": pw, "n_annotated": len(gs), "mean_jaccard": obs,
                     "null_mean": null.mean(), "p": (1 + (null >= obs).sum()) / (1 + len(null)),
                     "shared_terms": "; ".join(f"{names.get(t, t)} ({n})" for t, n in shared.items())})
    return pd.DataFrame(rows)


# --- 4. program association ------------------------------------------------------------

def rank_cols(A):
    R = rankdata(A, axis=0)
    R = (R - R.mean(axis=0)) / R.std(axis=0)
    return np.nan_to_num(R)


def program_identities() -> dict:
    """(context, program) -> identity (stage 11 for 50%; stage 12 best match for DS5)."""
    out = {}
    p11 = pd.read_csv(TABLES / "11_k_scan_programs.tsv", sep="\t")
    for k, ctx in ((9, "50pct"), (13, "50pct_k13")):
        for r in p11[p11["k"] == k].itertuples():
            for m in r.members.split(","):
                out[(ctx, m)] = r.identity
    b12 = pd.read_csv(TABLES / "12_ds5_best_50pct_match.tsv", sep="\t")
    for r in b12.itertuples():
        out[("ds5", r.ds5_program)] = r.identity if r.match else f"DS5:{r.ds5_mp}"
    return out


def associate(unit, scores, sets, ident) -> pd.DataFrame:
    ctx, libs, stem = UNITS[unit]
    src = TABLES if (TABLES / f"{stem}_usage.tsv").exists() else PROGRAMS_ALL_K
    usage = pd.read_csv(src / f"{stem}_usage.tsv", sep="\t", index_col=0)
    spectra = pd.read_csv(src / f"{stem}_spectra_score.tsv", sep="\t", index_col=0)
    top = {p: set(spectra[p].nlargest(N_TOP).index) for p in spectra.columns}
    rows = []
    for key, genes in sets.items():
        cells, s, N, obs, nul = [], [], [], [], []
        for lib in libs:
            d = scores[lib][key]
            cells += d["cells"]
            s.append(d["s"])
            N.append(d["N"])
            obs.append(d["obs"])
            nul.append(d["nul"])
        s, N = np.concatenate(s), np.concatenate(N)
        obs, nul = np.concatenate(obs), np.concatenate(nul)
        pos = pd.Index(cells).get_indexer(usage.index)
        keep = pos >= 0
        U = rank_cols(usage.to_numpy()[keep])
        pos = pos[keep]
        rs, rN = rank_cols(s[pos][:, None])[:, 0], rank_cols(N[pos])
        n = len(pos)
        rho = U.T @ rs / n
        rho_null = U.T @ rN / n
        mu, sd = rho_null.mean(axis=1), rho_null.std(axis=1)
        z = (rho - mu) / sd
        for j, prog in enumerate(usage.columns):
            circ = [g for g in genes if g in top[prog]]
            z_lo = np.nan
            if circ and key[1] == "readout":
                keepg = [i for i, g in enumerate(genes) if g not in circ]
                if len(keepg) >= MIN_LEAVE_OUT_GENES:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        s2 = np.nanmean(obs[pos][:, keepg], axis=-1)
                        N2 = np.nanmean(nul[pos][:, :, keepg], axis=-1)
                    r2 = U[:, j] @ rank_cols(s2[:, None])[:, 0] / n
                    r2n = U[:, j] @ rank_cols(N2) / n
                    z_lo = (r2 - r2n.mean()) / r2n.std()
            rows.append({"pathway": key[0], "tier": key[1], "context": ctx, "unit": unit,
                         "program": prog, "identity": ident.get((ctx, prog), ""),
                         "rho": rho[j], "null_mean": mu[j], "z": z[j],
                         "p": 2 * norm.sf(abs(z[j])),
                         "p_empirical": (1 + (np.abs(rho_null[j] - mu[j]) >= abs(rho[j] - mu[j])).sum())
                         / (1 + rho_null.shape[1]),
                         "readout_genes_in_program_top50": ",".join(circ), "z_leave_out": z_lo,
                         "p_leave_out": 2 * norm.sf(abs(z_lo)) if not np.isnan(z_lo) else np.nan})
    return pd.DataFrame(rows)


def replicated_call(pos, neg):
    return ("high" if pos >= MIN_REPLICATE_LIBS and neg == 0 else
            "low" if neg >= MIN_REPLICATE_LIBS and pos == 0 else "no consistent difference")


def bh(assoc):
    assoc = assoc.copy()
    assoc["q"] = np.nan
    assoc["q_leave_out"] = np.nan
    for tier, d in assoc.groupby("tier"):
        assoc.loc[d.index, "q"] = multipletests(d["p"], method="fdr_bh")[1]
        lo = d[d["p_leave_out"].notna()]
        if len(lo):
            assoc.loc[lo.index, "q_leave_out"] = multipletests(lo["p_leave_out"], method="fdr_bh")[1]
    return assoc


def score_libraries(sets, nulls, needed):
    """Per library: cell scores for each set (+ null), coherence rows, pseudobulk."""
    coh_rows, scores, pseudo = [], {}, {}
    for lib in LIBS:
        Z, cols, cells, pb = smoothed_z(lib, needed)
        pseudo[lib] = pb
        scores[lib] = {}
        for key, genes in sets.items():
            obs, nul, s, N = set_scores(Z, cols, genes, nulls[key])
            scores[lib][key] = {"cells": cells, "s": s, "N": N, "obs": obs, "nul": nul}
            if len(genes) < 2:
                continue
            r_obs = float(mean_pairwise_r(obs))
            r_null = mean_pairwise_r(nul)
            valid = ~np.isnan(r_null)
            coh_rows.append({
                "pathway": key[0], "tier": key[1], "library": lib, "n_cells": len(cells),
                "n_genes_present": int((~np.isnan(obs[0])).sum()), "mean_r": r_obs,
                "null_mean_r": float(np.nanmean(r_null)),
                "p": (1 + (r_null[valid] >= r_obs).sum()) / (1 + valid.sum())
                if not np.isnan(r_obs) else np.nan})
    return pd.DataFrame(coh_rows), scores, pseudo


def calibrate(scores_libs, bins, exclude, needed_fn, ident, ids, rng):
    pool = bins[~bins.index.isin(exclude)].index.to_numpy()
    cal_sets = {("CAL", f"cal{i:02d}"): list(rng.choice(pool, CAL_SET_SIZE, replace=False))
                for i in range(N_CAL)}
    cal_nulls = null_sets(cal_sets, bins, exclude | {g for v in cal_sets.values() for g in v}, rng)
    needed = {g for v in cal_sets.values() for g in v} | {g for n in cal_nulls.values() for g in n.ravel()}
    coh, scores, _ = score_libraries(cal_sets, cal_nulls, needed)
    units = [u for u, (ctx, _, _) in UNITS.items() if ctx == "50pct"]
    assoc = pd.concat([associate(u, scores, cal_sets, ident) for u in units], ignore_index=True)
    assoc["p_leave_out"] = np.nan
    assoc = bh(assoc)
    calls = summarise_50pct(assoc, ids, "50pct")
    return coh, calls


def per_gene_association(genes, bins, exclude, ident, rng):
    if not genes:
        return pd.DataFrame()
    gsets = {("BMP", f"antagonist:{g}"): [g] for g in genes}
    gnulls = null_sets(gsets, bins, exclude, rng)
    needed = set(genes) | {g for n in gnulls.values() for g in n.ravel()}
    _, scores, _ = score_libraries(gsets, gnulls, needed)
    units = [u for u, (ctx, _, _) in UNITS.items() if ctx in ("50pct", "ds5")]
    assoc = pd.concat([associate(u, scores, gsets, ident) for u in units], ignore_index=True)
    assoc["p_leave_out"] = np.nan
    return bh(assoc)


def summarise_50pct(assoc, ids, context="50pct") -> pd.DataFrame:
    a = assoc[(assoc["context"] == context) & assoc["identity"].fillna("").astype(bool)]
    rows = []
    for (pw, tier, ident), d in a.groupby(["pathway", "tier", "identity"]):
        best = d.loc[d.groupby("unit")["q"].idxmin()]
        pos = int(((best["q"] < Q) & (best["z"] > 0)).sum())
        neg = int(((best["q"] < Q) & (best["z"] < 0)).sum())
        call = replicated_call(pos, neg)
        circ = best["readout_genes_in_program_top50"].fillna("").astype(bool)
        # leave-out evidence: circular libraries use the leave-out test, others the main one
        z_eff = np.where(circ, best["z_leave_out"], best["z"])
        q_eff = np.where(circ, best["q_leave_out"], best["q"])
        pos_lo = int(((q_eff < Q) & (z_eff > 0)).sum())
        neg_lo = int(((q_eff < Q) & (z_eff < 0)).sum())
        call_lo = replicated_call(pos_lo, neg_lo)
        final = call
        if tier == "readout" and circ.any() and call != "no consistent difference":
            final = call if call_lo == call else "not separable from program genes"
        rows.append({"pathway": pw, "tier": tier, "identity": ident, "context": context,
                     "label": ids["label"].get(ident, ""),
                     "n_libraries": len(best), "n_sig_pos": pos, "n_sig_neg": neg,
                     "mean_z": best["z"].mean(), "call": call,
                     "circularity": bool(circ.any()),
                     "n_sig_pos_leave_out": pos_lo, "n_sig_neg_leave_out": neg_lo,
                     "mean_z_leave_out": float(np.nanmean(best["z_leave_out"]))
                     if circ.any() and best["z_leave_out"].notna().any() else np.nan,
                     "final_call": final})
    return pd.DataFrame(rows)


# --- figures ------------------------------------------------------------------------------

def plot_validation(coh, zf, decision) -> None:
    libs = [l for l in LIBS]
    keys = [f"{r.pathway} {r.tier}" for r in decision.itertuples()]
    mat = np.full((len(keys), len(libs)), np.nan)
    for i, r in enumerate(decision.itertuples()):
        d = coh[(coh["pathway"] == r.pathway) & (coh["tier"] == r.tier)].set_index("library")
        for j, l in enumerate(libs):
            if l in d.index:
                mat[i, j] = -np.log10(d.at[l, "p"])
    fig, ax = plt.subplots(figsize=(0.6 * len(libs) + 5, 0.32 * len(keys) + 2))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=3, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > -np.log10(COHERENCE_P):
                ax.text(j, i, "*", ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(range(len(libs)), libs, rotation=45, ha="right")
    ax.set_yticks(range(len(keys)),
                  [f"{k}  [{d}]" for k, d in zip(keys, decision["decision"])], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="-log10 empirical p (coherence vs matched random sets)")
    ax.set_title("Proxy coherence per library (* p < 0.05); grey = gene set not scorable/absent")
    fig.tight_layout()
    save_figure(fig, "13_proxy_coherence")
    plt.close(fig)


def plot_identity_heatmap(summ, ids, tier, stem) -> None:
    d = summ[(summ["tier"] == tier) & (summ["context"] == "50pct")]
    if d.empty:
        return
    idents = [i for i in ids.index if i in set(d["identity"]) and ids.at[i, "n_k_replicated"] > 0]
    pws = [p for p in PATHWAYS if p in set(d["pathway"])]
    mat = d.pivot_table(index="identity", columns="pathway", values="mean_z").reindex(
        index=idents, columns=pws)
    fig, ax = plt.subplots(figsize=(0.9 * len(pws) + 6, 0.36 * len(idents) + 1.8))
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto")
    for i, ident in enumerate(idents):
        for j, pw in enumerate(pws):
            r = d[(d["identity"] == ident) & (d["pathway"] == pw)]
            if len(r):
                r = r.iloc[0]
                mark = {"high": "▲", "low": "▼"}.get(r["final_call"], "")
                if r["final_call"] == "not separable from program genes":
                    mark = "c"
                ax.text(j, i, f"{mark}{r['n_sig_pos']}/{r['n_sig_neg']}", ha="center",
                        va="center", fontsize=6)
    ax.set_xticks(range(len(pws)), pws)
    ax.set_yticks(range(len(idents)), [f"{i} {ids.at[i, 'label'][:42]}" for i in idents], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="mean association z over 50% libraries")
    ax.set_title(f"{tier} proxies vs replicated programs (50% epiboly)\n"
                 f"cell text: ▲/▼ replicated call, c = not separable from program genes, "
                 f"n libraries significant +/−", fontsize=9)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def plot_time(tc) -> None:
    d = tc[tc["tier"] == "readout"]
    order = DS5_LIBS + [l for l in LIBS if l not in DS5_LIBS]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for pw in PATHWAYS:
        x = d[d["pathway"] == pw].set_index("library").reindex(order)
        if x["z"].notna().any():
            ax.plot(range(len(DS5_LIBS)), x["z"].iloc[:len(DS5_LIBS)], "o-", label=pw)
            ax.plot(range(len(DS5_LIBS) + 1, len(order) + 1), x["z"].iloc[len(DS5_LIBS):], "o",
                    color=ax.lines[-1].get_color(), alpha=0.5)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(len(DS5_LIBS), color="0.3", ls="--", lw=0.8)
    ax.set_xticks(list(range(len(DS5_LIBS))) + list(range(len(DS5_LIBS) + 1, len(order) + 1)),
                  order, rotation=45, ha="right")
    ax.set_ylabel("pseudobulk readout level\n(z vs matched random gene sets)")
    ax.set_title("Readout level over time within DS5 (left of dashed line); "
                 "50% libraries are another batch (reference only)")
    ax.legend(frameon=False, fontsize=7, ncol=4)
    fig.tight_layout()
    save_figure(fig, "13_ds5_time_course")
    plt.close(fig)


def plot_bmp_antagonist(assoc) -> None:
    a = assoc[assoc["pathway"] == "BMP"]
    piv = a.pivot_table(index=["unit", "program", "identity"], columns="tier", values="z").reset_index()
    if not {"readout", "antagonist"} <= set(piv.columns):
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    for unit, d in piv.groupby("unit"):
        ax.scatter(d["readout"], d["antagonist"], s=14, label=unit, alpha=0.8)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("BMP readout association z (per program)")
    ax.set_ylabel("BMP antagonist association z")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("BMP readout vs secreted antagonists across programs")
    fig.tight_layout()
    save_figure(fig, "13_bmp_readout_vs_antagonist")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    qc = sc.read_h5ad(PROCESSED / "dropseq_3.3-6.0hpf_qc.h5ad")

    print("[13] step 0: gene sets")
    res = resolve(list(qc.var_names))
    det = detection_by_library(qc)
    res, sets, set_table = build_gene_sets(res, det)
    print(set_table.to_string(index=False))

    print("[13] step 1: matched null sets")
    bins = gene_bins(qc)
    exclude = set(res.loc[res["matrix_gene"].astype(bool), "matrix_gene"])
    nulls = null_sets(sets, bins, exclude, rng)
    needed = set(exclude) | {g for n in nulls.values() for g in n.ravel()}

    print("[13] steps 2-3: per-library smoothing, coherence, scores")
    coh, scores, pseudo = score_libraries(sets, nulls, needed)
    big = coh[coh["n_cells"] >= MIN_CELLS_COHERENCE]
    dec = []
    for r in set_table.itertuples():
        if not r.scorable:
            dec.append({"pathway": r.pathway, "tier": r.tier, "n_pass": 0, "n_tested": 0,
                        "decision": "not scorable (<2 usable genes)"})
            continue
        d = big[(big["pathway"] == r.pathway) & (big["tier"] == r.tier) & big["p"].notna()]
        n_pass = int((d["p"] < COHERENCE_P).sum())
        ok = len(d) > 0 and n_pass >= len(d) / 2
        dec.append({"pathway": r.pathway, "tier": r.tier, "n_pass": n_pass, "n_tested": len(d),
                    "decision": "coherent" if ok else "not coherent"})
    decision = pd.DataFrame(dec)

    print("[13] step 2b: ZFIN spatial agreement")
    zf = zfin_agreement(sets, list(qc.var_names), rng)

    print("[13] step 4: program association")
    ident = program_identities()
    assoc = pd.concat([associate(u, scores, sets, ident) for u in UNITS], ignore_index=True)
    assoc = bh(assoc)
    ids = pd.read_csv(TABLES / "11_k_scan_identities.tsv", sep="\t").set_index("id")
    summ = pd.concat([summarise_50pct(assoc, ids, "50pct"),
                      summarise_50pct(assoc, ids, "50pct_k13")], ignore_index=True)

    print("[13] step 4b: calibration with random pseudo-pathways")
    cal_coh, cal_calls = calibrate(scores_libs=list(LIBS), bins=bins, exclude=exclude,
                                   needed_fn=lambda: needed, ident=ident, ids=ids, rng=rng)

    print("[13] step 6b: per-gene BMP antagonist association")
    anta_genes = per_gene_association(sets.get(("BMP", "antagonist"), []), bins, exclude,
                                      ident, rng)

    print("[13] step 5: DS5 time course")
    tc_rows = []
    for lib in LIBS:
        pb = pseudo[lib]
        for key, genes in sets.items():
            g = [x for x in genes if x in pb.index]
            if len(g) < MIN_SET_GENES:
                continue
            level = pb[g].mean()
            nl = np.array([pb.reindex(row).mean() for row in nulls[key]])
            tc_rows.append({"pathway": key[0], "tier": key[1], "library": lib,
                            "level": level, "null_mean": np.nanmean(nl),
                            "z": (level - np.nanmean(nl)) / np.nanstd(nl)})
    tc = pd.DataFrame(tc_rows)

    print("[13] step 6: BMP antagonist test")
    bmp = assoc[assoc["pathway"] == "BMP"].pivot_table(
        index=["unit", "program"], columns="tier", values="z").reset_index()
    anta = []
    if {"readout", "antagonist"} <= set(bmp.columns):
        for unit, d in bmp.groupby("unit"):
            anta.append({"unit": unit, "n_programs": len(d),
                         "spearman_readout_vs_antagonist": pd.Series(d["readout"]).corr(
                             pd.Series(d["antagonist"]), method="spearman")})
    anta = pd.DataFrame(anta)

    print("[13] step 7: state table")
    rep = [i for i in ids.index
           if bool(ids.at[i, "libs_k9"] >= MIN_REPLICATE_LIBS) or bool(ids.at[i, "libs_k13"] >= MIN_REPLICATE_LIBS)]
    state = []
    for i in rep:
        row = {"identity": i, "label": ids.at[i, "label"],
               "programs_from": "K9" if ids.at[i, "libs_k9"] >= MIN_REPLICATE_LIBS else "K13"}
        for pw in PATHWAYS:
            dd = decision[(decision["pathway"] == pw) & (decision["tier"] == "readout")]
            dd = dd["decision"].iloc[0] if len(dd) else "not scorable"
            if dd.startswith("not scorable"):
                row[pw] = "unknown: not detected"
            elif dd == "not coherent":
                row[pw] = "unknown: proxy not coherent"
            else:
                ctx = "50pct" if row["programs_from"] == "K9" else "50pct_k13"
                s = summ[(summ["pathway"] == pw) & (summ["tier"] == "readout")
                         & (summ["identity"] == i) & (summ["context"] == ctx)]
                s13 = summ[(summ["pathway"] == pw) & (summ["tier"] == "readout")
                           & (summ["identity"] == i) & (summ["context"] == "50pct_k13")]
                row[pw] = s["final_call"].iloc[0] if len(s) else "program absent"
                row[f"{pw}_K13"] = s13["final_call"].iloc[0] if len(s13) else "program absent"
        state.append(row)
    state = pd.DataFrame(state)

    fmt = dict(sep="\t", index=False, float_format="%.4g")
    res.to_csv(TABLES / "13_gene_resolution.tsv", **fmt)
    set_table.to_csv(TABLES / "13_gene_sets.tsv", **fmt)
    coh.to_csv(TABLES / "13_proxy_coherence.tsv", **fmt)
    decision.to_csv(TABLES / "13_proxy_decision.tsv", **fmt)
    zf.to_csv(TABLES / "13_proxy_zfin_agreement.tsv", **fmt)
    assoc.to_csv(TABLES / "13_program_association.tsv", **fmt)
    summ.to_csv(TABLES / "13_identity_pathway_50pct.tsv", **fmt)
    tc.to_csv(TABLES / "13_time_course.tsv", **fmt)
    anta.to_csv(TABLES / "13_bmp_antagonist_test.tsv", **fmt)
    state.to_csv(TABLES / "13_pathway_state_table.tsv", **fmt)
    cal_coh.to_csv(TABLES / "13_calibration_coherence.tsv", **fmt)
    cal_calls.to_csv(TABLES / "13_calibration_calls.tsv", **fmt)
    anta_genes.to_csv(TABLES / "13_bmp_antagonist_gene_association.tsv", **fmt)
    big_cal = cal_coh[cal_coh["n_cells"] >= MIN_CELLS_COHERENCE]
    print(f"[13] calibration: coherence p<{COHERENCE_P} in {(big_cal['p'] < COHERENCE_P).mean():.3f} "
          f"of (set, library) tests; pseudo-pathways passing the coherence rule: "
          f"{int((big_cal.groupby('tier')['p'].apply(lambda x: (x < COHERENCE_P).sum() >= len(x) / 2)).sum())}/{N_CAL}; "
          f"replicated calls: {int((cal_calls['call'] != 'no consistent difference').sum())}/{len(cal_calls)} "
          f"identity x pseudo-pathway pairs")

    plot_validation(coh, zf, decision)
    for tier in ["readout", "secondary", "sender", "receiver"]:
        plot_identity_heatmap(summ, ids, tier, f"13_identity_pathway_{tier}")
    plot_time(tc)
    plot_bmp_antagonist(assoc)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(decision.to_string(index=False))
    print(zf.to_string(index=False))
    print(state.to_string(index=False))
    print(anta.to_string(index=False))
    print("[13] done")


if __name__ == "__main__":
    main()
