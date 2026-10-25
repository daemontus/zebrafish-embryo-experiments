"""Consensus-NMF driver helpers shared by stage 09 (50% epiboly) and stage 12 (DS5).

See scripts/09_gene_programs_cnmf.py for the method description.
"""
from __future__ import annotations

import multiprocessing as mp
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

N_ITER = 100
N_HVG = 2000
MIN_CELLS = 3
DENSITY_THRESHOLD = 0.1
MIN_K = 7
N_WORKERS = 8


def _factorize_worker(args) -> None:
    from cnmf import cNMF

    out_dir, name, worker, total = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cNMF(output_dir=str(out_dir), name=name).factorize(
            worker_i=worker, total_workers=total, skip_completed_runs=True)


def factorize_counts(sub, name: str, out: Path, ks: list[int], seed: int, force: bool,
                     log_prefix: str = "[cnmf]"):
    """Run cNMF on the raw counts (layers['counts']) of `sub`, cached under out/name."""
    from cnmf import cNMF

    done_marker = out / name / "cnmf_tmp" / f"{name}.spectra.k_{ks[-1]}.merged.df.npz"
    obj = cNMF(output_dir=str(out), name=name)
    if done_marker.exists() and not force:
        return obj

    counts = sc.AnnData(sub.layers["counts"].copy(), obs=pd.DataFrame(index=sub.obs_names),
                        var=pd.DataFrame(index=sub.var_names))
    sc.pp.filter_genes(counts, min_cells=MIN_CELLS)
    (out / name).mkdir(parents=True, exist_ok=True)
    counts_fn = out / name / "counts.h5ad"
    counts.write_h5ad(counts_fn)

    print(f"{log_prefix} {name}: {counts.n_obs} cells x {counts.n_vars} genes; "
          f"K={ks[0]}..{ks[-1]}, {N_ITER} iterations each")
    obj.prepare(counts_fn=str(counts_fn), components=np.array(ks), n_iter=N_ITER,
                seed=seed, num_highvar_genes=N_HVG)
    with mp.get_context("spawn").Pool(N_WORKERS) as pool:
        pool.map(_factorize_worker, [(out, name, i, N_WORKERS) for i in range(N_WORKERS)])
    obj.combine()
    return obj


def k_selection(obj, lib: str, ks: list[int]) -> pd.DataFrame:
    rows = []
    for k in ks:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stats = obj.consensus(k, skip_density_and_return_after_stats=True,
                                  show_clustering=False, close_clustergram_fig=True)
        rows.append({"library": lib, "k": k,
                     "stability": float(stats.loc["silhouette", "stats"]),
                     "prediction_error": float(stats.loc["prediction_error", "stats"])})
    return pd.DataFrame(rows)


def choose_common_k(ks: pd.DataFrame, min_k: int = MIN_K) -> int:
    """The K >= min_k that maximises the worst library's consensus stability."""
    worst = ks[ks["k"] >= min_k].groupby("k")["stability"].min()
    return int(worst.idxmax())


def consensus(obj, prefix: str, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj.consensus(k, density_threshold=DENSITY_THRESHOLD, show_clustering=False,
                      close_clustergram_fig=True)
        usage, spectra_scores, _, _ = obj.load_results(K=k, density_threshold=DENSITY_THRESHOLD)
    usage.columns = [f"{prefix}_P{c}" for c in usage.columns]
    spectra_scores.columns = [f"{prefix}_P{c}" for c in spectra_scores.columns]
    return usage, spectra_scores
