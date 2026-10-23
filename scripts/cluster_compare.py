"""Cluster-comparison helpers shared by stage 07 and stage 08."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

KNN_MAPPING_K = 15


def library_colors(libs) -> dict[str, str]:
    cmap = plt.get_cmap("tab10" if len(libs) <= 10 else "tab20")
    return {lib: matplotlib.colors.to_hex(cmap(i)) for i, lib in enumerate(libs)}


def joint_composition(obs, libs, joint_labels, key: str = "joint") -> pd.DataFrame:
    """Library make-up of each joint cluster, with size-corrected evenness."""
    counts = pd.crosstab(obs[key], obs["library_short"]).reindex(
        index=joint_labels, columns=libs, fill_value=0)
    # Size-corrected evenness: share of each library's cells in this cluster,
    # renormalised over libraries, then Shannon entropy / log(n_libraries).
    per_lib = counts / counts.sum(axis=0)
    p = per_lib.div(per_lib.sum(axis=1), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(p * np.log(p)).fillna(0).sum(axis=1) / np.log(len(libs))
    out = counts.copy()
    out.insert(0, "n_cells", counts.sum(axis=1))
    out["library_evenness"] = ent.clip(lower=0)
    out["dominant_library"] = p.idxmax(axis=1)
    out["dominant_share"] = p.max(axis=1)
    out.index.name = "joint_cluster"
    return out


def knn_mapping(X, obs, libs, local_labels, k: int = KNN_MAPPING_K) -> pd.DataFrame:
    """Mean neighbour-label fractions from each local cluster into each other library.

    X is the embedding (cells x dims) in which neighbours are searched.
    """
    lib = obs["library_short"].to_numpy()
    local = obs["local"].to_numpy()
    rows = []
    for tgt in libs:
        tmask = lib == tgt
        tgt_clusters = [c for c in local_labels if c.startswith(f"{tgt}:")]
        kk = min(k, int(tmask.sum()))
        nn = NearestNeighbors(n_neighbors=kk).fit(X[tmask])
        idx = nn.kneighbors(X[~tmask], return_distance=False)
        neigh = local[tmask][idx]
        frac = np.stack([(neigh == c).mean(axis=1) for c in tgt_clusters], axis=1)
        frac = pd.DataFrame(frac, columns=tgt_clusters)
        frac["src"] = local[~tmask]
        mean = frac.groupby("src").mean()
        for src, row in mean.iterrows():
            for tc, v in row.items():
                rows.append({"src_cluster": src, "src_library": src.rsplit(":", 1)[0],
                             "tgt_cluster": tc, "tgt_library": tgt, "fraction": v})
    return pd.DataFrame(rows)


def best_matches(mapping: pd.DataFrame, sizes: pd.Series) -> pd.DataFrame:
    """Best target cluster per (source cluster, target library), with mutual flag."""
    best = (mapping.sort_values("fraction", ascending=False)
            .groupby(["src_cluster", "tgt_library"], sort=False).head(1))
    lookup = best.set_index(["src_cluster", "tgt_library"])["tgt_cluster"]
    best = best.copy()
    best["mutual"] = [
        lookup.get((t, s_lib)) == s
        for s, s_lib, t in zip(best["src_cluster"], best["src_library"], best["tgt_cluster"])
    ]
    best["src_size"] = best["src_cluster"].map(sizes)
    best["tgt_size"] = best["tgt_cluster"].map(sizes)
    return best.sort_values(["src_cluster", "tgt_library"]).reset_index(drop=True)
