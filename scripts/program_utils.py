"""Shared helpers of the program analyses (scripts 16-19).

Smoothing over library neighbourhoods, the held-out prediction test (one
logistic regression per evaluation gene; floor, feature sets and the knn
ceiling) and the gene modules of Farrell et al. (Table S2). Moved here from
the earlier scripts 04-06 (removed 2026-10-28).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from pipeline_utils import ROOT

N_PCS = 30
K_SMOOTH = 10  # neighbours for smoothing scores
K_CEILING = 20  # training neighbours of the knn ceiling
K_TARGET = 10  # neighbours of the neighbourhood target
TEST_FRACTION = 0.3
# Evaluation genes are detected in 2-98% of the stage's cells (a gene detected
# in every cell says nothing about cell state).
MIN_DETECTED, MAX_DETECTED = 0.02, 0.98
N_JOBS = 11
# L2 strength per gene and feature set, by 3-fold CV on the training cells,
# so that uninformative features shrink towards the floor model.
CS = [0.003, 0.03, 0.3, 3.0]
TARGETS = ["detection", "neighbourhood"]
Y_MIN = -0.2  # lower limit of figures (lower values are drawn at the limit)
MIN_SHARE = 0.04  # explainable share of the floor loss below which `explained` is NaN

FARRELL_S2 = ROOT / "paper" / "aar3131_farrell_table-s2.xlsx"  # not versioned
FARRELL_SHEETS = {
    "High": "high", "Oblong": "oblong", "Dome": "dome", "30% Epiboly": "30% epiboly",
    "50% Epiboly": "50% epiboly", "Shield": "shield", "60% Epiboly": "60% epiboly",
    "75% Epiboly": "75% epiboly", "90% Epiboly": "90% epiboly", "Bud": "bud",
    "3 Somites": "3-somite", "6 Somites": "6-somite",
}
FARRELL_TOP = 30  # genes per module (all that Table S2 lists)


def log_normalised(adata: sc.AnnData) -> sc.AnnData:
    out = adata.copy()
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    out.X = out.X.tocsc()
    return out


def pca(X: np.ndarray, n: int = N_PCS) -> np.ndarray:
    Z = StandardScaler().fit_transform(X)
    Z = np.clip(Z, -10, 10)
    u, s, _ = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    return u[:, :n] * s[:n]


def smooth(values: np.ndarray, library: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Mean over each cell and its K_SMOOTH nearest neighbours in its library."""
    out = np.empty_like(values, dtype=float)
    for lib in np.unique(library):
        rows = np.flatnonzero(library == lib)
        k = min(K_SMOOTH, len(rows) - 1)
        nb = NearestNeighbors(n_neighbors=k).fit(coords[rows]).kneighbors(return_distance=False)
        nb = np.column_stack([np.arange(len(rows)), nb])
        out[rows] = values[rows][nb].mean(axis=1)
    return out


def ceiling_feature(detect_b: np.ndarray, coords: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Detection frequency of every evaluation gene among each cell's training neighbours."""
    nn = NearestNeighbors(n_neighbors=K_CEILING + 1).fit(coords[train])
    nb = nn.kneighbors(coords, return_distance=False)
    train_idx = np.flatnonzero(train)
    out = np.empty(detect_b.shape, dtype=float)
    for i in range(len(coords)):
        neighbours = train_idx[nb[i]]
        neighbours = neighbours[neighbours != i][:K_CEILING]  # training cells exclude themselves
        out[i] = detect_b[neighbours].mean(axis=0)
    return out


def log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Cross-entropy minus the entropy of y (= log-loss for binary y)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    q = np.clip(y, 1e-12, 1 - 1e-12)
    return y * np.log(q / p) + (1 - y) * np.log((1 - q) / (1 - p))


def neighbourhood_target(norm: sc.AnnData, detect_b: np.ndarray, genes_b: np.ndarray,
                         library: np.ndarray, log_depth: np.ndarray) -> tuple[np.ndarray, ...]:
    """Detection frequency of each evaluation gene, and mean log depth, over each
    cell and its K_TARGET nearest neighbours in its library (PCA of the
    evaluation genes, so that the target shares no neighbourhood with a feature)."""
    coords = pca(norm.X[:, genes_b].toarray())
    target = np.empty_like(detect_b)
    depth = np.empty_like(log_depth)
    for lib in np.unique(library):
        rows = np.flatnonzero(library == lib)
        k = min(K_TARGET, len(rows) - 1)
        nb = NearestNeighbors(n_neighbors=k).fit(coords[rows]).kneighbors(return_distance=False)
        nb = rows[np.column_stack([np.arange(len(rows)), nb])]
        target[rows] = detect_b[nb].mean(axis=1)
        depth[rows] = log_depth[nb].mean(axis=1)
    return target, depth


_SHARED: dict = {}


def init_worker(bases: dict, feature_sets: list[np.ndarray], train: np.ndarray) -> None:
    _SHARED.update(bases=bases, feature_sets=feature_sets, train=train)


def fit_gene(task) -> tuple[np.ndarray, ...]:
    """Test loss per cell of the floor, every feature set and the ceiling, one gene.

    A soft target y in [0, 1] is fitted as a positive with weight y and a
    negative with weight 1 - y.
    """
    target, y, knn_p = task
    base, feature_sets, train = (_SHARED["bases"][target], _SHARED["feature_sets"],
                                 _SHARED["train"])
    test = ~train
    soft = target != "detection"
    out = []
    logit_knn = np.log((knn_p + 0.01) / (1 - knn_p + 0.01))[:, None]
    for extra in [None] + feature_sets + [logit_knn]:
        X = base if extra is None else np.hstack([base, extra])
        X = StandardScaler().fit(X[train]).transform(X)
        Xt, yt = X[train], y[train]
        weight = None
        if soft:
            Xt = np.vstack([Xt, Xt])
            weight = np.concatenate([yt, 1 - yt])
            yt = np.concatenate([np.ones(len(yt)), np.zeros(len(yt))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = LogisticRegressionCV(Cs=CS, cv=3, scoring="neg_log_loss", max_iter=500)
            model.fit(Xt, yt, sample_weight=weight)
        out.append(log_loss(y[test], model.predict_proba(X[test])[:, 1]))
    return tuple(out)


def farrell_modules() -> dict[str, list[pd.Series]]:
    """Table S2 of Farrell et al.: the FARRELL_TOP highest-weighted genes of
    every module of every stage, with weights normalised to sum to 1."""
    sheets = pd.read_excel(FARRELL_S2, sheet_name=None, header=None)
    out = {}
    for sheet, table in sheets.items():
        modules = []
        for col in range(0, table.shape[1] - 1, 2):
            genes = table.iloc[1:, col].dropna().astype(str).str.upper()
            weights = pd.to_numeric(table.iloc[1:len(genes) + 1, col + 1]).to_numpy()
            top = pd.Series(weights, index=genes.to_numpy()).sort_values(ascending=False)
            top = top.iloc[:FARRELL_TOP].clip(lower=0)
            modules.append(top / top.sum() if top.sum() > 0 else top)
        out[FARRELL_SHEETS[sheet]] = modules
    return out
