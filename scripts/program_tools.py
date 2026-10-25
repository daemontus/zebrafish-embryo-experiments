"""Gene-program replication and database annotation helpers (stages 10-12).

See scripts/10_program_annotation.py for the method description.
"""
from __future__ import annotations

import gzip
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from pipeline_utils import PROCESSED, ROOT

REFERENCE = ROOT / "data" / "reference"
PROGRAMS_DIR = PROCESSED / "09_programs"
LIBS = ["50% DS2", "50% DS3", "50% DS4", "50% DS4b"]
N_TOP = 50
N_SIGNATURE = 50
OVERLAP_Q = 1e-3
MIN_LIBRARIES = 3
ZFIN_WINDOW_HPF = (4.66, 6.0)
ZFIN_ASSAY = "mRNA in situ hybridization"
MIN_TERM_GENES = 5
MAX_GO_GENES = 500


def tag(lib: str) -> str:
    return "ZF50_" + lib.split()[-1]


def load_programs(k: int, libs=LIBS, tag_fn=tag, programs_dir=PROGRAMS_DIR,
                  prefix="09_programs"):
    scores, usages = {}, {}
    for lib in libs:
        stem = programs_dir / f"{prefix}_{tag_fn(lib)}_k{k}"
        s = pd.read_csv(f"{stem}_spectra_score.tsv", sep="\t", index_col=0)
        u = pd.read_csv(f"{stem}_usage.tsv", sep="\t", index_col=0)
        scores[lib], usages[lib] = s, u
    shared = sorted(set.intersection(*(set(s.index) for s in scores.values())))
    S = pd.concat([scores[lib].loc[shared] for lib in libs], axis=1)
    lib_of = {p: lib for lib in libs for p in scores[lib].columns}
    return S, usages, lib_of


def top_genes(S: pd.DataFrame, n: int = N_TOP) -> dict[str, list[str]]:
    return {p: list(S[p].nlargest(n).index) for p in S.columns}


def match_programs(S, lib_of, libs=LIBS) -> pd.DataFrame:
    tops = top_genes(S)
    M = len(S)
    R = np.corrcoef(S.to_numpy().T)
    R = pd.DataFrame(R, index=S.columns, columns=S.columns)
    rows = []
    for a, b in itertools.combinations(S.columns, 2):
        if lib_of[a] == lib_of[b]:
            continue
        k = len(set(tops[a]) & set(tops[b]))
        p = hypergeom.sf(k - 1, M, N_TOP, N_TOP)
        rows.append({"program_a": a, "program_b": b, "lib_a": lib_of[a], "lib_b": lib_of[b],
                     "pearson_r": R.at[a, b], "top50_overlap": k, "overlap_p": p})
    pairs = pd.DataFrame(rows)
    pairs["overlap_q"] = multipletests(pairs["overlap_p"], method="fdr_bh")[1]

    # mutual best match by r between each pair of libraries
    best = {}
    for a in S.columns:
        for lib in libs:
            if lib == lib_of[a]:
                continue
            cands = [b for b in S.columns if lib_of[b] == lib]
            best[(a, lib)] = max(cands, key=lambda b: R.at[a, b])
    pairs["mutual_best"] = [
        best[(a, lb)] == b and best[(b, la)] == a
        for a, b, la, lb in zip(pairs["program_a"], pairs["program_b"],
                                pairs["lib_a"], pairs["lib_b"])
    ]
    pairs["edge"] = pairs["mutual_best"] & (pairs["overlap_q"] < OVERLAP_Q)
    return pairs, R


def missing_library_matches(mp, pairs, lib_of, libs=LIBS) -> pd.Series:
    """For replicated MPs, the best (non-edge) partner in each library they miss."""
    out = {}
    for row in mp[mp["replicated"]].itertuples():
        members = row.members.split(",")
        missing = [l for l in libs if l not in row.libraries.split(",")]
        notes = []
        for lib in missing:
            d = pairs[(pairs["program_a"].isin(members) & (pairs["lib_b"] == lib))
                      | (pairs["program_b"].isin(members) & (pairs["lib_a"] == lib))]
            if len(d):
                b = d.sort_values("pearson_r", ascending=False).iloc[0]
                other = b["program_b"] if b["program_a"] in members else b["program_a"]
                notes.append(f"{other} (r={b['pearson_r']:.2f}, top50={int(b['top50_overlap'])})")
        out[row.mp] = "; ".join(notes)
    return pd.Series(out)


def meta_programs(S, pairs, lib_of, min_libraries=MIN_LIBRARIES) -> pd.DataFrame:
    parent = {p: p for p in S.columns}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs.loc[pairs["edge"], ["program_a", "program_b"]].itertuples(index=False):
        parent[find(a)] = find(b)
    comps = defaultdict(list)
    for p in S.columns:
        comps[find(p)].append(p)

    edges = pairs[pairs["edge"]]
    rows = []
    for members in comps.values():
        libs = sorted({lib_of[m] for m in members})
        e = edges[edges["program_a"].isin(members) & edges["program_b"].isin(members)]
        rows.append({
            "members": ",".join(sorted(members)),
            "n_programs": len(members),
            "n_libraries": len(libs),
            "libraries": ",".join(libs),
            "duplicate_library": len(libs) < len(members),
            "mean_edge_r": e["pearson_r"].mean() if len(e) else np.nan,
            "mean_edge_overlap": e["top50_overlap"].mean() if len(e) else np.nan,
        })
    mp = pd.DataFrame(rows).sort_values(["n_libraries", "mean_edge_r"], ascending=False)
    mp = mp.reset_index(drop=True)
    mp.insert(0, "mp", [f"MP{i + 1}" for i in range(len(mp))])
    mp["replicated"] = mp["n_libraries"] >= min_libraries
    return mp


def signatures(S, mp, min_libraries=MIN_LIBRARIES) -> tuple[pd.DataFrame, dict, dict]:
    tops = top_genes(S)
    mean_scores, sig, core = {}, {}, {}
    for row in mp.itertuples():
        members = row.members.split(",")
        m = S[members].mean(axis=1)
        mean_scores[row.mp] = m
        sig[row.mp] = list(m.nlargest(N_SIGNATURE).index)
        counts = pd.Series([g for p in members for g in tops[p]]).value_counts()
        core[row.mp] = [g for g in sig[row.mp] if counts.get(g, 0) >= min_libraries]
    return pd.DataFrame(mean_scores), sig, core


def parse_obo(path, keep_namespace=None) -> tuple[dict, dict]:
    """Minimal OBO parser: id -> name, id -> set(parents) via is_a and part_of."""
    names, parents = {}, defaultdict(set)
    cur, ns, obsolete = None, None, False

    def flush():
        if cur and not obsolete and (keep_namespace is None or ns == keep_namespace):
            names[cur["id"]] = cur["name"]
            parents[cur["id"]] |= cur["parents"]

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line == "[Term]" or line.startswith("[Typedef]"):
                flush()
                cur, ns, obsolete = ({"id": None, "name": None, "parents": set()}
                                     if line == "[Term]" else None), None, False
            elif cur is None:
                continue
            elif line.startswith("id: "):
                cur["id"] = line[4:]
            elif line.startswith("name: "):
                cur["name"] = line[6:]
            elif line.startswith("namespace: "):
                ns = line[11:]
            elif line.startswith("is_obsolete: true"):
                obsolete = True
            elif line.startswith("is_a: "):
                cur["parents"].add(line[6:].split(" ")[0])
            elif line.startswith("relationship: part_of "):
                cur["parents"].add(line.split(" ")[2])
        flush()
    return names, parents


def ancestors_fn(parents):
    cache = {}

    def anc(t):
        if t not in cache:
            out = {t}
            for p in parents.get(t, ()):
                out |= anc(p)
            cache[t] = out
        return cache[t]
    return anc


def zfin_annotations(window=ZFIN_WINDOW_HPF) -> tuple[dict[str, set], dict[str, str]]:
    cols = ["gene_id", "symbol", "fish", "super_id", "super_name", "sub_id", "sub_name",
            "start", "end", "assay", "mmo", "pub", "probe", "antibody", "fish_id"]
    z = pd.read_csv(REFERENCE / "zfin_wildtype_expression.txt", sep="\t", header=None,
                    names=cols, dtype=str)
    st = pd.read_csv(REFERENCE / "zfin_stage_ontology.txt", sep="\t", header=None,
                     names=["zdb", "zfs", "name", "h0", "h1"])
    h0 = dict(zip(st["name"], st["h0"]))
    h1 = dict(zip(st["name"], st["h1"]))
    z["h0"] = z["start"].map(h0)
    z["h1"] = z["end"].map(h1)
    lo, hi = window
    w = z[(z["assay"] == ZFIN_ASSAY) & (z["h0"] < hi) & (z["h1"] > lo)]
    names, parents = parse_obo(REFERENCE / "zfa.obo")
    anc = ancestors_fn(parents)
    gene_terms = defaultdict(set)
    for sym, sup, sub in w[["symbol", "super_id", "sub_id"]].itertuples(index=False):
        for t in (sup, sub):
            if isinstance(t, str) and t.startswith("ZFA:"):
                gene_terms[sym.upper()] |= anc(t)
    names = {**names, **dict(zip(w["super_id"], w["super_name"]))}
    drop = {t for t, n in names.items() if n in ("unspecified",)}
    gene_terms = {g: ts - drop for g, ts in gene_terms.items()}
    return gene_terms, names


def go_annotations() -> tuple[dict[str, set], dict[str, str]]:
    names, parents = parse_obo(REFERENCE / "go-basic.obo", keep_namespace="biological_process")
    anc = ancestors_fn(parents)
    gene_terms = defaultdict(set)
    with gzip.open(REFERENCE / "zfin.gaf.gz", "rt") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[8] != "P" or "NOT" in f[3] or f[4] not in names:
                continue
            gene_terms[f[2].upper()] |= anc(f[4])
    root = "GO:0008150"
    gene_terms = {g: ts - {root} for g, ts in gene_terms.items()}
    return gene_terms, names


def enrich(sig: dict[str, list[str]], universe_genes, gene_terms, names, max_genes=None):
    universe = [g for g in universe_genes if g in gene_terms]
    term_genes = defaultdict(set)
    for g in universe:
        for t in gene_terms[g]:
            term_genes[t].add(g)
    terms = {t: gs for t, gs in term_genes.items()
             if len(gs) >= MIN_TERM_GENES and (max_genes is None or len(gs) <= max_genes)}
    M = len(universe)
    rows = []
    for mp, genes in sig.items():
        hits = [g for g in genes if g in gene_terms]
        N = len(hits)
        hs = set(hits)
        for t, gs in terms.items():
            k = len(hs & gs)
            if k == 0:
                continue
            rows.append({"mp": mp, "term": t, "name": names.get(t, t), "overlap": k,
                         "signature_annotated": N, "term_size": len(gs), "universe": M,
                         "fold": (k / N) / (len(gs) / M),
                         "p": hypergeom.sf(k - 1, M, len(gs), N),
                         "genes": ",".join(sorted(hs & gs))})
    df = pd.DataFrame(rows)
    df["q"] = multipletests(df["p"], method="fdr_bh")[1]
    return df.sort_values(["mp", "p", "term"]).reset_index(drop=True), M
