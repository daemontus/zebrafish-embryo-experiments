"""Gene-set resolution for stage 13 (signalling-pathway proxies).

Resource symbols (scripts/pathway_proxies.tsv) are resolved to ZFIN genes and then
to the gene names used in the GEO Drop-seq matrix (2015-era ZFIN symbols):

  1. exact current ZFIN symbol                       -> status "current"
  2. else exact ZFIN previous name / alias           -> "alias" ("alias_ambiguous" if
                                                        it names several genes)
  3. else exact ZFIN full gene name (e.g. "lefty1")  -> "gene_name"
  4. else current symbols <symbol>a / <symbol>b       -> "paralog_expansion"
  5. else                                            -> "unresolved"; ZFIN current
     symbols sharing the first 3+ characters are listed in "suggestions" for manual
     review but are NOT used.

A resolved ZFIN gene is matched to the matrix through its current symbol if that is
in the matrix; otherwise through an alias, but never through an alias that is the
current symbol of a different ZFIN gene (e.g. wnt3a's alias "wnt3" is gene wnt3,
fgf8a's alias "ace" is gene ace). Matching is case-insensitive. Gene roles for sender/receiver sets come from GO
molecular function (ZFIN GAF): "receptor ligand activity" (GO:0048018) -> ligand,
"signaling receptor activity" (GO:0038023) -> receptor, including descendants.
"""
from __future__ import annotations

import gzip
from collections import defaultdict

import pandas as pd

from pipeline_utils import ROOT
from program_tools import REFERENCE, ancestors_fn, parse_obo

PROXY_TABLE = ROOT / "scripts" / "pathway_proxies.tsv"
GO_LIGAND = "GO:0048018"
GO_RECEPTOR = "GO:0038023"


def zfin_symbol_index():
    genes = pd.read_csv(REFERENCE / "zfin_gene.txt", sep="\t", header=None,
                        names=["zdb", "so", "symbol", "ncbi"], dtype=str)
    genes = genes[genes["zdb"].str.startswith("ZDB-GENE")]
    current = dict(zip(genes["zdb"], genes["symbol"]))
    al = pd.read_csv(REFERENCE / "zfin_aliases.txt", sep="\t", header=None,
                     names=["zdb", "name", "symbol", "alias", "so"], dtype=str,
                     on_bad_lines="skip")
    al = al[al["zdb"].str.startswith("ZDB-GENE", na=False)]
    # gene.txt omits some loci (e.g. wnt8a); the alias table also carries current symbols
    for z, sym in zip(al["zdb"], al["symbol"]):
        if isinstance(sym, str):
            current.setdefault(z, sym)
    full_name = {z: n for z, n in zip(al["zdb"], al["name"]) if isinstance(n, str)}
    al = al[al["alias"].notna()]
    aliases = defaultdict(set)
    for z, a in zip(al["zdb"], al["alias"]):
        aliases[z].add(a)
    by_current = defaultdict(set)
    for z, s in current.items():
        by_current[s.lower()].add(z)
    by_alias = defaultdict(set)
    for z, names in aliases.items():
        for a in names:
            by_alias[a.lower()].add(z)
    by_name = defaultdict(set)
    for z, n in full_name.items():
        by_name[n.lower()].add(z)
    return current, aliases, by_current, by_alias, by_name


def go_roles() -> dict[str, set]:
    names, parents = parse_obo(REFERENCE / "go-basic.obo", keep_namespace="molecular_function")
    anc = ancestors_fn(parents)
    roles = defaultdict(set)
    with gzip.open(REFERENCE / "zfin.gaf.gz", "rt") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[8] != "F" or "NOT" in f[3] or f[4] not in names:
                continue
            a = anc(f[4])
            if GO_LIGAND in a:
                roles[f[1]].add("ligand")
            if GO_RECEPTOR in a:
                roles[f[1]].add("receptor")
    return roles


def resolve(matrix_genes) -> pd.DataFrame:
    """One row per (resource entry, resolved ZFIN gene)."""
    current, aliases, by_current, by_alias, by_name = zfin_symbol_index()
    current_upper = {s.upper(): z for z, s in current.items()}
    roles = go_roles()
    upper = {g.upper(): g for g in matrix_genes}
    proxies = pd.read_csv(PROXY_TABLE, sep="\t", keep_default_na=False)
    rows = []
    for r in proxies.itertuples():
        s = r.symbol.lower()
        if s in by_current:
            zdbs, status = sorted(by_current[s]), "current"
        elif s in by_alias:
            zdbs = sorted(by_alias[s])
            status = "alias" if len(zdbs) == 1 else "alias_ambiguous"
        elif s in by_name:
            zdbs, status = sorted(by_name[s]), "gene_name"
        else:
            zdbs = sorted(z for suf in ("a", "b") for z in by_current.get(s + suf, ()))
            status = "paralog_expansion" if zdbs else "unresolved"
        if not zdbs:
            stem = s.rstrip("0123456789ab")[:max(3, len(s.rstrip("0123456789ab")))]
            sugg = sorted({sym for sym in by_current if len(stem) >= 3 and sym.startswith(stem)
                           and sym.upper() in upper})[:6]
            rows.append({**r._asdict(), "status": status, "zdb": "", "zfin_symbol": "",
                         "matrix_gene": "", "role": "", "suggestions": ",".join(sugg)})
            continue
        for z in zdbs:
            cur = current.get(z, "")
            blocked = []
            if cur.upper() in upper:
                hits = [upper[cur.upper()]]
            else:
                hits = sorted({upper[n.upper()] for n in aliases.get(z, set())
                               if n.upper() in upper
                               and current_upper.get(n.upper(), z) == z})
                blocked = sorted({upper[n.upper()] for n in aliases.get(z, set())
                                  if n.upper() in upper and current_upper.get(n.upper(), z) != z})
            role = ",".join(sorted(roles.get(z, ())))
            if not hits:
                note = (f"matrix name(s) {','.join(blocked)} are the current symbol of another "
                        "ZFIN gene (renamed since 2015); not used") if blocked else ""
                rows.append({**r._asdict(), "status": status, "zdb": z, "zfin_symbol": cur,
                             "matrix_gene": "", "role": role, "resolution_note": note})
            for h in hits:
                rows.append({**r._asdict(), "status": status, "zdb": z, "zfin_symbol": cur,
                             "matrix_gene": h, "role": role})
    out = pd.DataFrame(rows).drop(columns="Index")
    return out
