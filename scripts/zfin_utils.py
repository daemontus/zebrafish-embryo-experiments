"""ZFIN expression records: loading, stage hpf and matrix gene symbols.

Shared by 01c_heterogeneity_zfin, 09_variable_programs and 11_program_sets.
Needs data/raw/zfin/ (scripts/download_zfin.sh).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_utils import RAW, ROOT

ZFIN = RAW.parent / "zfin"
MAPPING = ROOT / "data" / "curated" / "zfin_anatomy_programs.tsv"
MAX_YEAR = 2018  # publications strictly before

ZFIN_COLUMNS = ["gene_id", "symbol", "fish", "super_id", "term", "sub_id", "subterm",
                "start", "end", "assay", "assay_id", "pub", "probe", "antibody", "fish_id"]
ZFIN_STAGE_HPF = {
    "Zygote:1-cell": 0.0, "Cleavage:2-cell": 0.75, "Cleavage:4-cell": 1.0,
    "Cleavage:8-cell": 1.25, "Cleavage:16-cell": 1.5, "Cleavage:32-cell": 1.75,
    "Cleavage:64-cell": 2.0, "Blastula:128-cell": 2.25, "Blastula:256-cell": 2.5,
    "Blastula:512-cell": 2.75, "Blastula:1k-cell": 3.0, "Blastula:High": 3.33,
    "Blastula:Oblong": 3.67, "Blastula:Sphere": 4.0, "Blastula:Dome": 4.33,
    "Blastula:30%-epiboly": 4.67, "Gastrula:50%-epiboly": 5.25, "Gastrula:Germ-ring": 5.67,
    "Gastrula:Shield": 6.0, "Gastrula:75%-epiboly": 8.0, "Gastrula:90%-epiboly": 9.0,
    "Gastrula:Bud": 10.0, "Segmentation:1-4 somites": 10.33,
    "Segmentation:5-9 somites": 11.67, "Segmentation:10-13 somites": 14.0,
    "Segmentation:14-19 somites": 16.0, "Segmentation:20-25 somites": 19.0,
    "Segmentation:26+ somites": 22.0, "Pharyngula:Prim-5": 24.0,
}


def zfin_records(max_year: int | None = MAX_YEAR) -> tuple[pd.DataFrame, dict[str, str]]:
    """Expression records published before `max_year` (None: all), with hpf
    ranges, and symbol -> ZFIN gene id.

    zfinpubs.txt lists journal articles only; direct data submissions (e.g.
    the Thisse et al. in situ screens, ~1,700 genes) are absent, so their
    year is taken from the date in the publication id (ZDB-PUB-YYMMDD-n).
    """
    e = pd.read_csv(ZFIN / "wildtype-expression_fish.txt", sep="\t", header=None,
                    names=ZFIN_COLUMNS, dtype=str)
    pubs = pd.read_csv(ZFIN / "zfinpubs.txt", sep="\t", header=None, dtype=str,
                       usecols=[0, 5], names=["pub", "year"])
    year = pd.to_numeric(pubs.set_index("pub")["year"], errors="coerce")
    yy = pd.to_numeric(e["pub"].str.extract(r"ZDB-PUB-(\d\d)")[0])
    e["year"] = e["pub"].map(year).fillna(yy.where(yy <= 80, yy - 100) + 2000)
    if max_year is not None:
        e = e[e["year"] < max_year]
    e = e.copy()
    e["start_hpf"] = e["start"].map(ZFIN_STAGE_HPF)
    e["end_hpf"] = e["end"].map(ZFIN_STAGE_HPF).fillna(np.inf)
    e = e[e["start_hpf"].notna()]
    current = dict(zip(e["symbol"].str.lower(), e["gene_id"]))
    aliases = pd.read_csv(ZFIN / "aliases.txt", sep="\t", header=None, dtype=str,
                          usecols=[0, 2, 3], names=["gene_id", "symbol", "alias"])
    aliases = aliases[aliases["gene_id"].str.startswith("ZDB-GENE")]
    symbols = dict(zip(aliases["symbol"].str.lower(), aliases["gene_id"]))
    symbols.update(current)
    by_alias = aliases.assign(alias=aliases["alias"].str.lower()).groupby("alias")["gene_id"]
    unique_alias = by_alias.nunique() == 1
    for alias, gene_id in by_alias.first()[unique_alias].items():
        symbols.setdefault(alias, gene_id)
    return e, symbols


def matrix_symbols(adata, symbols: dict[str, str], min_umis: int = 10) -> pd.Series:
    """ZFIN gene id -> matrix gene (the most expressed one if several), among
    genes with at least `min_umis` UMIs in `adata`."""
    total = pd.Series(np.asarray(adata.X.sum(axis=0)).ravel(), index=adata.var_names)
    total = total[total >= min_umis].sort_values(ascending=False)
    ids = pd.Series([symbols.get(g.lower()) for g in total.index], index=total.index)
    ids = ids.dropna()
    return pd.Series(ids.index, index=ids.to_numpy()).groupby(level=0).first()
