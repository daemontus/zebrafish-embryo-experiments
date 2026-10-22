#!/usr/bin/env python3
"""Build cell-level metadata tables for GEO SuperSeries GSE106587.

Reads the SOFT family files downloaded by scripts/download_gse106587.sh and
writes CSV/TSV tables to data/metadata/. Uses only the Python standard library.

Outputs
-------
data/metadata/dropseq_cell_metadata.csv
data/metadata/ss2_cell_metadata.csv
data/metadata/tenx_cell_metadata.csv
data/metadata/atlas_inventory.tsv
data/metadata/stage_summary.tsv
"""
from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "GSE106587"
META = ROOT / "data" / "metadata"
SOFT = META / "soft"

# Drop-seq cell names look like  ZF50_WT_DS4b_ACGTACGTACGT
# (stage code, genotype, sequencing batch, replicate suffix "b", barcode).
STAGE_INFO = {
    "ZFHIGH": ("high", 3.3),
    "ZFOBLONG": ("oblong", 3.8),
    "ZFDOME": ("dome", 4.3),
    "ZF30": ("30% epiboly", 4.7),
    "ZF50": ("50% epiboly", 5.3),
    "ZFS": ("shield", 6.0),
    "ZF60": ("60% epiboly", 7.0),
    "ZF75": ("75% epiboly", 8.0),
    "ZF90": ("90% epiboly", 9.0),
    "ZFB": ("bud", 10.0),
    "ZF3S": ("3-somite", 11.0),
    "ZF6S": ("6-somite", 12.0),
}
STAGE_ORDER = [
    "high", "oblong", "dome", "30% epiboly", "50% epiboly", "shield",
    "60% epiboly", "75% epiboly", "90% epiboly", "bud", "3-somite",
    "6-somite",
]


def parse_soft(path: Path) -> list[dict]:
    """Parse a GEO SOFT family file into a list of sample dictionaries."""
    samples: list[dict] = []
    cur: dict | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                if cur is not None:
                    samples.append(cur)
                cur = {
                    "gsm": line.split(" = ", 1)[1].strip(),
                    "title": "",
                    "characteristics": [],
                    "description": [],
                }
            elif cur is not None:
                if line.startswith("!Sample_title = "):
                    cur["title"] = line.split(" = ", 1)[1].strip()
                elif line.startswith("!Sample_characteristics_ch1 = "):
                    cur["characteristics"].append(
                        line.split(" = ", 1)[1].strip()
                    )
                elif line.startswith("!Sample_description = "):
                    cur["description"].append(line.split(" = ", 1)[1].strip())
    if cur is not None:
        samples.append(cur)
    for s in samples:
        chars: dict[str, str] = {}
        for item in s["characteristics"]:
            if ": " in item:
                k, v = item.split(": ", 1)
                chars[k.strip().lower()] = v.strip()
        s["chars"] = chars
    return samples


def read_header(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        return fh.readline().rstrip("\n").split("\t")


def read_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh]


def dropseq_metadata() -> tuple[list[dict], dict]:
    cells = read_header(RAW / "GSE106474" / "GSE106474_UMICounts.txt.gz")
    cells = [c for c in cells if c]

    samples = parse_soft(SOFT / "GSE106474_family.soft.gz")
    by_stage_batch: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in samples:
        by_stage_batch[
            (s["chars"].get("developmental stage", ""),
             s["chars"].get("sequencing batch", ""))
        ].append(s)
    for key in by_stage_batch:
        by_stage_batch[key].sort(key=lambda s: int(s["gsm"].replace("GSM", "")))

    # Prefix -> sample. Duplicate prefixes only differ by a trailing "b";
    # GEO lists the corresponding replicates in ascending GSM order.
    prefixes = sorted({c.rsplit("_", 1)[0] for c in cells})
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in prefixes:
        base = p[:-1] if p.endswith("b") else p
        stage_code, _genotype, batch = base.split("_")
        stage, _hpf = STAGE_INFO[stage_code]
        groups[(stage, batch)].append(p)
    prefix_to_sample: dict[str, dict] = {}
    for key, group in groups.items():
        group.sort(key=lambda p: (p.endswith("b"), p))
        geo = by_stage_batch[key]
        if len(group) != len(geo):
            raise RuntimeError(f"cannot match {key}: {group} vs {[g['gsm'] for g in geo]}")
        for p, s in zip(group, geo):
            prefix_to_sample[p] = s

    rows = []
    per_stage: Counter = Counter()
    for cell in cells:
        prefix = cell.rsplit("_", 1)[0]
        base = prefix[:-1] if prefix.endswith("b") else prefix
        stage_code, genotype, batch = base.split("_")
        stage, hpf = STAGE_INFO[stage_code]
        sample = prefix_to_sample[prefix]
        per_stage[stage] += 1
        rows.append({
            "cell_id": cell,
            "stage": stage,
            "hours_post_fertilization": hpf,
            "sequencing_batch": batch,
            "replicate": "b" if prefix.endswith("b") else "a",
            "genotype": "wild type" if genotype == "WT" else genotype,
            "geo_sample": sample["gsm"],
        })
    return rows, dict(per_stage)


def ss2_metadata() -> list[dict]:
    cells = read_header(RAW / "GSE106466" / "GSE106466_SS2_umi_cnts.txt.gz")
    cells = [c for c in cells if c]
    fpkm_cells = set(
        c for c in read_header(RAW / "GSE106466" / "GSE106466_SS2_fpkm.txt.gz")
        if c
    )
    samples = {s["title"]: s for s in parse_soft(SOFT / "GSE106466_family.soft.gz")}
    rows = []
    for cell in cells:
        s = samples[cell]
        rows.append({
            "cell_id": cell,
            "genotype": s["chars"].get("genotype", ""),
            "stage": s["chars"].get("developmental stage", ""),
            "library_batch": s["chars"].get("library preparation batch", ""),
            "geo_sample": s["gsm"],
            "protocol": "; ".join(s["description"]),
            "in_fpkm_matrix": cell in fpkm_cells,
        })
    return rows


def tenx_metadata() -> list[dict]:
    samples = {s["gsm"]: s for s in parse_soft(SOFT / "GSE106587_family.soft.gz")}
    rows = []
    for gsm, label in [("GSM3084402", "10xWT6S"), ("GSM3084403", "10xMZoep6S")]:
        barcodes = read_lines(
            RAW / "GSE106587" / f"{gsm}_{label}_barcodes.tsv.gz"
        )
        s = samples[gsm]
        for bc in barcodes:
            if not bc:
                continue
            rows.append({
                "barcode": bc,
                "geo_sample": gsm,
                "genotype": s["chars"].get("genotype/variation", ""),
                "stage": s["chars"].get("developmental stage", ""),
                "hours_post_fertilization": s["chars"].get(
                    "hours post-fertilization", ""
                ),
            })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    META.mkdir(parents=True, exist_ok=True)

    drop, _per_stage = dropseq_metadata()
    ss2 = ss2_metadata()
    tenx = tenx_metadata()

    write_csv(META / "dropseq_cell_metadata.csv", drop)
    write_csv(META / "ss2_cell_metadata.csv", ss2)
    write_csv(META / "tenx_cell_metadata.csv", tenx)

    # Stage summary -------------------------------------------------------
    stage_counts: dict[tuple[str, str], int] = Counter()
    stage_samples: dict[tuple[str, str], set] = defaultdict(set)
    for r in drop:
        key = ("Drop-seq", r["stage"])
        stage_counts[key] += 1
        stage_samples[key].add(r["geo_sample"])
    for r in ss2:
        key = ("SMART-seq2", r["stage"])
        stage_counts[key] += 1
        stage_samples[key].add(r["geo_sample"])
    for r in tenx:
        key = ("10x Genomics", r["stage"])
        stage_counts[key] += 1
        stage_samples[key].add(r["geo_sample"])

    hpf = {stage: h for stage, h in STAGE_INFO.values()}
    hpf["50% epiboly"] = 5.3
    hpf["6-somite"] = 12.0
    with (META / "stage_summary.tsv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["dataset", "stage", "hours_post_fertilization",
                    "n_cells_or_barcodes", "n_geo_samples"])
        for dataset in ["Drop-seq", "SMART-seq2", "10x Genomics"]:
            stages = [s for s in STAGE_ORDER if (dataset, s) in stage_counts]
            for stage in stages:
                key = (dataset, stage)
                w.writerow([dataset, stage, hpf[stage],
                            stage_counts[key], len(stage_samples[key])])

    # Inventory -----------------------------------------------------------
    inventory = [
        ("GSE106474", "Drop-seq, wild type, 12 stages", "UMI counts (genes x cells)",
         "GSE106474/GSE106474_UMICounts.txt.gz", "23,974 genes", f"{len(drop):,} cells",
         "Main atlas. Cell names encode stage, batch and replicate."),
        ("GSE106466", "SMART-seq2, WT + MZoep, 50% epiboly", "UMI counts (genes x cells)",
         "GSE106466/GSE106466_SS2_umi_cnts.txt.gz", "19,205 genes", f"{len(ss2):,} cells",
         "Full-length, 5' enriched. 364 MZoep + 52 WT cells."),
        ("GSE106466", "SMART-seq2, WT + MZoep, 50% epiboly", "FPKM (genes x cells)",
         "GSE106466/GSE106466_SS2_fpkm.txt.gz", "25,208 genes",
         f"{sum(r['in_fpkm_matrix'] for r in ss2):,} cells",
         "FPKM values; MZoep p0 batch is absent (no FPKM for those 88 cells)."),
        ("GSE106587", "10x Genomics, WT 6-somite", "sparse MatrixMarket (genes x barcodes)",
         "GSE106587/GSM3084402_10xWT6S_matrix.mtx.gz", "30,585 genes", "3,000 barcodes",
         "Genes have Ensembl ID + symbol."),
        ("GSE106587", "10x Genomics, MZoep 6-somite", "sparse MatrixMarket (genes x barcodes)",
         "GSE106587/GSM3084403_10xMZoep6S_matrix.mtx.gz", "30,585 genes", "2,200 barcodes",
         "Genes have Ensembl ID + symbol."),
    ]
    with (META / "atlas_inventory.tsv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["series", "description", "matrix_type", "relative_path",
                    "features", "observations", "notes"])
        w.writerows(inventory)

    print(f"Drop-seq:   {len(drop):,} cells")
    print(f"SMART-seq2: {len(ss2):,} cells")
    print(f"10x:        {len(tenx):,} barcodes")


if __name__ == "__main__":
    main()
