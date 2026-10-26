# AGENTS.md

Notes for agents working in this repository. Keep this file up to date when you
add tools, data, or conventions.

## Project

Development of a **Boolean model of cell differentiation in zebrafish early
development (up to ~12 hpf)**. The main focus is on the differentiation
decisions around **6 hpf**, i.e. germ ring and shield formation and the
associated differentiation processes (hypoblast/epiblast, dorsal–ventral).
The first milestone is a clean, reproducible copy of the
GSE106587 single-cell atlas.

Repository root: `/home/xpastva/zebrafish-early`

## Current status (2026-10-22)

- Downloaded the **complete GEO SuperSeries GSE106587** and its two SubSeries.
- Downloaded GEO SOFT/series-matrix metadata and built cell-level metadata tables.
- Verified gzip integrity, SHA-256 checksums, matrix dimensions, and ID
  uniqueness (`data/metadata/validation_report.txt`: **PASS**).
- **No QC, filtering, normalisation, or modelling has been run yet.** The files
  in `data/raw/` are the original GEO processed matrices.
- No Python virtual environment was needed for this step; all scripts use only
  the Python 3.12 standard library.
- `scripts/01_dataset_structure.py` describes the Drop-seq design (stage →
  sequencing batch → library, 28 libraries; two batch blocks sharing no stage)
  and writes the figures/table of the report's "Dataset overview" section.
  It removes no cells. It needs pandas/numpy/matplotlib (see Environment).
- `scripts/02_replicate_mixing.py` tests whether the six a/b library pairs
  (same stage and batch) mix in a joint kNN graph, against a permutation null,
  a random split of one library (negative control), an adjacent stage
  (positive control) and, where available, another batch. It runs once with
  library-size normalisation only and once after binomial thinning of the
  deeper library, and again after removing the top a/b-differing genes.
  Outputs: `results/tables/replicate_mixing*.tsv`, `results/figures/`
  (histograms, UMAPs) and `report/figures/replicate_mixing.pdf`.
  Result: no pair mixes (AUC 0.83-1.00), so a/b libraries are kept separate.
- `scripts/03_replicate_differences.py` collects descriptive gene-level
  evidence for why a/b libraries differ (pseudobulk fold changes, on/off
  genes with depth-matched detection, ribosomal fraction and stage-marker
  timing indicators) and writes `report/tables/replicate_{summary,genes}.tex`.
- No API keys or authentication were required (GEO is public).

## Dataset: GSE106587

**Farrell JA, Wang Y, Riesenfeld SJ, Shekhar K, Regev A, Schier AF (2018).
"Single-cell reconstruction of developmental trajectories during zebrafish
embryogenesis." Science 360:eaar3131. PMID: 29700225.**
DOI: https://doi.org/10.1126/science.aar3131

GSE106587 is a **SuperSeries**; most of the atlas is in the SubSeries:

| Series | Assay | Coverage | Matrix | Observations | Features |
|---|---|---|---|---|---|
| **GSE106474** | Drop-seq, wild type | high → 6-somite, 12 stages | `GSE106474_UMICounts.txt.gz` (genes × cells, TSV) | **39,505 cells** | 23,974 gene symbols |
| **GSE106466** | SMART-seq2, WT + MZoep | 50% epiboly only | `GSE106466_SS2_umi_cnts.txt.gz` (genes × cells) | **416 cells** (52 WT, 364 MZoep) | 19,205 genes |
| **GSE106466** | SMART-seq2, FPKM | 50% epiboly only | `GSE106466_SS2_fpkm.txt.gz` (genes × cells) | 328 cells (MZoep p0 batch absent) | 25,208 genes |
| **GSE106587** | 10x Genomics, WT | 6-somite | MatrixMarket (genes × barcodes) | 3,000 barcodes | 30,585 genes |
| **GSE106587** | 10x Genomics, MZoep | 6-somite | MatrixMarket (genes × barcodes) | 2,200 barcodes | 30,585 genes |

Important: the paper reports **38,731 cells**, while the downloaded Drop-seq
matrix contains **39,505 cell columns**. The difference is almost certainly
post-download QC/filtering in the paper; no cells have been removed here.

Only the **GEO processed matrices** were downloaded. Raw sequencing reads
(FASTQ/SRA, BioProject **PRJNA417291**) were **not** downloaded; those are much
larger and would require SRA tools. Ask the user before pulling raw reads.

### Stage coverage

Hours post-fertilisation (hpf) are the values supplied by GEO.

| Stage | hpf | Drop-seq cells | Drop-seq samples |
|---|---:|---:|---:|
| high | 3.3 | 311 | 2 |
| oblong | 3.8 | 200 | 2 |
| dome | 4.3 | 1,174 | 1 |
| 30% epiboly | 4.7 | 1,506 | 2 |
| 50% epiboly | 5.3 | 5,864 | 4 |
| shield | 6.0 | 1,046 | 1 |
| 60% epiboly | 7.0 | 4,201 | 3 |
| 75% epiboly | 8.0 | 6,324 | 3 |
| 90% epiboly | 9.0 | 5,626 | 3 |
| bud | 10.0 | 7,162 | 4 |
| 3-somite | 11.0 | 1,631 | 1 |
| 6-somite | 12.0 | 4,460 | 2 |

**Coverage of the ~12 hpf window:** GSE106587 spans high (3.3 hpf) to
6-somite (12 hpf), so the whole window from high onward is covered, with the
germ ring/shield focus (50% epiboly → shield → 60% epiboly, 5.3–7 hpf) well
sampled. Cleavage/early blastula stages (**<3.3 hpf) are not represented**;
pre-MBT maternal decisions (EVL/YSL, germ plasm, dorsal determinants) can only
be observed through their consequences. If those stages must be modelled, an
additional dataset will be required.

Note that the Drop-seq sampling is uneven across stages (e.g. only 311 high-stage
cells but 5,864 at 50% epiboly). Account for this in any stage-balanced model.

## Directory layout

```
data/
  raw/GSE106587/                 # original GEO files; gitignored, do not edit
    GSE106474/GSE106474_UMICounts.txt.gz
    GSE106466/GSE106466_SS2_umi_cnts.txt.gz
    GSE106466/GSE106466_SS2_fpkm.txt.gz
    GSE106587/GSE106587_RAW.tar
    GSE106587/GSM3084402_10xWT6S_*   # extracted 10x WT 6-somite
    GSE106587/GSM3084403_10xMZoep6S_* # extracted 10x MZoep 6-somite
    SHA256SUMS
  metadata/
    dropseq_cell_metadata.csv     # 39,505 cells
    ss2_cell_metadata.csv         # 416 cells
    tenx_cell_metadata.csv        # 5,200 barcodes
    stage_summary.tsv             # cells per dataset × stage
    atlas_inventory.tsv           # file/feature/observation inventory
    validation_report.txt         # PASS report from validate_data.py
    soft/                         # GEO SOFT family files (sample metadata)
    series_matrix/                # GEO series-matrix files
    filelist_GSE106587.txt        # GEO file listing/provenance
  processed/                      # for derived AnnData etc. (gitignored)
    dropseq_counts.h5ad           # sparse Drop-seq counts + metadata, built by
                                  # pipeline_utils.load_dropseq_counts()
scripts/
  download_gse106587.sh           # idempotent download + extract + checksums
  build_metadata.py               # SOFT → cell metadata tables (stdlib)
  validate_data.py                # integrity checks (stdlib)
  pipeline_utils.py               # shared paths, stage/batch constants, plot style
  01_dataset_structure.py         # stage x batch x library overview -> report
  02_replicate_mixing.py          # a/b library mixing (kNN, kBET, LISI, AUC)
  03_replicate_differences.py     # a/b gene-level differences, timing indicators
results/
  tables/                         # derived tables (TSV/CSV), versioned
    cell_qc_full_atlas.csv        # per-cell UMIs/genes, cache for 01 (delete to
                                  # recompute from the raw matrix)
  figures/                        # exploratory figures not (yet) in the report
report/
  main.tex                        # LaTeX report on the analysis
  references.bib                  # BibTeX entries (Farrell et al. 2018 included)
  figures/                        # figures included by main.tex (some generated)
  tables/                         # generated LaTeX tables (do not edit by hand)
  latexmkrc                       # build config; output goes to report/build/
```

Build the report with `cd report && latexmk main.tex` (TeX Live; PDF at
`report/build/main.pdf`, gitignored).

`data/raw/` is gitignored because of size. Keep derived objects in
`data/processed/`.

## Reproduce the download and metadata

```bash
cd /home/xpastva/zebrafish-early

# Download (~100 MB compressed), verify gzip, extract the 10x tar,
# and write data/raw/GSE106587/SHA256SUMS.
bash scripts/download_gse106587.sh

# Verify the raw files against the recorded checksums.
(cd data/raw/GSE106587 && sha256sum -c SHA256SUMS)

# Rebuild cell-level metadata tables from the GEO SOFT files.
python3 scripts/build_metadata.py

# Sanity-check dimensions and IDs; writes validation_report.txt.
python3 scripts/validate_data.py
```

The download script is safe to re-run: complete files are skipped. If a file
was partially downloaded, remove it and re-run.

Source URLs (public, no auth):

- `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE106nnn/GSE106474/suppl/`
- `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE106nnn/GSE106466/suppl/`
- `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE106nnn/GSE106587/suppl/`
- SOFT: `.../GSE106474/soft/GSE106474_family.soft.gz`, etc.

## Environment

This step deliberately uses only system tools:

| Tool | Version | Used for |
|---|---|---|
| Python | 3.12.3 (`/usr/bin/python3`) | metadata parsing, validation |
| curl | 8.5.0 | downloads |
| gzip | 1.12 | integrity checks |
| tar | GNU tar 1.35 | unpacking 10x archive |
| git | 2.43.0 | repository |

No `.venv` exists. No packages were installed by this step. The scripts use only
`csv`, `gzip`, `collections`, `pathlib`, and `sys`.

The **system** Python already provides some scientific packages (checked
2026-10-22): `numpy 2.3.5`, `pandas 3.0.5`, `scipy 1.18.0`. They were not used
for the download/metadata scripts and should not be assumed sufficient for the
analysis. `anndata`, `scanpy`, and `h5py` are **not installed**.

If later work needs single-cell tooling, create a project-local environment and
document the exact versions here:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install scanpy anndata numpy pandas scipy h5py
```

(That stack is **not installed yet**; do not assume it is available.) The
machine has 30 CPUs and ~125 GB RAM, and 4.6 TB free on `/home/xpastva`.

On the local macOS checkout, a `.venv` (Python 3.13) exists with the pinned
stack from the `initial-exploration` branch (pandas 3.0.6, numpy 2.5.4,
matplotlib 3.11.2, scanpy 1.12.4, ...). Analysis scripts run as
`.venv/bin/python scripts/01_dataset_structure.py` from the repository root.

## How to load the data

### Drop-seq (main atlas)

The matrix is **genes in rows, cells in columns**, tab-separated, with the first
column holding gene symbols:

```python
import pandas as pd
import scanpy as sc

path = "data/raw/GSE106587/GSE106474/GSE106474_UMICounts.txt.gz"
counts = pd.read_csv(path, sep="\t", index_col=0)  # genes × cells
adata = sc.AnnData(counts.T)                       # cells × genes
adata.var_names = counts.index
adata.obs_names = counts.columns

meta = pd.read_csv("data/metadata/dropseq_cell_metadata.csv", index_col="cell_id")
adata.obs = meta.loc[adata.obs_names].copy()
```

Expect roughly 39,505 × 23,974 mostly-zero values. Reading the dense TSV can
need several GB of RAM; convert to sparse CSR/AnnData once and store the result
in `data/processed/`.

### SMART-seq2

Same orientation (genes × cells). Use UMI counts for most analyses; FPKM is a
normalised alternative. The 88 `oep_p0_*` cells have UMI counts but are absent
from the FPKM matrix (`in_fpkm_matrix` column in the metadata).

### 10x Genomics

Two independent MatrixMarket datasets. `genes.tsv.gz` has **two columns**
(Ensembl gene ID and gene symbol); the `.mtx.gz` has a `%` comment line before
the dimensions. Barcode lists are **all barcodes in the matrix**, not
QC-filtered cells. Example with SciPy:

```python
import gzip
import scipy.io

base = "data/raw/GSE106587/GSE106587/GSM3084402_10xWT6S"
with gzip.open(base + "_matrix.mtx.gz", "rb") as fh:
    X = scipy.io.mmread(fh).tocsr()   # genes × barcodes (30,585 × 3,000)
```

`mmread` skips the `%` comment line. `scanpy.read_10x_mtx` expects the standard
names `matrix.mtx.gz`, `barcodes.tsv.gz`, and `features.tsv.gz`; the GEO files
use a GSM prefix and `genes.tsv.gz`, so rename/symlink them (or read manually)
before using it.

## Metadata schemas

`dropseq_cell_metadata.csv`

| Column | Meaning |
|---|---|
| `cell_id` | original matrix column name, e.g. `ZF50_WT_DS4b_ACGT...` |
| `stage` | high, oblong, dome, 30% epiboly, … |
| `hours_post_fertilization` | GEO hpf for the stage |
| `sequencing_batch` | DS2/DS3/DS4/DS5 |
| `replicate` | `a` or `b` (the `b` suffix in the cell-name prefix) |
| `genotype` | wild type |
| `geo_sample` | matching GSM accession |

`ss2_cell_metadata.csv`: `cell_id`, `genotype`, `stage`, `library_batch`,
`geo_sample`, `protocol`, `in_fpkm_matrix`.

`tenx_cell_metadata.csv`: `barcode`, `geo_sample`, `genotype`, `stage`,
`hours_post_fertilization`.

For Drop-seq, stage and sequencing batch are encoded exactly in the cell-name
prefix (`ZFHIGH_WT_DS5_...`). Prefix codes: `ZFHIGH`=high, `ZFOBLONG`=oblong,
`ZFDOME`=dome, `ZF30`=30% epiboly, `ZF50`=50% epiboly, `ZFS`=shield,
`ZF60`=60% epiboly, `ZF75`=75% epiboly, `ZF90`=90% epiboly, `ZFB`=bud,
`ZF3S`=3-somite, `ZF6S`=6-somite. The mapping of the `b` replicate to a specific
GSM is inferred from GEO replicate ordering; it does not affect stage assignment.
For SMART-seq2, `cell_id` equals the GEO sample title, so the GSM mapping is
exact.

## Caveats and gotchas

- **Raw counts only.** No cell/gene filtering, doublet removal, normalisation,
  batch correction, or cell-type labels have been applied.
- **Gene identifiers differ by assay.** Drop-seq uses gene symbols only.
  10x uses Ensembl IDs + symbols. SMART-seq2 uses symbols. Check orthology/
  symbol consistency before merging assays.
- **Cell-name parsing.** Drop-seq prefix format is
  `<stage>_<genotype>_<batch>[b]_<barcode>`; use the metadata table rather than
  re-parsing by hand.
- **10x barcodes ≠ cells.** 3,000 + 2,200 barcodes are provided; apply QC.
- **FPKM matrix is incomplete** (no `oep_p0` cells).
- **Stage sampling is imbalanced**, and the earliest stage is 3.3 hpf.
- **Do not edit `data/raw/`.** Re-download with the script if needed.
- **Do not commit raw or processed data**; see `.gitignore`.
- The paper's URD analysis and cell-type annotations are not included in these
  GEO files. If annotations/pseudotime are needed, look for the authors' URD
  objects/code (Farrell lab) or reproduce clustering/trajectory inference.

## Suggested next steps

1. Convert the Drop-seq matrix to a sparse AnnData object in `data/processed/`
   and record the conversion script.
2. QC-filter cells/genes, normalise, and inspect stage composition; consider
   downsampling per stage for a balanced Boolean model.
3. Use the full high → 6-somite range (≤12 hpf), but centre the initial model
   on the decisions around 6 hpf (50% epiboly → shield → 60% epiboly).
4. If stages before high (3.3 hpf) must be modelled, identify and download an
   additional early embryo dataset; GSE106587 cannot cover that interval.
5. Define binarisation thresholds and candidate regulator–target links before
   fitting the Boolean network.
