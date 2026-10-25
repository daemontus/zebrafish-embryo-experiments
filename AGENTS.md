# AGENTS.md

Notes for agents working in this repository. Keep this file up to date when you
add tools, data, or conventions.

## Project

Development of a **Boolean model of cell differentiation in zebrafish early
development (1–6 hpf)**. The first milestone is a clean, reproducible copy of the
GSE106587 single-cell atlas.

Repository root: `/home/xpastva/zebrafish-early`

## Current status (2026-10-23)

- Downloaded the **complete GEO SuperSeries GSE106587** and its two SubSeries.
- Downloaded GEO SOFT/series-matrix metadata and built cell-level metadata tables.
- Verified gzip integrity, SHA-256 checksums, matrix dimensions, and ID
  uniqueness (`data/metadata/validation_report.txt`: **PASS**).
- Built the **3.3–6.0 hpf Drop-seq subset**: 10,101 cells × 23,974 genes, all
  six stages from high through shield. Raw and processed AnnData objects are in
  `data/processed/`.
- Ran QC summaries and UMAP sanity checks (stages 01–04). QC is clean; the UMAP
  reveals strong **stage–batch confounding** (see caveats below).
- Stage 05 maps the **design structure of the full Drop-seq atlas** (stage →
  sequencing batch → library/GSM, 28 libraries) with per-library depth; see
  `results/05_dataset_structure.md`.
- Stage 06 clusters **each of the 12 window libraries separately** (Leiden
  r=0.5) with cluster QC. Recurring discrete populations: EVL, margin,
  endoderm, dorsal organizer, and a p53/"apoptotic-like" group. The ectoderm
  mass is over-split, and high/oblong libraries have no structure beyond EVL.
  See `results/06_library_clustering.md`. Clusters are not yet matched
  across libraries.
- Stage 07 merges related libraries without batch correction: DS5 across
  timepoints, and 50% epiboly across batches/replicates. Discrete
  populations merge across batches. The bulk ectoderm separates by library
  (DS2 most). DS5 is organised by time, with EVL and margin bridging stages.
  See `results/07_merged_contexts.md`.
- Stage 08 batch-corrects **only the 50% epiboly libraries**, with batch key
  = library. Harmony and scVI both remove the library split of the bulk
  ectoderm while keeping p53/endoderm/margin/EVL/prechordal intact.
  Harmony is the recommended default; scVI is sharper but varies by seed.
  See `results/08_batch_correction_50pct.md`.
- Stages 09–10 identify **gene programs** with consensus NMF per 50% epiboly
  library (K=9), keep those replicated in ≥3 libraries, and annotate them
  from ZFIN/GO enrichment. There are 8 replicated programs: EVL,
  ventrolateral margin, dorsal margin/shield, endoderm, p53, presumptive
  ectoderm, an unnamed early-declining program, and **primordial germ cells**
  (3–5 cells/library, never found by clustering). See
  `results/10_gene_programs.md`.
- Stage 11 scans K = 5–20. The replicated set **saturates at 10 programs**
  (K ≥ 13): the 8 above plus proliferation (histone/MKI67) and a
  NOTO/ADMP/CHD dorsal-organizer program. Forerunner-like cells are found
  only in DS4 at every K (candidate cells exist in all libraries). See
  `results/11_program_k_scan.md`.
- Stage 12 runs the program analysis on DS5. There are 11 programs recurrent
  across DS5 units, and every one matches a 50% identity. All 10 replicated
  50% programs reappear, appearing in developmental order (EVL/ectoderm
  first, ventrolateral mesoderm last). The DS4-only forerunner-like program
  reappears in DS5 shield (FOXJ1A rank 1; 18/50 genes shared), now 2
  libraries across 2 batches. See `results/12_ds5_programs.md`.
- Stage 13 scores **signalling-pathway proxies** (readout genes from a
  user-supplied resource, `scripts/pathway_proxies.tsv`, resolved via ZFIN).
  Nodal/BMP/Wnt/FGF/Notch readouts are coherent (calibrated false-positive
  rate 1/50); Hh and VEGF are undetectable (negative controls). The
  program × pathway state table is in `results/13_pathway_proxies.md`.
  Notch → ventrolateral mesoderm is not separable from that program's own
  genes.
- **No Boolean model has been fitted yet.**
- Created a project-local `.venv` with scanpy/anndata and documented exact
  versions in `requirements.lock.txt`.
- No API keys or authentication are required (GEO is public).

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

**Coverage gap for the 1–6 hpf goal:** the earliest stage in GSE106587 is
**high (3.3 hpf)**. Cleavage/blastula stages from **1–3 hpf are not represented**
in this atlas. If the Boolean model must start at 1 hpf, an additional dataset
(or new data) covering 1–3 hpf will be required. Within 1–6 hpf, GSE106587
covers high, oblong, dome, 30% epiboly, 50% epiboly, and shield (3.3–6.0 hpf);
60% epiboly (7 hpf) is just outside the requested window.

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
  processed/                      # derived AnnData objects (gitignored)
    dropseq_3.3-6.0hpf_raw.h5ad
    dropseq_3.3-6.0hpf_qc.h5ad
    dropseq_3.3-6.0hpf_umap.h5ad
    dropseq_3.3-6.0hpf_umap_harmony.h5ad
    libraries/<library>.h5ad      # stage 06, one per library (e.g. ZF50_WT_DS4b)
    07_ds5_joint.h5ad, 07_50pct_joint.h5ad  # stage 07 merged contexts
    08_50pct_corrected.h5ad       # stage 08: X_harmony, X_scvi_s{0,1,2}, joint_* labels
    09_cnmf/<library>[_ext]/      # cNMF working directories (K 5-15; _ext = K 16-20)
    09_programs/                  # consensus programs for every K (results/ keeps K=9)
    12_cnmf/, 12_programs/        # DS5 cNMF runs / programs for every K (results/ keeps K=13)
  reference/                      # ZFIN/GO downloads (gitignored), MANIFEST.tsv
results/
  figures/                        # 02_*, 03_*, 04_* QC/UMAP figures
  tables/                         # stage-wise QC, UMAP coords, cluster tables
  05_dataset_structure.md         # human-readable stage/batch/library overview
  06_library_clustering.md        # per-library clustering results + interpretation
  figures/06_libraries/           # one UMAP/QC panel per library
  07_merged_contexts.md           # merged-context results + interpretation
  08_batch_correction_50pct.md    # batch-correction comparison + verdict
  10_gene_programs.md             # replicated gene programs + interpretation
  11_program_k_scan.md            # how many replicated programs (K scan)
  12_ds5_programs.md              # DS5 programs and their 50% counterparts
  13_pathway_proxies.md           # pathway proxies: validation, calls, limits
scripts/
  download_gse106587.sh           # idempotent download + extract + checksums
  build_metadata.py               # SOFT → cell metadata tables (stdlib)
  validate_data.py                # integrity checks (stdlib)
  00_setup_env.sh                 # create .venv and install pinned requirements
  01_build_dropseq_subset.py      # 3.3–6.0 hpf subset → AnnData
  02_qc_summary.py                # QC metrics, tables, histograms
  03_umap_sanity.py               # normalise, PCA, UMAP, Leiden (uncorrected)
  04_umap_harmony.py              # diagnostic Harmony-corrected UMAP
  05_dataset_structure.py         # full-atlas stage x batch x library overview
  06_library_clustering.py        # per-library Leiden/UMAP + cluster QC
  07_merged_contexts.py           # DS5-across-time / 50%-across-batch joint embeddings
  08_batch_correction_50pct.py    # Harmony vs scVI on the 50% epiboly libraries
  09_gene_programs_cnmf.py        # cNMF programs per 50% epiboly library
  10_program_annotation.py        # replication, ZFIN/GO annotation, DS5 projection
  11_program_k_scan.py            # replication/annotation at every K, identities across K
  12_ds5_programs.py              # cNMF on DS5 units, recurrence, match to 50% identities
  13_pathway_proxies.py           # pathway proxy validation, program association, state table
  pathway_tools.py                # ZFIN symbol/alias resolution + GO roles (13)
  pathway_proxies.tsv             # resource gene lists (verbatim), the stage-13 input
  program_tools.py                # shared program matching + enrichment helpers (10-12)
  cnmf_tools.py                   # shared cNMF driver (09, 12)
  download_references.sh          # ZFIN expression/stages, GO GAF/obo, ZFA obo
  cluster_compare.py              # shared kNN-mapping/composition helpers (07, 08)
  run_pipeline.sh                 # run stages 01–04 in order
  pipeline_utils.py               # shared paths/constants/plot helpers
README.md                         # human-oriented pipeline instructions
requirements.txt                  # top-level pins + rationale
requirements.lock.txt             # exact full environment (pip freeze)
```

`data/raw/` and `data/processed/` are gitignored because of size. Keep derived
objects in `data/processed/`; small QC/UMAP tables and figures in `results/`.

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

### Run the analysis pipeline

```bash
# One-time environment setup (creates .venv, installs exact versions).
bash scripts/00_setup_env.sh

# Run stages 01-13. Existing stage-01/03/04 outputs are reused.
bash scripts/run_pipeline.sh

# Recompute everything from the raw matrix (~10 min).
bash scripts/run_pipeline.sh --force
```

Stage 01 reads the full 39,505-cell matrix and needs roughly 8–10 GB RAM; the
compressed TSV is ~1 GB and pandas parsing takes ~6.5 min on this machine.
Stages 02–04 are much faster. See `README.md` for the stage table and outputs.

Source URLs (public, no auth):

- `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE106nnn/GSE106474/suppl/`
- `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE106nnn/GSE106466/suppl/`
- `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE106nnn/GSE106587/suppl/`
- SOFT: `.../GSE106474/soft/GSE106474_family.soft.gz`, etc.

## Environment

System tools used for the download and metadata step:

| Tool | Version | Used for |
|---|---|---|
| Python | 3.12.3 (`/usr/bin/python3`) | metadata parsing, validation, pipeline |
| curl | 8.5.0 | downloads |
| gzip | 1.12 | integrity checks |
| tar | GNU tar 1.35 | unpacking 10x archive |
| git | 2.43.0 | repository |

The analysis pipeline uses a **project-local `.venv`** created with
`python3 -m venv .venv` (not `--system-site-packages`). Install/reinstall with:

```bash
bash scripts/00_setup_env.sh
```

Exact versions are frozen in `requirements.lock.txt`; top-level rationale is in
`requirements.txt`. Key packages:

| Package | Version | Why |
|---|---|---|
| scanpy | 1.12.4 | single-cell QC, PCA, neighbours, UMAP, Leiden |
| anndata | 0.13.4 | `.h5ad` containers |
| numpy | 2.5.4 | arrays |
| pandas | 3.0.6 | metadata tables |
| scipy | 1.18.1 | sparse matrices |
| matplotlib | 3.11.2 | QC/UMAP figures |
| seaborn | 0.13.2 | violin plots |
| h5py | 3.16.0 | AnnData I/O |
| scikit-learn | 1.9.1 | PCA/UMAP backends, silhouette scores |
| numba / llvmlite | 0.68.0 / 0.50.0 | UMAP acceleration |
| umap-learn / pynndescent | 0.5.12 / 0.6.0 | UMAP |
| leidenalg / igraph | 0.12.0 / 1.0.0 | Leiden clustering |
| harmonypy | 2.1.0 | batch correction (stage 04 diagnostic, stage 08) |
| scvi-tools / torch | 1.5.1 / 2.14.1 | scVI batch correction (stage 08, CPU) |
| scikit-misc | 0.5.3 | `seurat_v3` HVG selection on raw counts |
| cnmf | 1.7.1 | consensus NMF gene programs (stage 09) |

**harmonypy compatibility note:** scanpy 1.12's
`sc.external.pp.harmony_integrate` wrapper assumes harmonypy's old `Z_corr`
orientation. harmonypy 2.1.0 returns `(n_cells, n_pcs)`, so stage 04 calls
`harmonypy.run_harmony()` directly and normalises the orientation. Do not
replace it with the scanpy wrapper without checking this.

The machine has 30 CPUs and ~125 GB RAM, and 4.6 TB free on `/home/xpastva`.
`requirements.lock.txt` was generated with `pip freeze` on 2026-10-22.

## How to load the data

### Processed subset (recommended starting point)

```python
import scanpy as sc

adata = sc.read_h5ad("data/processed/dropseq_3.3-6.0hpf_umap.h5ad")
print(adata)                    # 10,101 cells x 18,893 genes (post min_cells filter)
adata.obs.head()                # stage, hpf, batch, QC metrics, Leiden
adata.obsm["X_umap"]            # UMAP coordinates
adata.layers["counts"]          # raw UMI counts (sparse)
adata.layers["lognorm"]         # log1p(CP10k) values
adata.X                         # log-normalised values (same as lognorm)
```

For raw, unfiltered counts of the same cells use
`data/processed/dropseq_3.3-6.0hpf_qc.h5ad` (10,101 × 23,974; `X` is raw
counts). Stage 03/04 objects store the full log-normalised matrix in `X`, raw
counts in `layers["counts"]`, and UMAP/PCA in `obsm`.

### Raw Drop-seq atlas

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

- **Strong stage–batch confounding in the 3.3–6.0 hpf subset.** All
  high/oblong/dome/30%-epiboly/shield cells are DS5; all 50%-epiboly cells are
  DS2–DS4. At the DS5 vs DS2–DS4 level, stage and batch are perfectly
  confounded (no stage appears in both groups), so their effects are not
  separately identifiable from this subset alone. The uncorrected UMAP
  therefore separates DS5 from DS2–DS4 rather than forming a clean
  developmental trajectory. Harmony reduces batch structure but also erases
  stage structure and creates disconnected islands (stage silhouette
  0.032 → 0.016; batch silhouette 0.015 → −0.032). Treat stage 04 as a
  diagnostic only; do not use it for biological interpretation.
- **The confounding extends to the whole atlas.** DS5 holds high, oblong,
  dome, 30% epiboly, shield, 3-somite and 6-somite. DS2–DS4 hold 50% epiboly
  to bud, and each of those stages is in all three batches. No stage is shared
  between DS5 and DS2–DS4, so adding later stages cannot separate that batch
  effect from stage. Within DS2–DS4 the design is balanced, so DS2/DS3/DS4
  effects can be estimated there.
- **Library is the finest technical unit.** 28 GEO samples = 28 libraries.
  The `b` replicate means a second library in the same stage and batch (high,
  oblong, 30% epiboly, 50% epiboly DS4, bud DS2, 6-somite). GEO's "Rep N"
  title numbering counts libraries across batches, so it is *not* the same
  as `replicate`. Within-batch libraries can differ more than 3× in median
  UMIs (oblong a vs b). Dome and shield are single, unreplicated libraries;
  high DS5b (48 cells) and oblong DS5a (72) are tiny.
- **Per-library cluster IDs are local.** `leiden` in
  `data/processed/libraries/*.h5ad` and `06_cell_clusters.csv` is numbered
  per library; there is no cross-library correspondence yet. Library names
  come from `pipeline_utils.library_of()` (the cell-ID prefix).
- **kNN mapping across timepoints is hub-dominated.** In stage 07, nearest
  neighbours always exist, so cross-stage "best matches" often point to
  whichever cluster is closest (hubs), not to the same cell type. Use
  library-centred centroid correlation, or a distance-aware matching, for
  cross-stage links.
- **Batch correction is only valid within 50% epiboly.** Stage 08's population
  labels (`POPULATIONS` in the script) are hand-assigned from the stage-07
  analysis; update them if stage 06 is re-run with different parameters
  (local cluster IDs would change). scVI is trained on CPU
  (`accelerator="cpu"`) for reproducibility, even though MPS is available.
- **Program interpretation must come from the reference databases.** Stage
  10 labels are derived from ZFIN/GO enrichment tables. Do not relabel
  programs from remembered markers. If a label is changed, cite the
  enrichment row that supports it. ZFIN/GO files change over time; record
  the `MANIFEST.tsv` date when quoting q-values.
- **Identity grouping across K can conflate nested programs.** Stage 11
  groups programs at r ≥ 0.6, so a specific program (e.g. PGC) can be
  joined with a broad program that contains its genes at low K. Check
  per-K membership in `11_k_scan_programs.tsv` before quoting "replicated
  from K = …".
- **Gene symbols: always resolve through ZFIN.** The Drop-seq matrix uses
  2015-era ZFIN symbols, with no Ensembl IDs. Use `pathway_tools.resolve()`
  (current symbol → alias → full gene name → a/b paralogs). Never accept an
  alias that is another gene's current symbol (e.g. `WNT11` is now a
  different gene from `wnt11f2`). Unresolved symbols are reported with
  suggestions, not substituted.
- **Pathway calls are relative and circularity-checked.** "high"/"low" in
  stage 13 means relative to other cells of the same library. Readout genes
  that are a program's own top genes are left out and the call re-tested.
- **Cross-context program matching uses top-gene overlap, not r.** Between
  DS5 and 50% epiboly, r over all genes is depressed by batch/stage noise
  (r ≈ 0.3–0.5 for clearly matching programs). Stage 12 requires >= 10/50
  shared top genes (q < 0.001) and counts each DS5 program only toward its
  best partner's identity; counting all pairs credits nested high-K
  sub-splits.
- **cNMF K is common across libraries** (worst-library stability, K ≥ 7).
  A per-library "best K" rule was tried and rejected: stability is flat for
  some libraries, which made the rule pick K=5 for DS2 and K=15 for
  DS4/DS4b.
- **Scrublet needs scikit-image only for automatic thresholds.** Stage 06
  passes a fixed `threshold`, uses only the continuous score, and adds no
  dependency.
- **QC state.** Stage 02 computes metrics but removes no cells. Stage 03 keeps
  all 10,101 cells (`min_genes=200`, `min_counts=500` are satisfied by every
  cell) and drops genes detected in <3 cells (18,893 remain). No doublet
  removal, no cell-type labels, no final Boolean-model preprocessing.
- **No mitochondrial genes** in the GEO Drop-seq feature list; `pct_counts_mt`
  cannot be computed. Ribosomal fraction (`RPL*`/`RPS*`, 94 genes) is provided
  as a technical proxy and is bimodal / stage-dependent.
- **Raw counts are preserved** in `layers["counts"]` in stage 03/04 objects;
  `X` and `layers["lognorm"]` are log1p(CP10k). In stage 01/02 objects `X` is
  raw UMI counts.
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

1. Decide how to handle stage–batch confounding before trajectory/Boolean
   inference (model stage explicitly, include batch as a covariate, or use a
   design that acknowledges DS5 vs DS2–DS4).
2. Define Boolean states from marker/regulator genes and candidate
   regulator–target links; use `layers["counts"]` for binarisation decisions.
3. If 1–3 hpf must be modelled, identify and download an additional early
   embryo dataset; GSE106587 cannot cover that interval.
4. Validate inferred networks against held-out stages and published zebrafish
   developmental annotations.
5. Keep the numbered pipeline convention: add new stages as
   `scripts/05_*.py`, document inputs/outputs in `README.md`, and update this
   file when the environment or data change.
