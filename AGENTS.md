# AGENTS.md

Notes for agents working in this repository. Keep this file up to date when you
add scripts, data, or conventions. Keep it about *how to work here*: results
belong in the report, history in git.

## Project

A **Boolean model of cell differentiation in early zebrafish development
(up to ~12 hpf)**, focused on the decisions around **6 hpf** (germ ring and
shield: hypoblast/epiblast, dorsal–ventral). The data is the single-cell atlas
of Farrell et al. 2018 (GEO GSE106587). The analysis so far: calibrated
heterogeneous genes (replacing HVGs), library alignment, expression programs
and replicated cell populations, written up in `report/main.tex`.

Checkouts: the server (`/home/xpastva/zebrafish-early`; 30 CPUs, ~125 GB RAM)
and a local macOS checkout.

## Rules

- **Do not edit `data/raw/`**; re-download with the scripts.
- **Do not commit raw or processed data** (`data/raw/`, `data/processed/` and
  `paper/` are gitignored). `results/tables`, `results/figures` and
  `report/` (including `report/main.pdf`) are versioned.
- **Ask before downloading raw reads** (FASTQ/SRA, BioProject PRJNA417291).
- Commit only when the user asks.
- `report/tables/*.tex` and generated report figures are written by the
  scripts; do not edit them by hand.
- Detail figures and tables go to the report's appendix ("Supplementary
  figures and tables", after the references, numbered S1, S2, ...), not the
  main text. Output figures can be included straight from `results/figures/`
  (`\graphicspath` includes it).
- Bibliography entries: check against Crossref; normalise author names to
  NFC (Crossref returns combining accents that pdflatex rejects).

## Environment

Local macOS: `.venv` (Python 3.13; pandas 3.0.6, numpy 2.5.4, matplotlib
3.11.2, scanpy 1.12.4, openpyxl 3.1.5, ...). Run scripts from the repository
root: `.venv/bin/python scripts/17_program_tests.py`. Scripts import each
other via `importlib.import_module("14_replicated_populations")`; to import
them elsewhere, put `scripts/` on `PYTHONPATH`.

The download/metadata scripts (`download_*.sh`, `build_metadata.py`,
`validate_data.py`) need only the Python standard library, curl, gzip and tar.

Build the report: `cd report && latexmk main.tex` (TeX Live; PDF in
`report/build/main.pdf`), then copy it to `report/main.pdf`, which is
committed so readers don't have to compile.

## Pipeline

Run order and dependencies (each script's docstring lists its outputs; tables
go to `results/tables/`, figures to `results/figures/` or `report/figures/`,
caches to `data/processed/`). Times are approximate, on the local machine.

| Script | Does | Needs | Time | Modes / caches |
|---|---|---|---|---|
| `01_dataset_structure.py` | stage × batch × library design, depth; report "Dataset overview" | counts | min | |
| `01b_heterogeneous_genes.py` | calibrated heterogeneous genes per library and stage (tiers weak/moderate/strong) | counts | ~25 min | `plot` (cache `heterogeneity_models.pkl`) |
| `01c_heterogeneity_zfin.py` | heterogeneous genes vs ZFIN in situ records | 01b, ZFIN | min | |
| `01d_hvg_flavours.py` | vs newer HVG methods (seurat_v3, Pearson residuals) | 01b, 01c | min | `plot` |
| `02_replicate_mixing.py` | do a/b libraries mix? (no) | 01b | min | |
| `03_replicate_differences.py` | why a/b libraries differ (genes, timing) | counts | min | |
| `14_replicated_populations.py` | per-library clusters replicated in another library; tracks across stages | 01b, 01c | few min | `[stages]`, `--seed`, `--tag`, `--tracks-only` |
| `15_alignment.py` | checks of the per-library alignment; UMAPs | 01b, 02 | ~20 min | `markers` (~2 min); cache `atlas_umap.tsv.gz` |
| `16_program_sets.py` | program sets → `programs.tsv` | 01b, 01c, ZFIN, `data/curated`, Farrell S2 | min | |
| `17_program_tests.py` | coherence/support and prediction tests of the sets | 16 | 30–90 min | `summarise` |
| `18_program_populations.py` | populations in program space vs 14 | 14, 17 | few min | |
| `19_program_atlas.py` | per-program UMAPs (exploration) | 14, 15, 17, 18 | min | `plot`, `atlas`, `combined`; caches `program_{scores,active}.tsv.gz` |
| `20_program_views.py` | dot plots, co-activity, tracks, genes (exploration) | 14, 19 | min | |
| `21_signal_coupling.py` | signalling readouts vs programs | 17, 18 | min | `plot` |
| `22_program_levels.py` | discretised program levels in the prediction test | 17 | ~45 min | `summarise` |
| `23_signal_atlas.py` | readouts vs expected programs on the atlas (exploration) | 17, 18, 19, 21 | few min | `plot`; caches `signal_atlas_*` |

Shared modules:
- `pipeline_utils.py`: paths, stages and batches, plot style,
  `load_dropseq_counts()` (builds/reads `data/processed/dropseq_counts.h5ad`),
  `nb_overdispersion` (per-gene NB dispersion vs the library's
  dispersion-mean trend phi_t(m) = a0 + a1 m^-b) and `pearson_residuals`
  (per-library residuals under that model, clipped at ±10, optionally
  centred = the alignment).
- `program_utils.py`: kNN smoothing, the held-out prediction test (floor /
  feature sets / ceiling) and the Farrell et al. Table S2 loader (all 30
  genes per module; reads `paper/aar3131_farrell_table-s2.xlsx`, not
  versioned).
- `zfin_utils.py`: ZFIN records, stage hpf, symbol matching
  (`zfin_records(max_year)`; ZFIN direct submissions get their year from the
  publication id).

Report sections: 3 dataset (01), 4 heterogeneous genes (01b–01d),
5 a/b libraries (02, 03), 6 alignment (15), 7 programs (16, 17, 21, 22),
8 populations (14, 18). 19, 20 and 23 are exploration only.

## Conventions and conclusions that affect downstream work

- **Heterogeneous genes replace HVGs everywhere.** "Heterogeneous" = weak tier
  (estimated FDR 50%); moderate/strong (10%/1%) where a reliable set is
  needed. A gene is in a tier at a stage if it is in >= half of the stage's
  libraries; genes with < 10 UMIs in a library are "not measured" there.
  `01b.heterogeneity_calls(counts, rng)` applies the calibration to any cells.
  Results depend on the simulation seed by a few % of calls.
- **Libraries are never pooled.** a/b libraries of the same stage and batch
  are distinguishable (02); quantitative work is done within a library.
  To compare cells across libraries use the alignment (per-library centred
  Pearson residuals, `pearson_residuals(..., centre=True)`); it removes all
  library averages, so use the original libraries for pseudobulk, library
  levels and composition. Stages ~1 h apart blur after alignment.
- **Program sets** (16): `farrell`, `zfin_pre`/`zfin_all` (+ `_het`),
  `signalling` (draft, `data/curated/signalling_programs.tsv`), `states`
  (six cell states from the archived literature table), `curated_staged`
  (stage-specific gene lists) and **`curated`, the set used downstream**:
  ZFIN terms grouped by `data/curated/zfin_anatomy_programs.tsv` (column
  `program`; column `fixed_program` splits groups that change identity), one
  gene list per program for its whole stage window, genes heterogeneous at
  >= 1 window stage, ranked by mean tier, cap 30. Known issue: pooling over
  the window moves dlx3b/foxi1 from the non-neural ectoderm to the placodes.
- **Program score** = mean centred Pearson residual of its genes;
  **supported** = coherent (vs the 95% quantile of 500 random sets of
  heterogeneous genes) in >= half of the stage's libraries; **active** =
  smoothed score above the 99.9% quantile of a gene-wise permutation null
  in the library.
- **Populations** (14): robust = found at >= 2 Leiden resolutions;
  single-library stages (dome, shield, 3-somite) are matched against adjacent
  stages, which is weaker evidence.
- **Signalling readouts**: Hedgehog readouts are never heterogeneous; the RA
  readouts (cyp26a1, dhrs3) do not measure RA activity well. Readout genes are
  removed from other programs before comparing (21, 23).
- Removed on 2026-10-28 (never committed): scripts 04–13 (first program
  scoring, cNMF, early prediction tests). Backup in
  `data/processed/archive/old_program_analyses_2026-10-28.tar.gz`.

## Data

### GSE106587

**Farrell JA, Wang Y, Riesenfeld SJ, Shekhar K, Regev A, Schier AF (2018).
"Single-cell reconstruction of developmental trajectories during zebrafish
embryogenesis." Science 360:eaar3131.** https://doi.org/10.1126/science.aar3131

| Series | Assay | Coverage | Matrix | Observations | Features |
|---|---|---|---|---|---|
| **GSE106474** | Drop-seq, wild type | high → 6-somite, 12 stages | `GSE106474_UMICounts.txt.gz` (genes × cells, TSV) | **39,505 cells** | 23,974 gene symbols |
| **GSE106466** | SMART-seq2, WT + MZoep | 50% epiboly only | `GSE106466_SS2_umi_cnts.txt.gz` (genes × cells) | **416 cells** (52 WT, 364 MZoep) | 19,205 genes |
| **GSE106466** | SMART-seq2, FPKM | 50% epiboly only | `GSE106466_SS2_fpkm.txt.gz` (genes × cells) | 328 cells (MZoep p0 batch absent) | 25,208 genes |
| **GSE106587** | 10x Genomics, WT | 6-somite | MatrixMarket (genes × barcodes) | 3,000 barcodes | 30,585 genes |
| **GSE106587** | 10x Genomics, MZoep | 6-somite | MatrixMarket (genes × barcodes) | 2,200 barcodes | 30,585 genes |

The Drop-seq atlas is the main data; SS2 and 10x are not used yet. The paper
reports 38,731 cells; the GEO matrix has 39,505 (the paper's outlier removal,
618 apoptotic-like and 139 poorly connected cells, is not applied in GEO; see
report section 3). Only GEO processed matrices are downloaded.

| Stage | hpf | Drop-seq cells | Libraries |
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

Stages before high (< 3.3 hpf) are not covered; pre-MBT maternal decisions
can only be seen through their consequences. Sampling is uneven across stages.

### Other inputs

- ZFIN expression, publication and alias files: `scripts/download_zfin.sh`
  → `data/raw/zfin/` (downloaded 2026-10-26, SHA256SUMS).
- `data/curated/`: `zfin_anatomy_programs.tsv` (ZFIN term → program; broad
  terms used by 01c), `signalling_programs.tsv` (pathway readouts; draft),
  `archive/` (earlier curated programs and per-gene evidence; only the six
  cell states are still used).
- `paper/` (local, gitignored): the paper, supplement and Table S2.

### Download and metadata

```bash
bash scripts/download_gse106587.sh               # idempotent; writes SHA256SUMS
(cd data/raw/GSE106587 && sha256sum -c SHA256SUMS)
python3 scripts/build_metadata.py                # SOFT -> data/metadata/*_cell_metadata.csv
python3 scripts/validate_data.py                 # writes data/metadata/validation_report.txt
```

Sources (public): `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE106nnn/{GSE106474,GSE106466,GSE106587}/suppl/`
and the SOFT files under `.../soft/`. A partially downloaded file must be
removed before re-running.

### Loading

Use `pipeline_utils.load_dropseq_counts()` for the Drop-seq atlas (sparse
AnnData with metadata; `obs` has `stage`, `sequencing_batch`, `replicate`,
`library`). The raw matrix is genes × cells TSV with gene symbols in the first
column (~39,505 × 23,974; reading it dense needs several GB).

SMART-seq2: genes × cells; the 88 `oep_p0_*` cells have UMI counts but are
absent from the FPKM matrix (`in_fpkm_matrix`).

10x: MatrixMarket per sample in `data/raw/GSE106587/GSE106587/`
(`GSM3084402_10xWT6S_*`, `GSM3084403_10xMZoep6S_*`); `genes.tsv.gz` has
Ensembl ID and symbol; `scipy.io.mmread` handles the `%` comment line;
`scanpy.read_10x_mtx` needs the files renamed. Barcodes are not QC-filtered.

### Metadata (`data/metadata/`)

`dropseq_cell_metadata.csv`: `cell_id` (matrix column, e.g.
`ZF50_WT_DS4b_ACGT...` = `<stage>_<genotype>_<batch>[b]_<barcode>`),
`stage`, `hours_post_fertilization`, `sequencing_batch` (DS2–DS5),
`replicate` (a/b), `genotype`, `geo_sample`. The b → GSM mapping is inferred
from GEO ordering; GEO "Rep N" numbers libraries across batches and is not
our a/b. Prefix codes: ZFHIGH, ZFOBLONG, ZFDOME, ZF30, ZF50, ZFS (shield),
ZF60, ZF75, ZF90, ZFB (bud), ZF3S, ZF6S.

`ss2_cell_metadata.csv`: `cell_id` (= GEO title), `genotype`, `stage`,
`library_batch`, `geo_sample`, `protocol`, `in_fpkm_matrix`.
`tenx_cell_metadata.csv`: `barcode`, `geo_sample`, `genotype`, `stage`,
`hours_post_fertilization`.

### Caveats

- Gene identifiers differ by assay (Drop-seq and SS2: symbols; 10x: Ensembl +
  symbol); the Drop-seq matrix often uses older symbols (ZFIN matching goes
  through aliases).
- No cell filtering, doublet removal or outlier removal has been applied;
  the paper's apoptotic-like cluster is still in the data.
- The paper's URD trajectories and cell-type annotations are not in GEO.
