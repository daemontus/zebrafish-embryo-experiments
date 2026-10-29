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
- `scripts/01b_heterogeneous_genes.py` (report section "markers of
  heterogeneity within libraries") selects heterogeneous genes per library
  with `pipeline_utils.nb_overdispersion`, a chance level calibrated against
  simulated homogeneous libraries (99.9% quantile of simulated residuals at
  the gene's mean; 9 simulations, seed per library from SEED and the name,
  so results depend on SEED by a few % of calls), and an effect-size floor
  from an empirical null of the well-expressed genes' residuals: three tiers
  weak / moderate / strong = estimated FDR 50% / 10% / 1% (calls table 0-3;
  "heterogeneous" = weak, the default downstream). Per stage: in the tier in
  >= half of the stage's libraries. It also
  compares with Seurat HVGs (most HVGs are near-undetected genes).
  `heterogeneity_calls(counts, rng)` applies the calibration to any cells.
  ~25 min. Outputs `results/tables/heterogeneous_genes*.tsv*`,
  `heterogeneity_libraries.tsv`, report figures/table, and
  `results/figures/heterogeneity_libraries.pdf` (overdispersion per library). Intended to replace
  HVGs everywhere downstream (02, 14-19 use them).
- `scripts/01c_heterogeneity_zfin.py` checks the heterogeneous genes against
  ZFIN in situ records (pre-2018): genes recorded in a localised structure
  at a stage are heterogeneous 3-12x more often than average, "whole
  organism" genes are depleted; HVGs recover fewer. ZFIN stage ranges are
  intervals (record covers start of first stage to start of the stage after
  the last). Outputs `results/tables/heterogeneity_zfin.tsv`, report table.
- `scripts/01d_hvg_flavours.py` (2026-10-28, report section 4 "newer HVG
  methods"): repeats 01b/01c's HVG comparisons for scanpy's seurat_v3 (vst)
  and analytic Pearson residuals (pooled stage and batch_key=library), plus
  the old "seurat" flavour with and without the usual min-3-cells filter
  (needed anyway: seurat_v3's loess fails on unfiltered genes; retried with
  wider spans). Result: most of the advantage over HVGs was the old flavour;
  newer methods are close to the NB residual ranking (ZFIN restricted genes
  in top k 25-56% vs 29-59%, replication 0.24-0.49 vs 0.29-0.50), still pick
  hundreds of unmeasurable genes, and the calibrated heterogeneous set beats
  every ranking of the same size by 2-9 points. Reads 01b's tables; a few
  minutes; `plot` redraws. Outputs `results/tables/hvg_flavours*.tsv`,
  report figure/table `hvg_flavours`. `01c.zfin_groups(records, stage)` is
  shared with 01c.
- `nb_overdispersion` (2026-10-28): the typical phi (`typical` column) is a
  dispersion-mean trend phi_t(m) = a0 + a1 m^-b (DESeq2-like, free exponent),
  fitted on log10 to the medians of raw phi in bins of 200 genes with >= 100
  UMIs and extrapolated to rarer genes; parameters in `.attrs["trend"]`.
  It replaced a running median with isotonic fit and boundary rules (same
  result within ~0.03 log10).
  01b calibrates on the residuals of simulated genes against the typical phi
  they were simulated with (since 2026-10-27; a running quantile of simulated
  log10 phi mixed in rarer genes at the upper end and overstated the cutoff).
- `scripts/02_replicate_mixing.py` tests whether the six a/b library pairs
  (same stage and batch) mix in a joint kNN graph, against a permutation null,
  a random split of one library (negative control), an adjacent stage
  (positive control) and, where available, another batch. It runs once with
  library-size normalisation only and once after binomial thinning of the
  deeper library, and again after removing the top a/b-differing genes.
  Outputs: `results/tables/replicate_mixing*.tsv`, `results/figures/`
  (histograms, UMAPs) and `report/figures/replicate_mixing.pdf`.
  Embeds on the heterogeneous genes of 01b (either library of a comparison),
  not HVGs (since 2026-10-27; joint HVGs include between-library genes).
  Result: no pair mixes (AUC 0.86-1.00), so a/b libraries are kept separate.
- `scripts/03_replicate_differences.py` collects descriptive gene-level
  evidence for why a/b libraries differ (pseudobulk fold changes, on/off
  genes with depth-matched detection, ribosomal fraction and stage-marker
  timing indicators) and writes `report/tables/replicate_{summary,genes}.tex`.
- `scripts/14_replicated_populations.py` (2026-10-28, exploratory) clusters
  each library on its own (Leiden 0.5/1/2/4) on shared PCs of per-library
  centred Pearson residuals (clipped at +-10 for every library) of the
  stage's heterogeneous genes and keeps
  clusters that match reciprocally in another library (MetaNeighbor-style
  neighbour voting, one-vs-best AUROC >= 0.8); single-library stages (dome,
  shield, 3-somite) are matched against the adjacent stages' libraries.
  Populations of different resolutions sharing >= 50% of cells are merged
  ("consensus"); "robust" = found at >= 2 resolutions (87-100% reproduce with
  another seed vs 56% for one resolution). Null: Voronoi clusters around
  random cells (same number) give 19-31% as many populations as the real
  clusters (per resolution, summed over stages; most at resolution 4).
  Centring removes all mean differences between libraries (the classifier
  AUC of 02 is ~0.4 for any pair, adjacent stages included), but locally
  stages >= 1.7 h apart stay distinct (80-97% of neighbourhoods unbalanced)
  while stages ~1 h apart blur to the level of a/b pairs; so adjacent-stage
  matches and stage links are weaker evidence. A clip of +-sqrt(cells) (used
  until 2026-10-28) made unequal random halves of one library differ. Robust populations of consecutive stages
  are linked by joint-PCA neighbour voting (AUROC >= 0.9) into tracks.
  A few minutes; `--seed`, `--tag`, `--tracks-only` (redraw tracks from tables).
  Outputs `results/tables/replicated_populations*.tsv*` (summary, matches,
  per-resolution, consensus, cells, links, tracks) and
  `results/figures/replicated_populations{,_tracks}.pdf`. A seed-1 run is in
  `..._seed1*`. Not in the report yet.
- `pipeline_utils.pearson_residuals(counts, genes, centre)` (2026-10-28):
  Pearson residuals of one library under its own NB model (trend phi_t),
  clipped at +-RESIDUAL_CLIP = 10 for every library, optionally centred per
  library (= alignment). Used by 14, 15, 17, 18.
- `scripts/15_alignment.py` (report section "aligning libraries") checks the
  alignment against the unaligned joint embedding of 02: same-stage library
  pairs vs random splits (unbalanced kNN neighbourhoods), same-batch pairs of
  different stages, kNN overlap with each library's own PCA, and transfer of
  marker-defined cell types (EVL, PGC, p53-like, endoderm, notochord,
  hatching gland) between libraries; per-stage and atlas UMAPs (atlas
  coordinates cached in `data/processed/atlas_umap.tsv.gz`). ~20 min.
  Figure `alignment_markers` (2026-10-28): the marker-called populations in
  every library on the per-stage UMAPs; `15_alignment.py markers` redraws it
  and `alignment_stages` without the checks (~2 min, same calls).
  Outputs `results/tables/alignment_*.tsv`, report figures `alignment_*`.
- `scripts/16_program_sets.py` builds stage-specific program sets
  (`results/tables/programs.tsv`): farrell (Table S2), zfin_pre / zfin_all
  (in situ terms covering the stage, via 01c.zfin_genes), *_het (restricted
  to heterogeneous genes), signalling (`data/curated/signalling_programs.tsv`,
  pathway readouts: feedback inhibitors and targets; draft, review sources),
  states (archive literature table), curated_staged (draft: ZFIN all-years
  terms grouped by `zfin_anatomy_programs.tsv`, genes heterogeneous at the
  stage, + signalling + states; ZFIN maternal/blastomere groups excluded; cap
  30 genes by publications at the stage) and curated (since 2026-10-29, the
  set used by 17-23): the same groups with ONE gene list per program for all
  stages of its window (first to last stage where the staged version exists):
  ZFIN records pooled over the window, gene heterogeneous at >= 1 window
  stage, ranked by mean heterogeneity tier over the window then
  publications, cap 30; groups that change identity are split via the
  `fixed_program` column of the mapping (margin / mesoderm, axial mesoderm /
  notochord, prechordal plate / hatching gland, presumptive neural / neural
  plate). Staged gene lists shared < half their genes between stages.
  Side effect: pooling moves dlx3b/foxi1 from the non-neural ectoderm to the
  placodes. Farrell modules: all 30 listed genes (since 2026-10-29).
- `scripts/17_program_tests.py`: program score = mean centred Pearson
  residual of its genes; coherence R = n Var(score)/mean gene variance vs
  random same-size sets of heterogeneous genes (95% quantile of N_RANDOM =
  500 sets since 2026-10-29; with 50, ~6% of calls flipped with the seed; the
  gene-wise permutation null is too weak), supported = coherent in >= half of the
  stage's libraries; library levels of state programs; prediction test of
  06/13 on all cells (evaluation genes: random measured and heterogeneous,
  never in any program); tests curated (fixed genes) and curated_staged
  side by side. ~30-90 min; `summarise` redraws. Outputs
  `results/tables/program_{coherence,support,library_levels}.tsv`,
  `program_tests_prediction*`, report figure `program_tests`.
- `scripts/18_program_populations.py`: script 14's procedure (via
  `run_stage(..., embed=...)`) in the space of supported curated program
  scores; compares with 14's robust populations (captured / named, not
  separated / unnamed = missing program) and computes coverage (>= 1 active
  program, population membership). Needs 14 and 17 outputs. Outputs
  `results/tables/program_populations*`, report figure `program_populations`,
  `report/tables/missing_programs.tex`.
- `scripts/19_program_atlas.py` (exploration, not in the report): for every
  curated program, its smoothed score on the per-stage UMAPs of 14 at every
  stage, the share of active cells, support and library-level pseudobulk,
  plus the program on the atlas UMAPs of 15 (aligned / unaligned, scores
  scaled per stage); overview heatmaps first. `plot` redraws from the cache
  (`data/processed/program_scores.tsv.gz`). Output
  `results/figures/program_atlas.pdf`, `results/tables/program_atlas.tsv`.
- `scripts/20_program_views.py` (exploration): from 19's cached scores and
  active calls (`data/processed/program_{scores,active}.tsv.gz`) and 14's
  populations and tracks: programs x robust populations dot plots per stage,
  program co-activity (score correlations) per stage, active share along
  each population track (>= 3 stages), and per-gene log2 detection in active
  vs other cells per program. Outputs `results/figures/program_{populations_dots,
  coactivity,tracks,genes}.pdf`.
- `scripts/21_signal_coupling.py` (report subsection "which programs co-occur
  with which signals"): per stage, correlation of the smoothed signalling
  readout scores with the other curated programs, after removing the readout
  genes from those programs (93 shared genes). Literature comparison in the
  report; bibliography entries for it were checked against Crossref
  (2026-10-28; author names normalised to NFC, since Crossref returns
  combining accents that pdflatex rejects). Hedgehog readouts are never heterogeneous; the RA readouts
  (cyp26a1, dhrs3) do not measure RA activity well. `plot` redraws. Outputs
  `results/tables/signal_coupling{,_removed}.tsv`, report figure
  `signal_coupling`.
- `scripts/22_program_levels.py` (2026-10-28, report subsection "how many
  activity levels does a program need?"): the prediction test of 17 (same
  evaluation genes, split, coordinates; floor/ceiling/random merged from
  17's per-gene table, so `curated` reproduces 17) with the curated scores
  discretised per stage (pooled aligned scores): k-means k = 2-6, encoded
  cumulatively (level >= v; one-hot k3/k6 as a check, worse with many
  levels), the Boolean null call of 18 and 3 states, and supported-only
  variants; plus per-program variance kept by the levels. Result: 2 levels
  keep ~0.92 of the continuous explained fraction, 3 levels ~0.98, 4+ ~1.0;
  the strict null call only ~0.85. ~45 min; `summarise` redraws. Outputs
  `results/tables/program_levels{,_genes,_fidelity}.tsv*`, report figure
  `program_levels`. Needs 17's outputs.
- `scripts/23_signal_atlas.py` (2026-10-29, exploration, not in the report):
  per pathway, two curated programs expected with high / low readouts
  (PAIRS: BMP non-neural ectoderm / neural plate, Nodal margin / presumptive
  neural, Wnt tailbud / forebrain-eye, FGF mesoderm / non-neural ectoderm,
  Notch presomitic / lateral plate); cells with exactly one of them active,
  coloured by readout score above / below the library average, on the atlas
  UMAPs. Readout genes removed from the programs as in 21. A few minutes;
  `plot` redraws. Outputs `results/figures/signal_atlas.pdf`,
  `results/tables/signal_atlas.tsv`, cache `data/processed/signal_atlas_*`.
- `scripts/19_program_atlas.py` also writes (2026-10-29)
  `results/figures/program_atlas_{aligned,unaligned}.pdf` (one atlas UMAP per
  program; `atlas` mode) and `program_atlas_combined.pdf` (groups of
  programs expected to be mutually exclusive, EXCLUSIVE_GROUPS, coloured by
  the active program, black = several; `combined` mode).
- `scripts/program_utils.py`: smoothing, the held-out prediction test
  (floor / feature sets / knn ceiling) and the Farrell et al. Table S2 loader
  (all 30 listed genes per module; until 2026-10-29
  the top 25, a leftover of the removed cNMF comparison), moved from the old scripts 04-06.
- `data/curated/archive/expression_programs.tsv` (+ `expression_program_genes.tsv`,
  per-gene literature evidence): the earlier hand-curated programs; only the
  six cell states (maternal, blastomere, cell cycle S/G2M, heat shock, p53)
  are still used (script 16, `states`).
- Removed 2026-10-28: scripts 04-13 (curated-program scoring, cNMF, the first
  prediction tests, combinations, hierarchy, ZFIN variable programs, the
  first program sets and activity calls), their tables and figures, the
  report sections built on them, and the caches `data/processed/cnmf`,
  `program_umap.tsv.gz`, `program_activity/`. They were never committed; a
  backup of the scripts and outputs (not the 3.3 GB cNMF cache) is in
  `data/processed/archive/old_program_analyses_2026-10-28.tar.gz` (local,
  gitignored).
- Report (2026-10-28): sections 6-8 are alignment (15), programs (16, 17,
  21, 22) and populations (14, 18).
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
  raw/zfin/                      # ZFIN downloads (2026-10-26), SHA256SUMS; gitignored
    GSE106474/GSE106474_UMICounts.txt.gz
    GSE106466/GSE106466_SS2_umi_cnts.txt.gz
    GSE106466/GSE106466_SS2_fpkm.txt.gz
    GSE106587/GSE106587_RAW.tar
    GSE106587/GSM3084402_10xWT6S_*   # extracted 10x WT 6-somite
    GSE106587/GSM3084403_10xMZoep6S_* # extracted 10x MZoep 6-somite
    SHA256SUMS
  curated/
    archive/expression_programs.tsv      # earlier curated programs (states used by 16)
    archive/expression_program_genes.tsv # per-gene literature evidence and rejections
    zfin_anatomy_programs.tsv     # ZFIN anatomy term -> program (scripts 01c, 16)
    signalling_programs.tsv       # signalling pathway readout genes (script 16; draft)
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
  zfin_utils.py                   # ZFIN expression records loader (01c)
  01_dataset_structure.py         # stage x batch x library overview -> report
  01b_heterogeneous_genes.py      # calibrated heterogeneous genes (replace HVGs)
  01c_heterogeneity_zfin.py       # heterogeneous genes vs ZFIN in situ records
  01d_hvg_flavours.py             # heterogeneous genes vs newer HVG methods
  02_replicate_mixing.py          # a/b library mixing (kNN, kBET, LISI, AUC)
  03_replicate_differences.py     # a/b gene-level differences, timing indicators
  14_replicated_populations.py    # per-library clusters replicated across libraries
  15_alignment.py                 # checks of the per-library alignment, UMAPs
  16_program_sets.py              # stage-specific program sets (ZFIN, Farrell, curated, ...)
  17_program_tests.py             # program coherence and prediction tests
  18_program_populations.py       # populations in program space, missing programs
  19_program_atlas.py             # per-program UMAPs and overview (exploration)
  20_program_views.py             # dot plots, co-activity, tracks, gene views
  21_signal_coupling.py           # signalling readouts vs programs over stages
  22_program_levels.py            # discretised programs: how many levels are needed
  23_signal_atlas.py              # signalling readouts vs expected programs (atlas UMAPs)
  program_utils.py                # smoothing, prediction test, Farrell modules
  download_zfin.sh                # ZFIN expression/publication/alias files
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
Added on 2026-10-26: `cnmf 1.7.1` (consensus NMF, script 05) and
`openpyxl 3.1.5` (reading Farrell et al. Table S2).

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
