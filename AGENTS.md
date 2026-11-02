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
and per-cell program calls (populations to follow), written up in
`report/main.tex`.

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
- Figures and tables (main and supplementary) are numbered in the order the
  text first refers to them; the appendix follows the same order. A figure
  taller than ~70% of a page needs `[tbp]`, not `[t]` (otherwise it and all
  later figures move to the end of the report).
- Outputs are reproducible: `pipeline_utils` fixes the PDF creation date
  (`SOURCE_DATE_EPOCH`), and versioned `.gz` tables are written with
  `compression=GZIP` (no timestamp), so a rerun that changes nothing changes
  no file. Keep it that way for new outputs.

## Environment

Local macOS: `.venv` (Python 3.13; pandas 3.0.6, numpy 2.5.4, matplotlib
3.11.2, scanpy 1.12.4, openpyxl 3.1.5, ...). Run scripts from the repository
root: `.venv/bin/python scripts/17_program_tests.py`. Scripts import each
other via `importlib.import_module("17_program_tests")`; to import
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

**Use the Makefile**: `make programs` (16, 17, 21, 22, 24, 25), `make populations`,
`make heterogeneity`, `make report`, `make all`; `make -n <target>` shows what
would run. It reruns a script when the script, a script it imports or an input
table is newer than its main output. Every script depends on
`pipeline_utils.py`, so after an edit there that cannot change a script's
results (e.g. a new helper), mark the outputs as current with
`make -t <target>` instead of rerunning 01b (~25 min).

**Caching** (`pipeline_utils.cached`, `data/processed/cache/<script>/`): 17 and
22 store every unit of work (coherence per stage and set, random-set quantiles
per library, the prediction setup per stage, the losses per model) under a
digest of its inputs, and every random draw has its own seed; an interrupted
run resumes. A change of the curated or control programs recomputes only
their coherence, but the whole prediction test of every stage whose program
genes change (the evaluation genes are in no program, so they change too):
about half an hour for 17 and 22 each. Increase a script's `CACHE_VERSION`
when a code change changes cached results (17: `COHERENCE_VERSION` for the
coherence part only); delete the cache directory to start over.

| Script | Does | Needs | Time | Modes / caches |
|---|---|---|---|---|
| `01_dataset_structure.py` | stage × batch × library design, depth; report "Dataset overview" | counts | min | |
| `01b_heterogeneous_genes.py` | calibrated heterogeneous genes per library and stage (tiers weak/moderate/strong) | counts | ~25 min | `plot` (cache `heterogeneity_models.pkl`) |
| `01c_heterogeneity_zfin.py` | heterogeneous genes vs ZFIN in situ records | 01b, ZFIN | min | |
| `01d_hvg_flavours.py` | vs newer HVG methods (seurat_v3, Pearson residuals) | 01b, 01c | min | `plot` |
| `02_replicate_mixing.py` | do a/b libraries mix? (no) | 01b | min | |
| `03_replicate_differences.py` | why a/b libraries differ (genes, timing) | counts | min | |
| `15_alignment.py` | checks of the per-library alignment; UMAPs | 01b, 02 | ~20 min | `markers` (~2 min); cache `atlas_umap.tsv.gz` |
| `16_program_sets.py` | program sets → `programs.tsv`, plus readable gene lists per set in `results/tables/program_sets/<set>.tsv` | 01b, 01c, ZFIN, `data/curated`, Farrell S2 | min | `tables` (rewrites the gene lists from `programs.tsv`) |
| `17_program_tests.py` | coherence/support (and relative coherence) and prediction tests of the sets (per stage and per library; explainable share per heterogeneity tier) | 01, 01b, 16 | cold ~1 h (coherence ~20 min); cached runs seconds to minutes | `summarise`; cache `cache/17` |
| `19_program_atlas.py` | per-program UMAPs (exploration; stage UMAPs of 26) | 15, 17, 25, 26 | min | `plot`, `atlas`, `combined`; caches `program_{scores,active}.tsv.gz` |
| `21_signal_coupling.py` | signalling profiles vs programs per library, relative to random gene sets (reading profiles: `signal_coupling`; all profiles: `signal_outputs`); expected senders, receivers and blockers as hypotheses (`HYPOTHESES`; AUROC on active cells against the parent programs); signals of every derivation at its emergence (phenotype transitions) | 17 | ~10 min | `plot` |
| `22_program_levels.py` | discretised program levels in the prediction test (own evaluation genes: in no curated program, may be in other sets) | 17 | cold ~30 min; cached runs seconds to minutes | `summarise`; cache `cache/22` |
| `23_signal_atlas.py` | readouts vs expected programs on the atlas (exploration) | 17, 19, 21 | few min | `plot`; caches `signal_atlas_*` |
| `24_program_overview.py` | report figures and tables: curated-program timeline (heterogeneous genes per stage vs the ZFIN window, derivations as indentation and arrows), programs per stage of the sets, Farrell modules named by the curated programs, the curated program table (from `data/curated/curated_programs.tsv`), the signalling evidence table and the genes removed to make the curated programs disjoint; the relative-coherence map in the rows of the timeline | 01b, 16, 17 | seconds | |
| `25_program_pseudobulk.py` | pseudobulk program strength per library (mean gene level relative to its peak, CP10k; the only library-level measure), in the layout of the timeline of 24 (squares) | 16, 24 | ~1 min | |
| `26_cell_programs.py` | per cell: all identity programs (supported or not) against their own permutation null (z, active), random gene sets as chance level, the top program and its margin, split-half reliability, which program wins where one is active but not on top (descendant/ancestor/sibling/unrelated); stress cells excluded (p53, dissociation, heat shock on top with margin >= 2); YSL/yolk and quality checks (nothing excluded); aligned stage UMAPs; per-program UMAPs of its active cells over all stages (`cell_programs_trajectories.pdf`); top programs vs the ZFIN windows (`cell_programs_zfin*.tsv`) | 16, 17 | cold ~30 min; cached runs ~2 min | `plot`; cache `cache/26` |
| `28_population_heterogeneity.py` | heterogeneous genes (01b's test) within top-program populations (one library, >= 100 cells) vs as many random cells of the library; calls count with >= 10 driver cells (residual > 3); power from the control (recovered share of the library's strong genes detectable at that size; < 5 detectable = inconclusive); ubiquitous genes (>= 1/4 of conclusive populations, >= 4 programs) and their co-varying genes | 01b, 26 | ~30 min cold (8 workers); cached seconds | `plot`; cache `cache/28` |
| `29_precursors.py` | precursors (cells with a program on top before it is supported): runner-up program and neighbours vs the expected parent, against the same share at supported stages (parent and child share genes); report table of where populations emerge | 26 | ~3 min | `plot` |

Shared modules:
- `pipeline_utils.py`: paths, stages and batches, plot style,
  `load_dropseq_counts()` (builds/reads `data/processed/dropseq_counts.h5ad`),
  `nb_overdispersion` (per-gene NB dispersion vs the library's
  dispersion-mean trend phi_t(m) = a0 + a1 m^-b) and `pearson_residuals`
  (per-library residuals under that model, clipped at ±10, optionally
  centred = the alignment), the cache helpers (`digest`, `cached`,
  `load_cached`, `save_cached`) and `GZIP` (timestamp-free gzip for `to_csv`).
- `program_utils.py`: kNN smoothing, `ProgramSpace` (smoothed scores and
  active calls of 19 and 23, constants of 21), the held-out prediction test (target:
  neighbourhood detection frequency only since 2026-10-29; floor /
  feature sets / ceiling; `fit_model_losses` keeps the loss per test cell) and the Farrell et al. Table S2 loader (all 30
  genes per module; reads `paper/aar3131_farrell_table-s2.xlsx`, not
  versioned).
- `zfin_utils.py`: ZFIN records, stage hpf, symbol matching
  (`zfin_records(max_year)`; ZFIN direct submissions get their year from the
  publication id).

Report sections: 3 dataset (01), 4 heterogeneous genes (01b–01d),
5 a/b libraries (02, 03), 6 alignment (15), 7 programs: sets, structure,
pseudobulk (16, 24, 25), 8 programs vs cells: support, prediction, levels
(17, 22; the coherence map is drawn by 24), 9 programs and signals (21), 10 cells and
populations (26, 28, 29; populations still to come). 19 and 23 are exploration only.

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
- **Program sets** (16): `farrell`, `zfin_all` and `zfin_all_het` (ZFIN terms,
  all years; the reported ZFIN sets), `zfin_pre_het` (pre-2018 records,
  heterogeneous genes; a sanity check, mentioned once in the report),
  `signalling` (the curated signalling profiles as a set of their own,
  fixed), `states` (six cell states from the archived
  literature table), `controls` (whole-organism genes, negative-control pathways and
  dissociation-stress genes, YSL; see 16's docstring) and **`curated`, the set
  used downstream**: ZFIN terms grouped by
  `data/curated/zfin_anatomy_programs.tsv` (column `program`; column
  `fixed_program` splits groups that change identity), **one gene list per
  program, defined at every stage** (no window: the data show when it
  varies): genes of the group's records from high to 6-somite (the group
  needs >= 1/3 of the gene's publications at some stage), heterogeneous at
  >= 1 stage, ranked by mean tier over all stages, cap 30. The ZFIN window
  (stages with >= 3 recorded genes, `results/tables/curated_zfin_windows.tsv`)
  is only compared with the data (boxes in the figures of 24 and 25). Plus the cell states (`state:`) and
  the signalling profiles, assigned per gene by the column `program` of
  `signalling_evidence.tsv`: general `signal:<pathway>` where readouts and
  ligands mark the same cells (Nodal, BMP, Wnt, FGF), otherwise
  `receiving:`, `sending:` (ligands, RA synthesis; `admp` as
  `sending:BMP_organizer`) and `blocking:` (secreted antagonists, RA
  degradation); readouts of several pathways (ved, vox, vent, nkd1, nrarpa/b)
  are not used. **Curated programs are disjoint**: signalling and state genes
  leave the anatomy programs, and two anatomy programs share no gene (more
  ZFIN publications wins; ties: where the gene ranks higher) unless one
  derives from the other, directly or not (column `derives_from` of
  `curated_programs.tsv`: prior knowledge; there is no separate
  sub-program column);
  removed genes in `results/tables/curated_removed_genes.tsv`. Dropped for lack of specific genes after that: `ventral_bmp` (ZFIN ventral mesoderm; the general BMP profile covers it), `floor_plate`, `head_mesoderm` and `hemato_vascular`; `presumptive_neural` is merged into `neural_plate` (`EXCLUDED_GROUPS` in 16). Genes can be assigned to a group by hand in `data/curated/gene_overrides.tsv` (gene, program, reason, references; keep it small and sourced): only `dharma` → dorsal organizer. ZFIN's maternal and blastomere terms
  are excluded (maternal records end before high; blastomere/blastoderm
  records are led by ventral and EVL genes even before shield). Yolk and YSL records are kept although the embryos
  were deyolked: as `control:YSL` in `controls` (the curated YSL group without
  any gene of the curated programs or other controls), and as terms in
  `zfin_*`; 01c still leaves them out of the heterogeneity
  validation.
- **Program score** = mean centred Pearson residual of its genes;
  **supported** = coherent (vs the 95% quantile of 500 random sets of
  heterogeneous genes) in >= half of the stage's libraries; **active** =
  smoothed score above the 99.9% quantile of a gene-wise permutation null
  in the library; **relative coherence** = (R - median) / (95% quantile -
  median) of the random sets (0 = random heterogeneous genes, 1 = the
  threshold), median over the stage's libraries: descriptive, for trends
  below the threshold.
- **Prediction test** (17): evaluation genes are in no program of any set,
  so no set predicts its own genes; losses are summed per stage and per
  library. 22 tests only the curated programs, so its evaluation genes need
  to be in no curated program only (more and stronger genes); its explained
  fractions are therefore not those of 17. With few heterogeneous
  evaluation genes (shield: 49 in 17) the test has little to explain.
- **Cell programs** (26): a cell's programs are scored against their own
  gene-wise permutation null (z = standardised excess; active above the
  99.9% quantile), smoothed over 10 neighbours found without the scored
  programs' genes (unsmoothed calls find an identity in only ~30-40% of the
  cells). Competing: all identity programs + maternal + blastomere, supported
  or not (since 2026-11-02: unsupported programs mark emerging progenitors,
  e.g. forebrain at 60% epiboly; random gene sets almost never win); support
  is still reported;
  top = active program with the largest z, margin = top z - next z (floored
  at 0). Low margins are mostly parent/child ties. Heat shock, dissociation
  and cell-cycle genes are left out of neighbourhoods and features, not
  cells; only stress cells (p53/dissociation/heat shock on top, margin >= 2)
  are excluded. The YSL control is not YSL (yolk-specific genes are nearly
  absent); doublets and low-quality cells need no filter beyond GEO's.
- **Top-program populations** (28) are largely homogeneous (median 0.19x the
  heterogeneous genes of random cells, in the 71 of 93 populations with power;
  small somite-stage populations are inconclusive); in groups of ~100
  cells a third of 01b's calls are carried by <= 3 cells, so count only calls
  with >= 10 driver cells. Ubiquitous genes, classified by what they co-vary
  with inside populations (`UBIQUITOUS_CLASSES` in 28): positional gradients
  (apoc1, apoeb, ved: ventral BMP; cdx4: posterior; irx7: anterior; msgn1:
  paraxial maturation) and states/technical axes (histh1l: mitosis; fosab:
  dissociation; zgc:110425 with the si:ch211-113a14 family); marcksl1b unclear.
  Positional ones tell where a cell is, not its population.
- **Signalling readouts**: Hedgehog readouts are never heterogeneous; the RA
  readouts (cyp26a1, dhrs3a) do not measure RA activity well (cyp26a1 is used
  as RA blocking; dhrs3b was dropped: no evidence); nkd1 and nrarpb
  are also BMP targets. Since the curated programs are disjoint, 21 and 23
  compare them directly.
- **Tried and dropped** (do not reintroduce without a new reason):
  scripts 04–13 (first program scoring, cNMF, early prediction tests;
  removed 2026-10-28, never committed, backup in
  `data/processed/archive/old_program_analyses_2026-10-28.tar.gz`); a
  stage-specific curated set (2026-10-29); embryo-age controls (genes
  changing between adjacent stages of one batch; 2026-11-01: few genes,
  none from 60% epiboly to bud, and the only coherent list was the
  si:ch211-113a14 gene family, not age); a per-library prediction script
  (merged into 17); replicated per-library clusters and program-space
  populations (14, 18, 20; removed 2026-11-02, in git history: Leiden
  clusters fragment graded populations; restarted from per-cell program
  calls, 26); closed frequent program patterns (27, never committed,
  archived in `data/processed/exploration/program_patterns/`); keeping a gene shared along a derivation only in the
  derived program (the parent programs lost their markers, e.g. the margin
  27 of 30 genes).

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
  terms used by 01c), `curated_programs.tsv` (hand-written annotation of
  every curated program: Table 1 link, derives_from (read by 16 for gene
  sharing, by 24 for indentation and arrows), location/signals, role,
  interpretation, bib keys; read by 24),
  `signalling_evidence.tsv` (readouts, negative-control genes, ligands and
  secreted antagonists with role,
  evidence type, strength, curated profile (`program`, `program_note`) and bib keys, references checked against Crossref
  and PubMed on 2026-10-29; read by 16 and 24),
  `gene_overrides.tsv` (manual gene → curated group, read by 16),
  `archive/` (earlier curated programs and per-gene evidence; only the six
  cell states are still used).
- `paper/` (local, gitignored): the paper, supplement and Table S2.
- `data/processed/grn_prior/` (experiment, unversioned, not in the
  Makefile): prior-knowledge GRN around 6 hpf. `core_edges.tsv` is a
  hand-curated table of zebrafish literature edges (abstracts checked in
  PubMed); `build_prior.py` combines it with signalling readouts, TFLink
  (GTRD ChIP), CollecTRI via HCOP orthology and CellOracle promoter motifs
  into `out/core_network.tsv` and `out/extended_network.tsv`, annotated
  with atlas correlation per library and, for repressions, a check in the
  replicated populations of the removed script 14 (`inhibition_*`; see the
  docstring; its tables are kept in `data/processed/archive/populations_2026-11-02/`,
  but `build_prior.py` and `model/` still read them from `results/tables`). When
  correlating genes within small groups, require >= 10 UMIs per gene in
  the group: a gene with no UMIs has Pearson residuals that only track
  depth, so undetected genes correlate spuriously. `model/` holds the first
  partially specified Boolean module (`dv_nodal.aeon`, literature edges only;
  `dv_nodal.md` lists its choices and expected attractors; `check_model.py`
  counts admissible functions with biodivine_aeon, which is not in `.venv`;
  `dv_nodal_v2.{aeon,md}` is the accepted revision: run any script with `DV_MODEL=v2`
  (`model_version.py`) for the v2 model, phenotypes and `results/v2/`; v1 stays the default;
  `PHENOTYPE_PROGRAMS.md` maps phenotypes to atlas programs; `REPORT.md` has the results).

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
