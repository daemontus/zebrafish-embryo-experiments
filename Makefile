# Analysis pipeline: every script's main output table depends on the script,
# the scripts it imports and its input tables, so `make` reruns only what is
# out of date, in order. `make -n <target>` lists what would run.
#
#   make heterogeneity   01-03 (dataset, heterogeneous genes, a/b libraries)
#   make programs        16, 17, 21, 22, 24, 25 (program sets and their tests)
#   make populations     15, 26 (cell programs and exclusions), 28 (population
#                        heterogeneity), 29 (precursors), 19, 23 (exploration)
#   make report          the report PDF (copied to report/main.pdf)
#   make all             everything
#
# Scripts 17 and 22 also cache their intermediate results (data/processed/cache),
# so a rerun after a small change of the programs recomputes only what changed.
# Raw data and data/processed/dropseq_counts.h5ad are not built here (see AGENTS.md).

PY ?= .venv/bin/python
T := results/tables
S := scripts
UTILS := $(S)/pipeline_utils.py

.PHONY: all heterogeneity programs populations report
all: heterogeneity programs populations report
heterogeneity: $(T)/libraries.tsv $(T)/heterogeneity_zfin.tsv $(T)/hvg_flavours.tsv \
	$(T)/replicate_mixing.tsv $(T)/replicate_timing.tsv
programs: $(T)/programs.tsv $(T)/program_support.tsv $(T)/signal_coupling.tsv \
	$(T)/program_levels.tsv $(T)/program_overlap.tsv $(T)/program_pseudobulk.tsv
populations: $(T)/alignment_stages.tsv $(T)/cell_programs.tsv $(T)/program_atlas.tsv \
	$(T)/signal_atlas.tsv $(T)/population_heterogeneity.tsv $(T)/precursors.tsv

# --- dataset and heterogeneous genes (01-03) -------------------------------------------------
$(T)/libraries.tsv: $(S)/01_dataset_structure.py $(UTILS)
	$(PY) $<
$(T)/heterogeneous_genes.tsv.gz: $(S)/01b_heterogeneous_genes.py $(UTILS)
	$(PY) $<
$(T)/heterogeneity_zfin.tsv: $(S)/01c_heterogeneity_zfin.py $(S)/01b_heterogeneous_genes.py \
		$(S)/zfin_utils.py $(UTILS) $(T)/heterogeneous_genes.tsv.gz \
		data/curated/zfin_anatomy_programs.tsv
	$(PY) $<
$(T)/hvg_flavours.tsv: $(S)/01d_hvg_flavours.py $(S)/01b_heterogeneous_genes.py \
		$(S)/01c_heterogeneity_zfin.py $(UTILS) $(T)/heterogeneous_genes.tsv.gz
	$(PY) $<
$(T)/replicate_mixing.tsv: $(S)/02_replicate_mixing.py $(UTILS) $(T)/heterogeneous_genes.tsv.gz
	$(PY) $<
$(T)/replicate_timing.tsv: $(S)/03_replicate_differences.py $(UTILS)
	$(PY) $<

# --- programs (16, 17, 21, 22, 24) -----------------------------------------------------------
$(T)/programs.tsv: $(S)/16_program_sets.py $(S)/01c_heterogeneity_zfin.py $(S)/zfin_utils.py \
		$(S)/program_utils.py $(UTILS) $(T)/heterogeneous_genes.tsv.gz \
		data/curated/zfin_anatomy_programs.tsv data/curated/signalling_evidence.tsv \
		data/curated/curated_programs.tsv data/curated/gene_overrides.tsv $(wildcard data/curated/archive/*.tsv)
	$(PY) $<
$(T)/program_support.tsv: $(S)/17_program_tests.py $(S)/program_utils.py $(UTILS) \
		$(T)/programs.tsv $(T)/heterogeneous_genes.tsv.gz $(T)/libraries.tsv \
		$(T)/heterogeneity_libraries.tsv
	$(PY) $<
$(T)/signal_coupling.tsv: $(S)/21_signal_coupling.py $(S)/program_utils.py $(UTILS) \
		$(T)/programs.tsv $(T)/program_support.tsv $(T)/heterogeneous_genes.tsv.gz \
		data/curated/curated_programs.tsv
	$(PY) $<
$(T)/program_levels.tsv: $(S)/22_program_levels.py $(S)/17_program_tests.py \
		$(S)/program_utils.py $(UTILS) $(T)/program_support.tsv
	$(PY) $<
$(T)/program_overlap.tsv: $(S)/24_program_overview.py $(S)/16_program_sets.py $(UTILS) \
		$(T)/programs.tsv $(T)/heterogeneous_genes.tsv.gz data/curated/curated_programs.tsv \
		data/curated/signalling_evidence.tsv $(T)/program_support.tsv
	$(PY) $<
$(T)/program_pseudobulk.tsv: $(S)/25_program_pseudobulk.py $(S)/24_program_overview.py \
		$(UTILS) $(T)/programs.tsv data/curated/curated_programs.tsv
	$(PY) $<

# --- cells and populations (15, 26, 28, 29; 19 and 23: exploration) ---------------------------------
$(T)/alignment_stages.tsv: $(S)/15_alignment.py $(S)/02_replicate_mixing.py $(UTILS) \
		$(T)/heterogeneous_genes.tsv.gz
	$(PY) $<
$(T)/cell_programs.tsv: $(S)/26_cell_programs.py $(S)/program_utils.py $(UTILS) \
		$(T)/programs.tsv $(T)/program_support.tsv $(T)/heterogeneous_genes.tsv.gz \
		data/curated/curated_programs.tsv $(T)/curated_zfin_windows.tsv
	$(PY) $<
$(T)/population_heterogeneity.tsv: $(S)/28_population_heterogeneity.py \
		$(S)/26_cell_programs.py $(S)/01b_heterogeneous_genes.py $(UTILS) \
		$(T)/cell_programs.tsv $(T)/programs.tsv
	$(PY) $<
$(T)/precursors.tsv: $(S)/29_precursors.py $(S)/26_cell_programs.py $(S)/program_utils.py \
		$(UTILS) $(T)/cell_programs.tsv $(T)/programs.tsv $(T)/heterogeneous_genes.tsv.gz \
		data/curated/curated_programs.tsv
	$(PY) $<
$(T)/program_atlas.tsv: $(S)/19_program_atlas.py $(S)/15_alignment.py $(S)/program_utils.py \
		$(UTILS) $(T)/cell_programs.tsv $(T)/alignment_stages.tsv $(T)/programs.tsv \
		$(T)/program_support.tsv $(T)/program_pseudobulk.tsv
	$(PY) $<
$(T)/signal_atlas.tsv: $(S)/23_signal_atlas.py $(S)/program_utils.py $(S)/19_program_atlas.py \
		$(UTILS) $(T)/program_support.tsv $(T)/program_atlas.tsv
	$(PY) $<

# --- report ----------------------------------------------------------------------------------
report: report/main.pdf
report/main.pdf: report/main.tex report/references.bib $(wildcard report/figures/*) \
		$(wildcard report/tables/*.tex)
	cd report && latexmk main.tex
	cp report/build/main.pdf report/main.pdf
