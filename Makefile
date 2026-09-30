.PHONY: test phase1 phase1-dry-run features eval-data eval eval-all results case-study bundle reproduce reproduce-refit bundle-upload demo-data

test:
	python -m pytest -q

data/cache/shard_plate_map.parquet:
	python scripts/shard_plate_map.py

data/pseudobulk/genes.parquet:
	python scripts/fetch_gene_metadata.py

# Rebuild every Phase 1 output from configs/slice.yaml. Resumes: finished shards are skipped.
phase1: data/cache/shard_plate_map.parquet data/pseudobulk/genes.parquet
	python scripts/run_phase1.py

phase1-dry-run: data/cache/shard_plate_map.parquet
	python scripts/run_phase1.py --dry-run

# Phase 3 features: drugs, drug_groups, cells, depmap_pca + the condition_features view,
# contracts 1-6 and reports/features.md. From a clean checkout this fetches the pinned Tahoe
# metadata and DepMap 24Q4, and builds phase 1 first if its conditions table is missing.
data/cache/hf/metadata/drug_metadata.parquet data/cache/cell_line_metadata.parquet:
	python scripts/fetch_tahoe_metadata.py

data/cache/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.parquet:
	python scripts/fetch_depmap.py

data/pseudobulk/conditions.parquet:
	$(MAKE) phase1

features: data/pseudobulk/conditions.parquet data/cache/hf/metadata/drug_metadata.parquet \
          data/cache/cell_line_metadata.parquet data/cache/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.parquet
	python scripts/build_features.py

# Phase 4a evaluation harness (docs/superpowers/specs/2026-09-27-phase4a-harness-design.md).
# eval-data streams Tahoe's DESeq2 rows once (~22 GB, one subprocess per line, resumable).
eval-data: features
	python scripts/build_eval_data.py

MODEL ?= dummy
eval:
	python scripts/run_eval.py --model $(MODEL)

# Every registered model (dummy + phase 4b baselines), then reports/results.md.
eval-all:
	python scripts/run_eval.py --all

results:
	python scripts/run_eval.py --report

# Phase 5a case study page (docs/superpowers/specs/2026-09-28-phase5a-case-study-design.md); needs make eval-all results.
case-study:
	python scripts/build_case_study.py

# Phase 5b reproduction bundle (docs/superpowers/specs/2026-09-29-phase5b-reproduction-design.md).
BUNDLE_VERSION ?= v1
bundle:
	python scripts/build_bundle.py --version $(BUNDLE_VERSION)
reproduce:
	python scripts/reproduce.py

reproduce-refit:
	python scripts/reproduce.py --refit
# Creates a DRAFT release only (needs GITHUB_TOKEN); the owner publishes it on GitHub.
bundle-upload:
	python scripts/upload_release.py --version $(BUNDLE_VERSION)

# Phase 5c demo data (docs/superpowers/specs/2026-09-29-phase5c-demo-design.md); refits every split, ~20-30 min.
demo-data:
	python scripts/build_demo_data.py
