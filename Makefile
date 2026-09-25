.PHONY: test phase1 phase1-dry-run

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
