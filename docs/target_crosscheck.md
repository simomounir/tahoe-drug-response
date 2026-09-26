# Target cross-check: our logFC vs. Tahoe's official DE table
Cross-checks the phase 1 target (natural-log CPM-ratio logFC in `data/pseudobulk/logfc.parquet`) against Tahoe-100M's own `pseudobulk_differential_expression` table (DESeq2 output, 1,026 Parquet shards on Hugging Face), per spec §4 task P3.1.
## Tahoe's DE shard layout (discovered, not documented upstream)
The 1,026 shards are not named by content, and a first attempt to locate them assumed a global sort by `(Cell_ID_Cellosaur, plate, drug, gene_name)` -- probing shards 0, 300, 500, 700, 900 directly disproved that (`CVCL_0023, CVCL_0334, CVCL_0293, CVCL_1056, CVCL_0131` -- not monotonic in any plausible key) and was caught before the real run wasted budget on it. The true structure: each cell line occupies one *contiguous* block of shards, sorted within the block by `(plate, drug, gene_name)`, but the *order of blocks* across the 1,026 shards is not the cell lines' alphabetical order (apparently whatever order Tahoe's export pipeline produced them in). `plate` runs well beyond 3 for each cell line (Tahoe's DE table also covers its broader ~1,100-compound program beyond our 92-drug, 3-plate slice), so a block starts fresh at `plate="1"` and runs through many more plates before ending; our slice only needs the first few shards of each ~20-shard block. Each shard's Parquet footer measured ~5.75 MB (thousands of small row groups), and a targeted remote query (`read_parquet(url) WHERE ...`) measured 40-50s per single-drug query -- apparently DuckDB's httpfs does not collapse a narrow predicate into few large HTTP range requests against a footer this size, whereas a plain full-shard download measured ~4s for ~90 MB. This script therefore downloads whole shards it needs (fast, and every byte is used) and locates them with one coarse, fixed-stride scan (stride 5) across all 1,026 shard footers -- any block spanning 5 or more shards is hit; a sampled line not found fails the run -- followed by a short local walk back to each needed cell line's true block start.
## Method
- Sampled `n_conditions=200` treated, QC-passing conditions from `data/pseudobulk/conditions.parquet` with seed `20260926` (`phase3.crosscheck.sample_conditions`).
- Built one coarse shard index (stride 5, ~205 footer probes), then processed cell lines in order of how many sampled conditions they cover (most first, so a network-cap stop loses as few matched conditions as possible): located and downloaded the shards covering each cell line's plates 1-3, extracted rows for every sampled drug on it, and deleted the shards immediately after.
- Join keys: `Cell_ID_Cellosaur = <our cell_line>` (exact; same Cellosaurus id both sides), `plate = <our plate, "plateN" stripped to "N">`, `trim(drug) = trim(<our drug>)` (Tahoe drug names carry stray whitespace, e.g. `"Erdafitinib "`; join key is `trim(drug)` per spec).
- Over genes with Tahoe `padj < 0.05`: sign agreement between our `logfc` and their `log2FoldChange`, and Spearman correlation (computed as Pearson correlation of ranks). Scale differs (natural log vs. log2, and ours is not variance-stabilised), so only sign and rank are compared, never magnitude (`phase3.crosscheck.compare_condition`).
- Pass rule (spec, fixed in advance): median per-condition sign agreement ≥ `min_median_sign_agreement=0.9`.
- Network cap: 25 GB (`crosscheck.network_cap_gb`). A running byte counter (probes + downloads) stops the script before exceeding it; any cell lines left unprocessed are reported below, not silently dropped.
## Tahoe's control
Confirmed directly against the DE table, not assumed: the example row for `CVCL_0023`/`4EGI-1`/0.05 µM/plate `1` has `n_cells_trt=1378`, `n_cells_ctrl=4862`, which match our own `conditions.parquet` `n_cells` for that exact condition and our own `controls.parquet` `n_cells` for `CVCL_0023`/`plate1` exactly. Tahoe's DE table control is the same plate-matched DMSO pseudobulk this project computes independently. Across the matched sample, our `n_cells` (treated) equalled Tahoe's `n_cells_trt` for 197/197 conditions (mismatches, if any, are noted below).
## Name mapping and coverage
- 48 distinct cell lines and 80 distinct drug names appear in the sample; cell line join needs no mapping (`Cell_ID_Cellosaur` is our own `cell_line`), plate join is a fixed string strip, drug join is `trim()` both sides.
### Cell lines not covered (0)

Every sampled cell line was located and processed.
### Unmatched conditions (3/200)
Cell line was covered but no Tahoe DE rows were matched for this (plate, drug, dose). **Correction
(final review, 2026-09-26):** the two Erdafitinib rows are a join bug, not missing data — the SQL
filter compared `trim(drug)` with the untrimmed name `'Erdafitinib '`. Fixed
(`crosscheck.drug_filter_values`, tested) but not re-run, since re-running means streaming ~25 GB
again and the result would only add 2 conditions to 197. BI-3406 on CVCL_1716 has no row in Tahoe's
table for that plate/dose.

| condition_id | cell_line | drug | dose | plate |
| --- | --- | --- | --- | --- |
| 0e8ba936737f7bcd | CVCL_0359 | Erdafitinib  | 0.0500 | plate1 |
| 7a0e5a671d22f720 | CVCL_0332 | Erdafitinib  | 0.0500 | plate1 |
| 03f4a19721033a32 | CVCL_1716 | BI-3406 | 0.5000 | plate2 |
## Results
- Matched 197/200 sampled conditions in Tahoe's DE table.
### Distribution of per-condition metrics
| metric | min | p25 | median | p75 | max | n |
|---|---|---|---|---|---|---|
| sign_agreement | 0.9628 | 0.9989 | 1.0000 | 1.0000 | 1.0000 | 197 |
| spearman | 0.8894 | 0.9459 | 0.9559 | 0.9618 | 1.0000 | 191 |

**Median sign agreement: 1.0000 -- PASS against `min_median_sign_agreement=0.9`.**

Median Spearman: 0.9559
### 5 worst conditions (lowest sign agreement)
| condition_id | cell_line | drug | dose | plate | n_genes | sign_agreement | spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e5c5efebe531f77 | CVCL_1119 | TAK-901 | 5.0000 | plate3 | 8209 | 0.9628 | 0.9642 |
| 29d9b1518b98e007 | CVCL_0218 | BI-78D3 | 0.5000 | plate2 | 3334 | 0.9913 | 0.9347 |
| fbe78a0e37292a5d | CVCL_1478 | HI-TOPK-032 | 0.5000 | plate2 | 4020 | 0.9955 | 0.9411 |
| cc3fdbfa7414519d | CVCL_1517 | AT7519 | 0.5000 | plate2 | 3400 | 0.9962 | 0.9590 |
| f8d194eb1d6e2b1d | CVCL_0218 | BAY1125976 | 0.5000 | plate2 | 2372 | 0.9962 | 0.9215 |

Duplicate gene symbols on Tahoe's side (dropped, all rows): 0 across all compared conditions. Dose conflicts (more than one Tahoe row per gene for one plate/drug/dose): 0.

**Coverage:** 197/200 sampled conditions were compared. The 3 others are excluded (reasons listed above). Cell lines were processed most-sampled-first, so an incomplete run covers a *selected*, not a uniformly random, subset of the seeded sample; the pass/fail call applies only to the compared conditions.
## Network
- 24823 MB of 25 GB cap spent: 447 Parquet-footer probes (~2682 MB, used only to locate shards) + 251 full shard downloads (22141 MB, used for the actual comparison). Every downloaded shard was deleted immediately after its rows were extracted; the scratch directory is removed at the end of the run.

## Known issue: memory footprint

The full run (2026-09-26, 2,021 s wall) peaked at 5.3 GB resident memory — within the project's
8 GB ceiling — but macOS reported an 18.6 GB peak *footprint* (memory compressed by the OS). A
diagnostic (`tmp/agents/footprint_local.py`) showed the footprint grows ~55 MB per 90 MB shard read
and is not released, whether or not DuckDB has a memory limit; remote footer probes add only ~5 MB
each. This is a one-off validation script, not part of `make phase1` / `make features`. If it is
re-run on a smaller machine, read each cell line's shards in a subprocess so the memory is returned.
