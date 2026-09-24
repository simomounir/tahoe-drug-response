# Step 0 findings

This document is the working record for the remote exploration required by [phase1.md](../phase1.md), §5. Everything below should be based on actual remote inspection of Tahoe-100M, not assumptions.

## 1. Actual schema of the expression subset

Verified from a live remote `DESCRIBE` on the first parquet shard in the dataset.

Expression shard (`data/train-00000-of-03388.parquet`):

- `genes`: `BIGINT[]`
- `expressions`: `FLOAT[]`
- `drug`: `VARCHAR`
- `sample`: `VARCHAR`
- `BARCODE_SUB_LIB_ID`: `VARCHAR`
- `cell_line_id`: `VARCHAR`
- `moa-fine`: `VARCHAR`
- `canonical_smiles`: `VARCHAR`
- `pubchem_cid`: `VARCHAR`
- `plate`: `VARCHAR`

This is a sparse-style per-cell array representation: a gene index array and a corresponding expression array, not a dense matrix. This means the ingest pipeline must aggregate by condition from the sparse arrays rather than assume a row-per-gene layout.

Metadata table (`metadata/obs_metadata.parquet`):

- `plate`
- `BARCODE_SUB_LIB_ID`
- `sample`
- `gene_count`
- `tscp_count`
- `mread_count`
- `drugname_drugconc`
- `drug`
- `cell_line`
- `sublibrary`
- `BARCODE`
- `pcnt_mito`
- `S_score`
- `G2M_score`
- `phase`
- `pass_filter`
- `cell_name`
- `__index_level_0__`

The likely join keys are `cell_line` / `cell_line_id` and `drug` / `canonical_smiles` / `drugname_drugconc`, with `plate` as the batch grouping column.

**Verified join (2026-09-24):** expression rows map to conditions via
`(sample, plate, cell_line_id = cell_line)`. `sample` is a **well**: one drug at one dose,
with all 50 cell lines pooled in it. Dose is only in the metadata (`drugname_drugconc`),
so this join is required to know the dose of an expression row. `BARCODE_SUB_LIB_ID` is a
per-cell key and is not needed.

**Expression encoding details (verified on real shards):**

- `genes` are integer gene indices in the range 1–62,712 (44,563 distinct seen on plate3).
  Mapping indices to gene symbols needs the gene metadata table — not done yet.
- Every cell's arrays start with **token `1` with value `-2`**. It is a marker, not a gene,
  and must be excluded before summing (it produced negative counts until it was removed).
- All other values are non-negative integers (raw counts stored as float).

## 2. Cells per cell line

The metadata table has 50 distinct cell lines.

Top 10 by count (from `COUNT(*) GROUP BY cell_line`):

- `CVCL_0546`: 6,364,321
- `CVCL_0459`: 5,943,411
- `CVCL_0480`: 4,170,586
- `CVCL_1285`: 3,420,378
- `CVCL_0399`: 3,274,272
- `CVCL_1056`: 2,785,040
- `CVCL_0293`: 2,766,680
- `CVCL_0334`: 2,744,482
- `CVCL_0023`: 2,664,482
- `CVCL_1119`: 2,495,566

This indicates a balanced but not uniform set: the dataset is already close to the target of 8–10 lines for a manageable slice, with plenty of room to choose a high-depth, diverse subset.

## 3. Cells per (cell line, drug, dose)

Not yet computed in full for the exact condition table, but the metadata table already confirms the dataset is structured around 50 cell lines and 380 unique drugs.

The next Step 0 task is to compute exact per-condition counts from the metadata and expression subset before deciding the initial slice.

**Update (2026-09-24):** across all 14 plates, the 8 slice lines have a median of ~2,550
metadata cells per (well, cell line). On plate3 after the full build, observed cells per
condition range 26–7,701 (median 1,133); 3 of 744 conditions fall below 100 cells (see
[phase1.md §13](../phase1.md)).

## 4. Control conditions

The obvious vehicle/control label is `DMSO_TF`, which appears as the highest-frequency drug in the metadata table:

- `DMSO_TF`: 2,330,156 cells

There is also a suspicious variant:

- `Trametinib (DMSO_TF solvate)`

This suggests the dataset uses a broad `drug` naming scheme and that control detection should not be purely string-based. We should normalize control IDs by checking both the `drug` field and the `drugname_drugconc` field, then confirm which rows correspond to vehicle controls.

**Update (2026-09-24):** each plate has 2–3 `DMSO_TF` wells (dose 0.0), so controls can be
plate-matched. **Known bug:** `is_control_drug` currently matches the substring `dmso`, so
`Trametinib (DMSO_TF solvate)` would be wrongly treated as a control. It does not occur on
plate3; fix before building other plates.

## 5. Dose levels

Dose values are not yet fully enumerated from the remote metadata, but the schema shows the relevant metadata field is `drugname_drugconc`, which appears to encode concentration in the description string (e.g. `[(Bestatin (hydrochloride), 0.05, 'uM')]`).

This means the real dose values should be extracted from the metadata field rather than guessed from a hardcoded list. The final slice config must record the actual values after verifying them in Step 0.

**Verified (2026-09-24):** three doses, all in µM: **0.05, 0.5, 5.0**. Each plate is
(almost entirely) one dose, so dose and plate are confounded:

| Dose | Plates |
|---|---|
| 0.05 µM | plate1, plate4, plate7, plate10, plate13 |
| 0.5 µM | plate2, plate5, plate8, plate11 |
| 5.0 µM | plate3, plate6, plate9, plate12, plate14 |

plate3 also has one well at 0.05 µM (8 conditions across the slice lines).

## 6. Shard layout

**Corrected 2026-09-24** (earlier figures of 4,419 files / 33,888 shards were wrong).
Verified from the Parquet footers of every shard (`scripts/shard_plate_map.py`,
output `data/cache/shard_plate_map.parquet`):

- **3,388** expression shards, `data/train-XXXXX-of-03388.parquet`
- ~28,200 cells per shard, ~70–100 MB per file
- row groups of 1,000 cells, each with plate min/max statistics in the footer
- shards are **sorted by plate**: only 13 of ~95,600 row groups contain more than one plate
- within a plate, cells from all wells and cell lines are **interleaved**, so any single
  shard holds a small piece of every condition on its plate (shard 0: ~15 cells per
  condition). A complete condition requires every shard of its plate.

Shards per plate:

| plate3 | plate7 | plate1 | plate9 | plate5 | plate14 | plate4 | plate11 | plate6 | plate2 | plate10 | plate13 | plate8 | plate12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 149 | 187 | 189 | 198 | 212 | 223 | 238 | 254 | 260 | 274 | 277 | 291 | 304 | 365 |

Consequence: files can be skipped **by plate**, not by cell line. The whole 8-line slice
across all plates would mean streaming all ~230 GB; one plate is ~10–30 GB.

## 7. Batch / plate structure

We verified the plate structure directly:

- `plate` is a real metadata column
- there are 14 distinct plates
- each plate includes all 50 cell lines and roughly 93–95 unique drugs

Plate-level counts:

- `plate1`: 5,481,420 cells, 50 lines, 93 drugs
- `plate10`: 8,044,908 cells, 50 lines, 95 drugs
- `plate11`: 7,435,869 cells, 50 lines, 95 drugs
- `plate12`: 10,487,057 cells, 50 lines, 95 drugs
- `plate13`: 8,501,658 cells, 50 lines, 44 drugs
- `plate14`: 6,518,806 cells, 50 lines, 95 drugs
- `plate2`: 8,064,658 cells, 50 lines, 93 drugs
- `plate3`: 4,705,402 cells, 50 lines, 93 drugs
- `plate4`: 7,005,356 cells, 50 lines, 95 drugs
- `plate5`: 6,419,498 cells, 50 lines, 95 drugs
- `plate6`: 7,545,393 cells, 50 lines, 95 drugs
- `plate7`: 5,692,117 cells, 50 lines, 94 drugs
- `plate8`: 8,880,979 cells, 50 lines, 94 drugs
- `plate9`: 5,866,669 cells, 50 lines, 94 drugs

This strongly suggests plate-matched control logic will matter for the final ingest design.

## 8. Slice recommendation

The dataset naturally suggests a first slice built on the most highly represented cell lines while keeping tissue diversity in mind.

Initial candidate lines from the top counts are:

- `CVCL_0546`
- `CVCL_0459`
- `CVCL_0480`
- `CVCL_1285`
- `CVCL_0399`
- `CVCL_1056`
- `CVCL_0293`
- `CVCL_0334`

These are the strongest starting options for a manageable 8–10 cell-line slice. The next step is to join these to DepMap metadata and verify tissue and mutational diversity before committing the exact list.

## 9. Open questions / decisions

Resolved:

- Controls: `DMSO_TF`, 2–3 wells per plate, plate-matched (§4).
- Doses: 0.05 / 0.5 / 5.0 µM, one dose per plate (§5).
- Slice: the 8 lines above, in `configs/slice.yaml`.

Still open:

- Which of the cell-line IDs map cleanly to DepMap IDs? Tissue diversity not yet checked.
- Why do expression shards contain only ~86% (median; range 47–99%) of the cells the
  metadata lists per condition? Hypothesis: shards hold only cells passing `pass_filter`.
  Not verified.
- Which plates to build beyond plate3 (e.g. one plate per dose).

---

## Notes from the first remote probe

- Date: 2026-09-23
- Environment: project-local `.venv` with DuckDB + PyArrow + pandas + Hugging Face Hub
- Remote repo checked: `tahoebio/Tahoe-100M`
- Verified facts: 4,419 parquet files in `data/`; 50 cell lines; 14 plates; 380 unique drugs; `DMSO_TF` appears as the strongest vehicle-control candidate.
- Current status: Step 0 remote schema exploration is underway and the initial slice candidates are identified; the remaining work is to validate the exact control logic, dose levels, and DepMap mapping before defining `configs/slice.yaml`.

## Status (2026-09-24)

Step 0 is complete except the DepMap/tissue check. Schema, join keys, controls, doses and
shard layout are verified; the plate3 build in [phase1.md §13](../phase1.md) was built on
these findings.
