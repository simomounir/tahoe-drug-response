# Phase 1 — Ingest and Pseudobulk

**Status:** in progress — pipeline runs end to end on real data for plate3 (see §13)
**Scope owner:** _you_
**Target:** a reproducible local dataset of pseudobulk drug-response profiles, built from Tahoe-100M, with a QC report.

---

## 1. Purpose

Turn a defined slice of the Tahoe-100M single-cell perturbation atlas into a small,
well-documented, reproducible table of **pseudobulk expression profiles per
(cell line, drug, dose)**, plus matched control profiles, on a single laptop, at zero cost.

Everything downstream (chemistry features, DepMap joins, baselines, models) depends on
this table being correct and rebuildable. Phase 1 is not done until it can be deleted
and regenerated with one command.

## 2. Non-goals

Explicitly **out of scope** for this phase:

- The C++ aggregation kernel (phase 2 — phase 1 ships the Python reference implementation)
- DepMap or chemistry features (phase 3)
- Any model, baseline or evaluation split (phase 4)
- Any cloud resource, orchestration tool or deployed service (phase 5)
- Cell-level modelling — this phase collapses cells to pseudobulk and does not preserve
  single-cell distributions

If a task does not contribute to "raw shards in, validated pseudobulk table out", it
belongs in a later phase.

## 3. Constraints

| Constraint | Value |
|---|---|
| RAM | 16 GB — peak process memory must stay under ~8 GB |
| Disk budget for raw subset | 120 GB hard ceiling |
| Network | Household broadband; downloads must be resumable and run over multiple sessions |
| Cost | €0 — local only, no cloud compute or storage |
| Accumulator precision | `float64` for all sums (see §7.3) |

## 4. Data sources

| Source | What | Notes |
|---|---|---|
| `tahoebio/Tahoe-100M` (Hugging Face) | Single-cell expression, Parquet shards | CC0 public domain. ~95.6M rows in the `expression_data` subset |
| Same repo, metadata subsets | Drug metadata (incl. canonical SMILES, PubChem CID), cell line metadata, gene metadata | Small — download in full |

**Licence note:** Tahoe-100M is released CC0, so the slice, the derived tables and the
results can all be published openly. Record the dataset revision/commit hash used, so
the build is reproducible if upstream changes.

## 5. Step 0 — Remote exploration (do this before downloading anything)

DuckDB can query remote Parquet over HTTP without a local copy. Use this to size the
slice and to confirm the real schema rather than assuming it.

Questions to answer and record in `docs/step0_findings.md`:

1. **Actual schema** of the expression subset: exact column names and types, how
   expression is encoded (sparse gene index + count arrays vs. dense), and what the
   join keys to drug/cell-line metadata are called.
2. **Cells per cell line** — full ranked table.
3. **Cells per (cell line, drug, dose)** — the distribution, and the fraction of
   conditions below candidate cell-count thresholds.
4. **Control conditions** — how controls (DMSO / vehicle) are labelled, and how many
   control cells exist per cell line and plate.
5. **Dose levels** — the actual dose values present, not assumed ones.
6. **Shard layout** — number of files, size per file, and whether shards are grouped by
   any useful key (plate, cell line). This decides whether filtering can skip whole files.
7. **Batch/plate structure** — which columns identify plate or run, since this affects
   whether controls must be matched within plate.

> Findings from step 0 override any assumption written elsewhere in this document.
> Update this spec rather than working around a mismatch.

## 6. Slice definition

Configured in `configs/slice.yaml`, not hardcoded.

**Selection criteria for cell lines** (target: 8–10):

- High cell count — prefer the most represented lines so conditions have depth
- Tissue diversity — at least 5 distinct tissues of origin, so context features have signal
- Mutational diversity — spread across common driver backgrounds, verified against
  DepMap identifiers so phase 3 can join cleanly
- Every chosen line must have a DepMap ID that resolves; drop any that does not

**Compounds:** all available compounds, no filtering.
**Doses:** all dose levels present.
**Genes:** keep all genes at ingest; gene filtering (e.g. highly variable genes) is a
modelling decision, made in phase 4 and not baked into this table.

`configs/slice.yaml` must contain: dataset revision, cell line list with DepMap IDs,
dose values, and QC thresholds. Changing this file and re-running must fully regenerate
the outputs.

## 7. Pipeline

### 7.1 Ingest

- Build a **shard manifest** first: one row per source file with URL, size and status
  (`pending` / `done` / `failed`).
- Process one shard at a time: read in batches, filter to the slice, append, mark done.
- **Resumable:** a crash or lost connection must lose at most one shard.
- Per-shard logging: rows read, rows kept, bytes, elapsed seconds, peak RSS.
  This log is a deliverable — it becomes the throughput evidence in the case study.
- Never hold more than one batch in memory. No full-shard `read_parquet` into pandas.

Output: `data/raw_subset/`, Parquet, partitioned by `cell_line` and `dose`, with a
target file size of 128–512 MB per part.

### 7.2 Condition table

One row per experimental condition, built from metadata and observed counts:

| Column | Type | Notes |
|---|---|---|
| `condition_id` | string | Stable hash of (cell_line, drug, dose, plate) — plate added after step 0: controls are per plate |
| `cell_line` | string | Slice member |
| `depmap_id` | string | For the phase 3 join |
| `drug` | string | Compound name |
| `pubchem_cid` | string | Nullable |
| `canonical_smiles` | string | Nullable; required for phase 3 |
| `dose` | float64 | Units recorded in step 0 findings |
| `is_control` | bool | Vehicle/DMSO |
| `n_cells` | int64 | Observed after filtering |
| `plate` | string | If present — needed for batch-matched controls |

### 7.3 Pseudobulk aggregation

Single streaming pass over `data/raw_subset/`, maintaining per-condition accumulators:

- `sum_counts[gene]` — **`float64`**. Summing millions of UMI counts in `float32`
  silently loses precision; this is non-negotiable.
- `n_cells`
- `sum_library_size`

Derived outputs, written separately from the raw sums:

- `mean_counts = sum_counts / n_cells`
- `cpm = sum_counts / sum(sum_counts) * 1e6`
- `log1p_cpm`

Memory estimate: ~7,600 conditions × ~20,000 genes × 8 bytes ≈ 1.2 GB. Fits in RAM,
so no multi-stage shuffle is needed. If step 0 shows a larger condition count, chunk by
cell line.

**Controls:** compute a control profile per (cell line, plate) if plate structure exists,
otherwise per cell line. Record which was used. Log fold-change against control is
computed and stored here, since every downstream phase needs it.

Output: `data/pseudobulk/` — one Parquet file per table (`sums`, `normalized`,
`logfc`, `controls`).

### 7.4 QC report

Generated as `reports/qc_phase1.md` plus the underlying Parquet, on every build:

- Cells per condition — distribution, and a list of conditions dropped for falling
  below `min_cells_per_condition`
- Conditions per cell line, before and after filtering
- Library size distribution per cell line, flagging outlier plates
- Control cell counts per cell line/plate, flagging any below `min_control_cells`
- Genes with zero counts across the entire slice
- A single summary line: conditions in, conditions out, % retained

Default thresholds (revisit after step 0, record the final values in `configs/slice.yaml`):

- `min_cells_per_condition`: 100
- `min_control_cells`: 500 per cell line/plate

Dropping conditions is expected. The report must say **which** and **why** — that
honesty is part of the deliverable.

## 8. Data contracts and validation

- A declared schema for every output table (column names, types, nullability), validated
  at runtime on write. A failed contract fails the build loudly; it does not warn.
- Key invariants asserted in code:
  - `condition_id` unique in the condition table
  - no negative counts
  - `n_cells` in the pseudobulk table matches the condition table
  - every non-control condition has a matching control profile
  - every `cell_line` present resolves to a `depmap_id`
- Schema changes require a version bump recorded in the table metadata.

## 9. Fixtures and tests

- **Committed fixture:** `tests/fixtures/` — a subsampled slice of ~2 cell lines,
  ~5 drugs, both doses, few hundred KB, in the same schema as the real data.
  Tests and the quickstart run against this and require no network access.
- Unit tests: filtering logic, accumulator maths (including a precision test that
  fails under `float32`), CPM/logFC correctness against a hand-computed example,
  control matching, contract violations.
- Integration test: full pipeline over the fixture, end to end, asserting the final
  table shape and a known value.
- CI runs the fixture path only. Target: under 2 minutes, no downloads.

## 10. Repository hygiene

- `data/` is gitignored from the **first** commit. No dataset file is ever committed.
- `reports/` output is gitignored except a committed example QC report for the case study.
- The raw subset is disposable: document how to delete it and rebuild.
- Config-driven, no magic constants in code.

## 11. Definition of done

Phase 1 is complete when **all** of the following hold:

1. `make phase1` rebuilds every output from nothing, given only the config, on a clean
   checkout.
2. The pseudobulk tables exist, pass all contracts, and are under 1 GB total.
3. The QC report is generated and its numbers are explainable.
4. The full test suite passes in CI against the committed fixture, without network access.
5. `docs/step0_findings.md` records the real schema and the slice sizing evidence.
6. The shard log gives concrete throughput numbers (rows/sec, GB processed, total wall time).
7. The raw subset can be deleted and regenerated without manual steps.

## 12. Open questions

Answered during step 0 (details in `docs/step0_findings.md`):

- **Sparse or dense?** Sparse: per-cell `genes` / `expressions` arrays. Aggregation
  unnests them; a leading marker token (`1`, value `-2`) must be dropped.
- **Plate-matched controls?** Yes: 2–3 `DMSO_TF` wells per plate, every cell line on every
  plate. Each plate is one dose.
- **DepMap mapping?** Still open.
- **Usable partitioning?** Yes, by plate: shards are sorted by plate, so whole files can be
  skipped. Not by cell line: every shard holds all lines.
- **Throughput?** ~15 s per ~80 MB shard (network-bound, ~5–7 MB/s). One plate is
  149–365 shards, i.e. roughly 40–90 minutes.

## 13. Current state (2026-09-24)

**Where we are:** the pipeline runs end to end on real data for **plate3 (5 µM)** and
the 8 configured cell lines. It produces pseudobulk profiles, plate-matched DMSO controls
and logFC, and output checks run on every build. Not done yet: DepMap IDs, CI, gene
symbols, zero-count gene list in the QC report.

### How to run

```bash
make phase1          # plates, cell lines, thresholds all from configs/slice.yaml
make phase1-dry-run  # how many shards / GB, no download
make test            # 20 offline tests
```

Outputs: `data/pseudobulk/{pseudobulk,logfc,conditions,controls}.parquet`,
`reports/qc_phase1.md`, per-shard log at `data/cache/partials/<key>/shard_log.csv`.
Shards already reduced are skipped, so `make phase1` resumes after a crash, and a
rebuild from cached partials takes ~3 minutes.

### Where the implementation differs from §7

| Spec | Implemented | Why |
|---|---|---|
| Shard manifest with status | `data/cache/shard_plate_map.parquet` (plate min/max per row group); a shard's partial file is its "done" marker | The map also picks which shards to download |
| `data/raw_subset/` partitioned by cell_line/dose | No raw copy kept. Each shard is reduced to per-(well, cell line) gene sums in `data/cache/partials/`, then deleted | The raw plate3 slice would be ~10 GB; the partials are 1.7 GB |
| In-memory accumulators | DuckDB with a 2 GB memory cap, spilling to disk | Stays far below the 8 GB ceiling |
| `condition_id` from (cell_line, drug, dose) | adds `plate` | Controls are per plate |
| `reports/qc_phase1.md` | as specified | — |
| logFC | `logfc = ln(1 + cpm) − ln(1 + cpm_control)` per gene, for QC-passing treated conditions vs the QC-passing DMSO of the same line and plate; genes seen in either profile | Natural log, pseudocount 1 on CPM |

Code: `src/phase1/stream.py` (shard reduce, combine, logFC in DuckDB),
`src/phase1/build.py` (everything after the shard loop),
`src/phase1/contracts.py` (schemas and invariants; a violation raises `ContractError`),
`src/phase1/pseudobulk.py` (pandas reference, same maths),
`src/phase1/conditions.py`, `src/phase1/qc.py`, `scripts/run_phase1.py`.
A test checks that the DuckDB path matches the pandas reference exactly; another checks
logFC against hand-computed values.

### plate3 result

| Measure | Value |
|---|---|
| Shards streamed | 149 (~10 GB), one at a time |
| Cells processed | 1,172,428 (metadata lists 1,326,434) |
| Conditions | 744 (736 drug + 8 DMSO), `condition_id` unique |
| Kept (≥100 cells) | 741 / 744 |
| DMSO controls kept (≥500 cells) | 8 / 8, 705–3,386 cells each |
| Treated conditions with logFC | 733 (736 treated − 3 dropped) |
| Cells per condition | min 26, median 1,133, max 7,701 |
| Output | `pseudobulk.parquet` 100 MB, `logfc.parquet` 161 MB, conditions + controls < 50 KB |
| Rebuild from partials | 2 min 41 s, peak RSS 2.1 GB |

### Findings to keep in mind

- **Dropped conditions are cytotoxic drugs, not pipeline errors:** Lonafarnib (CVCL_0293,
  CVCL_0334) and LY-2584702 (CVCL_0334) have few cells even in the metadata (51–182).
- **Coverage gap:** the expression shards hold a median 86% (range 47–99%) of the cells
  the metadata lists per condition. Probably a cell-level QC filter; unverified.
- **CVCL_0334's control** has 705 cells, close to the 500 threshold. The whole line is
  shallow: median 328 cells per condition vs 1,000–1,900 for the others. Consider
  replacing it when the DepMap/tissue check is done.
- **Control detection** now matches exact names from `control_drugs` in the config.
  `Trametinib (DMSO_TF solvate)` is no longer a control; the old test encoding that was
  wrong and has been corrected.
- **Output size:** ~260 MB per plate, so the 1 GB budget (§11.2) fits about 3–4 plates.
  Storing logFC as float32 or dropping genes absent from both profiles would help;
  decide before adding plates.

### Definition of done (§11) — current state

| # | Item | State |
|---|---|---|
| 1 | `make phase1` rebuilds everything | Yes, from config; needs the network the first time |
| 2 | Tables exist, pass contracts, < 1 GB | Yes: 261 MB, schemas + invariants checked on every build (DepMap invariant pending) |
| 3 | QC report explainable | Yes: dropped conditions with reasons, per-line and control tables; zero-count genes missing |
| 4 | Tests pass offline in CI | 20 tests pass offline; no CI yet |
| 5 | Step 0 findings recorded | Done, except DepMap |
| 6 | Shard throughput log | Implemented; plate3 was streamed before it existed, so its log is empty until the next fresh run |
| 7 | Raw subset deletable and rebuildable | Yes: raw shards are deleted after use, and a rerun re-downloads |

### Next steps

1. DepMap IDs + tissue for the 8 lines in `configs/slice.yaml`; add the invariant.
2. Gene index → symbol mapping from the gene metadata table.
3. Zero-count gene list in the QC report.
4. Decide output size policy, then add plates (e.g. plate4 = 0.05 µM, plate5 = 0.5 µM).
5. CI running `make test`.

---

## Appendix — what phase 2 will need from this

The C++ aggregation kernel replaces §7.3 and must reproduce the Python reference
implementation **exactly**. To make that testable, phase 1 must leave behind:

- The Python reference implementation, kept in the repo and not deleted
- The committed fixture, with expected outputs stored alongside it
- The per-shard timing log, to serve as the baseline for the speed comparison
