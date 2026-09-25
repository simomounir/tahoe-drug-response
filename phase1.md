# Phase 1 — Ingest and Pseudobulk

**Status:** complete — all §11 criteria met on plates 1–3 (see §13)
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
  plate. Each plate is one dose, and plates come in triplets sharing one drug set
  (plates 1/2/3 = the same 92 drugs at 0.05 / 0.5 / 5 µM).
- **DepMap mapping?** All 8 slice lines resolve, via `metadata/cell_line_metadata.parquet`.
- **Usable partitioning?** Yes, by plate: shards are sorted by plate, so whole files can be
  skipped. Not by cell line: every shard holds all lines.
- **Throughput?** ~6.7 s per ~100 MB shard (5.0 s download at ~20 MB/s + 1.7 s reduce).
  One plate is 149–365 shards, i.e. roughly 17–41 minutes.

## 13. Current state (2026-09-25)

**Where we are:** Phase 1 is complete. The pipeline builds pseudobulk profiles, plate-matched
DMSO controls and logFC for **plates 1, 2 and 3** (the same 92 drugs at 0.05, 0.5 and
5 µM) and the 8 configured cell lines. Output checks run on every build, 43 offline tests
run in CI on Ubuntu and macOS, and an example QC report is committed at
`reports/example_qc_phase1.md`.

### How to run

```bash
make phase1          # plates, cell lines, thresholds all from configs/slice.yaml
make phase1-dry-run  # shards per plate and how many still need downloading
make test            # 43 offline tests (also run by CI on every push)
```

Outputs: `data/pseudobulk/{pseudobulk,logfc,conditions,controls,genes}.parquet`,
`reports/qc_phase1.md`, per-shard log at `data/cache/partials/<plate key>/shard_log.csv`.
Shard partials are cached per plate, so adding a plate never re-downloads the others and
`make phase1` resumes after a crash. A rebuild from cached partials takes ~9 minutes.

### Slice

| Cell line | Name | DepMap ID | Tissue |
|---|---|---|---|
| CVCL_0546 | SW480 | ACH-000842 | Bowel |
| CVCL_0459 | NCI-H460 | ACH-000463 | Lung |
| CVCL_0480 | PANC-1 | ACH-000164 | Pancreas |
| CVCL_1285 | HOP62 | ACH-000861 | Lung |
| CVCL_0399 | LoVo | ACH-000950 | Bowel |
| CVCL_1056 | A498 | ACH-000555 | Kidney |
| CVCL_0293 | HEC-1-A | ACH-000954 | Uterus |
| CVCL_0371 | KATO III | ACH-000793 | Esophagus/Stomach |

6 tissues. KATO III replaced Hs 766T (CVCL_0334), which was shallow (median 328 processed
cells per condition on plate3) and duplicated pancreas.

### Where the implementation differs from §7

| Spec | Implemented | Why |
|---|---|---|
| Shard manifest with status | `data/cache/shard_plate_map.parquet` (plate min/max per row group); a shard's partial file is its "done" marker | The map also picks which shards to download |
| `data/raw_subset/` partitioned by cell_line/dose | No raw copy kept. Each shard is reduced to per-(well, cell line) gene sums in `data/cache/partials/<plate key>/`, then deleted | Raw plates 1–3 would be ~61 GB; the partials are 6.8 GB |
| In-memory accumulators | DuckDB, 2 GB memory cap, spilling to disk. Each plate's gene rows are split on disk by cell line, then pseudobulk and logFC are reduced per (plate, cell line) and merged with one sort | A plate-wide aggregation sat right at the 2 GB cap (plate2: ~23M condition × gene sums); per line is ~1/8 of that. Peak RSS 2.5 GB vs the 8 GB ceiling |
| `float64` sums | Summed in DOUBLE, then each gene count is checked to be whole and stored as INTEGER; `library_size` is integer addition of those | UMI counts are integers; a fractional count stops the build instead of being rounded |
| `sums` / `normalized` tables | `pseudobulk.parquet` holds `sum_counts`, `n_cells`, `library_size`; CPM and log1p-CPM are not stored (`phase1.pseudobulk.cpm()` recomputes them) | Derivable columns more than doubled the size (283 → 126 MB per plate, study in `tasks.md` T4) |
| logFC | `logfc = ln(1 + cpm) − ln(1 + cpm_control)` per gene, FLOAT, for QC-passing treated conditions vs the QC-passing DMSO of the same line and plate; genes seen in either profile | FLOAT changes values by ≤ 2.4e-7 |
| `condition_id` from (cell_line, drug, dose) | adds `plate` | Controls are per plate |
| Cell counts from metadata | Only `pass_filter == 'full'` cells | The shards hold only those cells; with the filter, processed = atlas exactly |
| Gene index | `genes.parquet`: `gene` (INTEGER, = Tahoe `token_id`), `gene_symbol`, `ensembl_id`; every observed gene must have a symbol | Readable, joinable outputs |
| Output budget | `max_output_mb` in the config: null = report only, a number = fail above it | §11.2 limit available as a hard check when wanted |

Code: `src/phase1/stream.py` (shard reduce, per-plate combine, logFC in DuckDB),
`src/phase1/build.py` (everything after the shard loop),
`src/phase1/contracts.py` (schemas and invariants; a violation raises `ContractError`),
`src/phase1/genes.py`, `src/phase1/pseudobulk.py` (pandas reference, same maths),
`src/phase1/conditions.py`, `src/phase1/qc.py`, `scripts/run_phase1.py`,
`scripts/fetch_gene_metadata.py`, `scripts/make_fixture.py`.

### Result (plates 1–3)

| Measure | Value |
|---|---|
| Shards streamed | 612 (61.5 GB), one at a time, each deleted after use |
| Cells processed | 5,313,322, equal to the metadata's `pass_filter == 'full'` count |
| Conditions | 2,213 (plate1 726, plate2 743, plate3 744), `condition_id` unique |
| Kept (≥100 cells) | 2,123 / 2,213 (plate1 666, plate2 715, plate3 742) |
| DMSO controls kept (≥500 cells) | 24 / 24, 2,057–14,701 cells each |
| Treated conditions with logFC | 2,099 |
| Cell line × drug pairs with all 3 doses | 619 / 736 (109 with 2, 8 with 1) |
| Cells per condition | min 1, median 1,978, max 14,701 |
| Genes | 52,933 of 62,710 observed, all with a symbol; 9,777 zero-count |
| Output | `pseudobulk.parquet` 144 MB, `logfc.parquet` 274 MB, total 418 MB |
| Rebuild from partials | 9.3 min, peak RSS 2.5 GB |

### Throughput (per-shard log, §7.1)

| Measure | Value |
|---|---|
| Rows read / kept | 17,273,090 / 5,313,322 |
| Total time | 69.2 min for 612 shards |
| Rate | ~4,160 rows/s, ~20 MB/s download |
| Per shard (median) | 5.0 s download + 1.7 s reduce |
| Peak RSS while streaming | 2.2 GB |

### Findings to keep in mind

- **Plate triplets:** a drug's dose response needs its plate triplet (1/2/3, 4/5/6, …).
  Plates 4 and 5 share only 1 drug with plate3.
- **plate1 drops are technical, not biological:** 60 conditions fail, mostly whole wells
  that fail on every line at the lowest dose (e.g. olaparib, median 3 cells per line). plate3's two drops
  are Lonafarnib, which is cytotoxic at 5 µM.
- **Source naming:** `Erdafitinib ` has a trailing space in Tahoe's metadata, identically on
  all plates, so it doesn't split conditions. Downstream joins on drug name should strip
  whitespace.
- **Control detection** matches exact names from `control_drugs`; `Trametinib (DMSO_TF
  solvate)` is a drug, not a control.

### Definition of done (§11) — current state

| # | Item | State |
|---|---|---|
| 1 | `make phase1` rebuilds everything | Yes, from config; needs the network the first time |
| 2 | Tables exist, pass contracts, < 1 GB | Yes: 418 MB for 3 plates; schemas and invariants (incl. DepMap) checked on every build |
| 3 | QC report explainable | Yes: dropped conditions with reasons, per-line and control tables, zero-count genes; example committed |
| 4 | Tests pass offline in CI | Yes: 43 tests, GitHub Actions on Ubuntu and macOS, no network |
| 5 | Step 0 findings recorded | Yes, `docs/step0_findings.md` |
| 6 | Shard throughput log | Yes, see Throughput above |
| 7 | Raw subset deletable and rebuildable | Yes: raw shards are deleted after use; deleting `data/` and rerunning rebuilds everything |

### Next

Phase 2: the C++ aggregation kernel, which must reproduce `tests/fixtures/expected_*.parquet`
exactly and is benchmarked against the throughput above.

---

## Appendix — what phase 2 will need from this

The C++ aggregation kernel replaces §7.3 and must reproduce the Python reference
implementation **exactly**. To make that testable, phase 1 must leave behind:

- The Python reference implementation, kept in the repo and not deleted
- The committed fixture, with expected outputs stored alongside it
- The per-shard timing log, to serve as the baseline for the speed comparison
