# Phase 3 — Features: design

**Status:** approved in conversation 2026-09-26, pending review of this written spec
**Roadmap entry:** Phase 3 — Features (`roadmap.md`)
**Binding constraints:** `evaluation.md` (leakage rules §4.1), `phase1.md` §3 (laptop, €0, rebuildable)
**Research behind the decisions:** `docs/research/phase3-drug-representation.md`,
`docs/research/phase3-cell-context.md`, `docs/research/phase3-tahoe-landscape.md`

---

## 1. Purpose

Give every condition *(cell line, drug, dose, plate)* a numeric description that a model can use
for drugs and cell lines it has never seen, without any of that description coming from the
response being predicted. Also give the phase 4 splitter the drug groups it needs to keep
near-duplicate compounds on one side of a split.

## 2. Scope

In scope:

- **P3.0** Expand the slice from 8 to all 50 cell lines (plates 1–3 unchanged).
- **P3.1** Cross-check the phase 1 target (our logFC) against Tahoe's published differential expression.
- **P3.2** Drug features from SMILES.
- **P3.3** Drug similarity groups.
- **P3.4** Cell-line features from DepMap and Tahoe metadata.
- **P3.5** Feature layer (DuckDB view) and contracts.
- **P3.6** Amendments to `evaluation.md`, made before any result exists.

Out of scope (later phases or never):

- Drawing train/validation/test splits — phase 4's harness does that from `drug_groups`.
- Pretrained molecular embeddings (ChemBERTa etc.) — at most a phase 4 ablation.
- Copy number and CRISPR dependency features — lowest marginal value in the literature
  reviewed; revisit only if a phase 4 ablation says cell features matter.
- Any model or baseline.

## 3. Decisions and why

| # | Decision | Why | Cost if wrong |
|---|---|---|---|
| D1 | Slice = all 50 lines, plates 1–3 | With 8 lines, `unseen_cell_line` and `both_unseen` (the pre-registered headline) test on 1–2 lines. Same 61.5 GB stream as before; outputs ~1.4–2.6 GB | ~1 h streaming, output budget raised to 3 GB |
| D2 | Morgan **count** fingerprint, radius 2, 2,048 slots, plus RDKit 2D descriptors | Standard in chemCPA/PRnet; a 25-dataset benchmark (Praski et al. 2025) finds pretrained embeddings give negligible gain over ECFP | A better representation exists; ablation later |
| D3 | SMILES standardised to the parent molecule (cleanup → metal disconnect → largest fragment → normalise/reionise → canonical) | 7 slice drugs are salts/complexes; counter-ion substructures would make unrelated salts look similar | Rare edge cases mis-standardised; caught by the cleaning log |
| D4 | Drug groups by **Butina clustering on Tanimoto similarity ≥ 0.6**, not Bemis–Murcko scaffolds | All 92 slice drugs have distinct Bemis–Murcko scaffolds (probe 2026-09-26), so scaffold grouping groups nothing; scaffold splits overestimate performance (Guo et al. 2024) | Threshold too loose/tight; sizes at 0.5/0.7 are reported |
| D5 | Dose encoded as log10(µM) | Three discrete doses; the common pattern in CPA/chemCPA/PRnet | None material |
| D6 | DepMap **24Q4** from Figshare+ (DOI 10.25452/figshare.plus.27993248.v1, CC BY 4.0) | Scriptable with stable file IDs and MD5s; newer releases are portal-only and not reliably scriptable | Features two releases old; basal profiles of established lines change little |
| D7 | Cell expression features = PCA fitted on **all DepMap lines** (~1,700), k = 10, only projected for our lines | Fitting any reduction on our own lines would learn from the evaluation population; external fit keeps held-out lines legitimate | k too small/large; k is config, ablation later |
| D8 | Mutation features = damaging and hotspot flags for genes damaged in ≥ 5% of DepMap lines, plus Tahoe driver flags | Mutations are the other primary cell feature family in the literature; a frequency rule fixed in advance avoids choosing genes after seeing results | Some informative rare drivers missed; Tahoe flags partly cover them |
| D9 | In-experiment DMSO profiles are **not** cell features in phase 3 | For held-out lines they are held-out data (`evaluation.md` §4.1); a control-based variant is a phase 4 ablation limited to `random`/`unseen_drug` | None for phase 3 |
| D10 | Drug features for all 379 compounds, groups for the slice's drugs only | Features are per-molecule and cheap; clusters depend on which drugs are in the set | None |

## 4. Tasks

Order: P3.0 first. Then P3.1, P3.2 + P3.3, and P3.4 are independent of each other. P3.5 needs
P3.2–P3.4. P3.6 can be written any time before phase 4 starts and must cite P3.3's numbers.

### P3.0 Expand the slice

- `configs/slice.yaml`: `selected_cell_lines` = all 50 lines in Tahoe `cell_line_metadata`;
  `cell_line_info` (DepMap ID, tissue) for each, generated from that table by a script, not by hand;
  `max_output_mb: 3000`.
- Stream plates 1–3 again (new partial cache key because the line set changes), rebuild, compare
  the 8 original lines' rows against the current build: they must match exactly, because the
  per-(plate, line) reduction does not depend on other lines.
- **Done when:** `make phase1` passes contracts on 50 lines; the 8-line rows match the previous
  build exactly; QC report shows per-line retention; output within budget; `phase1.md` §13 and
  `roadmap.md` numbers updated.

### P3.1 Target cross-check

- Download only the Tahoe `pseudobulk_differential_expression` files needed to cover a fixed,
  seeded random sample of 200 treated conditions from our slice.
- Confirm from the table and dataset card what their control group is (expected: DMSO on the same plate).
- Per condition, over genes with Tahoe `padj < 0.05`: sign agreement between our `logfc` (natural
  log of CPM ratio) and their `log2FoldChange`, and Spearman correlation. Scale differs, so only
  sign and rank are compared.
- **Pass rule (fixed now):** median per-condition sign agreement ≥ 0.90. Below that, phase 4 is
  blocked until the difference is explained in the report.
- **Output:** `docs/target_crosscheck.md` (method, sample, distribution, worst conditions inspected).

### P3.2 Drug features

- `src/phase3/drugs.py`: standardise (D3), featurise (D2), record `featurizable` and a reason.
- Descriptors undefined (NaN/inf) for any drug are dropped for all drugs; the dropped names are logged.
- **Output:** `data/features/drugs.parquet` — `drug` (trimmed name, matching `conditions.drug`),
  `smiles_input`, `smiles_parent`, `cleaning_log`, `featurizable`, `exclusion_reason`,
  `morgan_counts` (UTINYINT[2048], counts capped at 255), one FLOAT column per kept descriptor.

### P3.3 Drug similarity groups

- `src/phase3/groups.py`: Tanimoto on Morgan counts over the slice's featurizable drugs; Butina at
  the configured threshold; deterministic tie-breaking by drug name.
- **Output:** `data/features/drug_groups.parquet` — `drug`, `cluster_id`, `cluster_size`,
  `murcko_scaffold`, `nearest_drug`, `nearest_tanimoto`. The build log reports cluster-size
  distributions at 0.5 / 0.6 / 0.7.

### P3.4 Cell features

- `scripts/fetch_depmap.py`: download the four 24Q4 files by Figshare file ID, verify MD5,
  convert to Parquet in `data/cache/depmap/`.
- `src/phase3/cells.py`: PCA on the full DepMap expression panel (genes standardised on that panel),
  loadings saved to `data/features/depmap_pca.parquet`; project the slice's lines; mutation flags per D8;
  Tahoe driver flags from `cell_line_metadata`.
- **Output:** `data/features/cells.parquet` — `cell_line`, `depmap_id`, `depmap_release` (`24Q4`),
  `pc_1 … pc_k`, `dmg_<GENE>` and `hot_<GENE>` flags, `drv_<GENE>` flags.

### P3.5 Feature layer and contracts

- `src/phase3/layer.py`: DuckDB view `condition_features` over `conditions` ⋈ `drugs` ⋈
  `drug_groups` ⋈ `cells`, adding `log10_dose_um`. It is a view, not a stored table.
- `src/phase3/contracts.py` (reusing `phase1.contracts.ContractError`) — see §6.
- `make features` builds P3.2–P3.4 outputs, runs the contracts and writes a short
  `reports/features.md` (exclusions, cluster sizes, DepMap coverage).

### P3.6 Evaluation-spec amendments

Recorded in `evaluation.md` §12 with date and reason (no results exist yet, so these are design
changes, not post-hoc edits):

- §4.1 "Scaffold grouping" → drug-similarity grouping from `drug_groups` (D4), with the probe result as reason.
- §4 / §13: slice numbers (50 lines, 92 drugs, 3 doses) and the answered open question on scaffold count.
- §5.1: DE genes defined with Tahoe's `padj` as the primary rule if P3.1 passes, else our own rule;
  the choice and threshold written into `configs/eval.yaml` before phase 4 code exists.
- §6 baseline 4 (`nearest_chemical`) uses the Tanimoto defined here.
- Positioning: results are compared with in-house baselines only; published Tahoe numbers use
  the full atlas and are not comparable.

## 5. Configuration (`configs/features.yaml`)

```yaml
depmap:
  release: 24Q4
  figshare_article: 27993248
  files: [Model.csv, OmicsExpressionProteinCodingGenesTPMLogp1.csv,
          OmicsSomaticMutationsMatrixDamaging.csv, OmicsSomaticMutationsMatrixHotspot.csv]
drugs:
  morgan_radius: 2
  morgan_n_bits: 2048
  butina_similarity: 0.6
  report_similarities: [0.5, 0.6, 0.7]
cells:
  pca_components: 10
  mutation_min_frequency: 0.05
crosscheck:
  n_conditions: 200
  seed: 20260926
  padj: 0.05
  min_median_sign_agreement: 0.90
```

## 6. Contracts

The build fails with `ContractError` when:

1. A slice drug is neither `featurizable` nor carries an `exclusion_reason`.
2. A feature column of a featurizable drug or of a cell line contains null, NaN or inf.
3. `drug_groups` does not contain exactly the slice's featurizable drugs, or a drug has more than one cluster.
4. A slice cell line has no DepMap ID, or its ID is absent from the 24Q4 expression file (no imputation).
5. `condition_features` does not have exactly one row per condition in `conditions.parquet`.
6. PCA loadings on disk do not reproduce the stored projections.

## 7. Testing

Offline tests on tiny synthetic inputs, one per behaviour:

- Standardisation: salt → parent; unparseable SMILES → `featurizable=False` with reason; inorganic excluded.
- Fingerprint determinism, and equal vectors for two SMILES spellings of one molecule.
- Butina: near-duplicates share a cluster, an unrelated drug is a singleton, result independent of input order.
- PCA: projecting a panel member reproduces its fitted coordinates; loadings round-trip through Parquet.
- Each contract in §6 has a failing-case test.
- Provenance: the feature build succeeds with no `pseudobulk.parquet` / `logfc.parquet` present.
- View: one row per condition, no nulls for featurizable drugs.

Real-data checks are run by the dispatcher (not CI): `make phase1` on 50 lines, `make features`,
and the P3.1 report.

## 8. Definition of done

1. From a clean checkout, `make phase1` builds the 50-line slice and `make features` builds
   `drugs`, `drug_groups`, `cells` and the `condition_features` view, with contracts passing.
2. Every condition has complete features or is excluded with a reported reason.
3. No feature derives from the response (provenance test in CI).
4. Drug similarity groups are available to the phase 4 splitter.
5. `evaluation.md` carries the P3.6 amendments, dated.
6. `docs/target_crosscheck.md` exists and passes its rule, or explains why not.

## 9. Risks

- **Shallow lines.** The 42 new lines hold fewer cells on average (our 8 held 31% of cells); some
  conditions will fail `min_cells_per_condition`. Reported per line by the QC report; not a failure.
- **DepMap coverage.** If any Tahoe line is absent from 24Q4 expression, contract 4 fails the build.
  Resolution then is an explicit decision (drop the line from cell-feature models, or a newer release),
  not imputation.
- **Tahoe DE table semantics.** If its control is not plate-matched DMSO, P3.1 compares different
  things; the report must say so before any threshold is applied.
- **Disk.** Partials for 50 lines estimated 21–40 GB (394 GB free).
