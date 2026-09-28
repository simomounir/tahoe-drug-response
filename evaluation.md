# Evaluation Design

**Status:** draft, written before any model exists — this is deliberate
**Applies to:** all phases; the harness is built in phase 4 but the rules are fixed now

---

## 1. Purpose

Define how success is measured **before** any model is trained, so that the reported
results are a finding rather than a choice made after seeing the numbers.

Everything in this document should be readable by a reviewer as a commitment. If a rule
here is changed after results exist, the change is recorded in §12 with the date and the
reason, and the original rule stays visible.

## 2. Principles

1. **Baselines first.** The full baseline ladder (§6) is implemented and scored before
   any neural model is written.
2. **The harness precedes the models.** It is built against a dummy predictor that
   returns zeros. Every model is scored the moment it exists; nothing is eyeballed.
3. **Test data is touched once.** Model selection, hyperparameters and thresholds are
   decided on validation splits only. The test split is scored at the end, once, per model.
4. **All splits are reported, always.** No result is presented without its harder siblings
   alongside it.
5. **A negative result is a result.** See the pre-registered criteria in §8.

## 3. Prediction target

For a condition *(cell line, drug, dose)*, the model predicts a vector over genes of
**log fold-change relative to the matched control** for that cell line (and plate, if
plate-matched controls exist — see phase 1 §7.3).

Rationale: absolute expression is dominated by cell line identity, so a model can score
well on it while knowing nothing about drug response. Log fold-change isolates the effect
that is actually of interest.

Secondary target, optional and reported separately: absolute log1p CPM. Useful as a
sanity check, never as the headline number.

## 4. Splits

Four splits, all reported in every results table, in this order:

| Split | Held out | Purpose |
|---|---|---|
| `random` | Random conditions | Sanity check only — always labelled as optimistic |
| `unseen_drug` | All conditions for a set of compounds | Chemical generalization |
| `unseen_cell_line` | All conditions for a set of cell lines | Biological context generalization |
| `both_unseen` | Compounds × cell lines, neither seen | The honest headline number |

**Slice (amendment A2).** Plates 1–3: 92 compounds × 3 doses × 50 cell lines (13 tissues),
13,772 conditions including 150 DMSO controls, logFC for 11,535 treated conditions
(`phase1.md` §13). Only cell lines with logFC for at least
50% of their treated conditions are eligible to be held out or trained on. By this rule 45 lines
are eligible; CVCL_1531, CVCL_1571, CVCL_1577 (0%), CVCL_1716 (27%) and CVCL_1715 (29%) are excluded
before any split is drawn. The next-lowest line has 61%, so the cut falls in a natural gap.

Split assignment is deterministic given a seed, stored to disk, and versioned. The same
split files are used by every model. Re-drawing splits after seeing results is forbidden.

Each split has train / validation / test parts. The validation part is carved out of the
training side **using the same rule as the split itself** — a validation set for
`unseen_drug` holds out unseen drugs, not random conditions.

### 4.1 Leakage rules

These are the ways this evaluation can quietly become meaningless. Each must be enforced
in code and covered by a test.

- **Dose grouping.** All doses of the same compound live on the same side of a drug split.
- **Similarity grouping** *(amendment A1; replaced "Scaffold grouping", see §12)*. Compounds
  are grouped by Butina clustering on count-Tanimoto similarity ≥ 0.6 over Morgan count
  fingerprints (`data/features/drug_groups.parquet`, `cluster_id`); an entire cluster moves
  together. Every drug-split report states each test compound's highest Tanimoto similarity to
  any training compound. Holding out a compound while its close analogue remains in training
  tests memorization, not generalization.
- **Control leakage.** Control profiles for a held-out cell line are not available at
  training time in `unseen_cell_line`. If the model needs a control profile to predict,
  that dependency is documented as a limitation and evaluated separately.
- **Normalization statistics.** Any centring, scaling, gene filtering or highly-variable-gene
  selection is fitted on the training part only, then applied to validation and test.
- **Feature provenance.** No feature may be derived from the response being predicted, or
  from any aggregate computed over the test conditions.
- **Cell line features.** DepMap features are static and external, so they are permitted
  for held-out cell lines. This is a deliberate choice: it mirrors the real use case, where
  a new cell line has been characterized but not yet screened. *(Amendment A3)* Source is
  DepMap 24Q4 (PCA fitted on all DepMap lines, never on the slice). Declared gaps: hTERT-HPNE
  (non-cancer, no DepMap entry) and COLO 205 (no 24Q4 expression) have no or partial cell
  features; HepG2/C3A uses its parent HepG2's expression. Models that use cell features are
  scored on conditions with `features_complete`; the 551 treated conditions of the two gap
  lines are reported separately, never imputed.
- **Scoring set** *(amendment A7)*. Every model, baselines included, is trained and scored on the
  same conditions: eligible lines (A2) ∩ `features_complete` ∩ conditions with a Tahoe DE set
  (A5). hTERT-HPNE and COLO 205 are therefore never drawn as held-out lines. Their conditions are
  scored in a separate table, only for models that do not use cell features.

A test asserts that no `condition_id`, compound, similarity cluster or cell line appears on
both sides of any split.

## 5. Metrics

### 5.1 Differentially expressed gene set

Most genes do not respond to most drugs. Metrics computed over all genes are dominated
by unchanged genes and look impressive while meaning little. The primary metrics are
therefore restricted to the DE genes of each condition.

DE genes are derived from the **ground truth** (treated vs matched control), per condition,
with a fixed rule recorded in `configs/eval.yaml` (test, effect-size threshold, significance
threshold, and a cap of top *n* by absolute effect). This set is used for scoring only and
is never made available to a model as input.

*(Amendment A5)* The test is Tahoe's published per-condition DESeq2 result
(`metadata/pseudobulk_differential_expression`, plate-matched DMSO control): DE genes are those
with `padj < 0.05`, capped at the 200 with the smallest `padj` (ties broken by larger
|log2FoldChange|), with no separate effect-size threshold, and `min_de_genes = 20`
(`configs/eval.yaml`). Ranking by `padj` rather than by unshrunk |log2FoldChange| avoids favouring
noisy low-count genes. A condition with no row in Tahoe's table has no DE set: it is excluded from
DE-restricted metrics and counted in the report, like the `min_de_genes` exclusion. Our own logFC agrees with that table in direction for a median 100% (worst
96.3%) of its significant genes over 197 sampled conditions (`docs/target_crosscheck.md`), so the
two describe the same effect; a single pseudobulk per condition has no replicates for a test of
our own.

Conditions with fewer than `min_de_genes` DE genes are excluded from DE-restricted metrics
and counted separately in the report. This exclusion is reported, not silent.

### 5.2 Primary metrics

| Metric | Definition | Why |
|---|---|---|
| `de_pearson` | Pearson correlation between predicted and true logFC, over the condition's DE genes | Does the model get the direction and relative magnitude right? |
| `topk_overlap` | Jaccard overlap of predicted vs true top-*k* changed genes (k = 50 and 100, reported separately) | Would the prediction point a biologist at the right genes? |
| `discrimination` | Rank of the correct condition when the prediction is matched, by distance, against all held-out conditions; reported as normalized rank and as top-1 accuracy | Is the prediction condition-specific, or just an average drug effect? |

`discrimination` is the metric that catches the most common failure mode: a model that
predicts a generic stress response for every compound can score respectably on correlation
and near chance here.

### 5.3 Secondary metrics

- `de_mse` — mean squared error on DE genes
- `sign_accuracy` — fraction of DE genes with the correct direction
- `all_gene_pearson` — reported for completeness, always alongside `de_pearson`, never alone

### 5.4 Aggregation

Metrics are computed **per condition**, then aggregated as the median across conditions,
with the interquartile range shown. Per-condition scores are written to disk so the
distribution can be inspected and the worst cases examined by hand.

Aggregate means are not reported as the headline: a few easy conditions can carry them.

## 6. Baseline ladder

Implemented in this order, all scored on all splits, all kept in the final results table.

| # | Baseline | Definition |
|---|---|---|
| 0 | `no_change` | Predict a zero vector — no response at all |
| 1 | `global_mean` | Predict the mean logFC across all training conditions |
| 2 | `drug_mean` | Mean logFC of that drug across training cell lines (undefined for unseen drugs — report coverage) |
| 3 | `cell_mean` | Mean logFC of that cell line across training drugs |
| 4 | `nearest_chemical` | Response of the most similar training compound (count-Tanimoto on Morgan radius-2, 2,048-slot count fingerprints — the same measure as the similarity groups, amendment A4), in the same cell line where available |
| 5 | `ridge` | Ridge regression from [chemistry features ⊕ cell line features] to the logFC vector |

Baseline 0 is not a joke. A model that fails to beat "nothing happens" on a given metric
is telling you something important about that metric.

Baseline 5 is the one to beat. In perturbation prediction, linear models on good features
are frequently competitive with deep models, so `ridge` is the real bar, not `no_change`.

## 7. Uncertainty

- Every reported metric carries a **bootstrap confidence interval**, resampling conditions
  (not genes), with the number of resamples fixed in config.
- Model-vs-baseline comparisons are **paired** over conditions: compute the per-condition
  difference, then bootstrap that difference. Report the interval on the difference, not
  two separate intervals eyeballed for overlap.
- Report effect sizes with intervals. No p-value is reported as a pass/fail threshold.
- Where many comparisons are made, say so plainly rather than adjusting silently.

A difference whose paired interval contains zero is reported as "no detectable difference",
not as a win.

## 8. Pre-registered criteria

Written before any model exists. These define what the project will claim.

**Primary claim under test:** a conditional neural model, given chemistry and cell-context
features, predicts drug-induced expression changes better than ridge regression on the
`both_unseen` split.

- **Success:** the paired bootstrap interval on the difference in `de_pearson`
  (neural − ridge, `both_unseen`, test) excludes zero in favour of the neural model,
  and `discrimination` shows the same direction.
- **Failure:** the interval contains zero, or favours ridge.

**In the event of failure, the write-up leads with that finding.** The project's value is
the rigorous comparison, not the win. This sentence exists so that future-me cannot quietly
reframe the project after seeing the numbers.

Secondary questions, reported regardless of outcome:

- How much does performance degrade from `random` to `both_unseen`? (Expected: a lot.)
- Which split is harder, `unseen_drug` or `unseen_cell_line`? (This is genuinely interesting.)
- Do cell-context features help at all, measured by ablation?

## 9. Harness design

- Model interface: `predict(conditions) -> array[n_conditions, n_genes]`, nothing else.
  Baselines and neural models implement the same interface.
- A `dummy` predictor returning zeros is the first implementation and stays in the repo
  as a harness test.
- Scoring takes (predictions, split, ground truth) and emits a machine-readable
  `results.json` plus the per-condition scores.
- One command regenerates the full results table for every model and split.
- All randomness seeded and recorded; results reproducible on a clean checkout.
- Runtime and peak memory per model recorded alongside the metrics.

## 10. Reporting

The results table is generated, never hand-written. Rows are models (baselines included),
columns are splits, cells carry the metric and its interval.

The case study reports:

1. The `both_unseen` headline, with the ridge comparison next to it
2. The full table
3. The degradation curve across splits
4. At least one concrete failure case, inspected and explained

## 11. Forbidden practices

Listed explicitly so that a reviewer can see they were considered:

- Tuning anything on the test split
- Re-drawing splits after seeing results
- Reporting the best split and omitting the others
- Selecting genes or conditions after seeing which ones the model handles well
- Dropping "outlier" conditions without a rule defined in advance
- Comparing against a baseline that was not given the same features and tuning effort

## 12. Amendments

Any change to this document after results exist is logged here: date, what changed, why,
and what the rule was before.

All amendments below were made on **2026-09-26, before any model, baseline or split existed**
(phase 3 design, `docs/superpowers/specs/2026-09-26-phase3-features-design.md`).

- **A1 — §4.1 drug grouping.** *Before:* "Compounds are grouped by molecular scaffold
  (Bemis–Murcko or equivalent); an entire scaffold group moves together." *Now:* Butina clusters
  at count-Tanimoto ≥ 0.6. *Why:* all 92 slice compounds have distinct Bemis–Murcko scaffolds, so
  the old rule was a no-op; scaffold splits are known to overestimate performance (Guo et al.
  2024, arXiv:2406.00873). *Observed:* at 0.6 only one pair merges (EX229 / MK-3903, 0.65); the next
  most similar pair is 0.55, so near-duplicate leakage across drug splits is minimal and
  `unseen_drug` is close to holding out single compounds.
- **A2 — §4 slice and line eligibility.** *Before:* no coverage rule; §13 assumed ~380
  compounds. *Now:* the 92-compound, 50-line slice, and lines need logFC for ≥ 50%
  of treated conditions to enter any split. *Why:* five lines have logFC for 0–29% of treated
  conditions (too few cells, or no QC-passing control on most plates) and would add held-out
  lines with few or no scoreable conditions.
- **A3 — §4.1 cell line features.** *Before:* "DepMap features … are permitted for held-out cell
  lines." *Now:* same, plus the declared gaps and the rule that gap conditions are reported
  separately. *Why:* three of the 50 lines are not fully covered by DepMap 24Q4 (owner decisions).
- **A4 — §6 baseline 4.** *Before:* "Tanimoto on fingerprints". *Now:* count-Tanimoto on the
  phase 3 Morgan count fingerprints. *Why:* one similarity measure for grouping and baseline.
  *Note:* maximum pairwise similarity in the slice is 0.65, so this baseline has little signal
  to use.
- **A5 — §5.1 DE genes.** *Before:* "a fixed rule recorded in `configs/eval.yaml` (test,
  effect-size threshold, significance threshold, cap)", with the test left open (§13). *Now:*
  Tahoe's DESeq2 `padj < 0.05`, top 200 by smallest `padj`, `min_de_genes = 20`; conditions
  without a Tahoe row are excluded and counted. *Why:* the phase 3
  cross-check (P3.1) passed its pre-set rule (median sign agreement ≥ 0.90) at 1.00 on 197/200
  conditions, and our single pseudobulk per condition has no replicates for a test of its own.
  *Cost:* scoring needs Tahoe's DE rows for plates 1–3 of every eligible line (~22 GB streamed
  once, as in the cross-check).
- **A7 — §4.1 scoring set.** *Before:* not stated. *Now:* all models are trained and scored on
  eligible ∩ `features_complete` ∩ has-DE-set; the two gap lines are never held out and are
  reported separately. *Why:* otherwise models that use cell features and models that don't
  would be scored on different conditions and could not be compared (final review, 2026-09-26).
- **A6 — positioning.** Results are compared with this project's own baselines only. Published
  Tahoe-100M results use the full atlas (50 lines × 379 compounds, other splits) and are not
  comparable to this 92-compound slice.

Amendments made **after** results existed:

- **A8 — §6 baseline 5 (2026-09-27, post-hoc, owner decision).** *Before:* "Ridge regression from
  [chemistry features ⊕ cell line features] to the logFC vector." *Now:* ridge fits the residual from
  the per-dose mean of its fitting rows and adds that mean back, so full shrinkage gives exactly
  `global_mean`; `Ipc` enters as `log1p(Ipc)` (made before any ridge result was read). *Why:* on the
  first full run ridge scored 0.597 vs `global_mean` 0.666 (`de_pearson`, `both_unseen`), with α at
  the grid maximum in 3/5 repeats: an isotropic penalty shrinks the dose term along with ~2,340
  features, so full shrinkage gave the all-dose mean. The change makes the reference model stronger,
  so it cannot favour the neural model in §8. *Result after:* ridge 0.666 on `both_unseen`, no
  detectable difference from `global_mean`. Details: phase 4b spec amendments B7, B8.
- **A9 — §6 baseline 5 tuning range (2026-09-27, post-hoc, before any neural test result).** *Before:* α grid
  10^-2 … 10^6 (17 values). *Now:* 10^-4 … 10^6 (21 values). *Why:* ridge chose the grid minimum in all
  5 `random` repeats, so the grid edge, not the data, set its regularisation. Strengthens the reference
  only. The neural model's epoch cap was likewise raised (50 → 200, early stopping decides) after a
  train/val-only probe showed it still improving at 50 (phase 4c spec N9).

## 13. Open questions

- ~~Which DE test is appropriate for pseudobulk profiles with varying cell counts per
  condition?~~ Answered 2026-09-26: Tahoe's DESeq2 table (amendment A5).
- Should `discrimination` be computed within cell line, across all held-out conditions,
  or both? (Both is probably right; within-cell-line is the harder version.)
- ~~Is scaffold splitting too aggressive given only ~380 compounds?~~ Answered 2026-09-26: the
  92 slice compounds have 92 distinct Bemis–Murcko scaffolds, so scaffold grouping groups
  nothing; replaced by similarity grouping (amendment A1).
- How many conditions end up in `both_unseen`, and is that enough for stable intervals?
