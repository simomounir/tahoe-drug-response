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

Split assignment is deterministic given a seed, stored to disk, and versioned. The same
split files are used by every model. Re-drawing splits after seeing results is forbidden.

Each split has train / validation / test parts. The validation part is carved out of the
training side **using the same rule as the split itself** — a validation set for
`unseen_drug` holds out unseen drugs, not random conditions.

### 4.1 Leakage rules

These are the ways this evaluation can quietly become meaningless. Each must be enforced
in code and covered by a test.

- **Dose grouping.** All doses of the same compound live on the same side of a drug split.
- **Scaffold grouping.** Compounds are grouped by molecular scaffold (Bemis–Murcko or
  equivalent); an entire scaffold group moves together. Holding out a compound while its
  close analogue remains in training tests memorization, not generalization.
- **Control leakage.** Control profiles for a held-out cell line are not available at
  training time in `unseen_cell_line`. If the model needs a control profile to predict,
  that dependency is documented as a limitation and evaluated separately.
- **Normalization statistics.** Any centring, scaling, gene filtering or highly-variable-gene
  selection is fitted on the training part only, then applied to validation and test.
- **Feature provenance.** No feature may be derived from the response being predicted, or
  from any aggregate computed over the test conditions.
- **Cell line features.** DepMap features are static and external, so they are permitted
  for held-out cell lines. This is a deliberate choice: it mirrors the real use case, where
  a new cell line has been characterized but not yet screened.

A test asserts that no `condition_id`, compound, scaffold or cell line appears on both
sides of any split.

## 5. Metrics

### 5.1 Differentially expressed gene set

Most genes do not respond to most drugs. Metrics computed over all genes are dominated
by unchanged genes and look impressive while meaning little. The primary metrics are
therefore restricted to the DE genes of each condition.

DE genes are derived from the **ground truth** (treated vs matched control), per condition,
with a fixed rule recorded in `configs/eval.yaml` (test, effect-size threshold, significance
threshold, and a cap of top *n* by absolute effect). This set is used for scoring only and
is never made available to a model as input.

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
| 4 | `nearest_chemical` | Response of the most similar training compound (Tanimoto on fingerprints), in the same cell line where available |
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

_(none yet)_

## 13. Open questions

- Which DE test is appropriate for pseudobulk profiles with varying cell counts per
  condition — and does the cell count need to enter the weighting?
- Should `discrimination` be computed within cell line, across all held-out conditions,
  or both? (Both is probably right; within-cell-line is the harder version.)
- Is scaffold splitting too aggressive given only ~380 compounds? Check how many scaffold
  groups exist before committing — if the count is low, the unseen-drug split may be
  unstable, and that instability must be reported.
- How many conditions end up in `both_unseen`, and is that enough for stable intervals?
