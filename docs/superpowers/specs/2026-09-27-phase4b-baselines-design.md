# Phase 4b — Baseline ladder: design

**Status:** approved in conversation 2026-09-27, pending review of this written spec
**Roadmap entry:** Phase 4 — Evaluation and models (`roadmap.md`), second of three parts (4a harness,
4b baseline ladder, 4c neural model)
**Binding rules:** `evaluation.md` §6 (baseline ladder), §7 (uncertainty), §9 (harness); the phase 4a
harness (`docs/superpowers/specs/2026-09-27-phase4a-harness-design.md`) and `configs/eval.yaml`

---

## 1. Purpose

Implement the five baselines of `evaluation.md` §6 on the phase 4a harness, score each on every split and
repeat, and report them next to the dummy. The ladder shows how much of the response is the average
drug or cell effect; `ridge` is the bar the phase 4c neural model must beat.

## 2. Scope

In scope: `global_mean`, `drug_mean`, `cell_mean`, `nearest_chemical`, `ridge`; a validation scorer
passed to models; per-fit model info in `results.json`; coverage and paired-vs-ridge sections in
`reports/results.md`; `make eval-all`.

Out of scope: dose-pooled variants of the mean baselines ("ignore dose"; noted for later, owner
decision 2026-09-27), the neural model (4c), plots (phase 5), new dependencies.

## 3. Decisions

| # | Decision | Why | Cost if wrong |
|---|---|---|---|
| B1 | Mean baselines and `nearest_chemical` are **per dose**: a prediction for dose d uses training conditions at dose d only | Response grows with dose; pooling doses makes the baselines needlessly easy to beat (owner decision) | Dose-pooled numbers are not reported until added later |
| B2 | A baseline with no value to look up **falls back one level** (§4 table) and counts it; the report gives the fallback rate as coverage | The harness forbids NaN; `evaluation.md` §6 asks for coverage of `drug_mean` on unseen drugs | A fallback flatters the baseline on splits where it is undefined; coverage makes that visible |
| B3 | "Training rows" = train + val for every baseline; only `ridge` uses the val/train distinction (to choose α) | The baselines have nothing to tune; val is part of the fit side of each split (phase 4a §5) | None |
| B4 | `ridge` in **numpy via one SVD per fit**, all 18,995 genes predicted directly | No scikit-learn/scipy in the env; one SVD gives the solution for every α; a compressed target would cap what ridge can express and lower the bar | A few seconds per fit |
| B5 | `ridge` α chosen from a fixed log grid by **validation median `de_pearson`** (the headline metric), then refit on train + val | Tuning on the metric the claim is stated in; val never touches test | α tuned to one metric |
| B6 | Models see the metric only through a callback `TrainData.val_score(pred_val) -> float` | Keeps metric code in `metrics.py`; a model cannot peek at DE sets of test conditions | None |

## 4. The models (`src/phase4/baselines.py`)

For a test condition (line L, drug D, dose d); every mean is over training rows' logFC vectors.

| Model | Prediction | Fallback (counted) |
|---|---|---|
| `global_mean` | mean of all training conditions at dose d | none |
| `drug_mean` | mean of drug D at dose d across training lines | global_mean(d), when D has no training condition at d |
| `cell_mean` | mean of line L at dose d across training drugs | global_mean(d), when L has no training condition at d |
| `nearest_chemical` | response in line L at dose d of the most similar drug that has a training condition in L at d; ties in similarity averaged | line L has none at d → the nearest drug (among drugs with a training condition at d) averaged across its training lines at d |
| `ridge` | linear map from standardised features to the logFC vector, per B4–B5 | none |

**Similarity** (`nearest_chemical`): count-Tanimoto on Morgan radius-2, 2,048-slot count fingerprints
(`Σ min / Σ max`), the measure of the phase 3 similarity groups (amendment A4). Computed in `predict`
between the test drugs and the training drugs (at most 92 × 92) from `morgan_counts`, since a test drug
may be absent from the training rows.

**Ridge features:** `log1p(morgan_counts)` (2,048), the RDKit descriptor columns, `log10_dose_um`,
`pc_*` (10), `dmg_*`, `hot_*`, `drv_*` flags — about 2,340 columns. Standardised with the mean and SD of
the fitting rows; zero-SD columns are dropped. Targets are centred on the fitting rows' mean, which is
the intercept. With `X = U S Vᵀ`, `W(α) = V diag(s / (s² + α)) Uᵀ Y`. Grid: α = 10^k, k = −2, −1.5, …, 6
(17 values); ties go to the larger α. X is float64 (small); Y and `Uᵀ Y` stay float32.

**Model info:** each model exposes `info() -> dict`. Baselines return `{"fallback_rate": float}` over
the test conditions of that fit; `ridge` returns `{"alpha": float, "val_de_pearson": float}`.

## 5. Harness and report changes

- `predictors.TrainData` gains `val_score: Callable[[np.ndarray], float]`: given predictions for the
  rows with `part == "val"` (in their `features` order), returns their median `de_pearson` via
  `metrics.score_conditions`.
- `harness.evaluate` builds that callback per fit, calls `model.info()` if present after `predict`, and
  writes `model_info: [{repeat, split, ...info}]` to `results.json`.
- `predictors.PREDICTORS` registers the five models.
- `report.render_results` adds:
  - **Coverage:** mean fallback rate per model × split (models without fallbacks omitted).
  - **Paired vs ridge:** for each other model, model − ridge on `de_pearson` and `disc_rank_global` per
    split, from `per_condition.parquet` via `uncertainty.paired`; estimate [interval], marked
    "no detectable difference" when the interval contains 0. Omitted if ridge has no results.
- Makefile: `eval-all` runs `eval` for every registered model, then `results`.

## 6. Contracts

Unchanged from phase 4a (prediction shape, NaN/inf, split leakage). Added:

1. `val_score` receives exactly the val rows' shape; anything else raises.
2. A model's `info()` must be JSON-serialisable; `fallback_rate` in [0, 1].

## 7. Testing

Offline, tiny synthetic data (a few lines × drugs × 3 doses, a handful of genes):

- `global_mean`, `drug_mean`, `cell_mean` equal hand-computed means per dose; each fallback fires on an
  unseen drug / line and is counted in `fallback_rate`.
- `nearest_chemical`: picks the most similar drug in the same line and dose; averages ties; uses the
  across-lines fallback for an unseen line; Tanimoto matches a hand-computed value.
- `ridge`: recovers a known linear map on noise-free data; picks a smaller α on noise-free than on noisy
  data; identical output on two runs; zero-SD columns do not produce NaN.
- Harness: `val_score` is called with the val rows only; `model_info` lands in `results.json`.
- Report: coverage table and paired-vs-ridge section render from fake results.

## 8. Definition of done

1. `make eval-all` runs all six models (dummy + five) and each passes the harness contracts.
2. `reports/results.md` shows the six models on 4 splits × 5 repeats with intervals, the coverage table,
   and paired differences vs ridge.
3. Peak RSS < 6 GB and < 15 min per model for 20 fits; runtime and RSS recorded per model.
4. `make test` passes, CI included.

## 9. Risks

- **Ridge memory:** Y for ~8,600 fitting rows is ~650 MB float32, plus `Uᵀ Y` and one prediction matrix;
  expected ~3 GB peak. If exceeded, solve genes in blocks (same result).
- **Fallback-heavy splits:** `drug_mean` on `unseen_drug` is global_mean(d) for every test condition by
  construction; the coverage table states this rather than hiding it.
- **Ridge near the dummy:** on `both_unseen` a linear model may barely beat `global_mean`; that is a
  result, reported as such (`evaluation.md` §6).
