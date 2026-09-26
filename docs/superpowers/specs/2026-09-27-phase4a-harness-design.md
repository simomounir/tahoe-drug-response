# Phase 4a — Evaluation harness: design

**Status:** approved in conversation 2026-09-27, pending review of this written spec
**Roadmap entry:** Phase 4 — Evaluation and models (`roadmap.md`), first of three parts (4a harness,
4b baseline ladder, 4c neural model)
**Binding rules:** `evaluation.md` including amendments A1–A7 and `configs/eval.yaml`; the phase 3
feature layer (`docs/superpowers/specs/2026-09-26-phase3-features-design.md`)

---

## 1. Purpose

Build the referee before any player: one command that fits a model, scores it on every split with
the pre-registered metrics and intervals, and regenerates the results table. It is proven with a
dummy that predicts no change. Phase 4b and 4c only add models.

## 2. Scope

In scope: target matrix, DE sets, split files, the model interface, metrics, bootstrap intervals,
the dummy predictor, the results report, contracts and tests.

Out of scope: every real baseline (4b), the neural model and any tuning logic (4c), plots for the
case study (phase 5).

## 3. Decisions

| # | Decision | Why | Cost if wrong |
|---|---|---|---|
| E1 | **5 seeded repeats** of each split | With 45 lines and 91 drug clusters one draw holds out ~9 lines / ~18 clusters; repeats show whether a result depends on which ones landed in test (owner decision) | ~5× compute |
| E2 | Test ≈ 20%, validation ≈ 15% of the held-out unit, validation drawn from the training side by the same rule (`evaluation.md` §4) | Common proportions; keeps `both_unseen` test ≈ 9 lines × 18 clusters | Smaller or larger test sets |
| E3 | Gene universe = the **19,193 DepMap protein-coding genes** matched to our symbols | External and response-free; bounds the target matrix at ~0.9 GB (float32) | Non-coding responses ignored |
| E4 | Target = dense logFC matrix; genes absent from our sparse logFC are **exact zeros** (seen in neither treated nor control: ln 1 − ln 1) | Not imputation: the value is known | None |
| E5 | Scoring set = eligible lines (A2) ∩ `features_complete` ∩ has a Tahoe DE set (A5, A7), for **every** model | Comparable numbers across model families | Fewer scored conditions |
| E6 | Discrimination by Pearson distance over the genes that are DE in ≥ 1 test condition, reported **across all test conditions and within cell line** | Answers `evaluation.md` §13; within-line is the harder version | Choice of distance |
| E7 | Constant predictions: correlations count as 0; discrimination ties get the average rank | Makes the dummy scoreable and puts it at chance | None |
| E8 | Headline = mean over repeats of per-repeat medians; interval = bootstrap of conditions **within** each repeat (2,000 resamples), averaged over repeats; paired for model vs baseline | A condition tested in several repeats is not independent evidence | Wider or narrower intervals than a pooled bootstrap |
| E9 | DE rows fetched once per cell line **in a subprocess**, shards deleted after use; reusable shard-location code moved from `scripts/crosscheck_tahoe_de.py` into `src/phase4/tahoe_de.py` | The cross-check's footprint grew ~55 MB per shard and was never released | ~22 GB streamed once |

## 4. Components

`src/phase4/` (package `phase4`), each module with one job:

| Module | Job | Key interface |
|---|---|---|
| `tahoe_de.py` | Locate and download Tahoe DE shards, select a condition's rows (trimmed drug, µM dose, plate) | `fetch_line_de(cell_line, conditions, cfg) -> pd.DataFrame` |
| `targets.py` | Gene universe, dense target matrix, scoring set | `gene_universe(depmap_dir, genes) -> list[str]`, `build_targets(...) -> (matrix, condition_ids, genes)` |
| `splits.py` | Draw the four splits × 5 repeats | `draw_splits(conditions, drug_groups, cfg) -> pd.DataFrame` (repeat, split, condition_id, part ∈ {train, val, test}) |
| `metrics.py` | Per-condition metrics | `score_conditions(pred, truth, de_sets, conditions, cfg) -> pd.DataFrame` |
| `uncertainty.py` | Repeat-aware bootstrap, paired differences | `summarize(per_condition, cfg) -> pd.DataFrame`, `paired(a, b, cfg)` |
| `predictors.py` | `Predictor` protocol, `TrainData`, `DummyPredictor` | `fit(train: TrainData)`, `predict(conditions) -> np.ndarray` |
| `harness.py` | Fit + predict + score one model on all repeats × splits, record runtime and peak RSS | `evaluate(predictor_factory, cfg) -> path` |
| `report.py` | Regenerate `reports/results.md` from saved results | `render_results(results_dir) -> str` |
| `contracts.py` | Contracts in §6, reusing `phase1.contracts.ContractError` | one `validate_*` per artefact |

Scripts: `scripts/build_eval_data.py` (DE fetch + targets + splits), `scripts/run_eval.py --model <name>`.
Make targets: `eval-data`, `eval` (`MODEL=`), `results`.

### Model interface

```python
@dataclass
class TrainData:
    features: pd.DataFrame      # condition_features rows for train and val
    targets: np.ndarray         # [n_train_val, n_genes] logFC, same row order
    genes: list[str]
    part: np.ndarray            # 'train' or 'val' per row

class Predictor(Protocol):
    def fit(self, train: TrainData) -> None: ...
    def predict(self, conditions: pd.DataFrame) -> np.ndarray: ...   # [n, n_genes], no NaN
```

`predict` receives feature rows only. All preprocessing happens inside `fit` on training rows.

## 5. Data flow

1. `make eval-data`:
   - read `condition_features` (phase 3) and `configs/eval.yaml`;
   - scoring set per E5;
   - per eligible line, in a subprocess: fetch plates 1–3 DE shards, keep our conditions' rows with
     `padj < 0.05`, protein-coding, top 200 by `padj` (ties by |log2FC|) → `data/eval/de_sets.parquet`;
   - dense targets → `data/eval/targets.npy` + `data/eval/targets_index.parquet` (condition order, gene order);
   - splits → `data/eval/splits.parquet`; split sizes (per repeat, incl. `both_unseen`) → `reports/eval_data.md`.
2. `make eval MODEL=<name>`: for each repeat × split, fit on train+val, predict test, score, write
   `data/eval/results/<name>/per_condition.parquet` and `results.json` (metric summaries, runtime, peak RSS).
3. `make results`: `reports/results.md` — rows models, columns splits, cells median [95% interval],
   per-repeat table, exclusion counts.

## 6. Contracts

The build fails with `ContractError` when:

1. A split puts a condition, a drug cluster or a cell line on both train/val and test (and, for
   `both_unseen`, a test condition's drug or line appears in train).
2. Any of the 5 excluded lines (A2) or the 2 gap lines (A7) appears in any test part.
3. A DE set belongs to a condition outside the scoring set; conditions with < `min_de_genes` DE genes
   are counted as excluded, not scored.
4. A prediction is not exactly (test conditions × gene universe) or contains NaN/inf.
5. The target matrix's row or gene order differs from its index file.

## 7. Configuration additions (`configs/eval.yaml`)

```yaml
splits:
  repeats: 5
  seeds: [20260927, 20260928, 20260929, 20260930, 20261001]
  test_fraction: 0.20
  val_fraction: 0.15
metrics:
  topk: [50, 100]
bootstrap:
  resamples: 2000
  interval: 0.95
gene_universe: depmap_protein_coding
```

## 8. Testing

Offline, tiny synthetic data:

- Each split rule on 6 lines × 8 drugs (grouping, dose grouping, `both_unseen` discard rule, validation
  by the same rule, determinism per seed, excluded lines never in test).
- Each metric against hand-computed values; a perfect prediction gives Pearson 1, Jaccard 1, rank 1; a
  shuffled one is at chance on average.
- The dummy gives the E7 values.
- Paired bootstrap recovers a known difference; within-repeat resampling is used.
- Each contract has a failing case.
- `tahoe_de` selection: trimmed names, µM dose match, ambiguity raises (reusing the P3.1 cases).

## 9. Definition of done

1. From a clean checkout, `make eval-data` and `make eval MODEL=dummy` run and pass their contracts.
2. `reports/results.md` shows the dummy on 4 splits × 5 repeats with intervals.
3. `reports/eval_data.md` gives split sizes per repeat, including `both_unseen`, and exclusion counts.
4. `make test` passes, CI included.

## 10. Risks

- **Too few `both_unseen` conditions** for stable intervals: reported per repeat; if a repeat has
  fewer than 200 scored test conditions, that is stated next to the headline.
- **DE fetch time and memory:** ~22 GB, ~35 min; subprocess per line (E9) bounds memory.
- **Tahoe DE missing for some conditions:** excluded and counted (A5).
