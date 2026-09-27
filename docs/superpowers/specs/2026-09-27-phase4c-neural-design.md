# Phase 4c — Conditional neural model: design

**Status:** approved in conversation 2026-09-27, pending review of this written spec
**Roadmap entry:** Phase 4 — Evaluation and models (`roadmap.md`), third of three parts (4a harness,
4b baseline ladder, 4c neural model)
**Binding rules:** `evaluation.md` §2 (principles), §8 (pre-registered claim), §9 (harness), §11
(forbidden practices), §12 (amendments); the phase 4a harness and phase 4b baselines specs; `configs/eval.yaml`

---

## 1. Purpose

Answer the pre-registered claim (`evaluation.md` §8): does a conditional neural model, given the same chemistry
and cell-context features as ridge, predict drug-induced expression changes better than ridge on `both_unseen`?
Also answer the §8 secondary question "do cell-context features help at all?" by ablation. The goal is an honest
answer, not the highest score; a failure is reported first (§8).

## 2. Scope

In scope: an MLP predictor (`neural`), cell-feature ablations `neural_nocell` and `ridge_nocell`, a per-fit
hyperparameter search on val, a "Pre-registered claim" section in `reports/results.md`, the `evaluation.md` §12
entry for phase 4b amendment B8, PyTorch as a dependency.

Out of scope: factorised (chemCPA-style) encoders and low-rank outputs (possible follow-ups, not needed to answer
§8); pretrained molecular embeddings; plots and the case study (phase 5); any change to splits, metrics or the
scoring set.

## 3. Decisions

| # | Decision | Why | Cost if wrong |
|---|---|---|---|
| N1 | **MLP on ridge's exact inputs** (`baselines.scalar_columns` + log1p Morgan counts, standardised on the fitting rows) | Same features as the bar (§11 "same features"); any gain comes from non-linearity, i.e. drug × cell interactions ridge cannot express (owner decision) | A structured encoder might do better; out of scope |
| N2 | **Residual on the per-dose mean**, as ridge after B8 | Same floor as ridge: a network that learns nothing predicts `global_mean` | None |
| N3 | Architecture: input → 512 → 512 → n_genes, ReLU, dropout after each hidden layer, linear output | Standard; width matters less than regularisation with ~74 training drugs | Under- or over-capacity |
| N4 | Loss MSE over all genes; AdamW, lr 1e-3, batch 256, ≤ 50 epochs, early stopping on val median `de_pearson` with patience 5 | Val metric = the headline metric, as ridge's α (B5) | Other losses might favour DE genes more |
| N5 | **Search per fit** over 4 configs: dropout {0.2, 0.5} × weight decay {1e-4, 1e-2}; best val `de_pearson` wins (ties → larger weight decay, then larger dropout); refit on train + val for the chosen config's best epoch count | Repeats redraw splits, so one repeat's val can be another's test: tuning once per split and reusing it would let test data steer selection (§2.3). Per fit mirrors ridge's per-fit α. 4 configs vs ridge's 17 α values is comparable effort for the §11 parity rule | A wider search could find a better config; budget-limited (owner: ≤ 1 h) |
| N6 | Ablation = same class with `use_cell_features: false`: drops `pc_*`, `dmg_*`, `hot_*`, `drv_*`; registered as `neural_nocell` and `ridge_nocell` | Answers §8's secondary question for both model families in one table (owner decision) | +~1 h compute |
| N7 | PyTorch, trained on the M1 GPU (MPS) when available, else CPU; the device is recorded in `results.json` | Fits the 1 h budget; CPU alone is ~5× slower | MPS is not bit-reproducible (N8) |
| N8 | Seeds from config (`torch.manual_seed`, numpy); MPS runs may differ in the last digits between runs, stated in the report; CPU runs are bit-identical and are what the tests check | `evaluation.md` §9 asks for reproducibility; the GPU cannot fully guarantee it | Last-digit differences on rerun |
| N9 | **Test data touched once**: development and the timing probe use only train/val of one repeat × split; the real `make eval MODEL=neural` runs once, after the code is final. If the probe predicts > 1 h for both neural models, `max_epochs` is lowered before the real run, never after | §2.3 | A lower epoch cap could underfit |

## 4. Components

| Unit | Job | Interface |
|---|---|---|
| `src/phase4/neural.py` | MLP module, training loop with early stopping, per-fit search, predictor class | `NeuralPredictor(hidden, dropouts, weight_decays, lr, batch_size, max_epochs, patience, seed, use_cell_features=True)` with `fit(train)`, `predict(conditions)`, `info() -> {"dropout", "weight_decay", "epochs", "val_de_pearson", "device"}` |
| `src/phase4/baselines.py` | `RidgeBaseline` gains `use_cell_features: bool = True`; `scalar_columns(features, use_cell_features=True)` | unchanged otherwise |
| `src/phase4/predictors.py` | registry: `neural`, `neural_nocell`, `ridge_nocell` | as before; per-model kwargs from `configs/eval.yaml` `models.<name>` |
| `src/phase4/report.py` | "Pre-registered claim" section first | `claim_verdict(paired_de, paired_disc) -> str` |
| `configs/eval.yaml` | `models.neural`, `models.neural_nocell`, `models.ridge_nocell` blocks with every N3–N8 value | no numbers in code |
| `requirements.txt` | pin `torch` | CI installs the CPU wheel on Linux |
| `evaluation.md` §12 | dated entry for B8 (post-hoc, owner decision) | text |

The harness is unchanged: `neural` is an ordinary predictor using `TrainData.val_score`.

## 5. Claim verdict (report)

For `both_unseen`, from `uncertainty.paired(neural, ridge)`:

- **SUCCESS** if the `de_pearson` difference interval lies entirely above 0 **and** the `disc_rank_global`
  difference estimate is below 0 (lower rank is better).
- **FAILURE** otherwise, including a tie. The section shows both differences with intervals, the rule text
  from §8 verbatim, and the ablation rows (`neural_nocell`, `ridge_nocell`) next to their full-feature models.

The section is omitted when `neural` or `ridge` results are missing.

## 6. Contracts

Unchanged harness contracts (prediction shape, finite values, split leakage, model info JSON). Added:

1. `use_cell_features: false` leaves no `pc_*`, `dmg_*`, `hot_*`, `drv_*` column in the design matrix.
2. Early stopping uses only `val_score`; the refit never sees a val score.

## 7. Testing

Offline, tiny synthetic data, CPU:

- The MLP fits a pure interaction (y = drug feature × cell feature) that ridge cannot (MLP val Pearson clearly
  above ridge's).
- With its output layer zeroed, the predictor returns exactly the per-dose mean.
- The search picks the config with the best val score (stubbed `val_score`), ties go to larger weight decay then larger dropout,
  and the refit trains for the recorded epoch count.
- `use_cell_features=False` drops the cell columns (both `neural` and `ridge`).
- Two seeded CPU fits give identical predictions.
- `claim_verdict`: success, tie (FAILURE), loss (FAILURE), and de_pearson win with discrimination against it (FAILURE).

## 8. Definition of done

1. `make eval-all` runs all nine models and each passes the harness contracts.
2. `neural` + `neural_nocell` take ≤ 1 h together; peak RSS < 6 GB per model.
3. `reports/results.md` opens with the claim verdict and shows the ablation rows.
4. `evaluation.md` §12 has the B8 entry.
5. `make test` passes locally and in CI.

## 9. Risks

- **Few drugs:** ~74 training drugs per `both_unseen` fit against 2,048 fingerprint slots; the MLP may only
  memorise drugs. Strong regularisation is in the search; a failure is a result (§8).
- **Budget:** estimated ~10–20 s per training run on MPS, 5 per fit × 20 fits per model; the timing probe (N9)
  checks this before the real run.
- **CI install size:** the default Linux torch wheel includes CUDA (~2 GB); the workflow installs the CPU
  wheel from the PyTorch index on Linux.
