# Phase 5a — Case study page: design

**Status:** approved in conversation 2026-09-28, pending review of this written spec
**Roadmap entry:** Phase 5 — Delivery (`roadmap.md`), first of four sub-projects (5a case study, then reproduction
bundle + scheduled CI, interactive demo, portfolio site — each with its own spec)
**Binding rules:** `evaluation.md` §8 (claim), §10 (what the case study reports), §11 (no selection after seeing results);
roadmap principle "no hand-written results tables"

---

## 1. Purpose

One web page that tells the project's result to a stranger: a 60-second top layer for hiring managers, a methods
layer for scientists (owner decision: both, layered). Every number and figure on it is generated from the saved
evaluation results, and the headline wording follows the numbers, so the page cannot drift from the evidence.

## 2. Scope

In scope: `make case-study` → `reports/case_study.html`, a self-contained page (inline CSS and SVG, no network
requests); the pre-stated failure-case rule and its analysis; a small harness refactor so the page reuses the exact
fitting inputs; matplotlib as a dependency.

Out of scope (later sub-projects): hosting and the portfolio site, the interactive demo, the reproduction bundle and
scheduled CI, custom domains.

## 3. Decisions

| # | Decision | Why | Cost if wrong |
|---|---|---|---|
| C1 | **Generated static HTML**, one file, inline CSS + SVG | Looks like a web page, drops into GitHub Pages later, works offline (owner decision) | A later site generator may want a different shape |
| C2 | Layered content (§4): 60-second top, methods below | Two audiences (owner decision) | Longer page |
| C3 | **Failure case = the `both_unseen` test condition of repeat 0 whose measured logFC is farthest from the per-dose mean of the training rows** (Euclidean distance over that condition's DE genes); the rule is printed on the page | §11 forbids picking examples after seeing model scores; the rule uses truth and the simplest baseline only, and selects the most drug-specific response — what the models miss by falling back to the average | Another rule might show a more typical case |
| C4 | Predictions for the failure case come from **refitting** `global_mean`, `ridge`, `neural` on repeat 0 `both_unseen` fitting rows with the same config and seed, predicting only that condition | The harness stores scores, not predictions (~3 GB per model); a refit of one split takes about a minute | Neural on MPS may differ in the last digits from the scored run; stated on the page |
| C5 | `harness.fit_inputs(data, repeat, split) -> (TrainData-without-val_score parts, test_rows)` extracted from `evaluate` and used by both | One definition of "what a fit sees"; no duplicated split logic | Small refactor of committed code, covered by an equality test |
| C6 | Headline and comparison sentences are **chosen from the numbers**: "no detectable difference" if the paired interval contains 0, "beats"/"falls short of" otherwise; the verdict sentence follows `report.claim_verdict` | Prose cannot contradict a rerun | Wording less natural than hand-written |
| C7 | Figures with matplotlib (pinned), rendered to SVG strings; colour-blind-safe palette; light/dark via CSS variables; readable at 375 px width | Accessible, self-contained, one dependency | matplotlib styling limits |

## 4. Page content

Top layer (60 seconds):

1. Title and the question: can a model predict how a new drug changes gene expression in a new cancer cell line?
2. Verdict block: the pre-registered claim, the verdict, and the `both_unseen` numbers (neural, ridge, per-dose mean)
   with the generated sentence (C6).
3. "Why the result is trustworthy": rules fixed before any model; baselines first with comparable tuning; 5 repeated
   splits with repeat-aware bootstrap intervals; amendments logged (count from `evaluation.md` §12).
4. Degradation curve: median `de_pearson` with 95% intervals for `global_mean`, `ridge`, `neural` across `random`,
   `unseen_cell_line`, `unseen_drug`, `both_unseen`.

Methods layer:

5. Data: slice size (drugs, doses, lines, scoring conditions, genes) read from `data/eval/` and `reports/eval_data.md` inputs.
6. Evaluation design: splits, metrics, uncertainty, the claim rule verbatim (`report.CLAIM_RULE`).
7. Full results table (all models × splits, `de_pearson` with intervals) and the ablation table (full − no cell features).
8. Failure case (C3): the rule, the condition (line, drug, dose), a chart of the top 20 DE genes — measured vs
   `global_mean`, `ridge`, `neural` — and each model's `de_pearson` on it; a short explanation generated from those numbers.
9. Amendments and limitations: A1–A9 one line each (titles from `evaluation.md` §12); fixed limitations text (92 drugs,
   one dataset, pseudobulk, plates 1–3).
10. Reproduce: the `make` targets in order, with measured runtimes from `results.json`.

## 5. Components

| Unit | Job | Interface |
|---|---|---|
| `src/phase4/harness.py` | `fit_inputs` extracted; `evaluate` uses it | `fit_inputs(data, repeat, split) -> FitInputs(features, targets, part, fit_rows, test_rows)` |
| `src/phase5/failure_case.py` | C3 selection + C4 refit | `select_condition(data, repeat, split) -> str`; `predictions_for(condition_id, models, cfg, data) -> dict[str, np.ndarray]` |
| `src/phase5/figures.py` | SVG figures | `degradation_svg(summaries) -> str`; `failure_case_svg(genes, truth, preds) -> str` |
| `src/phase5/case_study.py` | gather numbers, choose sentences, render | `build_context(results_dir, eval_dir, cfg) -> dict`; `render(context) -> str` |
| `src/phase5/templates/case_study.html` | layout, CSS, prose with `string.Template` placeholders | — |
| `scripts/build_case_study.py`, Makefile `case-study` | entry point | writes `reports/case_study.html` |

## 6. Contracts

The build fails when:

1. A placeholder is left unfilled, or the page contains "nan"/"None".
2. A model required by the page (`global_mean`, `ridge`, `neural`) has no results.
3. The failure-case condition is not a repeat 0 `both_unseen` test condition.

## 7. Testing

Offline, tiny synthetic eval and results directories:

- Page renders; every headline number equals its `results.json` value.
- The headline sentence follows the verdict and intervals (win / tie / loss).
- `select_condition` picks the condition farthest from the per-dose mean on synthetic data, and only test rows.
- `fit_inputs` reproduces exactly the rows the harness used (`evaluate` output unchanged on the synthetic eval dir).
- Figures are well-formed SVG; contracts 1–3 each have a failing case.

## 8. Definition of done

1. `make case-study` builds `reports/case_study.html` from the real results in < 3 minutes.
2. Its numbers match `reports/results.md`.
3. Screenshots in light mode, dark mode and at phone width are shown to the owner.
4. `make test` passes.
