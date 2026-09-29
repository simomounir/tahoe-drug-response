# Phase 5c — Interactive demo: design

**Status:** approved in conversation 2026-09-29
**Roadmap entry:** Phase 5 — Delivery, third sub-project ("interactive demo, precomputed lookup first").
**Binding rules:** `evaluation.md` §2.3 (test data touched once — the demo shows predictions made when the condition
was a test condition), §9; phase 5a page conventions (tokens, palette, escaping).

## 1. Purpose

Let a visitor pick a cell line, drug, dose and split and see, for that condition, the measured expression change next
to what the per-dose mean, ridge and the neural network predicted when the condition was held out — the case study's
result one condition at a time.

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | All four splits; models = per-dose mean, ridge, neural (owner decisions) | Shows degradation per condition; readable chart |
| D2 | Predictions come from refits via `harness.fit_inputs` (ridge full α search, neural with each fit's recorded config); for each (condition, split) the **first repeat in which the condition is a test condition**; conditions never held out in a split are absent for that split | The harness stored scores, not predictions; one deterministic prediction per (condition, split) |
| D3 | **Contract:** each refit's per-condition `de_pearson` equals the scored run's within 0.003 (the MPS tolerance, phase 4c N8); otherwise the export fails | The demo cannot show predictions that differ from what was evaluated |
| D4 | Values stored on each condition's DE genes only (≤ 200, rank order), as integers ×100; measured stored once per condition; per-condition scored `de_pearson` of each model per split | Size: ~30 MB total instead of GB |
| D5 | One JSON shard per drug + `index.json` (lines with names, drugs, doses, per split the medians of each model, which (line, drug, dose) exist) | A visitor loads < 1 MB |
| D6 | Static page `site/demo/` (HTML + one JS file + SVG drawn in JS), no framework or library; same tokens and palette as the case study; values rendered with `textContent` (no HTML injection) | Free hosting in 5d; consistent look; safe |

## 3. Components

- `src/phase5/demo_data.py`: `export(eval_dir, results_dir, cfg, out_dir, cell_meta=None) -> dict` (summary: conditions, bytes).
- `scripts/build_demo_data.py`, Makefile `demo-data` → `site/demo/data/`.
- `site/demo/index.html`, `site/demo/demo.js`.

## 4. Testing and done-when

Tests (tiny fixture): shard values equal targets and model predictions at the DE genes; first test repeat chosen; a
condition never tested in a split is absent; the refit/scored contract fails on a tampered score; index lists every
shard. Page: headless-Chrome smoke check through a local HTTP server (chart rendered for a sample condition).

Done when `make demo-data` runs in < 30 min and reports its size, the page works locally (light, dark, 375 px
screenshots shown to the owner), a sampled condition's numbers equal `per_condition.parquet`, and `make test` passes.
