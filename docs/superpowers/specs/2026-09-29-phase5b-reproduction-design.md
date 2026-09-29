# Phase 5b — Reproduction bundle and scheduled CI: design

**Status:** approved in conversation 2026-09-29, pending review of this written spec
**Roadmap entry:** Phase 5 — Delivery (`roadmap.md`), second sub-project (5a case study done; then 5c demo, 5d site).
Done-when of Phase 5 this serves: "reproduce the headline number from the repo in under 10 minutes".
**Binding rules:** `evaluation.md` §8 (claim), §9 (reproducibility); roadmap principle "no data in the repo".

---

## 1. Purpose

Let a stranger check the headline without the ~10 h full pipeline: download a small versioned snapshot of the
evaluation data and results, recompute, and compare with numbers committed in the repo. Keep that check alive with
scheduled CI.

## 2. Scope

In scope: `make bundle`, `make reproduce` (level 1), `make reproduce-refit` (level 2), a draft-release upload script,
committed expected values, a scheduled CI workflow, README reproduce section and badge.

Out of scope: Zenodo/DOI (possible later through Zenodo's GitHub integration, owner decision 2026-09-29), the full
pipeline (level 3 already exists as `make phase1 … eval-all`), the demo and the site.

## 3. Decisions

| # | Decision | Why | Cost if wrong |
|---|---|---|---|
| R1 | Two levels: **1 = statistics** from the saved per-condition scores (13 MB); **2 = refit** the claim's models on `both_unseen` from the eval set (~805 MB) | Level 2 retrains the models behind the claim; level 1 is a seconds-long check that CI can run cheaply (owner decision) | — |
| R2 | Hosted as **GitHub Release assets** (`level1.zip`, `level2.zip`, `MANIFEST.json`), not Zenodo | Reversible; the project is a portfolio piece first; a DOI can be minted later from a release (owner decision) | No DOI for now |
| R3 | Upload via `scripts/upload_release.py` with `GITHUB_TOKEN` from the environment, creating a **draft** release; the owner reviews and publishes. The script never publishes, never stores the token | Publishing is the owner's call; drafts are invisible to the public | — |
| R4 | `reproduce/expected.json` (committed) holds the headline values: per split and model (`global_mean`, `ridge`, `neural`) the `de_pearson` estimate and interval, the paired neural − ridge `de_pearson` and `disc_rank_global` on `both_unseen`, and the verdict | The repo states what must come out; any drift fails loudly | Must be regenerated deliberately (`--write-expected`) if results ever change |
| R5 | Level 1 compares **exactly** at the reported precision (3 decimals); level 2 compares estimates within **±0.003** and the verdict exactly | Level 1 is deterministic (seeded bootstrap); level 2 retrains, and MPS/CPU differ in the last digits (phase 4c N8) | A real change smaller than 0.003 passes level 2 (level 1 still catches it) |
| R6 | Level 2 refits `global_mean` and `ridge` fully (α search included) and `neural` with **each fit's recorded dropout, weight decay and epoch count** from the bundle's `results.json`; `REFIT_SEARCH=1` reruns the full neural search | ~5× faster (< 10 min target on the M1); the recorded choices were made on val only | The default skips re-validating the search itself |
| R7 | `reproduce/bundle.json` (committed) holds the release tag, asset URLs and sha256; downloads are verified before use; `BUNDLE_DIR=<path>` uses a local copy instead | Integrity; offline use and tests | — |
| R8 | CI: `.github/workflows/reproduce.yml` — **monthly** schedule + manual dispatch runs tests and level 1; level 2 only on manual dispatch (input `refit: true`), CPU, 90 min timeout | Catches dependency and link rot; monthly is enough (owner decision); level 2 is too heavy to schedule. Note: GitHub disables scheduled workflows after 60 days without repo activity | Silent pause after inactivity (re-enable with one click) |
| R9 | Bundle README: contents, versions, sources and licences — Tahoe-100M (CC0), DepMap 24Q4-derived features (CC BY 4.0, attribution), RDKit-derived descriptors — and how to use it | Redistribution terms respected | — |

## 4. Components

| Unit | Job | Interface |
|---|---|---|
| `src/phase5/bundle.py` | build archives + manifest; verify checksums; fetch (URL or `BUNDLE_DIR`) | `build(eval_dir, out_dir, version) -> Path`; `verify(dir, manifest) -> None`; `fetch(level, bundle_json, cache_dir) -> Path` |
| `src/phase5/reproduce.py` | level 1 and level 2 recomputation, comparison with expected | `headline(results_dir, cfg) -> dict`; `refit_both_unseen(eval_dir, results_dir, cfg, search=False) -> Path`; `compare(got, expected, tol) -> list[str]` |
| `src/phase4/neural.py` | optional fixed config: `NeuralPredictor(..., fixed=None)` where `fixed={"dropout", "weight_decay", "epochs"}` skips the search | — |
| `scripts/build_bundle.py`, `scripts/reproduce.py` (`--refit`, `--write-expected`), `scripts/upload_release.py` | entry points | Makefile: `bundle`, `reproduce`, `reproduce-refit`, `bundle-upload` |
| `reproduce/expected.json`, `reproduce/bundle.json` | committed expectations and bundle location | — |
| `.github/workflows/reproduce.yml`, README section + badge | scheduled check | — |

## 5. Contracts

The command fails (non-zero exit, message naming the value) when:

1. A downloaded or local file's sha256 differs from the manifest.
2. Any recomputed value differs from `expected.json` beyond its tolerance, or the verdict differs.
3. `upload_release.py` is run without `GITHUB_TOKEN`, or the API reports the release is not a draft.

## 6. Testing

Offline, tiny synthetic data (reusing the phase 5a fixtures):

- `build` writes both archives and a manifest whose checksums verify; a flipped byte fails `verify`.
- Level 1 passes on a matching fake bundle and fails when one per-condition score is altered.
- Level 2 with recorded neural configs reproduces a tiny fake run within tolerance; `fixed` skips the search (one training per fit).
- The upload client against a stubbed HTTP layer: creates a draft, uploads the assets, never sends `draft: false`;
  a missing token raises a clear error.
- `compare` reports every mismatch, not only the first.

## 7. Definition of done

1. `make bundle` builds `dist/bundle-v1/` (~820 MB) and its manifest.
2. `BUNDLE_DIR=dist/bundle-v1 make reproduce` passes in seconds; `make reproduce-refit` passes with a measured time
   (target < 10 min on the M1; CPU-only time measured and stated in the README).
3. `make bundle-upload` creates a draft release when the owner runs it with a token; publishing and pushing are the
   owner's actions.
4. The workflow and README section exist; `make test` passes.
