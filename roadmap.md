# Roadmap

Predicting drug-induced transcriptional responses in cancer cell lines — and measuring
honestly whether a learned model beats simple baselines.

**Data:** Tahoe-100M (CC0, ~100M single cells, 50 cancer cell lines, ~1,100 compounds)
· DepMap (cell line genetics) · PubChem/SMILES chemistry, via the dataset's own annotations.

**Question:** given a compound and a cell line, can we predict how gene expression changes —
and does that hold for compounds and cell lines never seen during training?

**Constraints:** single laptop, €0 infrastructure, everything reproducible from a clean checkout.

---

## Phases

### Phase 1 — Ingest and pseudobulk
Stream the Tahoe shards, filter to a defined slice, aggregate cells into pseudobulk
profiles per (cell line, drug, dose) with matched controls, and produce a QC report.

**Done when:** one command rebuilds every output from nothing; contracts pass; QC report
explains what was dropped and why.
**Spec:** [`phase1.md`](phase1.md)

### Phase 2 — C++ aggregation kernel
Replace the Python aggregation with a C++ implementation wrapped via nanobind, built into
wheels for Linux and macOS in CI.

**Done when:** output matches the Python reference exactly on the fixture; a benchmark
reports a real speedup with the measurement method documented; wheels build and install
from CI.

### Phase 3 — Features
Chemistry features from SMILES (fingerprints, descriptors, scaffolds), cell-line context
from DepMap (baseline expression, mutations, dependencies), joined through a DuckDB layer
with contracts on every table.

**Done when:** every condition has complete features or is explicitly excluded; no feature
derives from the prediction target; scaffold groups are computed and available to the
splitter.

### Phase 4 — Evaluation and models
Build the harness first, against a dummy predictor. Then the baseline ladder. Then a
conditional neural model. All four splits, all metrics, bootstrap intervals throughout.

**Done when:** the results table regenerates with one command and the pre-registered
criterion in the evaluation spec has been answered — in either direction.
**Spec:** [`evaluation.md`](evaluation.md)

### Phase 5 — Delivery
Interactive demo (precomputed lookup first, live inference as an upgrade), case study page,
portfolio site, scheduled CI.

**Done when:** a stranger can open the demo, read the case study in 60 seconds, and
reproduce the headline number from the repo in under 10 minutes.

---

## Sequencing

| Phase | Estimate |
|---|---|
| 1 — Ingest | 2–3 weeks |
| 2 — C++ kernel | 1–2 weeks |
| 3 — Features | 2–3 weeks |
| 4 — Evaluation and models | 3 weeks |
| 5 — Delivery | 2 weeks |

Evenings and weekends. Phase 4 is where the interesting surprises live — protect its time
rather than letting phase 1 expand into it.

## Principles

- **Baselines before models.** The bar is ridge regression on good features, not zero.
- **Evaluation rules are fixed in advance.** See the evaluation spec; amendments are logged.
- **A negative result is a result**, and is reported as the headline if that is what the
  numbers say.
- **Infrastructure serves the result.** If a task does not change a number or save real
  time, it is not phase work.
- **Nothing is committed that cannot be rebuilt** — no data in the repo, no hand-written
  results tables.

## Status

- [x] Step 0 — remote schema exploration and slice sizing → `docs/step0_findings.md`
- [x] Phase 1 — `make phase1` builds plates 1–3 (92 drugs × 3 doses, 8 lines, 6 tissues) with
  contracts: 2,123/2,213 conditions, 24/24 DMSO controls, logFC for 2,099, 418 MB; 43 tests
  in CI. See [`phase1.md` §13](phase1.md).
- [ ] Phase 2
- [ ] Phase 3
- [ ] Phase 4
- [ ] Phase 5
