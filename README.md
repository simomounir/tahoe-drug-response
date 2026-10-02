# Predicting drug responses on Tahoe-100M — measured honestly

[![tests](https://github.com/simomounir/tahoe-drug-response/actions/workflows/test.yml/badge.svg)](https://github.com/simomounir/tahoe-drug-response/actions/workflows/test.yml)
[![reproduce](https://github.com/simomounir/tahoe-drug-response/actions/workflows/reproduce.yml/badge.svg)](https://github.com/simomounir/tahoe-drug-response/actions/workflows/reproduce.yml)

Given a compound and a cancer cell line, can a model predict how gene expression changes — including for compounds
and cell lines it has never seen? This project builds pseudobulk drug responses from the Tahoe-100M single-cell atlas,
fixes its evaluation rules before any model exists, and compares a neural network with a ladder of simple baselines.

**Site:** https://simomounir.github.io/tahoe-drug-response/ — landing page, case study and interactive demo.

**Result (pre-registered claim: FAILURE).** On new drugs in new cell lines, the neural network, ridge regression and
the average response at each dose score the same (median DE-gene Pearson 0.666); ridge beats the neural network on
every split. Details and figures: the case study (`make case-study` → `reports/case_study.html`).

| | |
|---|---|
| Question, data, phases | [`roadmap.md`](roadmap.md) |
| Evaluation rules and amendments | [`evaluation.md`](evaluation.md) |
| Design specs per phase | [`docs/superpowers/specs/`](docs/superpowers/specs/) |

<a id="reproduce"></a>
## Reproduce

Python 3.12 environment: `pip install -r requirements.txt`, then

| command | what it does | time |
|---|---|---|
| `make reproduce` | downloads the bundle's saved scores and recomputes every interval, the paired differences and the verdict; compares with [`reproduce/expected.json`](reproduce/expected.json) | seconds |
| `make reproduce-refit` | downloads the evaluation set (845 MB) and refits the per-dose mean, ridge and the neural network on all five "new drug + new line" splits, then compares | ~100 s on an Apple M1 (GPU or CPU) plus the download; slower on smaller machines |
| `make phase1 features eval-data eval-all` | the full pipeline from Tahoe-100M | ~10 h, ~22 GB streamed |

Until the bundle release is published, build it locally with `make bundle` and run
`BUNDLE_DIR=dist/bundle-v1 make reproduce` (or `reproduce-refit`). The bundle (release `bundle-v1`) contains data derived from Tahoe-100M (CC0) and DepMap 24Q4 (CC BY 4.0); see its README
for sources and attribution. `make test` runs the offline test suite.
