# Phase 5d — Project site: design

**Status:** approved in conversation 2026-10-02 (owner accepted all recommendations)
**Roadmap entry:** Phase 5 — Delivery, last sub-project. Done-when of Phase 5: "a stranger can open the demo, read the case
study in 60 seconds, and reproduce the headline number from the repo in under 10 minutes."

## Decisions

| # | Decision | Why |
|---|---|---|
| S1 | **Project site** at `simomounir.github.io/tahoe-drug-response`: landing page, case study, demo; author shown as the GitHub handle only | Owner decisions |
| S2 | `make site` assembles `dist/site-<version>/site/` locally (landing generated from the results, the generated case study, the demo + its data, `.nojekyll`) and zips it to `site.zip` with `SITE_MANIFEST.json` (sha256) | Generated content and 71 MB of demo data never enter git |
| S3 | Published as a **draft release** `site-v<N>` by the existing upload script (`--kind site`); the owner publishes | Same reversible, owner-controlled pattern as the bundle |
| S4 | `.github/workflows/pages.yml`: on a published `site-*` release (or manual dispatch with a tag), download `site.zip`, verify sha256, deploy with the official Pages actions. One-time owner step: Settings → Pages → Source "GitHub Actions" | Hosting free; deploy reproducible from a release |
| S5 | Landing numbers and verdict come from `reproduce.headline` on the saved results; figure from `figures.degradation_svg`; shared nav (Home · Case study · Demo · GitHub) on all three pages | No hand-written numbers; one navigation |
| S6 | Build fails if any internal link (href/src) in `dist/site` points to a missing file | A stranger never hits a dead link |

## Testing and done-when

Tests: site assembles with all files; landing numbers equal the headline; link check passes and catches a broken link;
`--kind site` uploads `site.zip` + manifest under tag `site-v1` as a draft; workflow uses `actions/deploy-pages`.
Done when the site builds and is checked locally (screenshots), the draft release exists, and after the owner publishes and
enables Pages the live `/`, `/case-study/`, `/demo/` and a demo data file return 200.
