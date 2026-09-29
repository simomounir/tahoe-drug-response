# Tahoe drug-response reproduction bundle $version

Snapshot of the evaluation data and results behind the case study of
https://github.com/simomounir/tahoe-drug-response (built from commit `$commit`).

| file | contents | size |
|---|---|---|
| `level1.zip` | per-condition scores and summaries of every model ($models) | $level1_mb MB |
| `level2.zip` | the evaluation set: conditions with drug and cell-line features, genes, logFC targets, DE gene sets, splits, and the evaluation config | $level2_mb MB |
| `MANIFEST.json` | sha256 and size of each archive, git commit, package versions | — |

## Use

From a clone of the repository:

    make reproduce          # level 1: recompute intervals, paired differences and the verdict (seconds)
    make reproduce-refit    # level 2: refit the per-dose mean, ridge and the neural network on both_unseen

Both compare against `reproduce/expected.json` in the repository and fail on any difference.

## Sources and licences

- **Tahoe-100M** (Vevo Therapeutics / Arc Institute), CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/): single-cell profiles from which the pseudobulk log
  fold-changes, conditions and splits are derived; differential-expression gene sets from its DESeq2 table.
- **DepMap 24Q4** (Broad Institute), CC BY 4.0: cell-line expression principal components and mutation flags are
  derived (principal components and flags computed by this project) from DepMap Public 24Q4. Attribution: DepMap, Broad
  (2024). DepMap 24Q4 Public. Figshare+. Dataset. https://doi.org/10.25452/figshare.plus.27993248 — licence:
  https://creativecommons.org/licenses/by/4.0/
- Drug descriptors and fingerprints computed with RDKit (BSD-3-Clause) from the dataset's SMILES.

This bundle is released under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) with the attributions above.
