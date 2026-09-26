"""Cross-check our from-scratch pseudobulk logFC against Tahoe's official DE table (spec P3.1).

Pure, offline-testable functions live here. Network I/O (sampling conditions from the real
build, locating and querying the remote DE parquet files) lives in
scripts/crosscheck_tahoe_de.py, which imports this module.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from phase3.config import DOSE_UNIT_TO_UM as UNIT_TO_UM, load_features_config

def load_crosscheck_config(features_yaml_path: Path | str | None = None) -> dict:
    """The `crosscheck` section of configs/features.yaml (via phase3.config; missing keys raise)."""
    return load_features_config(features_yaml_path)["crosscheck"]


def sample_conditions(conditions: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """A seeded random sample of up to `n` treated, QC-passing conditions.

    Excludes controls (`is_control`) and anything that failed QC (`qc_pass`), so every sampled
    condition has a real logFC to compare and a real plate-matched control behind it.
    """
    pool = conditions[(~conditions["is_control"].astype(bool)) & (conditions["qc_pass"].astype(bool))]
    n = min(n, len(pool))
    return pool.sample(n=n, random_state=seed).reset_index(drop=True)


def match_drug_name(ours_drug: str, theirs_drugs) -> str | None:
    """Match our drug name against Tahoe's DE drug names, trimmed on both sides (spec rule).

    Tahoe drug names carry stray whitespace (e.g. "Erdafitinib "); an exact match is preferred,
    falling back to a trim-insensitive match. Returns None, never raises, when nothing matches --
    callers are responsible for reporting unmatched names rather than dropping them silently.
    """
    if ours_drug in theirs_drugs:
        return ours_drug
    trimmed = ours_drug.strip()
    for name in theirs_drugs:
        if name.strip() == trimmed:
            return name
    return None


def drug_filter_values(drugs) -> list[str]:
    """Distinct trimmed drug names, sorted, for a `trim(drug) IN (...)` filter on Tahoe's DE table."""
    return sorted({str(d).strip() for d in drugs})


def compare_condition(ours: pd.DataFrame, theirs: pd.DataFrame, padj: float) -> dict:
    """Compare one condition's logFC against Tahoe's DE table, over Tahoe's significant genes.

    ours: columns `gene_symbol`, `logfc` (our natural-log CPM-ratio logFC).
    theirs: columns `gene_symbol`, `log2FoldChange`, `padj` (Tahoe's DESeq2 output).

    Only genes with Tahoe `padj < padj` are considered; genes present on only one side are
    dropped from the comparison (not an error -- this is exactly the "unmatched, not crashing"
    behaviour the spec asks for). Scale differs between natural-log and log2 fold changes, so
    only sign and rank are compared, never magnitude.

    Tahoe rows with NaN `padj` or NaN `log2FoldChange` are excluded (never counted as a sign
    disagreement). A gene symbol that appears more than once on Tahoe's side is ambiguous: all its
    rows are dropped and counted in `n_duplicate_symbols`. Signs are compared with `np.sign`, so an
    exact zero agrees only with an exact zero.

    Returns a dict with `n_genes` (genes actually compared), `sign_agreement` (fraction with
    matching sign), `spearman` (rank correlation, computed as Pearson correlation of ranks) and
    `n_duplicate_symbols`. `sign_agreement`/`spearman` are NaN, not raised errors, when there are
    zero matched genes or no variance to rank.
    """
    valid = theirs.dropna(subset=["padj", "log2FoldChange"])
    duplicated = valid["gene_symbol"].duplicated(keep=False)
    n_duplicate_symbols = int(valid.loc[duplicated, "gene_symbol"].nunique())
    valid = valid[~duplicated]
    sig = valid[valid["padj"] < padj]
    merged = ours.merge(sig[["gene_symbol", "log2FoldChange"]], on="gene_symbol", how="inner")
    n_genes = len(merged)

    if n_genes == 0:
        return {"n_genes": 0, "sign_agreement": float("nan"), "spearman": float("nan"), "n_duplicate_symbols": n_duplicate_symbols}

    ours_sign = np.sign(merged["logfc"].to_numpy())
    theirs_sign = np.sign(merged["log2FoldChange"].to_numpy())
    sign_agreement = float(np.mean(ours_sign == theirs_sign))

    ours_ranks = merged["logfc"].rank()
    theirs_ranks = merged["log2FoldChange"].rank()
    if ours_ranks.nunique() < 2 or theirs_ranks.nunique() < 2:
        spearman = float("nan")
    else:
        spearman = float(ours_ranks.corr(theirs_ranks, method="pearson"))

    return {"n_genes": n_genes, "sign_agreement": sign_agreement, "spearman": spearman, "n_duplicate_symbols": n_duplicate_symbols}


def select_condition_rows(de: pd.DataFrame, plate: str, drug: str, dose_um: float) -> pd.DataFrame:
    """Tahoe DE rows for one (plate, drug, dose), with the drug name trimmed on both sides.

    Tahoe reports dose as `concentration` + `concentration_unit`; it is converted to µM and matched
    with a relative tolerance of 1e-6. An unknown unit raises. Returns an empty frame when nothing
    matches; raises ValueError when the same gene appears more than once for the selected
    (plate, drug, dose), because that would mix two measurements into one comparison.
    """
    units = set(de["concentration_unit"].dropna().unique())
    unknown = units - set(UNIT_TO_UM)
    if unknown:
        raise ValueError(f"unknown concentration unit(s) in Tahoe DE rows: {sorted(unknown)}")
    dose = de["concentration"] * de["concentration_unit"].map(UNIT_TO_UM)
    rows = de[(de["plate"] == plate) & (de["drug"].str.strip() == drug.strip()) & np.isclose(dose, dose_um, rtol=1e-6, atol=0)]
    if rows["gene_symbol"].duplicated().any():
        raise ValueError(f"more than one Tahoe DE row per gene for plate={plate} drug={drug} dose={dose_um} uM")
    return rows
