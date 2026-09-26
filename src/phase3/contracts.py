"""Contracts 1-6 of the Phase 3 features design (spec §6).

Reuses `phase1.contracts.ContractError` -- no new error type. Each `validate_*` function checks
the contracts that apply to one output table; `scripts/build_features.py` calls all four after
building the corresponding table(s).

Controller ruling (recorded 2026-09-26): contract 2 ("no null/NaN/inf in a feature column of a
featurizable drug or of a cell line") holds except where a declared gap explains it --
`expression_available = False` may leave `pc_*` null, `depmap_available = False` may leave
`dmg_*`/`hot_*` null, and controls (`is_control`) leave drug columns null. Any other null/NaN/inf
is a contract violation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from phase1.contracts import ContractError
from phase3.cells import loadings_frame_to_pca, project

# Metadata/reason columns that are *expected* to be null exactly when there is nothing to report
# (e.g. `exclusion_reason` is null for a featurizable drug -- that is the good case, not a gap).
# These are excluded from the "must not be null" checks below; every other column in the table is
# a feature and is held to contract 2.
_DRUG_REASON_COLS = {"exclusion_reason"}
_CELL_REASON_COLS = {"depmap_absent_reason"}
_VIEW_REASON_COLS = {"exclusion_reason", "depmap_absent_reason"}

# Columns identifying a drug row rather than describing it; not floats, but still required to be
# present (non-null) for a featurizable drug.
_DRUG_ID_COLS = {"drug", "smiles_input", "smiles_parent", "cleaning_log", "featurizable"}


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def _assert_no_null_or_inf(df: pd.DataFrame, cols: list[str], label: str) -> None:
    for col in cols:
        series = df[col]
        n_null = int(series.isna().sum())
        _check(n_null == 0, f"{label}: column {col!r} has {n_null} null value(s)")
        if pd.api.types.is_float_dtype(series):
            n_inf = int(np.isinf(series.to_numpy(dtype=float)).sum())
            _check(n_inf == 0, f"{label}: column {col!r} has {n_inf} inf value(s)")


# ---------------------------------------------------------------------------
# Contract 1 + drug half of contract 2
# ---------------------------------------------------------------------------


def validate_drug_features(drugs: pd.DataFrame, slice_drugs: set[str]) -> None:
    """Contract 1: every slice drug is featurizable or carries an exclusion_reason.
    Contract 2 (drugs): no null/NaN/inf feature column for a featurizable drug."""
    present = set(drugs["drug"])
    missing = slice_drugs - present
    _check(not missing, f"drug_features: slice drug(s) missing from drugs table: {sorted(missing)}")

    subset = drugs[drugs["drug"].isin(slice_drugs)]
    featurizable = subset["featurizable"].astype(bool)
    unexplained = subset[~featurizable & subset["exclusion_reason"].isna()]
    _check(
        unexplained.empty,
        f"drug_features: slice drug(s) neither featurizable nor exclusion_reason: {sorted(unexplained['drug'])}",
    )

    feat = drugs[drugs["featurizable"].astype(bool)]
    feature_cols = [c for c in drugs.columns if c not in _DRUG_REASON_COLS]
    _assert_no_null_or_inf(feat, feature_cols, "drug_features")


# ---------------------------------------------------------------------------
# Contract 3
# ---------------------------------------------------------------------------


def validate_drug_groups(drug_groups: pd.DataFrame, slice_featurizable_drugs: set[str]) -> None:
    """Contract 3: drug_groups is exactly the slice's featurizable drugs, one cluster each."""
    _check(drug_groups["drug"].is_unique, "drug_groups: drug appears in more than one cluster")
    actual = set(drug_groups["drug"])
    missing = slice_featurizable_drugs - actual
    extra = actual - slice_featurizable_drugs
    _check(
        not missing and not extra,
        f"drug_groups: does not match the slice's featurizable drugs exactly (missing={sorted(missing)}, extra={sorted(extra)})",
    )
    # Contract 2 on the table itself (the splitter reads it directly). nearest_* is undefined only
    # when a single drug is grouped.
    cols = ["cluster_id", "cluster_size", "murcko_scaffold"]
    if len(drug_groups) > 1:
        cols += ["nearest_drug", "nearest_tanimoto"]
    _assert_no_null_or_inf(drug_groups, cols, "drug_groups")


# ---------------------------------------------------------------------------
# Contract 4 + cell half of contract 2 + contract 6
# ---------------------------------------------------------------------------


def validate_cell_features(
    cells: pd.DataFrame,
    loadings: pd.DataFrame | None = None,
    expression: pd.DataFrame | None = None,
    rtol: float = 1e-5,
) -> None:
    """Contract 4: a slice line missing DepMap coverage is a declared gap, not a silent null.
    Contract 2 (cells): no null/NaN/inf outside the declared gaps (`depmap_available = False` =>
    `dmg_*`/`hot_*`/`pc_*` may be null; `expression_available = False` => `pc_*` may be null).
    Contract 6 (only when `loadings`/`expression` are given): the stored `pc_*` for every line
    with `expression_available` reproduce `project(loadings, expression.loc[expression_source])`.
    """
    _check(cells["cell_line"].is_unique, "cell_features: cell_line is not unique")

    pc_cols = [c for c in cells.columns if c.startswith("pc_")]
    dmg_hot_cols = [c for c in cells.columns if c.startswith("dmg_") or c.startswith("hot_")]
    drv_cols = [c for c in cells.columns if c.startswith("drv_")]

    depmap_available = cells["depmap_available"].astype(bool)
    expression_available = cells["expression_available"].astype(bool)

    # Contract 4: depmap_available flags a declared reason either way.
    no_depmap = cells[~depmap_available]
    _check(
        no_depmap["depmap_absent_reason"].notna().all(),
        "cell_features: depmap_available = False without a depmap_absent_reason",
    )
    _check(
        no_depmap["depmap_id"].isna().all(),
        "cell_features: depmap_available = False but depmap_id is set",
    )
    has_depmap = cells[depmap_available]
    _check(
        has_depmap["depmap_id"].notna().all(),
        "cell_features: depmap_available = True but depmap_id is null",
    )
    _check(
        has_depmap["depmap_absent_reason"].isna().all(),
        "cell_features: depmap_available = True but depmap_absent_reason is set",
    )

    # dmg_/hot_: null only where depmap_available is False.
    for col in dmg_hot_cols:
        bad = cells[depmap_available & cells[col].isna()]
        _check(bad.empty, f"cell_features: {col} null for {len(bad)} line(s) with depmap_available = True")
        _assert_no_null_or_inf(cells[depmap_available], [col], "cell_features")

    # pc_: null only where expression_available is False; also requires depmap_available (no
    # expression without a DepMap ID).
    _check(
        (~expression_available | depmap_available).all(),
        "cell_features: expression_available = True but depmap_available = False",
    )
    for col in pc_cols:
        bad = cells[expression_available & cells[col].isna()]
        _check(bad.empty, f"cell_features: {col} null for {len(bad)} line(s) with expression_available = True")
        _assert_no_null_or_inf(cells[expression_available], [col], "cell_features")
    no_expr = cells[~expression_available]
    _check(
        no_expr["expression_source"].isna().all(),
        "cell_features: expression_available = False but expression_source is set",
    )
    has_expr = cells[expression_available]
    _check(
        has_expr["expression_source"].notna().all(),
        "cell_features: expression_available = True but expression_source is null",
    )

    # drv_: Tahoe driver flags never depend on DepMap coverage -- always required.
    _assert_no_null_or_inf(cells, drv_cols, "cell_features")

    if loadings is not None and expression is not None:
        pca = loadings_frame_to_pca(loadings)
        for _, row in has_expr.iterrows():
            source = row["expression_source"]
            _check(source in expression.index, f"cell_features: expression_source {source!r} not in the expression panel")
            projected = project(pca, expression.loc[[source]])[0]
            stored = row[pc_cols].to_numpy(dtype=float)
            _check(
                np.allclose(projected, stored, rtol=rtol, atol=1e-8),
                f"cell_features: stored pc_* for {row['cell_line']} does not reproduce from the loadings file "
                f"(stored={stored}, projected={projected})",
            )


# ---------------------------------------------------------------------------
# Contract 5 + view half of contract 2
# ---------------------------------------------------------------------------


def validate_condition_features(view: pd.DataFrame, conditions: pd.DataFrame) -> None:
    """Contract 5: exactly one condition_features row per row of conditions.parquet.
    Contract 2 (view): every condition is either `features_complete` with no null/NaN/inf feature
    column, or excluded with a non-null `feature_gap`."""
    _check(
        len(view) == len(conditions),
        f"condition_features: {len(view)} rows, expected {len(conditions)} (one per condition)",
    )
    _check(view["condition_id"].is_unique, "condition_features: condition_id is not unique")
    _check(
        set(view["condition_id"]) == set(conditions["condition_id"]),
        "condition_features: condition_id set differs from conditions.parquet",
    )
    _check(
        {"features_complete", "feature_gap"} <= set(view.columns),
        "condition_features: missing features_complete/feature_gap columns",
    )

    complete = view["features_complete"].astype(bool)
    gap_null = view["feature_gap"].isna()
    inconsistent = view[complete != gap_null]
    _check(
        inconsistent.empty,
        f"condition_features: features_complete/feature_gap disagree for {len(inconsistent)} row(s)",
    )

    raw_cols = set(conditions.columns) | {"features_complete", "feature_gap", "log10_dose_um"}
    feature_cols = [c for c in view.columns if c not in raw_cols and c not in _VIEW_REASON_COLS]
    complete_rows = view[complete]
    _assert_no_null_or_inf(complete_rows, feature_cols, "condition_features")
    # log10_dose_um is null exactly for controls (not a feature column, but still required for
    # every non-control 'complete' condition).
    non_control_complete = complete_rows[~complete_rows["is_control"].astype(bool)]
    _assert_no_null_or_inf(non_control_complete, ["log10_dose_um"], "condition_features")
