"""The feature layer: a DuckDB view joining conditions to drugs, drug_groups and cells (spec P3.5, §6).

`condition_features` is *only* ever a view (never a stored table, per the brief): conditions LEFT
JOINed to drugs and drug_groups on `trim(drug)`, and to cells on `cell_line`, plus a derived
`log10_dose_um`, `features_complete` and `feature_gap` column (controller ruling, recorded
2026-09-26): every condition is either complete (`feature_gap` null) or excluded with a reason --
'control', 'drug_not_featurizable', 'no_depmap' or 'no_expression', in that priority order when
more than one applies.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from phase1.stream import connect, sql_str
from phase3.config import DOSE_UNIT_TO_UM

# Dose units seen in conditions.parquet, converted to micromolar. Anything not listed here fails
# loudly (ValueError) rather than silently producing a wrong log10_dose_um -- the brief requires
# checking `unit` and converting, never assuming.


def _dose_case_sql(units: list[str]) -> str:
    unknown = sorted(set(units) - set(DOSE_UNIT_TO_UM))
    if unknown:
        raise ValueError(
            f"conditions: unknown dose unit(s) {unknown}; add a conversion to µM in phase3.config.DOSE_UNIT_TO_UM"
        )
    whens = " ".join(f"WHEN {sql_str(u)} THEN c.dose * {DOSE_UNIT_TO_UM[u]}" for u in DOSE_UNIT_TO_UM)
    return f"CASE c.unit {whens} END"


def create_condition_features_view(con: duckdb.DuckDBPyConnection, conditions_path: Path | str, features_dir: Path | str) -> None:
    """Create (or replace) the `condition_features` view on `con`. Never persisted as a table."""
    conditions_path = Path(conditions_path)
    features_dir = Path(features_dir)
    drugs_path = features_dir / "drugs.parquet"
    groups_path = features_dir / "drug_groups.parquet"
    cells_path = features_dir / "cells.parquet"

    units = con.execute(f"SELECT DISTINCT unit FROM read_parquet({sql_str(conditions_path)})").fetchdf()["unit"].tolist()
    dose_case = _dose_case_sql(units)

    # features_complete is derived from feature_gap (null <=> complete) in an outer SELECT over a
    # subquery -- not a second CREATE OR REPLACE VIEW referencing the view's own name, which would
    # make the view definition self-referential (infinite recursion on every later query).
    sql = f"""
    CREATE OR REPLACE VIEW condition_features AS
    SELECT *, feature_gap IS NULL AS features_complete
    FROM (
        SELECT
            c.*,
            CASE WHEN c.is_control THEN NULL ELSE log10({dose_case}) END AS log10_dose_um,
            d.* EXCLUDE (drug),
            g.* EXCLUDE (drug),
            ce.* EXCLUDE (cell_line, depmap_id),
            CASE
                WHEN c.is_control THEN 'control'
                WHEN NOT COALESCE(d.featurizable, FALSE) THEN 'drug_not_featurizable'
                WHEN NOT COALESCE(ce.depmap_available, FALSE) THEN 'no_depmap'
                WHEN NOT COALESCE(ce.expression_available, FALSE) THEN 'no_expression'
                ELSE NULL
            END AS feature_gap
        FROM read_parquet({sql_str(conditions_path)}) c
        LEFT JOIN read_parquet({sql_str(drugs_path)}) d ON trim(c.drug) = trim(d.drug)
        LEFT JOIN read_parquet({sql_str(groups_path)}) g ON trim(c.drug) = trim(g.drug)
        LEFT JOIN read_parquet({sql_str(cells_path)}) ce ON c.cell_line = ce.cell_line
    ) AS joined
    """
    con.execute(sql)


def load_condition_features(conditions_path: Path | str, features_dir: Path | str) -> pd.DataFrame:
    """The `condition_features` view, materialised as a DataFrame -- for Phase 4 and for reporting.

    Opens and closes its own DuckDB connection; `condition_features` itself is never written to
    disk as a table, only ever defined as a view for the duration of one connection.
    """
    con = connect()
    try:
        create_condition_features_view(con, conditions_path, features_dir)
        return con.execute("SELECT * FROM condition_features").fetchdf()
    finally:
        con.close()
