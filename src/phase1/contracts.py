from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from phase1.pseudobulk import SPECIAL_TOKENS
from phase1.stream import sql_str


class ContractError(Exception):
    pass


PSEUDOBULK_SCHEMA = {
    "condition_id": "VARCHAR",
    "gene": "BIGINT",
    "sum_counts": "DOUBLE",
    "n_cells": "BIGINT",
    "library_size": "DOUBLE",
    "cpm": "DOUBLE",
    "log1p_cpm": "DOUBLE",
}
LOGFC_SCHEMA = {
    "condition_id": "VARCHAR",
    "control_condition_id": "VARCHAR",
    "gene": "BIGINT",
    "cpm": "DOUBLE",
    "cpm_control": "DOUBLE",
    "logfc": "DOUBLE",
}
CONDITIONS_REQUIRED = ["condition_id", "cell_line", "depmap_id", "drug_name", "dose", "is_control", "plate", "n_cells", "qc_pass"]


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def _scalar(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


def check_parquet_schema(con: duckdb.DuckDBPyConnection, path: Path, schema: dict[str, str]) -> None:
    actual = dict(con.execute(f"SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM read_parquet({sql_str(path)}))").fetchall())
    _check(actual == schema, f"{path.name}: schema {actual} != declared {schema}")
    nulls = " + ".join(f"COUNT(*) - COUNT({col})" for col in schema)
    _check(_scalar(con, f"SELECT {nulls} FROM read_parquet({sql_str(path)})") == 0, f"{path.name}: contains nulls")


def validate_conditions(conditions: pd.DataFrame) -> None:
    missing = set(CONDITIONS_REQUIRED) - set(conditions.columns)
    _check(not missing, f"conditions: missing columns {sorted(missing)}")
    _check(conditions["condition_id"].is_unique, "conditions: condition_id is not unique")
    no_depmap = sorted(conditions.loc[conditions["depmap_id"].isna(), "cell_line"].unique())
    _check(not no_depmap, f"conditions: cell line(s) without depmap_id: {no_depmap}")
    _check(conditions[CONDITIONS_REQUIRED].notna().all().all(), "conditions: nulls in required columns")
    _check((conditions["n_cells"] >= 0).all(), "conditions: negative n_cells")
    orphans = conditions[~conditions["is_control"] & conditions["control_condition_id"].isna()]
    _check(orphans.empty, f"conditions: {len(orphans)} treated condition(s) have no control on their plate")


def validate_pseudobulk(con: duckdb.DuckDBPyConnection, path: Path, conditions: pd.DataFrame) -> None:
    check_parquet_schema(con, path, PSEUDOBULK_SCHEMA)
    src = f"read_parquet({sql_str(path)})"
    _check(_scalar(con, f"SELECT COUNT(*) FROM {src} WHERE sum_counts < 0") == 0, f"{path.name}: negative counts")
    tokens = ", ".join(str(t) for t in SPECIAL_TOKENS)
    _check(_scalar(con, f"SELECT COUNT(*) FROM {src} WHERE gene IN ({tokens})") == 0, f"{path.name}: marker token present")
    dup = _scalar(con, f"SELECT COUNT(*) FROM (SELECT condition_id, gene FROM {src} GROUP BY ALL HAVING COUNT(*) > 1)")
    _check(dup == 0, f"{path.name}: {dup} duplicate (condition_id, gene) rows")

    observed = con.execute(f"SELECT condition_id, ANY_VALUE(n_cells) AS n FROM {src} GROUP BY condition_id").fetchdf()
    merged = observed.merge(conditions[["condition_id", "n_cells"]], on="condition_id", how="left")
    _check(merged["n_cells"].notna().all(), f"{path.name}: condition_id not in the condition table")
    mismatch = merged[merged["n"] != merged["n_cells"]]
    _check(mismatch.empty, f"{path.name}: n_cells differs from the condition table for {len(mismatch)} condition(s)")


def validate_logfc(con: duckdb.DuckDBPyConnection, path: Path, expected_conditions: int) -> None:
    check_parquet_schema(con, path, LOGFC_SCHEMA)
    n = _scalar(con, f"SELECT COUNT(DISTINCT condition_id) FROM read_parquet({sql_str(path)})")
    _check(n == expected_conditions, f"{path.name}: {n} conditions, expected {expected_conditions}")
