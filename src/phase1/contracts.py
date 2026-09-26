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
    "gene": "INTEGER",
    "sum_counts": "INTEGER",
    "n_cells": "INTEGER",
    "library_size": "BIGINT",
}
LOGFC_SCHEMA = {
    "condition_id": "VARCHAR",
    "control_condition_id": "VARCHAR",
    "gene": "INTEGER",
    "logfc": "FLOAT",
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


def validate_conditions(conditions: pd.DataFrame, depmap_absent: frozenset[str] = frozenset()) -> None:
    """depmap_absent: lines whose missing DepMap ID is declared in the config with a reason."""
    missing = set(CONDITIONS_REQUIRED) - set(conditions.columns)
    _check(not missing, f"conditions: missing columns {sorted(missing)}")
    _check(conditions["condition_id"].is_unique, "conditions: condition_id is not unique")
    no_depmap = conditions["depmap_id"].isna()
    undeclared = sorted(set(conditions.loc[no_depmap, "cell_line"]) - depmap_absent)
    _check(not undeclared, f"conditions: cell line(s) without depmap_id: {undeclared}")
    required = [c for c in CONDITIONS_REQUIRED if c != "depmap_id"]
    _check(conditions[required].notna().all().all(), "conditions: nulls in required columns")
    _check((conditions["n_cells"] >= 0).all(), "conditions: negative n_cells")
    orphans = conditions[~conditions["is_control"] & conditions["control_condition_id"].isna()]
    _check(orphans.empty, f"conditions: {len(orphans)} treated condition(s) have no control on their plate")


def validate_pseudobulk(
    con: duckdb.DuckDBPyConnection, path: Path, conditions: pd.DataFrame, rows_per_bucket: int = 20_000_000
) -> None:
    """rows_per_bucket bounds memory of the duplicate check (one hash group per row); it does not change the result."""
    check_parquet_schema(con, path, PSEUDOBULK_SCHEMA)
    src = f"read_parquet({sql_str(path)})"
    _check(_scalar(con, f"SELECT COUNT(*) FROM {src} WHERE sum_counts < 0") == 0, f"{path.name}: negative counts")
    tokens = ", ".join(str(t) for t in SPECIAL_TOKENS)
    _check(_scalar(con, f"SELECT COUNT(*) FROM {src} WHERE gene IN ({tokens})") == 0, f"{path.name}: marker token present")
    # Duplicates share a condition_id, so bucketing by its hash keeps the check exact while each
    # GROUP BY holds only ~rows_per_bucket groups (a single pass OOMs at 2 GB on ~300M rows).
    n_rows = _scalar(con, f"SELECT COUNT(*) FROM {src}")
    n_buckets = max(1, -(-n_rows // rows_per_bucket))
    dup = sum(
        _scalar(
            con,
            f"SELECT COUNT(*) FROM (SELECT condition_id, gene FROM {src} "
            f"WHERE hash(condition_id) % {n_buckets} = {bucket} GROUP BY ALL HAVING COUNT(*) > 1)",
        )
        for bucket in range(n_buckets)
    )
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


def validate_output_size(size_mb: float, max_output_mb: float | None) -> None:
    """max_output_mb = None only reports; otherwise pseudobulk + logfc must fit within it."""
    if max_output_mb is not None:
        _check(size_mb <= max_output_mb, f"outputs: pseudobulk + logfc = {size_mb:.6g} MB exceeds max_output_mb = {max_output_mb}")
