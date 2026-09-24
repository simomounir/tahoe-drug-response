#!/usr/bin/env python3
"""Step 0 remote schema and slice-sizing probe for Tahoe-100M.

This script is intentionally lightweight and designed to answer the questions in
phase1.md §5 before any large local download. It uses DuckDB's remote Parquet support
with the HTTPFS extension to inspect live dataset files.
"""

from __future__ import annotations

import duckdb
from huggingface_hub import HfApi

DATASET_REPO = "tahoebio/Tahoe-100M"
EXPRESSION_FILE = "data/train-00000-of-03388.parquet"
METADATA_FILE = "metadata/obs_metadata.parquet"


def list_parquet_files() -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".parquet"))


def remote_parquet_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{path}"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    return con


def schema_for(url: str) -> list[tuple]:
    con = connect()
    query = "DESCRIBE SELECT * FROM read_parquet(?)"
    rows = con.execute(query, [url]).fetchall()
    con.close()
    return rows


def count_rows(url: str) -> int:
    con = connect()
    query = "SELECT COUNT(*) FROM read_parquet(?)"
    value = int(con.execute(query, [url]).fetchone()[0])
    con.close()
    return value


def top_cell_lines(url: str, limit: int = 10) -> list[tuple[str, int]]:
    con = connect()
    query = "SELECT cell_line, COUNT(*) AS n FROM read_parquet(?) GROUP BY cell_line ORDER BY n DESC LIMIT ?"
    rows = con.execute(query, [url, limit]).fetchall()
    con.close()
    return rows


def top_drugs(url: str, limit: int = 15) -> list[tuple[str, int]]:
    con = connect()
    query = "SELECT drug, COUNT(*) AS n FROM read_parquet(?) GROUP BY drug ORDER BY n DESC LIMIT ?"
    rows = con.execute(query, [url, limit]).fetchall()
    con.close()
    return rows


def summary(url: str) -> dict[str, int]:
    con = connect()
    query = "SELECT COUNT(DISTINCT cell_line), COUNT(DISTINCT plate), COUNT(DISTINCT drug) FROM read_parquet(?)"
    vals = con.execute(query, [url]).fetchone()
    con.close()
    return {
        "distinct_cell_lines": int(vals[0]),
        "distinct_plates": int(vals[1]),
        "distinct_drugs": int(vals[2]),
    }


def main() -> None:
    files = list_parquet_files()
    print(f"Dataset parquet files found: {len(files)}")
    print("First 10 paths:")
    for path in files[:10]:
        print(" -", path)

    expression_url = remote_parquet_url(EXPRESSION_FILE)
    metadata_url = remote_parquet_url(METADATA_FILE)

    print(f"\nExpression sample file URL: {expression_url}")
    print("\nExpression schema for sample file:")
    for row in schema_for(expression_url):
        print(row)
    print(f"\nEstimated row count for sample file: {count_rows(expression_url)}")

    print("\nMetadata summary:")
    for key, value in summary(metadata_url).items():
        print(f" - {key}: {value}")

    print("\nTop cell lines:")
    for row in top_cell_lines(metadata_url, 10):
        print(" -", row)

    print("\nTop drugs:")
    for row in top_drugs(metadata_url, 15):
        print(" -", row)

    print("\nLikely control labels:")
    con = connect()
    q = "SELECT DISTINCT drug FROM read_parquet(?) WHERE lower(drug) LIKE '%dmso%' OR lower(drug) LIKE '%vehicle%' OR lower(drug) LIKE '%control%' ORDER BY drug LIMIT 20"
    print(con.execute(q, [metadata_url]).fetchdf().to_string(index=False))
    con.close()

    print("\nPlate counts:")
    con = connect()
    q = "SELECT plate, COUNT(*) AS cells, COUNT(DISTINCT cell_line) AS lines, COUNT(DISTINCT drug) AS drugs FROM read_parquet(?) GROUP BY plate ORDER BY plate"
    print(con.execute(q, [metadata_url]).fetchdf().to_string(index=False))
    con.close()

    print("\nNext steps:")
    print("1. Confirm cell-line diversity and candidate slice lines against DepMap IDs.")
    print("2. Estimate condition counts per (cell_line, drug, dose) before choosing the slice.")
    print("3. Define the initial slice and record the decision in docs/step0_findings.md.")


if __name__ == "__main__":
    main()
