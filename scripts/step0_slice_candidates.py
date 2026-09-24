#!/usr/bin/env python3
"""Compute the first candidate slice for Tahoe-100M from remote metadata.

This script answers the Step 0 questions that drive the initial cell-line slice:
- which lines are most represented,
- how many drugs exist per line,
- whether controls are plate-matched,
- which 8-10 lines are a reasonable draft slice before DepMap validation.
"""

from __future__ import annotations

import duckdb

DATASET_REPO = "tahoebio/Tahoe-100M"
METADATA_PATH = "metadata/obs_metadata.parquet"
URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{METADATA_PATH}"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    return con


def fetch_all(query: str, *args) -> list[tuple]:
    con = connect()
    rows = con.execute(query, list(args)).fetchall()
    con.close()
    return rows


def cell_line_counts() -> list[tuple[str, int]]:
    q = "SELECT cell_line, COUNT(*) AS n_cells FROM read_parquet(?) GROUP BY cell_line ORDER BY n_cells DESC"
    return fetch_all(q, URL)


def cell_line_drug_counts() -> list[tuple[str, int]]:
    q = "SELECT cell_line, COUNT(DISTINCT drug) AS n_drugs FROM read_parquet(?) GROUP BY cell_line ORDER BY n_drugs DESC"
    return fetch_all(q, URL)


def dmsotf_by_plate() -> list[tuple[str, str, int]]:
    q = """
    SELECT cell_line, plate, SUM(CASE WHEN lower(drug) = 'dmso_tf' THEN 1 ELSE 0 END) AS dmso_cells
    FROM read_parquet(?)
    GROUP BY cell_line, plate
    ORDER BY cell_line, plate
    """
    return fetch_all(q, URL)


def candidate_slice(n: int = 10) -> list[tuple[str, int]]:
    return cell_line_counts()[:n]


def main() -> None:
    lines = cell_line_counts()
    print("Distinct cell lines:", len(lines))
    print("\nTop 10 cell lines by cell count:")
    for cell_line, n_cells in lines[:10]:
        print(f" - {cell_line}: {n_cells}")

    print("\nTop 10 cell lines by number of distinct drugs:")
    for cell_line, n_drugs in cell_line_drug_counts()[:10]:
        print(f" - {cell_line}: {n_drugs} unique drugs")

    print("\nDraft candidate slice (top 10 by cell count):")
    for i, (cell_line, n_cells) in enumerate(candidate_slice(10), start=1):
        print(f" {i}. {cell_line} ({n_cells} cells)")

    print("\nDMSO_TF counts by cell_line and plate (sample):")
    for cell_line, plate, dmso_cells in dmsotf_by_plate()[:20]:
        print(f" - {cell_line} / {plate}: {dmso_cells} DMSO_TF cells")

    print("\nInterpretation:")
    print("- These are the strongest candidate lines for the initial slice based on actual metadata counts.")
    print("- The final slice should still be checked against DepMap identities and tissue diversity before being frozen.")


if __name__ == "__main__":
    main()
