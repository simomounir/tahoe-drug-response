from __future__ import annotations

from pathlib import Path

import duckdb

from phase1.contracts import ContractError
from phase1.stream import sql_str


def validate_gene_coverage(con: duckdb.DuckDBPyConnection, pseudobulk_path: str | Path, genes_path: str | Path) -> dict:
    """Raise ContractError if any pseudobulk gene index lacks a symbol in genes.parquet; return coverage counts."""
    n_observed, n_unmapped, examples = con.execute(
        f"""
        WITH observed AS (SELECT DISTINCT gene FROM read_parquet({sql_str(pseudobulk_path)})),
        symbols AS (
            SELECT gene FROM read_parquet({sql_str(genes_path)})
            WHERE gene_symbol IS NOT NULL AND gene_symbol <> ''
        ),
        unmapped AS (SELECT gene FROM observed ANTI JOIN symbols USING (gene))
        SELECT (SELECT COUNT(*) FROM observed),
               (SELECT COUNT(*) FROM unmapped),
               (SELECT list(gene ORDER BY gene)[1:10] FROM unmapped)
        """
    ).fetchone()
    if n_unmapped:
        raise ContractError(f"{n_unmapped} of {n_observed} pseudobulk genes have no symbol, e.g. {examples}")
    return {"n_observed": int(n_observed), "n_unmapped": 0}
