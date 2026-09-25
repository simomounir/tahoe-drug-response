import importlib.util
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from phase1.contracts import PSEUDOBULK_SCHEMA, ContractError
from phase1.genes import validate_gene_coverage

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path, pb_genes, gene_rows):
    pb = tmp_path / "pseudobulk.parquet"
    genes = tmp_path / "genes.parquet"
    pd.DataFrame({"condition_id": ["c1"] * len(pb_genes), "gene": pd.Series(pb_genes, dtype="int64")}).to_parquet(pb)
    pd.DataFrame(gene_rows, columns=["gene", "gene_symbol", "ensembl_id"]).astype({"gene": "int64"}).to_parquet(genes)
    return pb, genes


def test_all_genes_mapped(tmp_path):
    pb, genes = _write(
        tmp_path,
        [2, 3, 3],
        [(2, "TP53", "ENSG00000141510"), (3, "MYC", "ENSG00000136997"), (4, "EGFR", "ENSG00000146648")],
    )
    result = validate_gene_coverage(duckdb.connect(), pb, genes)
    assert result == {"n_observed": 2, "n_unmapped": 0}


def test_unmapped_gene_raises(tmp_path):
    pb, genes = _write(
        tmp_path,
        [2, 3, 5, 6],
        [(2, "TP53", "ENSG00000141510"), (3, "MYC", "ENSG00000136997"), (6, None, "ENSG00000000003")],
    )
    with pytest.raises(ContractError, match=r"2 of 4 .*\[5, 6\]"):
        validate_gene_coverage(duckdb.connect(), pb, genes)


def test_write_genes_gene_type_matches_pseudobulk(tmp_path):
    spec = importlib.util.spec_from_file_location("fetch_gene_metadata", ROOT / "scripts" / "fetch_gene_metadata.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, out = tmp_path / "gene_metadata.parquet", tmp_path / "genes.parquet"
    pd.DataFrame({"gene_symbol": ["B", "A"], "ensembl_id": ["E2", "E1"], "token_id": pd.Series([4, 3], dtype="int64")}).to_parquet(source)

    con = duckdb.connect()
    assert module.write_genes(con, source, out) == 2
    types = dict(con.execute(f"SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM read_parquet('{out}'))").fetchall())
    assert types["gene"] == PSEUDOBULK_SCHEMA["gene"]
    assert pd.read_parquet(out)["gene"].tolist() == [3, 4]
