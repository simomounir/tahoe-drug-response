import duckdb
import pandas as pd
import pytest

from phase1.contracts import ContractError
from phase1.genes import validate_gene_coverage


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
