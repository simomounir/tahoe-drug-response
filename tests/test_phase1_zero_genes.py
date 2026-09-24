from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from phase1.build import build_outputs
from phase1.contracts import ContractError
from phase1.stream import aggregate_shard, connect

CONFIG = {
    "selected_cell_lines": ["L1", "L2"],
    "cell_line_info": {
        "L1": {"depmap_id": "ACH-000001", "tissue": "Lung"},
        "L2": {"depmap_id": "ACH-000002", "tissue": "Bowel"},
    },
    "min_cells_per_condition": 1,
    "min_control_cells": 1,
    "control_drugs": ["DMSO_TF"],
}

# Cells here only ever emit gene tokens 10, 11, 12 (plus the excluded marker token 1).
SHARDS = [
    pd.DataFrame(
        [
            {"sample": "s1", "plate": "plate3", "cell_line_id": "L1", "genes": [1, 10, 11], "expressions": [-2.0, 3.0, 1.0]},
            {"sample": "s2", "plate": "plate3", "cell_line_id": "L1", "genes": [1, 10, 12], "expressions": [-2.0, 2.0, 2.0]},
        ]
    ),
    pd.DataFrame(
        [
            {"sample": "s1", "plate": "plate3", "cell_line_id": "L1", "genes": [1, 10], "expressions": [-2.0, 1.0]},
            {"sample": "s1", "plate": "plate3", "cell_line_id": "L2", "genes": [1, 10], "expressions": [-2.0, 4.0]},
            {"sample": "s2", "plate": "plate3", "cell_line_id": "L2", "genes": [1, 11], "expressions": [-2.0, 4.0]},
        ]
    ),
]


def _metadata() -> pd.DataFrame:
    rows = [
        ("s1", "L1", "DrugA", "[('DrugA', 0.5, 'uM')]"),
        ("s2", "L1", "DMSO_TF", "[('DMSO_TF', 0.0, 'uM')]"),
        ("s1", "L2", "DrugA", "[('DrugA', 0.5, 'uM')]"),
        ("s2", "L2", "DMSO_TF", "[('DMSO_TF', 0.0, 'uM')]"),
    ]
    return pd.DataFrame(
        [{"sample": s, "plate": "plate3", "cell_line": l, "drug": d, "drugname_drugconc": dc, "n_cells": 5} for s, l, d, dc in rows]
    )


def _genes_parquet(path: Path, rows: list[tuple]) -> None:
    pd.DataFrame(rows, columns=["gene", "gene_symbol", "ensembl_id"]).astype({"gene": "int64"}).to_parquet(path)


def _run(tmp_path, genes_path=None):
    con = connect(temp_dir=tmp_path)
    gene_files, cell_files = [], []
    for i, shard in enumerate(SHARDS):
        path = tmp_path / f"shard{i}.parquet"
        shard.to_parquet(path)
        g, c = tmp_path / f"p{i}_genes.parquet", tmp_path / f"p{i}_cells.parquet"
        aggregate_shard(con, path, CONFIG["selected_cell_lines"], ["plate3"], g, c)
        gene_files.append(g)
        cell_files.append(c)
    out = tmp_path / "out"
    result = build_outputs(con, gene_files, cell_files, _metadata(), CONFIG, out, tmp_path / "qc.md", genes_path=genes_path)
    return result, out


def test_zero_count_genes_reported_when_genes_parquet_present(tmp_path):
    # Observed genes across the whole slice: 10, 11, 12. ZZZ1/ZZZ2 (genes 20, 21) are never observed.
    genes_path = tmp_path / "genes.parquet"
    _genes_parquet(
        genes_path,
        [
            (10, "GENE10", "ENSG10"),
            (11, "GENE11", "ENSG11"),
            (12, "GENE12", "ENSG12"),
            (20, "ZZZ1", "ENSG20"),
            (21, "ZZZ2", "ENSG21"),
        ],
    )

    result, out = _run(tmp_path, genes_path=genes_path)

    coverage = result["summary"]["gene_coverage"]
    assert coverage["skipped"] is False
    assert coverage["n_zero_count_genes"] == 2
    assert coverage["zero_count_gene_sample"] == ["ZZZ1", "ZZZ2"]

    report = (tmp_path / "qc.md").read_text()
    assert "Zero-count genes across the slice: 2" in report
    assert "ZZZ1" in report and "ZZZ2" in report


def test_zero_count_gene_sample_is_capped_and_sorted(tmp_path):
    genes_path = tmp_path / "genes.parquet"
    observed = [(10, "GENE10", "ENSG10"), (11, "GENE11", "ENSG11"), (12, "GENE12", "ENSG12")]
    # 25 unobserved genes, symbols named so alphabetical sort != gene-id sort.
    unobserved = [(100 + i, f"Z{25 - i:02d}", f"ENSGZ{i}") for i in range(25)]
    _genes_parquet(genes_path, observed + unobserved)

    result, _ = _run(tmp_path, genes_path=genes_path)

    coverage = result["summary"]["gene_coverage"]
    assert coverage["n_zero_count_genes"] == 25
    assert len(coverage["zero_count_gene_sample"]) == 20
    assert coverage["zero_count_gene_sample"] == sorted(coverage["zero_count_gene_sample"])
    assert coverage["zero_count_gene_sample"][0] == "Z01"


def test_gene_check_skipped_when_genes_parquet_absent(tmp_path):
    # No genes.parquet written anywhere; default lookup (output_dir/genes.parquet) misses too.
    result, out = _run(tmp_path, genes_path=None)

    coverage = result["summary"]["gene_coverage"]
    assert coverage == {"skipped": True}
    assert not (out / "genes.parquet").exists()

    report = (tmp_path / "qc.md").read_text()
    assert "Zero-count genes: skipped (genes.parquet not found)" in report


def test_gene_check_uses_default_path_when_genes_parquet_in_output_dir(tmp_path):
    # genes_path=None should default to output_dir / "genes.parquet"; but build_outputs
    # only mkdirs output_dir itself, so pre-create it before dropping genes.parquet there.
    out = tmp_path / "out"
    out.mkdir(parents=True)
    _genes_parquet(out / "genes.parquet", [(10, "GENE10", "ENSG10"), (11, "GENE11", "ENSG11"), (12, "GENE12", "ENSG12")])

    result, _ = _run(tmp_path, genes_path=None)

    coverage = result["summary"]["gene_coverage"]
    assert coverage["skipped"] is False
    assert coverage["n_zero_count_genes"] == 0


def test_unmapped_pseudobulk_gene_raises_contract_error(tmp_path):
    # genes.parquet is missing a symbol for observed gene 12 -> ContractError from validate_gene_coverage.
    genes_path = tmp_path / "genes.parquet"
    _genes_parquet(genes_path, [(10, "GENE10", "ENSG10"), (11, "GENE11", "ENSG11")])

    with pytest.raises(ContractError, match="no symbol"):
        _run(tmp_path, genes_path=genes_path)
