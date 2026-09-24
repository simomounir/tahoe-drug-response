from __future__ import annotations

import math

import pandas as pd
import pytest

from phase1.build import build_outputs
from phase1.contracts import ContractError
from phase1.stream import aggregate_shard, connect

CONFIG = {
    "selected_cell_lines": ["L1", "L2"],
    "min_cells_per_condition": 1,
    "min_control_cells": 1,
    "control_drugs": ["DMSO_TF"],
}


def _metadata(include_l2_control: bool = True) -> pd.DataFrame:
    rows = [
        ("s1", "L1", "DrugA", "[('DrugA', 0.5, 'uM')]"),
        ("s2", "L1", "DMSO_TF", "[('DMSO_TF', 0.0, 'uM')]"),
        ("s1", "L2", "DrugA", "[('DrugA', 0.5, 'uM')]"),
    ]
    if include_l2_control:
        rows.append(("s2", "L2", "DMSO_TF", "[('DMSO_TF', 0.0, 'uM')]"))
    return pd.DataFrame(
        [{"sample": s, "plate": "plate3", "cell_line": l, "drug": d, "drugname_drugconc": dc, "n_cells": 5} for s, l, d, dc in rows]
    )


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


def _run(tmp_path, metadata):
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
    result = build_outputs(con, gene_files, cell_files, metadata, CONFIG, out, tmp_path / "qc.md")
    return result, out


def test_build_outputs_writes_validated_logfc_against_plate_matched_dmso(tmp_path):
    result, out = _run(tmp_path, _metadata())

    conditions = pd.read_parquet(out / "conditions.parquet")
    assert conditions["qc_pass"].all()
    assert result["summary"]["n_logfc_conditions"] == 2
    assert len(pd.read_parquet(out / "controls.parquet")) == 2

    logfc = pd.read_parquet(out / "logfc.parquet")
    names = conditions.set_index("condition_id")[["cell_line", "is_control"]]
    l1 = logfc[logfc["condition_id"].map(names["cell_line"]) == "L1"].set_index("gene")["logfc"]
    # L1 DrugA: gene10 = 4, gene11 = 1 (cpm 8e5, 2e5); L1 DMSO: gene10 = 2, gene12 = 2 (cpm 5e5, 5e5)
    assert l1.to_dict() == pytest.approx(
        {10: math.log(800_001) - math.log(500_001), 11: math.log(200_001), 12: -math.log(500_001)}
    )
    assert "## Dropped conditions" in (tmp_path / "qc.md").read_text()


def test_build_outputs_fails_when_a_treated_condition_has_no_control(tmp_path):
    with pytest.raises(ContractError, match="no control"):
        _run(tmp_path, _metadata(include_l2_control=False))
