from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from phase1.build import build_outputs
from phase1.config import load_slice_config
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
SLICE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "slice.yaml"


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


def _run(tmp_path, metadata, config=CONFIG):
    con = connect(temp_dir=tmp_path)
    gene_files, cell_files = [], []
    for i, shard in enumerate(SHARDS):
        path = tmp_path / f"shard{i}.parquet"
        shard.to_parquet(path)
        g, c = tmp_path / f"p{i}_genes.parquet", tmp_path / f"p{i}_cells.parquet"
        aggregate_shard(con, path, config["selected_cell_lines"], ["plate3"], g, c)
        gene_files.append(g)
        cell_files.append(c)
    out = tmp_path / "out"
    result = build_outputs(con, gene_files, cell_files, metadata, config, out, tmp_path / "qc.md")
    return result, out


def test_build_outputs_writes_validated_logfc_against_plate_matched_dmso(tmp_path):
    result, out = _run(tmp_path, _metadata())

    conditions = pd.read_parquet(out / "conditions.parquet")
    assert conditions["qc_pass"].all()
    assert conditions.groupby("cell_line")["depmap_id"].first().to_dict() == {"L1": "ACH-000001", "L2": "ACH-000002"}
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


def test_build_outputs_fails_when_a_cell_line_has_no_depmap_id(tmp_path):
    config = {**CONFIG, "cell_line_info": {"L1": CONFIG["cell_line_info"]["L1"], "L2": {"tissue": "Bowel"}}}
    with pytest.raises(ContractError, match=r"without depmap_id: \['L2'\]"):
        _run(tmp_path, _metadata(), config)


def test_slice_config_has_depmap_id_and_tissue_for_every_selected_line():
    config = load_slice_config(SLICE_CONFIG)
    info = config["cell_line_info"]
    for line in config["selected_cell_lines"]:
        assert info[line]["depmap_id"].startswith("ACH-"), line
        assert info[line]["tissue"], line


def _output_mb(out: Path) -> float:
    return round(sum((out / f).stat().st_size for f in ("pseudobulk.parquet", "logfc.parquet")) / 1e6, 2)


def test_output_size_is_only_reported_when_no_limit_is_set(tmp_path):
    result, out = _run(tmp_path, _metadata(), {**CONFIG, "max_output_mb": None})
    summary = result["summary"]
    assert summary["max_output_mb"] is None
    assert summary["output_size_mb"] == _output_mb(out)
    report = (tmp_path / "qc.md").read_text()
    assert f"Output size: {summary['output_size_mb']} MB\n" in report
    assert "limit" not in report


def test_output_size_within_limit_passes(tmp_path):
    result, _ = _run(tmp_path, _metadata(), {**CONFIG, "max_output_mb": 100})
    assert result["summary"]["max_output_mb"] == 100
    assert f"Output size: {result['summary']['output_size_mb']} MB (limit 100 MB)" in (tmp_path / "qc.md").read_text()


def test_output_size_over_limit_fails(tmp_path):
    with pytest.raises(ContractError, match="exceeds max_output_mb = 1e-06"):
        _run(tmp_path, _metadata(), {**CONFIG, "max_output_mb": 1e-6})


def test_slice_config_declares_max_output_mb():
    assert "max_output_mb" in load_slice_config(SLICE_CONFIG)
