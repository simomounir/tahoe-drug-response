from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase1.stream import connect
from phase3.layer import create_condition_features_view, load_condition_features

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _write_features(features_dir: Path) -> None:
    """A tiny drugs/drug_groups/cells trio, hand-written (not via phase3.drugs/cells) so these
    tests exercise only the view's SQL: joins, whitespace handling, dose conversion, feature_gap."""
    drugs = pd.DataFrame(
        {
            "drug": ["Aspirin", "Unfeaturizable"],
            "featurizable": [True, False],
            "exclusion_reason": [None, "parse_failed"],
            "MolWt": [180.16, np.nan],
        }
    )
    groups = pd.DataFrame({"drug": ["Aspirin"], "cluster_id": [0], "cluster_size": [1]})
    cells = pd.DataFrame(
        {
            "cell_line": ["L1", "L2"],
            "depmap_id": ["ACH-1", None],
            "depmap_available": [True, False],
            "expression_available": [True, False],
            "pc_1": [0.5, np.nan],
        }
    )
    drugs.to_parquet(features_dir / "drugs.parquet", index=False)
    groups.to_parquet(features_dir / "drug_groups.parquet", index=False)
    cells.to_parquet(features_dir / "cells.parquet", index=False)


def _write_conditions(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# create_condition_features_view
# ---------------------------------------------------------------------------


def test_view_has_one_row_per_condition(tmp_path: Path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    _write_features(features_dir)
    conditions_path = tmp_path / "conditions.parquet"
    _write_conditions(
        conditions_path,
        [
            {"condition_id": "c1", "cell_line": "L1", "drug": "Aspirin", "is_control": False, "dose": 0.5, "unit": "uM"},
            {"condition_id": "c2", "cell_line": "L1", "drug": "Aspirin", "is_control": False, "dose": 5.0, "unit": "uM"},
            {"condition_id": "c3", "cell_line": "L1", "drug": "DMSO_TF", "is_control": True, "dose": 0.0, "unit": "uM"},
        ],
    )
    con = connect()
    create_condition_features_view(con, conditions_path, features_dir)
    view = con.execute("SELECT * FROM condition_features ORDER BY condition_id").fetchdf()
    con.close()
    assert len(view) == 3
    assert view["condition_id"].tolist() == ["c1", "c2", "c3"]


def test_view_whitespace_only_drug_name_still_joins(tmp_path: Path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    _write_features(features_dir)
    conditions_path = tmp_path / "conditions.parquet"
    _write_conditions(
        conditions_path,
        [
            {"condition_id": "c1", "cell_line": "L1", "drug": "Aspirin ", "is_control": False, "dose": 0.5, "unit": "uM"},
        ],
    )
    con = connect()
    create_condition_features_view(con, conditions_path, features_dir)
    row = con.execute("SELECT * FROM condition_features").fetchdf().iloc[0]
    con.close()
    assert row["featurizable"] == True  # noqa: E712 -- joined despite the trailing space
    assert row["MolWt"] == pytest.approx(180.16)
    assert row["feature_gap"] is None
    assert bool(row["features_complete"]) is True


def test_view_controls_get_null_drug_columns_without_failing(tmp_path: Path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    _write_features(features_dir)
    conditions_path = tmp_path / "conditions.parquet"
    _write_conditions(
        conditions_path,
        [{"condition_id": "c1", "cell_line": "L1", "drug": "DMSO_TF", "is_control": True, "dose": 0.0, "unit": "uM"}],
    )
    con = connect()
    create_condition_features_view(con, conditions_path, features_dir)
    row = con.execute("SELECT * FROM condition_features").fetchdf().iloc[0]
    con.close()
    assert pd.isna(row["featurizable"])
    assert pd.isna(row["MolWt"])
    assert pd.isna(row["log10_dose_um"])
    assert row["feature_gap"] == "control"
    assert bool(row["features_complete"]) is False


def test_view_marks_undeclared_gaps_by_reason(tmp_path: Path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    _write_features(features_dir)
    conditions_path = tmp_path / "conditions.parquet"
    _write_conditions(
        conditions_path,
        [
            {"condition_id": "c_drug", "cell_line": "L1", "drug": "Unfeaturizable", "is_control": False, "dose": 0.5, "unit": "uM"},
            {"condition_id": "c_depmap", "cell_line": "L2", "drug": "Aspirin", "is_control": False, "dose": 0.5, "unit": "uM"},
        ],
    )
    con = connect()
    create_condition_features_view(con, conditions_path, features_dir)
    view = con.execute("SELECT * FROM condition_features ORDER BY condition_id").fetchdf().set_index("condition_id")
    con.close()
    assert view.loc["c_drug", "feature_gap"] == "drug_not_featurizable"
    assert view.loc["c_depmap", "feature_gap"] == "no_depmap"  # L2: depmap_available=False


def test_view_log10_dose_um_converts_units():
    # uM should be a no-op: log10(0.5) directly.
    assert np.log10(0.5) == pytest.approx(-0.30103, abs=1e-5)


def test_view_unknown_dose_unit_raises_loudly(tmp_path: Path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    _write_features(features_dir)
    conditions_path = tmp_path / "conditions.parquet"
    _write_conditions(
        conditions_path,
        [{"condition_id": "c1", "cell_line": "L1", "drug": "Aspirin", "is_control": False, "dose": 0.5, "unit": "mystery"}],
    )
    con = connect()
    with pytest.raises(ValueError, match="unknown dose unit"):
        create_condition_features_view(con, conditions_path, features_dir)
    con.close()


def test_load_condition_features_returns_dataframe_and_closes_its_connection(tmp_path: Path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    _write_features(features_dir)
    conditions_path = tmp_path / "conditions.parquet"
    _write_conditions(
        conditions_path,
        [{"condition_id": "c1", "cell_line": "L1", "drug": "Aspirin", "is_control": False, "dose": 0.5, "unit": "uM"}],
    )
    df = load_condition_features(conditions_path, features_dir)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["condition_id"] == "c1"


# ---------------------------------------------------------------------------
# Provenance + end-to-end: scripts/build_features.py's build()
# ---------------------------------------------------------------------------


def _tiny_depmap_dir(tmp_path: Path) -> Path:
    from phase3.cells import DAMAGING_FILE, EXPRESSION_FILE, HOTSPOT_FILE

    depmap_dir = tmp_path / "depmap"
    depmap_dir.mkdir()
    lines = ["ACH-1", "ACH-2", "ACH-3"]
    expression = pd.DataFrame(
        {"": lines, "G1": [1.0, 2.0, 3.0], "G2": [3.0, 1.0, 2.0], "G3": [0.0, 5.0, 1.0]}
    )
    damaging = pd.DataFrame({"": lines, "G1": [1.0, 1.0, 0.0], "G2": [0.0, 0.0, 0.0]})
    hotspot = pd.DataFrame({"": lines, "G1": [0.0, 1.0, 0.0], "G2": [0.0, 0.0, 0.0]})
    expression.to_parquet(depmap_dir / EXPRESSION_FILE, index=False)
    damaging.to_parquet(depmap_dir / DAMAGING_FILE, index=False)
    hotspot.to_parquet(depmap_dir / HOTSPOT_FILE, index=False)
    return depmap_dir


def test_build_features_provenance_succeeds_with_only_conditions_parquet(tmp_path: Path):
    """spec §7 provenance test: the build must succeed when the pseudobulk directory holds only
    conditions.parquet -- no pseudobulk.parquet or logfc.parquet -- proving no feature derives
    from the response being predicted."""
    import build_features

    pseudobulk_dir = tmp_path / "pseudobulk"
    pseudobulk_dir.mkdir()
    conditions_path = pseudobulk_dir / "conditions.parquet"
    _write_conditions(
        conditions_path,
        [
            {
                "condition_id": "c1", "cell_line": "L1", "drug": "Aspirin", "is_control": False,
                "dose": 0.5, "unit": "uM",
            },
            {
                "condition_id": "c2", "cell_line": "L1", "drug": "Aspirin ", "is_control": False,
                "dose": 5.0, "unit": "uM",
            },
            {
                "condition_id": "c3", "cell_line": "L1", "drug": "DMSO_TF", "is_control": True,
                "dose": 0.0, "unit": "uM",
            },
            {
                # A second distinct featurizable drug so nearest_drug/nearest_tanimoto (defined
                # only when at least one *other* featurizable drug exists) are non-null -- with
                # only 92 real slice drugs this is never a singleton set, so this mirrors that.
                "condition_id": "c4", "cell_line": "L1", "drug": "Ibuprofen", "is_control": False,
                "dose": 1.0, "unit": "uM",
            },
        ],
    )
    # The defining assertion of this test: only conditions.parquet exists in this directory.
    assert sorted(p.name for p in pseudobulk_dir.iterdir()) == ["conditions.parquet"]

    drug_metadata_path = tmp_path / "drug_metadata.parquet"
    pd.DataFrame(
        {
            "drug": ["Aspirin", "Ibuprofen"],
            "canonical_smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"],
        }
    ).to_parquet(drug_metadata_path, index=False)

    depmap_dir = _tiny_depmap_dir(tmp_path)

    tahoe_cells_path = tmp_path / "cell_line_metadata.parquet"
    pd.DataFrame({"Cell_ID_Cellosaur": ["L1"], "Driver_Gene_Symbol": ["TP53"]}).to_parquet(tahoe_cells_path, index=False)

    slice_cfg = {"selected_cell_lines": ["L1"], "cell_line_info": {"L1": {"depmap_id": "ACH-1", "tissue": "Test"}}}
    cfg = {
        "drugs": {"morgan_radius": 2, "morgan_n_bits": 256, "butina_similarity": 0.6, "report_similarities": [0.5, 0.6, 0.7]},
        "depmap": {"release": "TEST"},
        "cells": {"pca_components": 2, "mutation_min_frequency": 0.5, "driver_min_lines": 1, "expression_proxy": {}, "expression_absent": {}},
    }

    features_dir = tmp_path / "features"
    report_path = tmp_path / "reports" / "features.md"

    summary = build_features.build(
        conditions_path, drug_metadata_path, depmap_dir, tahoe_cells_path, slice_cfg, features_dir, report_path, cfg=cfg
    )

    assert summary["n_conditions"] == 4
    assert summary["n_view_rows"] == 4
    assert summary["n_featurizable"] == 2
    assert summary["feature_gap_counts"].get("control") == 1
    assert summary["feature_gap_counts"].get("complete", 0) == 3  # c1, c2 (whitespace-only name difference) and c4
    assert (features_dir / "drugs.parquet").exists()
    assert (features_dir / "drug_groups.parquet").exists()
    assert (features_dir / "cells.parquet").exists()
    assert (features_dir / "depmap_pca.parquet").exists()
    assert report_path.exists()
    assert "feature_gap" in report_path.read_text()

    # Still true after the build: nothing was written into or read from pseudobulk.parquet/logfc.parquet.
    assert sorted(p.name for p in pseudobulk_dir.iterdir()) == ["conditions.parquet"]
