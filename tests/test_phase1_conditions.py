from __future__ import annotations

from pathlib import Path

from phase1.conditions import attach_condition_ids, build_condition_table, load_fixture_metadata


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "metadata_small.csv"


def test_fixture_loads_and_selects_required_rows():
    df = load_fixture_metadata(FIXTURE_PATH)
    selected = ["CVCL_0546", "CVCL_0459", "CVCL_0480"]
    table = build_condition_table(df, selected)

    assert len(table) == 6
    assert table["condition_id"].is_unique
    assert set(table["cell_line"]) == set(selected)
    assert set(table["condition_id"]) == set(table["condition_id"])


def test_condition_table_marks_controls_and_extracts_dose():
    df = load_fixture_metadata(FIXTURE_PATH)
    table = build_condition_table(df, ["CVCL_0546", "CVCL_0459"])

    control_rows = table[table["is_control"]]
    assert len(control_rows) == 2
    assert control_rows.loc[control_rows["cell_line"] == "CVCL_0546", "n_cells"].iloc[0] == 150 + 160
    assert set(control_rows["drug_name"]) == {"DMSO_TF"}
    assert table.loc[table["drug_name"] == "Adagrasib", "dose"].iloc[0] == 0.05
    assert table.loc[table["drug_name"] == "DMSO_TF", "dose"].iloc[0] == 0.0


def test_attach_condition_ids_for_sparse_expression_rows():
    metadata = load_fixture_metadata(FIXTURE_PATH)
    metadata = metadata[metadata["cell_line"].isin(["CVCL_0546", "CVCL_0459"])].copy()
    expression = metadata[["sample", "plate", "cell_line"]].copy()
    expression["genes"] = [[0, 1], [1, 2], [0, 2], [1, 3], [2, 3], [1, 4], [0, 3], [2, 4]]
    expression["expressions"] = [[3.0, 4.0], [2.0, 5.0], [1.0, 1.0], [2.0, 2.0], [4.0, 1.0], [3.0, 2.0], [2.0, 3.0], [1.0, 4.0]]

    merged = attach_condition_ids(expression, metadata)

    assert "condition_id" in merged.columns
    assert merged["condition_id"].notna().all()
    assert len(merged) == len(expression)
    expected = set(build_condition_table(metadata, ["CVCL_0546", "CVCL_0459"])["condition_id"])
    assert set(merged["condition_id"]) == expected
