from __future__ import annotations

from pathlib import Path

import pandas as pd

from phase1.conditions import attach_condition_ids, build_condition_table


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "metadata_small.parquet"
LINES = ["CVCL_0546", "CVCL_0459"]
# Fixture: plate3, 2 lines x (5 drug wells + 2 DMSO wells).
N_WELLS_PER_LINE = 7
N_CONDITIONS_PER_LINE = 6


def load_metadata() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE_PATH)


def test_fixture_loads_and_selects_required_rows():
    df = load_metadata()
    table = build_condition_table(df, LINES)

    assert len(table) == len(LINES) * N_CONDITIONS_PER_LINE
    assert table["condition_id"].is_unique
    assert set(table["cell_line"]) == set(LINES)

    one_line = build_condition_table(df, ["CVCL_0546"])
    assert len(one_line) == N_CONDITIONS_PER_LINE
    assert set(one_line["cell_line"]) == {"CVCL_0546"}
    assert build_condition_table(df, ["CVCL_9999"]).empty


def test_condition_table_marks_controls_and_extracts_dose():
    df = load_metadata()
    table = build_condition_table(df, LINES)

    control_rows = table[table["is_control"]]
    assert len(control_rows) == len(LINES)
    dmso_wells = df[(df["cell_line"] == "CVCL_0546") & (df["drug"] == "DMSO_TF")]
    assert len(dmso_wells) == 2
    assert control_rows.loc[control_rows["cell_line"] == "CVCL_0546", "n_cells"].iloc[0] == dmso_wells["n_cells"].sum()
    assert set(control_rows["drug_name"]) == {"DMSO_TF"}
    assert table.loc[table["drug_name"] == "AZD-8055", "dose"].iloc[0] == 5.0
    assert table.loc[table["drug_name"] == "DMSO_TF", "dose"].iloc[0] == 0.0


def test_attach_condition_ids_for_sparse_expression_rows():
    metadata = load_metadata()
    assert len(metadata) == len(LINES) * N_WELLS_PER_LINE
    expression = metadata[["sample", "plate", "cell_line"]].copy()
    expression["genes"] = [[i % 5, i % 5 + 1] for i in range(len(expression))]
    expression["expressions"] = [[float(i % 3 + 1), float(i % 4 + 1)] for i in range(len(expression))]

    merged = attach_condition_ids(expression, metadata)

    assert "condition_id" in merged.columns
    assert merged["condition_id"].notna().all()
    assert len(merged) == len(expression)
    expected = set(build_condition_table(metadata, ["CVCL_0546", "CVCL_0459"])["condition_id"])
    assert set(merged["condition_id"]) == expected
