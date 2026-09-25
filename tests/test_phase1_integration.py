from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from phase1.contracts import LOGFC_SCHEMA, PSEUDOBULK_SCHEMA, check_parquet_schema, validate_conditions
from phase1.pseudobulk import SPECIAL_TOKENS
from phase1.stream import connect

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
KEYS = ["condition_id", "gene"]
REAL_SHARD_SCHEMA = [
    ("genes", pa.list_(pa.int64())),
    ("expressions", pa.list_(pa.float32())),
    ("drug", pa.string()),
    ("sample", pa.string()),
    ("BARCODE_SUB_LIB_ID", pa.string()),
    ("cell_line_id", pa.string()),
    ("moa-fine", pa.string()),
    ("canonical_smiles", pa.string()),
    ("pubchem_cid", pa.string()),
    ("plate", pa.string()),
]


def _load_make_fixture():
    spec = importlib.util.spec_from_file_location("make_fixture_under_test", ROOT / "scripts" / "make_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MF = _load_make_fixture()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    work = tmp_path_factory.mktemp("integration")
    result, out = MF.build_fixture_outputs(MF.EXPRESSION_PATH, MF.METADATA_PATH, work)
    return result, out


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(KEYS).reset_index(drop=True)


def test_fixture_files_are_small_and_in_the_real_schema():
    paths = [MF.EXPRESSION_PATH, MF.METADATA_PATH, MF.EXPECTED_PSEUDOBULK, MF.EXPECTED_LOGFC]
    assert sum(p.stat().st_size for p in paths) < 500_000

    schema = pq.read_schema(MF.EXPRESSION_PATH)
    assert [(f.name, f.type) for f in schema] == REAL_SHARD_SCHEMA
    expression = pd.read_parquet(MF.EXPRESSION_PATH)
    assert set(expression["cell_line_id"]) == set(MF.CELL_LINES)
    assert set(expression["plate"]) == {MF.PLATE}
    assert all(g[0] in SPECIAL_TOKENS and v[0] == -2 for g, v in zip(expression["genes"], expression["expressions"]))

    metadata = pd.read_parquet(MF.METADATA_PATH)
    assert list(metadata.columns) == MF.METADATA_COLUMNS
    counts = expression.groupby(["cell_line_id", "plate", "sample"]).size()
    assert metadata.set_index(["cell_line", "plate", "sample"])["n_cells"].to_dict() == counts.to_dict()


def test_conditions_qc_and_shapes(built):
    result, out = built
    conditions = result["conditions"]
    validate_conditions(conditions)

    # 2 lines x (N_DRUGS drugs + 1 DMSO condition pooling the plate's DMSO wells)
    assert len(conditions) == len(MF.CELL_LINES) * (MF.N_DRUGS + 1)
    assert conditions["is_control"].sum() == len(MF.CELL_LINES)
    assert (conditions["n_cells"] == conditions["n_cells_atlas"]).all()
    dropped = conditions[~conditions["qc_pass"]]
    assert len(dropped) == len(MF.CELL_LINES)
    assert (dropped["n_cells"] == MF.CELLS_LOW_DRUG_WELL).all()

    treated_ok = set(conditions.loc[conditions["qc_pass"] & ~conditions["is_control"], "condition_id"])
    assert result["summary"]["n_logfc_conditions"] == len(treated_ok) == len(MF.CELL_LINES) * (MF.N_DRUGS - 1)
    logfc = pd.read_parquet(out / "logfc.parquet")
    assert set(logfc["condition_id"]) == treated_ok

    pseudobulk = pd.read_parquet(out / "pseudobulk.parquet")
    assert set(pseudobulk["condition_id"]) == set(conditions["condition_id"])
    assert not pseudobulk["gene"].isin(SPECIAL_TOKENS).any()
    assert pseudobulk.duplicated(KEYS).sum() == 0 and logfc.duplicated(KEYS).sum() == 0


@pytest.mark.parametrize(
    ("name", "expected_path", "schema"),
    [("pseudobulk", MF.EXPECTED_PSEUDOBULK, PSEUDOBULK_SCHEMA), ("logfc", MF.EXPECTED_LOGFC, LOGFC_SCHEMA)],
)
def test_outputs_equal_stored_expected_exactly(built, name, expected_path, schema):
    _, out = built
    check_parquet_schema(connect(), expected_path, schema)
    actual = _sorted(pd.read_parquet(out / f"{name}.parquet"))
    expected = _sorted(pd.read_parquet(expected_path))
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
