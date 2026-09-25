from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from phase1.conditions import attach_condition_ids, condition_lookup
from phase1.pseudobulk import build_pseudobulk
from phase1.stream import aggregate_shard, combine_partials, connect, shards_for_plates

LINES = ["L1", "L2"]
PLATES = ["plate4"]

METADATA = pd.DataFrame(
    [
        {"sample": "s1", "plate": "plate4", "cell_line": "L1", "drugname_drugconc": "[('DrugA', 0.5, 'uM')]"},
        {"sample": "s1", "plate": "plate4", "cell_line": "L2", "drugname_drugconc": "[('DrugA', 0.5, 'uM')]"},
        # two DMSO wells on the same plate must collapse into one condition
        {"sample": "s2", "plate": "plate4", "cell_line": "L1", "drugname_drugconc": "[('DMSO_TF', 0.0, 'uM')]"},
        {"sample": "s3", "plate": "plate4", "cell_line": "L1", "drugname_drugconc": "[('DMSO_TF', 0.0, 'uM')]"},
    ]
)

SHARD_A = pd.DataFrame(
    [
        {"sample": "s1", "plate": "plate4", "cell_line_id": "L1", "genes": [1, 0, 2], "expressions": [-2.0, 3.0, 4.0]},
        {"sample": "s1", "plate": "plate4", "cell_line_id": "L2", "genes": [2], "expressions": [5.0]},
        {"sample": "s2", "plate": "plate4", "cell_line_id": "L1", "genes": [0, 2], "expressions": [1.0, 2.0]},
        {"sample": "s1", "plate": "plate4", "cell_line_id": "OTHER", "genes": [0], "expressions": [99.0]},
    ]
)
SHARD_B = pd.DataFrame(
    [
        {"sample": "s1", "plate": "plate4", "cell_line_id": "L1", "genes": [1, 3], "expressions": [2.0, 1.0]},
        {"sample": "s3", "plate": "plate4", "cell_line_id": "L1", "genes": [0], "expressions": [6.0]},
        {"sample": "s9", "plate": "plate7", "cell_line_id": "L1", "genes": [0], "expressions": [99.0]},
    ]
)


def test_streamed_pseudobulk_matches_in_memory_pseudobulk(tmp_path):
    con = connect(temp_dir=tmp_path)
    gene_files, cell_files = [], []
    for name, shard in (("a", SHARD_A), ("b", SHARD_B)):
        shard_path = tmp_path / f"{name}.parquet"
        shard.to_parquet(shard_path)
        gene_out, cell_out = tmp_path / f"{name}_genes.parquet", tmp_path / f"{name}_cells.parquet"
        aggregate_shard(con, shard_path, LINES, PLATES, gene_out, cell_out)
        gene_files.append(gene_out)
        cell_files.append(cell_out)

    out_path = tmp_path / "pseudobulk.parquet"
    per_condition = combine_partials(con, gene_files, cell_files, condition_lookup(METADATA), out_path)
    streamed = pd.read_parquet(out_path)

    cells = pd.concat([SHARD_A, SHARD_B]).rename(columns={"cell_line_id": "cell_line"})
    cells = cells[cells["cell_line"].isin(LINES) & cells["plate"].isin(PLATES)]
    expected = build_pseudobulk(attach_condition_ids(cells, METADATA))

    cols = ["condition_id", "gene", "sum_counts", "n_cells", "library_size"]
    assert list(streamed.columns) == cols
    pd.testing.assert_frame_equal(
        streamed[cols].reset_index(drop=True), expected[cols].reset_index(drop=True), check_dtype=False
    )
    assert len(per_condition) == 3
    assert per_condition["n_cells"].sum() == 5
    assert 1 not in set(streamed["gene"])
    assert (streamed["sum_counts"] > 0).all()


def test_multi_plate_combine_matches_in_memory_and_cleans_up(tmp_path):
    con = connect(temp_dir=tmp_path)
    metadata = pd.concat([METADATA, METADATA.assign(plate="plate5")], ignore_index=True)
    # Shard B straddles both plates, as real boundary shards do.
    shard_a = SHARD_A
    shard_b = pd.concat([SHARD_B, SHARD_A.assign(plate="plate5")], ignore_index=True)
    gene_files, cell_files = [], []
    for name, shard in (("a", shard_a), ("b", shard_b)):
        shard_path = tmp_path / f"{name}.parquet"
        shard.to_parquet(shard_path)
        g, c = tmp_path / f"{name}_genes.parquet", tmp_path / f"{name}_cells.parquet"
        aggregate_shard(con, shard_path, LINES, ["plate4", "plate5"], g, c)
        gene_files.append(g)
        cell_files.append(c)

    out_path = tmp_path / "out" / "pseudobulk.parquet"
    out_path.parent.mkdir()
    per_condition = combine_partials(con, gene_files, cell_files, condition_lookup(metadata), out_path)
    streamed = pd.read_parquet(out_path)

    cells = pd.concat([shard_a, shard_b]).rename(columns={"cell_line_id": "cell_line"})
    cells = cells[cells["cell_line"].isin(LINES) & cells["plate"].isin(["plate4", "plate5"])]
    expected = build_pseudobulk(attach_condition_ids(cells, metadata)).sort_values(["condition_id", "gene"])

    cols = ["condition_id", "gene", "sum_counts", "n_cells", "library_size"]
    pd.testing.assert_frame_equal(
        streamed[cols].reset_index(drop=True), expected[cols].reset_index(drop=True), check_dtype=False
    )
    assert len(per_condition) == 6
    assert list(out_path.parent.iterdir()) == [out_path]


def test_combine_partials_refuses_to_round_non_integer_counts(tmp_path):
    con = connect(temp_dir=tmp_path)
    shard_path = tmp_path / "a.parquet"
    SHARD_A.assign(expressions=[[-2.0, 3.5, 4.0], [5.0], [1.0, 2.0], [99.0]]).to_parquet(shard_path)
    g, c = tmp_path / "g.parquet", tmp_path / "c.parquet"
    aggregate_shard(con, shard_path, LINES, PLATES, g, c)
    with pytest.raises(duckdb.Error, match="non-integer sum_counts"):
        combine_partials(con, [g], [c], condition_lookup(METADATA), tmp_path / "pseudobulk.parquet")


def test_aggregate_shard_reports_kept_cells(tmp_path):
    con = connect(temp_dir=tmp_path)
    shard_path = tmp_path / "a.parquet"
    SHARD_A.to_parquet(shard_path)
    kept = aggregate_shard(con, shard_path, LINES, PLATES, tmp_path / "g.parquet", tmp_path / "c.parquet")
    assert kept == 3
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "plates, expected",
    [(["plate4"], [0, 2]), (["plate10"], [1, 2]), (["plate7"], [])],
)
def test_shards_for_plates_uses_row_group_ranges(plates, expected):
    plate_map = pd.DataFrame(
        {
            "shard": [0, 1, 2],
            "plate_min": ["plate4", "plate10", "plate1"],
            "plate_max": ["plate4", "plate10", "plate5"],
        }
    )
    assert shards_for_plates(plate_map, plates) == expected
