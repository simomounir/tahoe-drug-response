from __future__ import annotations

import duckdb
import pandas as pd

from phase4 import tahoe_de

REV = "2dc57900b7981cfcf5e211527169a0b006546a95"


def test_file_url_is_pinned_to_the_revision():
    url = tahoe_de.file_url(5, REV)
    assert f"/resolve/{REV}/" in url and url.endswith("train-00005-of-01026.parquet")


def test_probe_cache_round_trip(tmp_path):
    cache = {
        3: pd.DataFrame({"cell_line": ["A", "A", "B"], "plate": ["1", "2", "1"]}, index=pd.Index([0, 1, 2], name="row_group_id")),
        7: None,
    }
    path = tmp_path / "index.parquet"
    tahoe_de.save_probe_cache(cache, path)
    loaded = tahoe_de.load_probe_cache(path)
    assert set(loaded) == {3, 7}
    assert loaded[7] is None
    pd.testing.assert_frame_equal(loaded[3], cache[3])


def test_load_probe_cache_missing_file_is_empty(tmp_path):
    assert tahoe_de.load_probe_cache(tmp_path / "absent.parquet") == {}


def test_extract_rows_filters_line_plates_and_trimmed_drugs(tmp_path):
    shard = tmp_path / "shard.parquet"
    pd.DataFrame(
        {
            "gene_name": ["G1", "G2", "G3", "G4", "G5"],
            "log2FoldChange": [1.0, -1.0, 2.0, 0.5, 0.1],
            "padj": [0.01, 0.2, 0.01, 0.01, 0.01],
            "n_cells_trt": [10] * 5,
            "n_cells_ctrl": [100] * 5,
            "drug": ["Erdafitinib ", "Erdafitinib ", "Erdafitinib ", "Other", "Erdafitinib "],
            "plate": ["1", "1", "4", "1", "2"],
            "concentration": [0.05] * 5,
            "concentration_unit": ["uM"] * 5,
            "Cell_ID_Cellosaur": ["A", "A", "A", "A", "B"],
        }
    ).to_parquet(shard)
    rows = tahoe_de.extract_rows(duckdb.connect(), shard, cell_line="A", drugs=["Erdafitinib"])
    # line A only, plates 1-3 only, trimmed drug match; padj filtering is left to prepare_de_sets
    assert sorted(rows["gene_name"]) == ["G1", "G2"]
    assert set(rows.columns) >= {"gene_name", "log2FoldChange", "padj", "drug", "plate", "concentration", "concentration_unit", "n_cells_trt"}
