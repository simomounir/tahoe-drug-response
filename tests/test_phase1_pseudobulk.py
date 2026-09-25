from __future__ import annotations

import pandas as pd

from phase1.pseudobulk import build_pseudobulk, cpm


def test_build_pseudobulk_sums_genes():
    expr = pd.DataFrame(
        [
            {"condition_id": "c1", "genes": [10, 11], "expressions": [3.0, 4.0], "n_cells": 1},
            {"condition_id": "c1", "genes": [11, 12], "expressions": [2.0, 5.0], "n_cells": 1},
            {"condition_id": "c2", "genes": [10, 11], "expressions": [1.0, 1.0], "n_cells": 1},
        ]
    )

    out = build_pseudobulk(expr)

    c1 = out[out["condition_id"] == "c1"].sort_values("gene")
    c2 = out[out["condition_id"] == "c2"].sort_values("gene")

    assert c1["sum_counts"].tolist() == [3.0, 6.0, 5.0]
    assert c2["sum_counts"].tolist() == [1.0, 1.0]
    assert c1["library_size"].iloc[0] == 14.0
    assert c2["library_size"].iloc[0] == 2.0
    assert "cpm" not in out.columns and "log1p_cpm" not in out.columns
    assert cpm(c1["sum_counts"], c1["library_size"]).tolist() == [3 / 14 * 1e6, 6 / 14 * 1e6, 5 / 14 * 1e6]
    assert out["sum_counts"].dtype.kind == "f"
