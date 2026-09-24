from __future__ import annotations

import pandas as pd

from phase1.pseudobulk import build_pseudobulk


def test_build_pseudobulk_sums_genes_and_cpm():
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
    assert c1["cpm"].tolist()[0] > 0
    assert c1["log1p_cpm"].tolist()[0] > 0
    assert out["sum_counts"].dtype.kind == "f"
