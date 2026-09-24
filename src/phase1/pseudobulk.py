from __future__ import annotations

import numpy as np
import pandas as pd

# Tahoe prepends token 1 (value -2) to every cell's gene list; it is a marker, not a gene.
SPECIAL_TOKENS = (1,)


def build_pseudobulk(df: pd.DataFrame) -> pd.DataFrame:
    """Sum sparse (genes, expressions) rows per condition and compute CPM / log1p CPM.

    Rows may be single cells or pre-aggregated groups; n_cells is summed per condition.
    """
    if df.empty:
        return pd.DataFrame()

    long = df[["condition_id", "genes", "expressions"]].explode(["genes", "expressions"]).dropna()
    long = pd.DataFrame(
        {
            "condition_id": long["condition_id"].to_numpy(),
            "gene": long["genes"].astype("int64").to_numpy(),
            "sum_counts": long["expressions"].astype("float64").to_numpy(),
        }
    )
    grouped = long[~long["gene"].isin(SPECIAL_TOKENS)].groupby(["condition_id", "gene"], as_index=False)["sum_counts"].sum()

    cells = df.get("n_cells", pd.Series(1, index=df.index))
    n_cells = cells.groupby(df["condition_id"]).sum()
    grouped["n_cells"] = grouped["condition_id"].map(n_cells).astype("int64")

    grouped["library_size"] = grouped.groupby("condition_id")["sum_counts"].transform("sum")
    grouped["cpm"] = grouped["sum_counts"] / grouped["library_size"] * 1_000_000.0
    grouped["log1p_cpm"] = np.log1p(grouped["cpm"])
    return grouped.sort_values(["condition_id", "gene"]).reset_index(drop=True)
