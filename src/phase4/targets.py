"""Gene universe, DE sets, scoring set and the dense target matrix (phase 4a spec E3–E5, task P4a.2)."""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from phase1.stream import sql_str


def gene_universe(expression_path: Path | str, genes: pd.DataFrame) -> pd.DataFrame:
    """DepMap protein-coding symbols (expression-file columns, ` (ENTREZ)` stripped) ∩ our gene symbols.

    One row per symbol, sorted by symbol; a symbol listed under two Entrez IDs counts once.
    `.attrs`: n_depmap, n_duplicate_symbols, n_not_in_ours.
    """
    columns = pq.read_schema(expression_path).names[1:]  # first column is the ACH model ID
    symbols = [c.rsplit(" (", 1)[0] for c in columns]
    unique = list(dict.fromkeys(symbols))
    ours = genes.drop_duplicates("gene_symbol").set_index("gene_symbol")["gene"]
    kept = sorted(s for s in unique if s in ours.index)
    universe = pd.DataFrame({"gene": [int(ours[s]) for s in kept], "gene_symbol": kept})
    universe.attrs.update(
        n_depmap=len(symbols), n_duplicate_symbols=len(symbols) - len(unique), n_not_in_ours=len(unique) - len(kept)
    )
    return universe


def eligible_lines(conditions: pd.DataFrame, logfc_condition_ids: set[str], min_fraction: float) -> set[str]:
    """Lines with logFC for at least `min_fraction` of their treated conditions (evaluation.md A2)."""
    treated = conditions[~conditions["is_control"].astype(bool)]
    has = treated["condition_id"].isin(logfc_condition_ids)
    fraction = has.groupby(treated["cell_line"]).mean()
    return set(fraction[fraction >= min_fraction].index)


def prepare_de_sets(con: duckdb.DuckDBPyConnection, raw_paths: list[Path], universe_symbols: set[str], cfg: dict) -> pd.DataFrame:
    """Per condition: protein-coding genes with padj < padj_below, ranked by padj then larger |log2FC|, capped at top_n.

    Filtered and ranked in DuckDB one raw file (= one cell line) at a time: the raw rows (681M on the real
    build) never reach pandas, only the capped sets do. Exact, since a condition's rows sit in one file.
    """
    de_cfg = cfg["de_genes"]
    con.register("p4_universe", pd.DataFrame({"gene_symbol": sorted(universe_symbols)}))
    parts = []
    try:
        for path in raw_paths:
            parts.append(con.execute(f"""
                SELECT gene_symbol, log2FoldChange, padj, condition_id, rank FROM (
                    SELECT r.gene_symbol, r.log2FoldChange, r.padj, r.condition_id,
                           row_number() OVER (PARTITION BY r.condition_id
                                              ORDER BY r.padj, abs(r.log2FoldChange) DESC, r.gene_symbol) AS rank
                    FROM read_parquet({sql_str(path)}) r SEMI JOIN p4_universe u ON r.gene_symbol = u.gene_symbol
                    WHERE r.padj < ? AND NOT isnan(r.padj) AND NOT isnan(r.log2FoldChange)
                )
                WHERE rank <= ?
            """, [de_cfg["padj_below"], de_cfg["top_n"]]).df())
    finally:
        con.unregister("p4_universe")
    columns = ["gene_symbol", "log2FoldChange", "padj", "condition_id", "rank"]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    return df.sort_values(["condition_id", "rank"]).reset_index(drop=True)


def scoring_set(view: pd.DataFrame, eligible: set[str], de_sets: pd.DataFrame, min_de_genes: int) -> tuple[pd.DataFrame, dict]:
    """Conditions every model is trained and scored on (evaluation.md A7), and counts of each exclusion reason.

    Rules apply in order; a condition is counted under the first rule it fails.
    """
    n_de = de_sets.groupby("condition_id").size()
    rules = [
        ("control", ~view["is_control"].astype(bool)),
        ("ineligible_line", view["cell_line"].isin(eligible)),
        ("features_incomplete", view["features_complete"].astype(bool)),
        ("no_de_set", view["condition_id"].isin(n_de.index)),
        ("below_min_de_genes", view["condition_id"].map(n_de).fillna(0) >= min_de_genes),
    ]
    keep = pd.Series(True, index=view.index)
    excluded = {}
    for reason, passes in rules:
        excluded[reason] = int((keep & ~passes).sum())
        keep &= passes
    return view[keep].reset_index(drop=True), excluded


def build_targets(con: duckdb.DuckDBPyConnection, logfc_path: Path | str, conditions: pd.DataFrame, universe: pd.DataFrame) -> np.ndarray:
    """Dense float32 logFC, rows in `conditions` order, columns in `universe` order.

    Genes absent from the sparse logFC were seen in neither the treated nor the control profile, so their
    logFC is exactly ln(1) - ln(1) = 0 (spec E4). Queried per cell line so the 1.3 GB file is never loaded whole.
    """
    matrix = np.zeros((len(conditions), len(universe)), dtype=np.float32)
    want_cols = pd.DataFrame({"gene": universe["gene"].astype("int64").to_numpy(), "col_idx": np.arange(len(universe), dtype=np.int32)})
    con.register("p4_want_cols", want_cols)
    rows_all = pd.DataFrame({"condition_id": conditions["condition_id"].to_numpy(), "row_idx": np.arange(len(conditions), dtype=np.int32),
                             "cell_line": conditions["cell_line"].to_numpy()})
    try:
        for _, group in rows_all.groupby("cell_line", sort=True):
            con.register("p4_want_rows", group[["condition_id", "row_idx"]])
            reader = con.execute(
                f"""
                SELECT r.row_idx, c.col_idx, l.logfc
                FROM read_parquet({sql_str(logfc_path)}) l
                JOIN p4_want_rows r ON l.condition_id = r.condition_id
                JOIN p4_want_cols c ON CAST(l.gene AS BIGINT) = c.gene
                """
            ).to_arrow_reader(1_000_000)
            for batch in reader:
                matrix[batch.column(0).to_numpy(), batch.column(1).to_numpy()] = batch.column(2).to_numpy().astype(np.float32)
            con.unregister("p4_want_rows")
    finally:
        con.unregister("p4_want_cols")
    return matrix
