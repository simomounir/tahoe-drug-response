from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from phase1.conditions import assign_controls, build_condition_table, condition_lookup
from phase1.contracts import validate_conditions, validate_logfc, validate_output_size, validate_pseudobulk
from phase1.genes import validate_gene_coverage
from phase1.parse import CONTROL_DRUGS
from phase1.pseudobulk import SPECIAL_TOKENS
from phase1.qc import flag_conditions, render_qc_report, summarize_qc
from phase1.stream import combine_partials, sql_str, write_logfc

CONTROL_COLUMNS = ["cell_line", "plate", "condition_id", "n_cells", "library_size", "qc_pass"]


def build_outputs(
    con: duckdb.DuckDBPyConnection,
    gene_files: list[Path],
    cell_files: list[Path],
    metadata: pd.DataFrame,
    config: dict,
    output_dir: Path,
    report_path: Path,
    genes_path: Path | None = None,
) -> dict:
    """Shard partials + metadata -> validated pseudobulk, logfc, conditions, controls and QC report."""
    selected_lines = list(config["selected_cell_lines"])
    control_drugs = config.get("control_drugs", CONTROL_DRUGS)
    min_cells = int(config.get("min_cells_per_condition", 100))
    min_control = int(config.get("min_control_cells", 500))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata[metadata["cell_line"].isin(selected_lines)]

    pseudobulk_path = output_dir / "pseudobulk.parquet"
    per_condition = combine_partials(con, gene_files, cell_files, condition_lookup(metadata), pseudobulk_path)

    # n_cells = cells actually processed; n_cells_atlas = what the metadata says exists.
    conditions = build_condition_table(metadata, selected_lines, control_drugs).rename(columns={"n_cells": "n_cells_atlas"})
    line_info = config.get("cell_line_info") or {}
    conditions["depmap_id"] = conditions["cell_line"].map(lambda line: (line_info.get(line) or {}).get("depmap_id"))
    conditions = conditions.merge(per_condition, on="condition_id", how="left")
    conditions["n_cells"] = conditions["n_cells"].fillna(0).astype("int64")
    conditions["qc_pass"] = flag_conditions(conditions, min_cells, min_control)
    conditions = assign_controls(conditions)
    summary = summarize_qc(conditions, per_condition, min_cells_per_condition=min_cells, min_control_cells=min_control)

    logfc_path = output_dir / "logfc.parquet"
    n_logfc = write_logfc(con, pseudobulk_path, conditions, logfc_path)
    summary["n_logfc_conditions"] = n_logfc

    validate_conditions(conditions)
    validate_pseudobulk(con, pseudobulk_path, conditions)
    validate_logfc(con, logfc_path, n_logfc)

    max_output_mb = config.get("max_output_mb")
    output_mb = (pseudobulk_path.stat().st_size + logfc_path.stat().st_size) / 1e6
    summary["output_size_mb"] = round(output_mb, 2)
    summary["max_output_mb"] = max_output_mb
    validate_output_size(output_mb, max_output_mb)

    if genes_path is None:
        genes_path = output_dir / "genes.parquet"
    if genes_path.exists():
        validate_gene_coverage(con, pseudobulk_path, genes_path)
        summary["gene_coverage"] = _zero_count_genes(con, pseudobulk_path, genes_path)
    else:
        summary["gene_coverage"] = {"skipped": True}

    conditions.to_parquet(output_dir / "conditions.parquet", index=False)
    conditions[conditions["is_control"]][CONTROL_COLUMNS].rename(columns={"condition_id": "control_condition_id"}).to_parquet(
        output_dir / "controls.parquet", index=False
    )
    report_path.write_text(render_qc_report(summary), encoding="utf-8")
    return {"summary": summary, "conditions": conditions}


def _zero_count_genes(con: duckdb.DuckDBPyConnection, pseudobulk_path: Path, genes_path: Path) -> dict:
    """Vocabulary genes (excluding SPECIAL_TOKENS) with zero counts across the whole pseudobulk slice."""
    tokens = ", ".join(str(t) for t in SPECIAL_TOKENS)
    n_zero, symbols = con.execute(
        f"""
        WITH vocab AS (
            SELECT gene, gene_symbol FROM read_parquet({sql_str(genes_path)})
            WHERE gene NOT IN ({tokens})
        ),
        observed AS (SELECT DISTINCT gene FROM read_parquet({sql_str(pseudobulk_path)})),
        zero AS (SELECT gene, gene_symbol FROM vocab ANTI JOIN observed USING (gene))
        SELECT (SELECT COUNT(*) FROM zero),
               (SELECT list(gene_symbol ORDER BY gene_symbol) FROM zero WHERE gene_symbol IS NOT NULL AND gene_symbol <> '')
        """
    ).fetchone()
    return {"n_zero_count_genes": int(n_zero), "zero_count_gene_sample": (symbols or [])[:20], "skipped": False}
