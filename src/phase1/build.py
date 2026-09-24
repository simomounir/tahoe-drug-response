from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from phase1.conditions import assign_controls, build_condition_table, condition_lookup
from phase1.contracts import validate_conditions, validate_logfc, validate_pseudobulk
from phase1.parse import CONTROL_DRUGS
from phase1.qc import flag_conditions, render_qc_report, summarize_qc
from phase1.stream import combine_partials, write_logfc

CONTROL_COLUMNS = ["cell_line", "plate", "condition_id", "n_cells", "library_size", "qc_pass"]


def build_outputs(
    con: duckdb.DuckDBPyConnection,
    gene_files: list[Path],
    cell_files: list[Path],
    metadata: pd.DataFrame,
    config: dict,
    output_dir: Path,
    report_path: Path,
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

    conditions.to_parquet(output_dir / "conditions.parquet", index=False)
    conditions[conditions["is_control"]][CONTROL_COLUMNS].rename(columns={"condition_id": "control_condition_id"}).to_parquet(
        output_dir / "controls.parquet", index=False
    )
    report_path.write_text(render_qc_report(summary), encoding="utf-8")
    return {"summary": summary, "conditions": conditions}
