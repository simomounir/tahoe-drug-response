from __future__ import annotations

from pathlib import Path

import pandas as pd

from phase1.conditions import attach_condition_ids, build_condition_table
from phase1.config import load_slice_config
from phase1.pseudobulk import build_pseudobulk
from phase1.qc import render_qc_report, summarize_qc


def run_phase1(
    metadata: pd.DataFrame,
    expression: pd.DataFrame,
    selected_lines: list[str],
    min_cells_per_condition: int = 100,
    min_control_cells: int = 500,
    output_dir: str | Path | None = None,
) -> dict:
    """Run the small Phase 1 pipeline for the fixture dataset."""
    conditions = build_condition_table(metadata, selected_lines)

    if "condition_id" not in expression.columns:
        expression = attach_condition_ids(expression, metadata)

    # The fixture expression table can contain condition IDs that are already present
    # in the condition table; this keeps the orchestration simple and testable.
    pseudobulk = build_pseudobulk(expression)

    summary = summarize_qc(
        conditions,
        pseudobulk,
        min_cells_per_condition=min_cells_per_condition,
        min_control_cells=min_control_cells,
    )
    report = render_qc_report(summary)

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        conditions.to_csv(out_dir / "conditions.csv", index=False)
        pseudobulk.to_csv(out_dir / "pseudobulk.csv", index=False)
        (out_dir / "qc_report.md").write_text(report, encoding="utf-8")

    return {
        "conditions": conditions,
        "pseudobulk": pseudobulk,
        "summary": summary,
        "report": report,
    }


def run_phase1_from_config(
    config: dict,
    metadata: pd.DataFrame,
    expression: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> dict:
    """Run Phase 1 using a slice config dictionary, not hardcoded values."""
    selected_lines = config.get("selected_cell_lines", [])
    if not selected_lines:
        raise ValueError("Slice config must include selected_cell_lines")
    return run_phase1(
        metadata=metadata,
        expression=expression,
        selected_lines=list(selected_lines),
        min_cells_per_condition=int(config.get("min_cells_per_condition", 100)),
        min_control_cells=int(config.get("min_control_cells", 500)),
        output_dir=output_dir,
    )


def run_phase1_from_yaml(
    config_path: str | Path,
    metadata: pd.DataFrame,
    expression: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> dict:
    """Run Phase 1 from a YAML slice config path."""
    config = load_slice_config(config_path)
    return run_phase1_from_config(config, metadata, expression, output_dir=output_dir)
