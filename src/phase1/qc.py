from __future__ import annotations

import pandas as pd


def flag_conditions(conditions: pd.DataFrame, min_cells_per_condition: int, min_control_cells: int) -> pd.Series:
    """True where a condition has enough cells (controls use the stricter threshold)."""
    required = conditions["is_control"].map(lambda c: min_control_cells if c else min_cells_per_condition)
    return conditions["n_cells"] >= required


def summarize_qc(
    conditions: pd.DataFrame,
    pseudobulk: pd.DataFrame,
    min_cells_per_condition: int = 100,
    min_control_cells: int = 500,
) -> dict:
    """Summarize condition-level QC and return a small status dictionary.

    The logic is intentionally compact and deterministic so that downstream report
    generation can build a richer markdown report from this baseline summary.
    """
    if conditions.empty:
        return {
            "n_conditions_total": 0,
            "n_conditions_kept": 0,
            "dropped_conditions": [],
            "n_control_conditions_total": 0,
            "n_control_conditions_kept": 0,
            "mean_library_size": 0.0,
            "warnings": ["no conditions"],
            "status": "warn",
        }

    conditions = conditions.copy()
    pseudobulk = pseudobulk.copy()

    if "condition_id" not in conditions.columns:
        raise ValueError("conditions must include condition_id")
    if "condition_id" not in pseudobulk.columns:
        raise ValueError("pseudobulk must include condition_id")

    library_by_condition = pseudobulk.groupby("condition_id")["library_size"].max().to_dict()
    conditions["library_size"] = conditions["condition_id"].map(library_by_condition).fillna(0.0)

    kept_mask = flag_conditions(conditions, min_cells_per_condition, min_control_cells)
    kept = conditions[kept_mask].copy()
    dropped = conditions[~kept_mask & ~conditions["is_control"]]["condition_id"].tolist()

    control_rows = conditions[conditions["is_control"] == True]
    control_kept = control_rows[control_rows["n_cells"] >= min_control_cells]

    mean_library_size = float(kept["library_size"].mean()) if not kept.empty else 0.0

    warnings = []
    if kept.empty:
        warnings.append("no conditions passed the cell-count threshold")
    if control_rows.empty:
        warnings.append("no DMSO control conditions found")
    elif len(control_kept) < len(control_rows):
        warnings.append(
            f"{len(control_rows) - len(control_kept)} control(s) below {min_control_cells} cells; "
            "their cell lines have no usable baseline on that plate"
        )
    status = "warn" if warnings else "pass"

    tables = {}
    detail_cols = [c for c in ["cell_line", "plate", "drug_name", "dose", "n_cells", "n_cells_atlas"] if c in conditions.columns]
    if "cell_line" in conditions.columns:
        dropped_rows = conditions[~kept_mask].copy()
        dropped_rows["reason"] = [
            f"{'control ' if c else ''}n_cells {n} < {min_control_cells if c else min_cells_per_condition}"
            for c, n in zip(dropped_rows["is_control"], dropped_rows["n_cells"])
        ]
        tables["Dropped conditions"] = dropped_rows[detail_cols + ["reason"]]
        per_line = conditions.assign(kept=kept_mask).groupby("cell_line").agg(
            conditions=("condition_id", "size"),
            kept=("kept", "sum"),
            median_cells=("n_cells", "median"),
            median_library_size=("library_size", "median"),
        )
        tables["Per cell line"] = per_line.reset_index()
        tables["Controls"] = control_rows[[c for c in detail_cols if c != "drug_name"] + ["library_size"]]

    return {
        "n_conditions_total": int(len(conditions)),
        "n_conditions_kept": int(len(kept)),
        "dropped_conditions": dropped,
        "n_control_conditions_total": int(len(control_rows)),
        "n_control_conditions_kept": int(len(control_kept)),
        "mean_library_size": mean_library_size,
        "warnings": warnings,
        "status": status,
        "tables": tables,
    }


def render_qc_report(summary: dict) -> str:
    """Render a compact markdown QC summary for Phase 1."""
    dropped = ", ".join(summary.get("dropped_conditions", [])) if summary.get("dropped_conditions") else "none"
    lines = [
        "# Phase 1 QC report",
        "",
        f"Conditions retained: {summary.get('n_conditions_kept', 0)} / {summary.get('n_conditions_total', 0)}",
        f"Dropped conditions: {dropped}",
        f"Controls retained: {summary.get('n_control_conditions_kept', 0)} / {summary.get('n_control_conditions_total', 0)}",
        f"Mean library size: {summary.get('mean_library_size', 0.0)}",
        f"Status: {summary.get('status', 'warn')}",
    ]
    lines += [f"- WARNING: {w}" for w in summary.get("warnings", [])]
    if "n_logfc_conditions" in summary:
        lines.append(f"Treated conditions with logFC vs plate-matched DMSO: {summary['n_logfc_conditions']}")
    gene_coverage = summary.get("gene_coverage")
    if gene_coverage is not None:
        if gene_coverage.get("skipped"):
            lines.append("Zero-count genes: skipped (genes.parquet not found)")
        else:
            sample = gene_coverage.get("zero_count_gene_sample", [])
            lines.append(f"Zero-count genes across the slice: {gene_coverage.get('n_zero_count_genes', 0)}")
            if sample:
                lines.append(f"Zero-count gene sample: {', '.join(sample)}")
    for title, table in summary.get("tables", {}).items():
        lines += ["", f"## {title}", "", _markdown_table(table) if len(table) else "none"]
    return "\n".join(lines)


def _markdown_table(df: pd.DataFrame) -> str:
    def fmt(v) -> str:
        return f"{v:,.0f}" if isinstance(v, float) and v == v and v >= 1000 else str(v)

    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(rows)
