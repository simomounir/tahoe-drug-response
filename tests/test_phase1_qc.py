from __future__ import annotations

import pandas as pd

from phase1.qc import render_qc_report, summarize_qc


def test_summarize_qc_flags_low_cell_conditions_and_controls():
    conditions = pd.DataFrame(
        [
            {"condition_id": "c1", "is_control": False, "n_cells": 10},
            {"condition_id": "c2", "is_control": True, "n_cells": 7},
            {"condition_id": "c3", "is_control": False, "n_cells": 2},
            {"condition_id": "c4", "is_control": True, "n_cells": 5},
        ]
    )

    pseudobulk = pd.DataFrame(
        [
            {"condition_id": "c1", "library_size": 1500.0},
            {"condition_id": "c2", "library_size": 2200.0},
            {"condition_id": "c3", "library_size": 700.0},
            {"condition_id": "c4", "library_size": 900.0},
        ]
    )

    summary = summarize_qc(
        conditions,
        pseudobulk,
        min_cells_per_condition=5,
        min_control_cells=6,
    )

    assert summary["n_conditions_total"] == 4
    assert summary["n_conditions_kept"] == 2
    assert summary["dropped_conditions"] == ["c3"]
    assert summary["n_control_conditions_total"] == 2
    assert summary["n_control_conditions_kept"] == 1
    assert summary["mean_library_size"] == 1850.0
    assert summary["status"] == "warn"
    assert "1 control(s) below 6 cells" in summary["warnings"][0]
    assert "WARNING" in render_qc_report(summary)


def test_render_qc_report_includes_key_summary_lines():
    summary = {
        "n_conditions_total": 4,
        "n_conditions_kept": 2,
        "dropped_conditions": ["c3"],
        "n_control_conditions_total": 2,
        "n_control_conditions_kept": 1,
        "mean_library_size": 1850.0,
        "status": "pass",
    }

    report = render_qc_report(summary)

    assert "# Phase 1 QC report" in report
    assert "Conditions retained: 2 / 4" in report
    assert "Dropped conditions: c3" in report
    assert "Controls retained: 1 / 2" in report
    assert "Status: pass" in report
