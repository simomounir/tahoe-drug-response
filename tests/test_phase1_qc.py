from __future__ import annotations

import pandas as pd

from phase1.conditions import build_condition_table, load_fixture_metadata
from phase1.pipeline import run_phase1, run_phase1_from_config
from phase1.qc import render_qc_report, summarize_qc

FIXTURE_PATH = "tests/fixtures/metadata_small.csv"


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


def test_full_phase1_fixture_flow_condition_to_qc():
    df = load_fixture_metadata(FIXTURE_PATH)
    selected = ["CVCL_0546", "CVCL_0459"]

    conditions = build_condition_table(df, selected)
    library_lookup = {
        row["condition_id"]: float(row["n_cells"]) * 10.0
        for _, row in conditions.iterrows()
    }
    pseudobulk = pd.DataFrame(
        [{"condition_id": k, "library_size": v} for k, v in library_lookup.items()]
    )

    summary = summarize_qc(
        conditions,
        pseudobulk,
        min_cells_per_condition=100,
        min_control_cells=100,
    )
    report = render_qc_report(summary)

    assert summary["n_conditions_total"] == len(conditions)
    assert summary["n_conditions_kept"] == len(conditions)
    assert summary["n_control_conditions_total"] == 2
    assert summary["n_control_conditions_kept"] == 2
    assert "Conditions retained" in report
    assert "Controls retained" in report
    assert summary["status"] == "pass"


def test_run_phase1_pipeline_orchestrates_all_stages():
    metadata = load_fixture_metadata(FIXTURE_PATH)
    expression = pd.DataFrame(
        [
            {"condition_id": "c1", "genes": [0, 1], "expressions": [3.0, 4.0], "n_cells": 1},
            {"condition_id": "c1", "genes": [1, 2], "expressions": [2.0, 5.0], "n_cells": 1},
            {"condition_id": "c2", "genes": [0, 1], "expressions": [1.0, 1.0], "n_cells": 1},
            {"condition_id": "c3", "genes": [0, 1], "expressions": [2.0, 2.0], "n_cells": 1},
        ]
    )

    result = run_phase1(
        metadata,
        expression,
        selected_lines=["CVCL_0546", "CVCL_0459"],
        min_cells_per_condition=5,
        min_control_cells=6,
    )

    assert set(result["conditions"]["cell_line"]) == {"CVCL_0546", "CVCL_0459"}
    assert "condition_id" in result["pseudobulk"].columns
    assert set(result["summary"].keys()) >= {"n_conditions_total", "n_conditions_kept", "status"}
    assert "# Phase 1 QC report" in result["report"]
    assert result["summary"]["status"] == "pass"


def test_run_phase1_writes_fixture_outputs_to_disk(tmp_path):
    metadata = load_fixture_metadata(FIXTURE_PATH)
    expression = pd.DataFrame(
        [
            {"condition_id": "c1", "genes": [0, 1], "expressions": [3.0, 4.0], "n_cells": 1},
            {"condition_id": "c1", "genes": [1, 2], "expressions": [2.0, 5.0], "n_cells": 1},
            {"condition_id": "c2", "genes": [0, 1], "expressions": [1.0, 1.0], "n_cells": 1},
        ]
    )

    result = run_phase1(
        metadata,
        expression,
        selected_lines=["CVCL_0546"],
        min_cells_per_condition=5,
        min_control_cells=6,
        output_dir=tmp_path,
    )

    assert (tmp_path / "conditions.csv").exists()
    assert (tmp_path / "pseudobulk.csv").exists()
    assert (tmp_path / "qc_report.md").exists()
    assert "# Phase 1 QC report" in result["report"]


def test_run_phase1_from_config_uses_slice_values():
    metadata = load_fixture_metadata(FIXTURE_PATH)
    expression = pd.DataFrame(
        [
            {"condition_id": "c1", "genes": [0, 1], "expressions": [3.0, 4.0], "n_cells": 1},
            {"condition_id": "c1", "genes": [1, 2], "expressions": [2.0, 5.0], "n_cells": 1},
            {"condition_id": "c2", "genes": [0, 1], "expressions": [1.0, 1.0], "n_cells": 1},
        ]
    )
    config = {
        "selected_cell_lines": ["CVCL_0546", "CVCL_0459"],
        "min_cells_per_condition": 5,
        "min_control_cells": 6,
    }

    result = run_phase1_from_config(config, metadata, expression)

    assert set(result["conditions"]["cell_line"]) == {"CVCL_0546", "CVCL_0459"}
    assert result["summary"]["n_conditions_total"] == len(result["conditions"])
    assert result["summary"]["status"] == "pass"


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
