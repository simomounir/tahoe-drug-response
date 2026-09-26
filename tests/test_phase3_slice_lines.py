from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_slice_lines.py"
spec = importlib.util.spec_from_file_location("make_slice_lines", SCRIPT)
make_slice_lines = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_slice_lines)

SCREENED = pd.DataFrame({"cell_line": ["CVCL_A", "CVCL_B", "CVCL_C"], "n_cells": [300, 200, 100]})
TAHOE = pd.DataFrame(
    {
        # Tahoe's table has one row per driver mutation, so lines repeat.
        "Cell_ID_Cellosaur": ["CVCL_A", "CVCL_A", "CVCL_B", "CVCL_C", "CVCL_X"],
        "cell_name": ["Alpha", "Alpha", "Beta", "Gamma", "Unscreened"],
        "Cell_ID_DepMap": ["ACH-000001", "ACH-000001", "ACH-000002", None, "ACH-000009"],
        "Organ": ["Lung", "Lung", "Bowel", "Pancreas", "Skin"],
    }
)
CONFIG_TEXT = """dataset_revision: main
# kept comment
{begin}
selected_cell_lines:
  - OLD
{end}

min_cells_per_condition: 100
""".format(begin=make_slice_lines.BEGIN, end=make_slice_lines.END)


def test_lines_follow_screened_order_and_ignore_unscreened_lines():
    lines = make_slice_lines.slice_lines(SCREENED, TAHOE, {"CVCL_C": "no DepMap entry"})
    assert list(lines["cell_line"]) == ["CVCL_A", "CVCL_B", "CVCL_C"]


def test_missing_depmap_id_fails_unless_allowed():
    with pytest.raises(ValueError, match="CVCL_C"):
        make_slice_lines.slice_lines(SCREENED, TAHOE, {})


def test_screened_line_absent_from_tahoe_metadata_fails():
    screened = pd.concat([SCREENED, pd.DataFrame({"cell_line": ["CVCL_Z"], "n_cells": [5]})])
    with pytest.raises(ValueError, match="CVCL_Z"):
        make_slice_lines.slice_lines(screened, TAHOE, {"CVCL_C": "no DepMap entry"})


def test_rendered_block_parses_to_lines_and_info():
    lines = make_slice_lines.slice_lines(SCREENED, TAHOE, {"CVCL_C": "no DepMap entry"})
    config = yaml.safe_load(make_slice_lines.render_block(lines))
    assert config["selected_cell_lines"] == ["CVCL_A", "CVCL_B", "CVCL_C"]
    assert config["cell_line_info"]["CVCL_A"] == {"depmap_id": "ACH-000001", "tissue": "Lung"}
    assert config["cell_line_info"]["CVCL_C"] == {"depmap_id": None, "depmap_absent": "no DepMap entry", "tissue": "Pancreas"}


def test_replace_block_keeps_everything_outside_the_markers_and_is_idempotent():
    lines = make_slice_lines.slice_lines(SCREENED, TAHOE, {"CVCL_C": "no DepMap entry"})
    block = make_slice_lines.render_block(lines)
    once = make_slice_lines.replace_block(CONFIG_TEXT, block)
    assert make_slice_lines.replace_block(once, block) == once
    assert "# kept comment" in once and "min_cells_per_condition: 100" in once and "OLD" not in once
    assert yaml.safe_load(once)["selected_cell_lines"] == ["CVCL_A", "CVCL_B", "CVCL_C"]


def test_replace_block_requires_markers():
    with pytest.raises(ValueError, match="marker"):
        make_slice_lines.replace_block("selected_cell_lines: []\n", "x: 1\n")
