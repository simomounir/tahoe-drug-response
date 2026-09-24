from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase1.config import load_slice_config
from phase1.parse import is_control_drug, parse_drugname_drugconc


def test_load_slice_config_reads_selected_cell_lines():
    config = load_slice_config(Path(__file__).resolve().parents[1] / "configs" / "slice.yaml")

    assert config["status"] == "validated_candidate"
    assert config["selected_cell_lines"][0] == "CVCL_0546"
    assert len(config["selected_cell_lines"]) == 8


def test_parse_drugname_drugconc_extracts_dose_and_unit():
    parsed = parse_drugname_drugconc("[('Adagrasib', 0.05, 'uM')]")

    assert parsed == {"drug": "Adagrasib", "dose": 0.05, "unit": "uM"}


def test_is_control_drug_matches_vehicle_only():
    assert is_control_drug("DMSO_TF") is True
    assert is_control_drug("Trametinib (DMSO_TF solvate)") is False
    assert is_control_drug("Adagrasib") is False
