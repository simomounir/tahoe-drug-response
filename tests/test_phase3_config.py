from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from phase3 import config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "features.yaml"


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_repo_config_loads_with_every_section():
    cfg = config.load_features_config(REPO_CONFIG)
    assert set(cfg) >= {"depmap", "drugs", "cells", "crosscheck"}
    assert cfg["crosscheck"]["network_cap_gb"] == 25


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.load_features_config(tmp_path / "absent.yaml")


def test_missing_required_key_raises_and_names_it(tmp_path):
    data = yaml.safe_load(REPO_CONFIG.read_text())
    del data["drugs"]["morgan_radius"]
    with pytest.raises(ValueError, match="drugs.morgan_radius"):
        config.load_features_config(_write(tmp_path, data))


def test_optional_declarations_default_to_empty(tmp_path):
    data = yaml.safe_load(REPO_CONFIG.read_text())
    data["cells"].pop("expression_proxy", None)
    data["cells"].pop("expression_absent", None)
    cfg = config.load_features_config(_write(tmp_path, data))
    assert cfg["cells"]["expression_proxy"] == {} and cfg["cells"]["expression_absent"] == {}


def test_dose_units_convert_to_micromolar():
    assert config.DOSE_UNIT_TO_UM["nM"] == pytest.approx(1e-3)
    assert config.DOSE_UNIT_TO_UM["uM"] == config.DOSE_UNIT_TO_UM["µM"] == 1.0
