"""The one loader for configs/features.yaml, and the dose-unit table shared by phase 3 modules.

Every setting comes from the file: a missing file, section or required key raises instead of
falling back to a default, so a typo cannot silently change a pre-registered value.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

FEATURES_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "features.yaml"

REQUIRED_KEYS = {
    "depmap": ["release", "figshare_article", "files"],
    "drugs": ["morgan_radius", "morgan_n_bits", "butina_similarity", "report_similarities"],
    "cells": ["pca_components", "mutation_min_frequency", "driver_min_lines"],
    "crosscheck": ["n_conditions", "seed", "padj", "min_median_sign_agreement", "network_cap_gb"],
    "tahoe": ["repo", "revision"],
}

# Declarations, not tunables: absent means nothing is declared.
OPTIONAL_DECLARATIONS = {"cells": ["expression_proxy", "expression_absent"]}

DOSE_UNIT_TO_UM = {"uM": 1.0, "µM": 1.0, "nM": 1e-3, "mM": 1e3}


def load_features_config(path: Path | str | None = None) -> dict:
    """The whole configs/features.yaml, validated against REQUIRED_KEYS."""
    path = Path(path) if path is not None else FEATURES_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"features config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = copy.deepcopy(yaml.safe_load(fh) or {})
    missing = [
        f"{section}.{key}"
        for section, keys in REQUIRED_KEYS.items()
        for key in keys
        if not isinstance(cfg.get(section), dict) or key not in cfg[section]
    ]
    if missing:
        raise ValueError(f"{path}: missing required setting(s) {missing}")
    for section, keys in OPTIONAL_DECLARATIONS.items():
        for key in keys:
            cfg[section][key] = cfg[section].get(key) or {}
    return cfg
