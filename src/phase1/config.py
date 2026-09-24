from __future__ import annotations

from pathlib import Path

import yaml


def load_slice_config(path: Path | str) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Slice config at {path} is not a mapping")
    return config
