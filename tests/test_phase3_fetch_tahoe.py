from __future__ import annotations

import importlib.util
from pathlib import Path

from phase3.config import load_features_config

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_tahoe_metadata.py"
spec = importlib.util.spec_from_file_location("fetch_tahoe_metadata", SCRIPT)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def test_downloads_are_pinned_to_the_configured_revision():
    cfg = load_features_config()
    plan = fetch.planned_downloads(cfg)
    assert {item["repo_file"] for item in plan} == {"metadata/drug_metadata.parquet", "metadata/cell_line_metadata.parquet"}
    assert all(item["revision"] == cfg["tahoe"]["revision"] and len(item["revision"]) == 40 for item in plan)
    dests = {item["dest"].relative_to(fetch.ROOT).as_posix() for item in plan}
    assert dests == {"data/cache/hf/metadata/drug_metadata.parquet", "data/cache/cell_line_metadata.parquet"}


def test_sha256_of_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert fetch.sha256_of(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
