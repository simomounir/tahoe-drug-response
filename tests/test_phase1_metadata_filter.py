from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
LINES = ["L1", "L2"]


@pytest.fixture()
def run_phase1(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_phase1_under_test", ROOT / "scripts" / "run_phase1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    return module


def _cells(with_pass_filter: bool) -> pd.DataFrame:
    rows = []
    # (cell_line, sample, n_full, n_minimal); L3 is outside the slice.
    for line, sample, n_full, n_min in [("L1", "s1", 3, 2), ("L1", "s2", 4, 0), ("L2", "s1", 1, 5), ("L3", "s1", 7, 7)]:
        for pf, n in (("full", n_full), ("minimal", n_min)):
            for _ in range(n):
                rows.append({
                    "plate": "plate3", "sample": sample, "cell_line": line, "drug": "DrugA",
                    "drugname_drugconc": "[('DrugA', 5.0, 'uM')]", "pass_filter": pf, "gene_count": 100,
                })
    df = pd.DataFrame(rows)
    return df if with_pass_filter else df.drop(columns="pass_filter")


def _counts(df: pd.DataFrame) -> dict:
    return {(r.cell_line, r.sample): int(r.n_cells) for r in df.itertuples()}


def test_counts_only_full_cells_when_pass_filter_present(run_phase1, tmp_path):
    src = tmp_path / "obs.parquet"
    _cells(with_pass_filter=True).to_parquet(src, index=False)

    out = run_phase1.load_metadata_source(src, LINES)

    assert _counts(out) == {("L1", "s1"): 3, ("L1", "s2"): 4, ("L2", "s1"): 1}
    assert list(out.columns) == ["cell_line", "plate", "sample", "drug", "drugname_drugconc", "n_cells"]


def test_counts_all_cells_when_pass_filter_absent(run_phase1, tmp_path):
    src = tmp_path / "obs.parquet"
    _cells(with_pass_filter=False).to_parquet(src, index=False)

    out = run_phase1.load_metadata_source(src, LINES)

    assert _counts(out) == {("L1", "s1"): 5, ("L1", "s2"): 4, ("L2", "s1"): 6}


def test_cache_key_includes_filter_and_ignores_old_unfiltered_cache(run_phase1, tmp_path):
    src = tmp_path / "obs.parquet"
    _cells(with_pass_filter=True).to_parquet(src, index=False)
    cache = tmp_path / "cache"
    old_key = hashlib.sha1((str(src) + "|" + ",".join(sorted(LINES))).encode()).hexdigest()[:12]
    stale = pd.DataFrame([{"cell_line": "L1", "plate": "plate3", "sample": "s1", "drug": "DrugA",
                           "drugname_drugconc": "x", "n_cells": 999}])
    stale.to_parquet(cache / f"metadata_by_sample_{old_key}.parquet", index=False)

    full = run_phase1.load_metadata_source(src, LINES)
    minimal = run_phase1.load_metadata_source(src, LINES, pass_filter="minimal")

    assert _counts(full)[("L1", "s1")] == 3
    assert _counts(minimal) == {("L1", "s1"): 2, ("L2", "s1"): 5}
    assert len(list(cache.glob("metadata_by_sample_*.parquet"))) == 3

    # Second call is served from the new cache even if the source disappears.
    src.unlink()
    assert _counts(run_phase1.load_metadata_source(src, LINES)) == _counts(full)


def test_csv_branch_without_pass_filter_still_works(run_phase1, tmp_path):
    src = tmp_path / "meta.csv"
    pd.DataFrame([
        {"cell_line": "L1", "drug": "DrugA", "drugname_drugconc": "[('DrugA', 5.0, 'uM')]", "plate": "plate3", "sample": "s1", "n_cells": 10},
        {"cell_line": "L3", "drug": "DrugA", "drugname_drugconc": "[('DrugA', 5.0, 'uM')]", "plate": "plate3", "sample": "s1", "n_cells": 20},
    ]).to_csv(src, index=False)

    out = run_phase1.load_metadata_source(src, LINES)

    assert _counts(out) == {("L1", "s1"): 10}
