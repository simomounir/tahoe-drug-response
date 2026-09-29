from __future__ import annotations

import json

import pandas as pd
import pytest

from phase4 import harness
from phase5 import reproduce
from test_phase5_case_study import CS_CFG, _eval_dir, _results

CFG = {**CS_CFG, "metrics": {"topk": [2]}}


def test_level1_passes_and_detects_tamper(tmp_path):
    _results(tmp_path / "r", {"global_mean": 0.6, "ridge": 0.62, "neural": 0.61})
    expected = reproduce.headline(tmp_path / "r", CFG)
    assert reproduce.compare(reproduce.headline(tmp_path / "r", CFG), expected, 0.0005) == []
    p = tmp_path / "r" / "ridge" / "per_condition.parquet"
    per = pd.read_parquet(p)
    per.loc[per["split"] == "random", "de_pearson"] += 0.05
    per.to_parquet(p)
    bad = reproduce.compare(reproduce.headline(tmp_path / "r", CFG), expected, 0.0005)
    assert bad and any("ridge" in b and "random" in b for b in bad)


def test_compare_reports_all():
    exp = {"verdict": "FAILURE", "de_pearson": {"ridge": {"random": [0.5, 0.4, 0.6], "both_unseen": [0.5, 0.4, 0.6]}}}
    got = {"verdict": "SUCCESS", "de_pearson": {"ridge": {"random": [0.6, 0.4, 0.6], "both_unseen": [0.5, 0.4, 0.7]}}}
    bad = reproduce.compare(got, exp, 0.001)
    assert len(bad) == 3 and any("verdict" in b for b in bad)


def _scored(tmp_path):
    ev = _eval_dir(tmp_path)
    for m in ["global_mean", "ridge", "neural"]:
        harness.evaluate(m, CFG, ev, ev / "results")
    return ev


def test_refit_recorded_config_within_tol(tmp_path):
    ev = _scored(tmp_path)
    original = reproduce.headline(ev / "results", CFG, splits=["both_unseen"])
    out = reproduce.refit_both_unseen(ev, ev / "results", CFG, tmp_path / "refit")
    assert reproduce.compare(reproduce.headline(out, CFG, splits=["both_unseen"]), original, 0.003) == []


def test_refit_missing_info(tmp_path, capsys):
    ev = _scored(tmp_path)
    p = ev / "results" / "neural" / "results.json"
    res = json.loads(p.read_text())
    res["model_info"] = []
    p.write_text(json.dumps(res))
    reproduce.refit_both_unseen(ev, ev / "results", CFG, tmp_path / "refit")
    assert "no recorded neural config" in capsys.readouterr().out


def test_missing_expected(tmp_path):
    with pytest.raises(reproduce.ReproduceError, match="--write-expected"):
        reproduce.load_expected(tmp_path / "expected.json")


def test_compare_shorter_list_fails():
    assert reproduce.compare({"x": [0.5]}, {"x": [0.5, 0.6, 0.7]}, 0.01)


def test_level2_uses_bundled_config(tmp_path):
    (tmp_path / "eval.yaml").write_text("models: {neural: {seed: 1}}\n")
    assert reproduce.bundle_config(tmp_path, {"models": {"neural": {"seed": 2}}})["models"]["neural"]["seed"] == 1
    assert reproduce.bundle_config(tmp_path / "missing", {"a": 1}) == {"a": 1}
