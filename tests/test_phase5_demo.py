from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from phase4 import harness
from phase5 import demo_data
from test_phase5_case_study import GENES, _eval_dir
from test_phase5_reproduce import CFG


def _setup(tmp_path, repeats=1):
    ev = _eval_dir(tmp_path)
    if repeats > 1:  # repeat 1 = same split again, so every test condition's first test repeat is 0
        s = pd.read_parquet(ev / "splits.parquet")
        pd.concat([s, s.assign(repeat=1)]).to_parquet(ev / "splits.parquet")
    for m in demo_data.MODELS:
        harness.evaluate(m, CFG, ev, ev / "results")
    return ev


def _export(tmp_path, ev):
    return demo_data.export(ev, ev / "results", CFG, tmp_path / "out")


def _shard(tmp_path, drug):
    idx = json.loads((tmp_path / "out" / "index.json").read_text())
    return idx, json.loads((tmp_path / "out" / idx["drugs"][drug]["shard"]).read_text())


def test_shard_values(tmp_path):
    ev = _setup(tmp_path)
    _export(tmp_path, ev)
    idx, shard = _shard(tmp_path, "D3")
    c = shard["conditions"]["L3-D3-0"]
    assert [idx["genes"][g] for g in c["genes"]] == GENES[:3]  # DE genes in rank order
    data = harness.load_eval_dir(ev)
    row = data["conditions"].index[data["conditions"]["condition_id"] == "L3-D3-0"][0]
    np.testing.assert_allclose(np.array(c["measured"]) / 100, np.asarray(data["targets"][row])[:3], atol=0.006)
    s = c["splits"]["both_unseen"]
    assert s["repeat"] == 0 and set(s["pred"]) == set(demo_data.MODELS)
    per = pd.read_parquet(ev / "results" / "ridge" / "per_condition.parquet").query("condition_id == 'L3-D3-0'")
    assert s["score"]["ridge"] == pytest.approx(float(per["de_pearson"].iloc[0]), abs=1e-6)


def test_first_test_repeat(tmp_path):
    ev = _setup(tmp_path, repeats=2)
    _export(tmp_path, ev)
    _, shard = _shard(tmp_path, "D3")
    assert shard["conditions"]["L3-D3-0"]["splits"]["both_unseen"]["repeat"] == 0


def test_never_tested_absent(tmp_path):
    ev = _setup(tmp_path)
    _export(tmp_path, ev)
    idx, shard = _shard(tmp_path, "D0")
    assert shard["conditions"]["L0-D0-0"]["splits"] == {}  # a training condition: never held out


def test_contract_refit_vs_scored(tmp_path):
    ev = _setup(tmp_path)
    p = ev / "results" / "ridge" / "per_condition.parquet"
    per = pd.read_parquet(p)
    per["de_pearson"] = per["de_pearson"] + 0.05
    per.to_parquet(p)
    with pytest.raises(demo_data.DemoError, match="ridge"):
        _export(tmp_path, ev)


def test_index_lists_shards(tmp_path):
    ev = _setup(tmp_path)
    summary = _export(tmp_path, ev)
    idx = json.loads((tmp_path / "out" / "index.json").read_text())
    assert set(idx["drugs"]) == {"D0", "D1", "D2", "D3"} and all((tmp_path / "out" / d["shard"]).exists() for d in idx["drugs"].values())
    assert summary["bytes"] > 0 and "both_unseen" in idx["medians"]
    assert idx["featured"] == "L3-D3-0"  # the case study's failure case opens the demo


def test_shard_names_safe():
    names = {demo_data.shard_name(d) for d in ["A/B drug", "A B drug", "Café-1", "x" * 200]}
    assert len(names) == 4 and all(n.endswith(".json") and "/" not in n and " " not in n and len(n) < 80 for n in names)


def test_neural_checked_by_split_median(tmp_path):
    ev = _setup(tmp_path)
    p = ev / "results" / "neural" / "per_condition.parquet"
    per = pd.read_parquet(p)
    per["de_pearson"] = per["de_pearson"] + 0.05  # shifts the split median
    per.to_parquet(p)
    with pytest.raises(demo_data.DemoError, match="neural.*median"):
        _export(tmp_path, ev)


def test_stores_refit_and_evaluated_scores(tmp_path):
    ev = _setup(tmp_path)
    _export(tmp_path, ev)
    _, shard = _shard(tmp_path, "D3")
    s = shard["conditions"]["L3-D3-0"]["splits"]["both_unseen"]
    assert set(s["score"]) == set(s["scored"]) == set(demo_data.MODELS)


def test_index_lists_tested_per_split(tmp_path):
    ev = _setup(tmp_path)
    _export(tmp_path, ev)
    idx, shard = _shard(tmp_path, "D3")
    assert "L3-D3-0" in idx["tested"]["both_unseen"] and "L0-D0-0" not in idx["tested"]["both_unseen"]
    assert idx["tested"]["random"] == []  # the fixture only has a both_unseen split


def test_missing_recorded_config_raises(tmp_path):
    ev = _setup(tmp_path)
    p = ev / "results" / "neural" / "results.json"
    res = json.loads(p.read_text())
    res["model_info"] = []
    p.write_text(json.dumps(res))
    with pytest.raises(demo_data.DemoError, match="recorded"):
        _export(tmp_path, ev)


def test_nan_on_one_side_raises(tmp_path):
    ev = _setup(tmp_path)
    p = ev / "results" / "neural" / "per_condition.parquet"
    per = pd.read_parquet(p)
    per.loc[per["condition_id"] == "L3-D3-0", "de_pearson"] = np.nan
    per.to_parquet(p)
    with pytest.raises(demo_data.DemoError, match="NaN"):
        _export(tmp_path, ev)
