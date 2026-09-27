from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from phase1.contracts import ContractError
from phase4 import contracts, harness, splits
from phase4.predictors import PREDICTORS, DummyPredictor, TrainData

GENES = [f"G{i}" for i in range(5)]
CFG = {
    "splits": {"repeats": 2, "seeds": [1, 2], "test_fraction": 0.2, "val_fraction": 0.15},
    "metrics": {"topk": [2]},
    "bootstrap": {"resamples": 50, "interval": 0.95, "seed": 1},
    "min_de_genes": 2,
}


def _write_eval_dir(tmp_path):
    rng = np.random.default_rng(0)
    rows = [{"condition_id": f"{l}-{d}", "cell_line": l, "drug": d, "dose": 0.5, "features_complete": True, "pc_1": rng.normal()}
            for l in ["L0", "L1", "L2", "L3", "L4", "L5"] for d in ["D0", "D1", "D2", "D3", "D4", "D5"]]
    cond = pd.DataFrame(rows)
    groups = pd.DataFrame({"drug": [f"D{i}" for i in range(6)], "cluster_id": range(6)})
    de = pd.DataFrame([{"condition_id": c, "gene_symbol": g, "rank": r + 1} for c in cond["condition_id"] for r, g in enumerate(GENES[:3])])
    d = tmp_path / "eval"
    d.mkdir()
    cond.to_parquet(d / "conditions.parquet")
    pd.DataFrame({"gene_symbol": GENES}).to_parquet(d / "genes.parquet")
    np.save(d / "targets.npy", rng.normal(size=(len(cond), len(GENES))).astype(np.float32))
    de.to_parquet(d / "de_sets.parquet")
    splits.draw_splits(cond, groups, CFG).to_parquet(d / "splits.parquet")
    return d


def test_dummy_predicts_zeros_of_the_right_shape():
    model = DummyPredictor()
    model.fit(TrainData(features=pd.DataFrame({"x": [1, 2]}), targets=np.ones((2, 5), np.float32), genes=GENES, part=np.array(["train", "val"])))
    pred = model.predict(pd.DataFrame({"x": [1, 2, 3]}))
    assert pred.shape == (3, 5) and not pred.any()
    assert PREDICTORS["dummy"] is DummyPredictor


@pytest.mark.parametrize(
    "pred,match",
    [(np.zeros((2, 4), np.float32), "shape"), (np.array([[np.nan] * 5, [0.0] * 5], np.float32), "NaN")],
)
def test_prediction_contract(pred, match):
    with pytest.raises(ContractError, match=match):
        contracts.validate_prediction(pred, n_rows=2, n_genes=5)


def test_split_contract_catches_a_line_on_both_sides():
    cond = pd.DataFrame({"condition_id": ["a", "b"], "cell_line": ["L0", "L0"], "drug": ["D0", "D1"]})
    groups = pd.DataFrame({"drug": ["D0", "D1"], "cluster_id": [0, 1]})
    bad = pd.DataFrame({"repeat": [0, 0], "split": ["unseen_cell_line"] * 2, "condition_id": ["a", "b"], "part": ["train", "test"]})
    with pytest.raises(ContractError, match="unseen_cell_line"):
        contracts.validate_splits(bad, cond, groups, forbidden_test_lines=set())


def test_split_contract_forbidden_test_line():
    cond = pd.DataFrame({"condition_id": ["a"], "cell_line": ["GAP"], "drug": ["D0"]})
    groups = pd.DataFrame({"drug": ["D0"], "cluster_id": [0]})
    bad = pd.DataFrame({"repeat": [0], "split": ["random"], "condition_id": ["a"], "part": ["test"]})
    with pytest.raises(ContractError, match="GAP"):
        contracts.validate_splits(bad, cond, groups, forbidden_test_lines={"GAP"})


def test_de_set_contract_rejects_conditions_outside_scoring_set():
    de = pd.DataFrame({"condition_id": ["x", "x"], "gene_symbol": ["G0", "G1"], "rank": [1, 2]})
    with pytest.raises(ContractError, match="outside"):
        contracts.validate_de_sets(de, scoring_ids={"a"}, min_de_genes=2)


def test_harness_end_to_end_with_dummy(tmp_path):
    d = _write_eval_dir(tmp_path)
    out = harness.evaluate("dummy", CFG, d, tmp_path / "results")
    per = pd.read_parquet(out / "per_condition.parquet")
    sp = pd.read_parquet(d / "splits.parquet")
    assert len(per) == (sp["part"] == "test").sum()
    assert (per["de_pearson"] == 0).all()
    res = json.loads((out / "results.json").read_text())
    assert res["model"] == "dummy" and res["runtime_s"] >= 0 and res["peak_rss_mb"] > 0
    assert {"de_pearson", "topk_overlap_2", "disc_rank_global"} <= set(res["summaries"])
    assert {r["split"] for r in res["summaries"]["de_pearson"]} == set(splits.SPLITS)


def test_harness_rejects_a_wrong_shape_predictor(tmp_path, monkeypatch):
    class Bad(DummyPredictor):
        def predict(self, conditions):
            return np.zeros((len(conditions), 3), np.float32)

    monkeypatch.setitem(PREDICTORS, "bad", Bad)
    with pytest.raises(ContractError, match="shape"):
        harness.evaluate("bad", CFG, _write_eval_dir(tmp_path), tmp_path / "results")
