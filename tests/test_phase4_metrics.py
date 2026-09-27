from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase4 import metrics

GENES = ["G0", "G1", "G2", "G3", "G4", "G5"]
CFG = {"metrics": {"topk": [2]}}


def _de(sets: dict) -> pd.DataFrame:
    rows = [{"condition_id": c, "gene_symbol": g, "rank": r + 1} for c, genes in sets.items() for r, g in enumerate(genes)]
    return pd.DataFrame(rows)


TRUTH = np.array(
    [
        [3.0, -2.0, 1.0, 0.0, 0.0, 0.5],
        [-1.0, 2.0, 0.0, 3.0, 0.2, 0.0],
        [0.0, 0.5, -3.0, 1.0, 2.0, 0.0],
    ],
    dtype=np.float32,
)
CONDS = pd.DataFrame({"condition_id": ["a", "b", "c"], "cell_line": ["L1", "L1", "L2"]})
DE = _de({"a": ["G0", "G1", "G2"], "b": ["G3", "G1", "G0"], "c": ["G2", "G4", "G3"]})


def _score(pred):
    return metrics.score_conditions(pred, TRUTH, GENES, DE, CONDS, CFG).set_index("condition_id")


def test_perfect_prediction():
    s = _score(TRUTH.copy())
    assert np.allclose(s["de_pearson"], 1.0)
    assert np.allclose(s["topk_overlap_2"], 1.0)
    assert np.allclose(s["disc_rank_global"], 0.0) and (s["disc_top1_global"] == 1).all()
    assert np.allclose(s["de_mse"], 0.0) and np.allclose(s["sign_accuracy"], 1.0)


def test_sign_flip_gives_minus_one():
    s = _score(-TRUTH)
    assert np.allclose(s["de_pearson"], -1.0)
    assert np.allclose(s["sign_accuracy"], 0.0)


def test_zero_prediction_has_the_fixed_constant_values():
    s = _score(np.zeros_like(TRUTH))
    assert (s["de_pearson"] == 0).all() and (s["all_gene_pearson"] == 0).all()
    assert (s["topk_overlap_2"] == 0).all()
    assert np.allclose(s["disc_rank_global"], 0.5) and (s["disc_top1_global"] == 0).all()


def test_topk_jaccard_hand_computed():
    pred = TRUTH.copy()
    pred[0] = [0.1, 0.0, 5.0, 4.0, 0.0, 0.0]  # predicted top-2 {G2, G3}; true top-2 of "a" = {G0, G1}
    pred[1] = [5.0, 0.0, 0.0, 4.0, 0.0, 0.0]  # {G0, G3} vs {G3, G1} -> 1/3
    s = _score(pred)
    assert s.loc["a", "topk_overlap_2"] == 0.0
    assert s.loc["b", "topk_overlap_2"] == pytest.approx(1 / 3)


def test_within_line_uses_only_same_line_candidates():
    s = _score(TRUTH.copy())
    # "c" is alone in L2: no other candidate, rank undefined
    assert np.isnan(s.loc["c", "disc_rank_within_line"])
    assert s.loc["a", "disc_rank_within_line"] == 0.0


def test_rows_align_with_conditions_order():
    out = metrics.score_conditions(TRUTH.copy(), TRUTH, GENES, DE, CONDS, CFG)
    assert list(out["condition_id"]) == ["a", "b", "c"]
    assert list(out["n_de_genes"]) == [3, 3, 3]


def test_de_pearson_fast_path_matches_score_conditions():
    rng = np.random.default_rng(3)
    genes = [f"G{i}" for i in range(30)]
    conds = pd.DataFrame({"condition_id": [f"c{i}" for i in range(6)], "cell_line": ["A", "A", "B", "B", "C", "C"]})
    de = pd.DataFrame([{"condition_id": c, "gene_symbol": g, "rank": r + 1}
                       for c in conds["condition_id"] for r, g in enumerate(rng.choice(genes, 8, replace=False))])
    pred, truth = rng.normal(size=(6, 30)), rng.normal(size=(6, 30))
    full = metrics.score_conditions(pred, truth, genes, de, conds, {"metrics": {"topk": [2]}})
    idx = metrics.de_index(genes, de, set(conds["condition_id"]))
    np.testing.assert_allclose(metrics.de_pearson(pred, truth, idx, list(conds["condition_id"])), full["de_pearson"])
