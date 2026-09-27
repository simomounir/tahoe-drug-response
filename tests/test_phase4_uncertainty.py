from __future__ import annotations

import numpy as np
import pandas as pd

from phase4 import uncertainty

CFG = {"bootstrap": {"resamples": 500, "interval": 0.95, "seed": 1}}


def _per_condition(values_by_repeat: dict, split="unseen_drug", metric="de_pearson") -> pd.DataFrame:
    rows = [
        {"repeat": r, "split": split, "condition_id": f"c{i}", metric: v}
        for r, values in values_by_repeat.items()
        for i, v in enumerate(values)
    ]
    return pd.DataFrame(rows)


def test_headline_is_mean_of_repeat_medians_with_interval():
    rng = np.random.default_rng(0)
    per = _per_condition({r: rng.normal(0.3, 0.1, 200) for r in range(5)})
    s = uncertainty.summarize(per, "de_pearson", CFG).iloc[0]
    assert abs(s["estimate"] - 0.3) < 0.03
    assert s["ci_low"] < s["estimate"] < s["ci_high"]
    assert len(s["repeat_medians"]) == 5


def test_resampling_stays_within_each_repeat():
    # repeat 0 all 0.0, repeat 1 all 1.0: within-repeat resampling can only give 0.5
    per = _per_condition({0: [0.0] * 50, 1: [1.0] * 50})
    s = uncertainty.summarize(per, "de_pearson", CFG).iloc[0]
    assert s["estimate"] == 0.5 and s["ci_low"] == 0.5 and s["ci_high"] == 0.5


def test_paired_recovers_a_known_difference():
    rng = np.random.default_rng(1)
    base = {r: rng.normal(0.2, 0.2, 300) for r in range(5)}
    a = _per_condition({r: v + 0.1 for r, v in base.items()})
    b = _per_condition(base)
    d = uncertainty.paired(a, b, "de_pearson", CFG).iloc[0]
    assert abs(d["estimate"] - 0.1) < 1e-9
    assert d["ci_low"] > 0 and not d["no_detectable_difference"]


def test_paired_identical_models_contain_zero():
    rng = np.random.default_rng(2)
    base = _per_condition({r: rng.normal(0.2, 0.2, 100) for r in range(3)})
    d = uncertainty.paired(base, base.copy(), "de_pearson", CFG).iloc[0]
    assert d["ci_low"] <= 0 <= d["ci_high"] and d["no_detectable_difference"]


def test_nan_scores_are_dropped_and_counted():
    per = _per_condition({0: [0.1, np.nan, 0.3]})
    s = uncertainty.summarize(per, "de_pearson", CFG).iloc[0]
    assert s["n_scored"] == 2 and s["n_nan"] == 1


def test_deterministic():
    rng = np.random.default_rng(3)
    per = _per_condition({r: rng.normal(size=50) for r in range(2)})
    pd.testing.assert_frame_equal(uncertainty.summarize(per, "de_pearson", CFG), uncertainty.summarize(per, "de_pearson", CFG))
