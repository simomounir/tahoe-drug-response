"""The case study's inspected failure case (evaluation.md §10.4; phase 5a spec C3, C4).

Rule, fixed before looking at any model's score on it: among the test conditions of repeat 0 `both_unseen`, the one whose
measured logFC is farthest (Euclidean, over its DE genes) from the per-dose mean of that fit's training rows; ties go to
the smallest condition_id. Predictions come from refitting each model on exactly that fit's inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from phase4.baselines import GlobalMean
from phase4.harness import fit_inputs
from phase4.metrics import de_index
from phase4.predictors import PREDICTORS


def distances(data: dict, repeat: int = 0, split: str = "both_unseen") -> pd.Series:
    """Per test condition of (repeat, split): Euclidean distance over its DE genes between measured logFC and the per-dose
    mean of that fit's training rows (train + val). Conditions without a DE set get -1."""
    train, test_rows = fit_inputs(data, repeat, split)
    base = GlobalMean()
    base.fit(train)
    test = data["conditions"].iloc[test_rows].reset_index(drop=True)
    mean = base.predict(test)
    idx = de_index(list(data["genes"]["gene_symbol"]), data["de_sets"], set(test["condition_id"]))
    truth = np.asarray(data["targets"][test_rows])
    ids = list(test["condition_id"])
    dist = [float(np.linalg.norm(truth[i, idx[c]] - mean[i, idx[c]])) if c in idx else -1.0 for i, c in enumerate(ids)]
    return pd.Series(dist, index=ids, name="distance")


def select_condition(data: dict, repeat: int = 0, split: str = "both_unseen") -> str:
    d = distances(data, repeat, split)
    return min(d.index, key=lambda c: (-d[c], c))


def predictions_for(condition_id: str, models: list[str], cfg: dict, data: dict, repeat: int = 0,
                    split: str = "both_unseen") -> dict[str, np.ndarray]:
    train, test_rows = fit_inputs(data, repeat, split)
    test = data["conditions"].iloc[test_rows].reset_index(drop=True)
    if condition_id not in set(test["condition_id"]):
        raise ValueError(f"{condition_id} is not a test condition of repeat {repeat} {split}")
    row = test[test["condition_id"] == condition_id].reset_index(drop=True)
    out = {}
    for name in models:
        model = PREDICTORS[name](**cfg.get("models", {}).get(name, {}))
        model.fit(train)
        out[name] = model.predict(row)[0]
    return out
