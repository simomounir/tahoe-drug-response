from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from phase4 import baselines, neural
from phase4.predictors import PREDICTORS, TrainData

KW = {"hidden": 32, "dropouts": [0.0, 0.1], "weight_decays": [1e-4, 1e-2], "lr": 1e-2, "batch_size": 64, "max_epochs": 60,
      "patience": 10, "seed": 7, "device": "cpu"}
N_GENES = 12


def _data(n=400, seed=0):
    """y_g = w_g * z(MolWt) * pc_1 + dose offset: a pure drug x cell interaction on top of a per-dose mean."""
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"cell_line": [f"L{i % 7}" for i in range(n)], "drug": [f"D{i % 11}" for i in range(n)],
                      "log10_dose_um": rng.choice([-1.0, 0.0], n), "morgan_counts": [np.zeros(4, np.uint8)] * n,
                      "MolWt": rng.normal(300, 50, n), "pc_1": rng.normal(size=n), "dmg_TP53": rng.integers(0, 2, n).astype(float)})
    inter = ((f["MolWt"] - 300) / 50 * f["pc_1"]).to_numpy()
    w = rng.normal(size=N_GENES)
    y = (inter[:, None] * w[None, :] + 2.0 * f["log10_dose_um"].to_numpy()[:, None]).astype(np.float32)
    part = np.array(["train"] * int(n * 0.6) + ["val"] * int(n * 0.2) + ["test"] * (n - int(n * 0.6) - int(n * 0.2)))
    return f, y, part


def _row_pearson(p, t):
    p = p - p.mean(axis=1, keepdims=True)
    t = t - t.mean(axis=1, keepdims=True)
    return (p * t).sum(1) / np.sqrt((p * p).sum(1) * (t * t).sum(1) + 1e-12)


def _train_data(f, y, part, val_score=None):
    fit = part != "test"
    truth = y[part == "val"]
    score = val_score or (lambda p: float(np.median(_row_pearson(p, truth))))
    return TrainData(features=f[fit].reset_index(drop=True), targets=y[fit], genes=[f"G{i}" for i in range(N_GENES)], part=part[fit], val_score=score)


def _fit(f, y, part, val_score=None, **kw):
    model = neural.NeuralPredictor(**{**KW, **kw})
    model.fit(_train_data(f, y, part, val_score))
    return model


def test_neural_learns_interaction_ridge_cannot():
    f, y, part = _data()
    test = f[part == "test"].reset_index(drop=True)
    mlp = np.median(_row_pearson(_fit(f, y, part).predict(test), y[part == "test"]))
    ridge = baselines.RidgeBaseline(alpha_log10_start=-2.0, alpha_log10_stop=6.0, alpha_log10_step=0.5)
    ridge.fit(_train_data(f, y, part))
    lin = np.median(_row_pearson(ridge.predict(test), y[part == "test"]))
    assert mlp > 0.8 and mlp > lin + 0.3
    assert PREDICTORS["neural"] is neural.NeuralPredictor and PREDICTORS["neural_nocell"] is neural.NeuralPredictor


def test_neural_zero_output_is_per_dose_mean():
    f, y, part = _data()
    model = _fit(f, y, part, max_epochs=2)
    with torch.no_grad():
        model.model.out.weight.zero_()
        model.model.out.bias.zero_()
    test = f[part == "test"].reset_index(drop=True)
    gm = baselines.GlobalMean()
    gm.fit(_train_data(f, y, part))
    np.testing.assert_allclose(model.predict(test), gm.predict(test), atol=1e-6)


def test_neural_search_picks_best_and_refits_recorded_epochs(monkeypatch):
    f, y, part = _data()
    seen = []

    def fake_train(x, t, dropout, weight_decay, p, device, x_val=None, score=None, epochs=None):
        seen.append((dropout, weight_decay, epochs))
        best = 1.0 if (dropout, weight_decay) == (0.1, 1e-4) else 0.5
        return neural._MLP(x.shape[1], p.hidden, t.shape[1], dropout), 4 if x_val is not None else epochs, best

    monkeypatch.setattr(neural, "_train", fake_train)
    model = _fit(f, y, part)
    assert (model.info()["dropout"], model.info()["weight_decay"], model.info()["epochs"]) == (0.1, 1e-4, 4)
    assert seen[-1] == (0.1, 1e-4, 4)  # refit trains exactly the recorded epoch count, without val
    assert len(seen) == 5


def test_neural_search_tie_rule(monkeypatch):
    f, y, part = _data()
    monkeypatch.setattr(neural, "_train", lambda x, t, dropout, weight_decay, p, device, x_val=None, score=None, epochs=None:
                        (neural._MLP(x.shape[1], p.hidden, t.shape[1], dropout), 3, 0.7))
    info = _fit(f, y, part).info()
    assert (info["weight_decay"], info["dropout"]) == (1e-2, 0.1)  # largest weight decay, then largest dropout


def test_neural_early_stop_at_first_epoch():
    f, y, part = _data()
    scores = iter(range(1000, 0, -1))  # every later epoch scores worse than the first
    model = _fit(f, y, part, val_score=lambda p: float(next(scores)), patience=2)
    assert model.info()["epochs"] == 1
    assert np.isfinite(model.predict(f[part == "test"].reset_index(drop=True))).all()


def test_neural_odd_batch():
    f, y, part = _data(n=325)  # 195 train rows = 3 batches of 64 + 3; train+val 260 = 4 x 64 + 4
    model = _fit(f, y, part, batch_size=64, max_epochs=2)
    model2 = _fit(f, y, part, batch_size=194, max_epochs=2)  # last training batch of size 1
    assert np.isfinite(model.predict(f.head(3))).all() and np.isfinite(model2.predict(f.head(3))).all()


def test_neural_records_device():
    f, y, part = _data()
    assert _fit(f, y, part, max_epochs=1).info()["device"] == "cpu"
    assert neural.pick_device(None).type in {"cpu", "mps"}


def test_neural_is_deterministic_on_cpu():
    f, y, part = _data()
    test = f[part == "test"].reset_index(drop=True)
    np.testing.assert_array_equal(_fit(f, y, part, max_epochs=5).predict(test), _fit(f, y, part, max_epochs=5).predict(test))


def test_neural_nocell_drops_cell_columns():
    f, y, part = _data()
    model = _fit(f, y, part, max_epochs=1, use_cell_features=False)
    assert not [c for c in model.cols if c.startswith(baselines.CELL_PREFIXES)]
    g = f.copy()
    g["pc_1"] = g["pc_1"] + 5.0
    np.testing.assert_array_equal(model.predict(f.head(5)), model.predict(g.head(5)))


def test_neural_all_nan_val_scores_do_not_crash():
    f, y, part = _data()
    model = _fit(f, y, part, val_score=lambda p: float("nan"), max_epochs=3, patience=2)
    assert model.info()["epochs"] == 1
    assert np.isfinite(model.predict(f.head(3))).all()
