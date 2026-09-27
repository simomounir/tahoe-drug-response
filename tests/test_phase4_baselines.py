from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase4 import baselines
from phase4.predictors import PREDICTORS, TrainData

GENES = ["A", "B"]
DOSES = [-1.0, 0.0]  # log10 µM
FP = {"D0": [2, 1, 0, 0, 0, 0, 0, 0], "D1": [2, 1, 1, 0, 0, 0, 0, 0], "D2": [0, 0, 0, 0, 3, 3, 0, 0]}


def _frame():
    """3 lines x 3 drugs x 2 doses; target A = 100*line + 10*drug + dose_idx, B = -line."""
    rows = [{"condition_id": f"L{l}-D{d}-{k}", "cell_line": f"L{l}", "drug": f"D{d}", "log10_dose_um": dose,
             "morgan_counts": np.array(FP[f"D{d}"], np.uint8), "MolWt": 100.0 + 50 * d, "pc_1": float(l), "dmg_TP53": float(l == 2)}
            for l in range(3) for d in range(3) for k, dose in enumerate(DOSES)]
    f = pd.DataFrame(rows)
    line = f["cell_line"].str[1].astype(int)
    drug = f["drug"].str[1].astype(int)
    dose_idx = f["log10_dose_um"].map({DOSES[0]: 0, DOSES[1]: 1})
    y = np.stack([100 * line + 10 * drug + dose_idx, -line], axis=1).astype(np.float32)
    return f, y


def _train(frame, targets, part=None, val_score=None):
    part = np.array(["train"] * len(frame)) if part is None else part
    return TrainData(features=frame.reset_index(drop=True), targets=targets, genes=GENES, part=part, val_score=val_score)


def _split(mask):
    f, y = _frame()
    return f[~mask].reset_index(drop=True), y[~mask.to_numpy()], f[mask].reset_index(drop=True), y[mask.to_numpy()]


def _fit(cls, frame, targets):
    model = cls()
    model.fit(_train(frame, targets))
    return model


def test_mean_baselines_are_per_dose_means():
    f, _ = _frame()
    ftr, ytr, fte, _ = _split((f["cell_line"] == "L0") & (f["drug"] == "D0"))
    test = fte.sort_values("log10_dose_um").reset_index(drop=True)  # dose -1 then 0
    g = _fit(baselines.GlobalMean, ftr, ytr).predict(test)
    np.testing.assert_allclose(g[0], [990 / 8, -9 / 8], rtol=1e-6)  # 8 training rows at dose -1
    np.testing.assert_allclose(g[1], [998 / 8, -9 / 8], rtol=1e-6)
    np.testing.assert_allclose(_fit(baselines.DrugMean, ftr, ytr).predict(test), [[150, -1.5], [151, -1.5]])
    np.testing.assert_allclose(_fit(baselines.CellMean, ftr, ytr).predict(test), [[15, 0], [16, 0]])
    assert PREDICTORS["global_mean"] is baselines.GlobalMean
    assert PREDICTORS["drug_mean"] is baselines.DrugMean and PREDICTORS["cell_mean"] is baselines.CellMean


@pytest.mark.parametrize("cls,col,unseen", [(baselines.DrugMean, "drug", "D2"), (baselines.CellMean, "cell_line", "L2")])
def test_mean_falls_back_to_global_for_unseen_unit(cls, col, unseen):
    f, y = _frame()
    ftr, ytr = f[f[col] != unseen].reset_index(drop=True), y[(f[col] != unseen).to_numpy()]
    other = "cell_line" if col == "drug" else "drug"
    test = f[(f[other] == f[other].iloc[0]) & (f["log10_dose_um"] == -1.0)].reset_index(drop=True)
    model = _fit(cls, ftr, ytr)
    pred = model.predict(test)
    assert model.info() == {"fallback_rate": pytest.approx(1 / 3)}
    fell = test[col] == unseen
    np.testing.assert_allclose(pred[fell.to_numpy()][0], ytr[(ftr["log10_dose_um"] == -1.0).to_numpy()].mean(axis=0), rtol=1e-6)


def test_mean_unknown_dose_raises():
    f, y = _frame()
    low = (f["log10_dose_um"] == -1.0).to_numpy()
    model = _fit(baselines.GlobalMean, f[low], y[low])
    with pytest.raises(ValueError, match="log10_dose_um"):
        model.predict(f[~low])


def _one(f, line, drug, dose=-1.0):
    return f[(f["cell_line"] == line) & (f["drug"] == drug) & (f["log10_dose_um"] == dose)].reset_index(drop=True)


def test_nearest_picks_most_similar_drug_in_same_line_and_dose():
    f, _ = _frame()
    ftr, ytr, _, _ = _split((f["cell_line"] == "L0") & (f["drug"] == "D0") & (f["log10_dose_um"] == -1.0))
    model = _fit(baselines.NearestChemical, ftr, ytr)
    # in (L0, dose -1) the candidates are D1 (Tanimoto 3/4 to D0) and D2 (0) -> D1's response there
    np.testing.assert_allclose(model.predict(_one(f, "L0", "D0")), [[10, 0]])
    assert model.info() == {"fallback_rate": 0.0}
    assert PREDICTORS["nearest_chemical"] is baselines.NearestChemical


def test_nearest_averages_tied_drugs():
    f, _ = _frame()
    ftr, ytr, _, _ = _split((f["cell_line"] == "L0") & (f["drug"] == "D2"))
    # D2 shares no bits with D0 or D1: both at similarity 0 -> mean of (L0, D0) and (L0, D1) at dose -1
    np.testing.assert_allclose(_fit(baselines.NearestChemical, ftr, ytr).predict(_one(f, "L0", "D2")), [[5, 0]])


def test_nearest_unseen_line_uses_nearest_drug_mean_across_lines():
    f, _ = _frame()
    ftr, ytr, _, _ = _split(f["cell_line"] == "L0")
    model = _fit(baselines.NearestChemical, ftr, ytr)
    # no training row in L0: nearest drug at dose -1 is D0 itself (similarity 1), mean over L1, L2
    np.testing.assert_allclose(model.predict(_one(f, "L0", "D0")), [[150, -1.5]])
    assert model.info() == {"fallback_rate": 1.0}


def test_nearest_all_zero_fingerprint():
    f, y = _frame()
    f["morgan_counts"] = [np.zeros(8, np.uint8) if d == "D2" else c for d, c in zip(f["drug"], f["morgan_counts"])]
    mask = ((f["cell_line"] == "L0") & (f["drug"] == "D2")).to_numpy()
    model = _fit(baselines.NearestChemical, f[~mask].reset_index(drop=True), y[~mask])
    pred = model.predict(_one(f, "L0", "D2"))
    assert np.isfinite(pred).all()
    np.testing.assert_allclose(pred, [[5, 0]])  # similarity 0 to every drug -> all candidates tie


RIDGE_KW = {"alpha_log10_start": -2.0, "alpha_log10_stop": 6.0, "alpha_log10_step": 0.5}


def _ridge_data(noise: float, n: int = 70, seed: int = 0):
    """40 Morgan slots and 40 genes with 40 training rows: enough dimensions for noise to reward a larger alpha."""
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({
        "cell_line": [f"L{i % 5}" for i in range(n)], "drug": [f"D{i % 9}" for i in range(n)],
        "log10_dose_um": rng.choice(DOSES, n), "morgan_counts": list(rng.integers(0, 4, size=(n, 40)).astype(np.uint8)),
        "MolWt": rng.normal(300, 50, n), "pc_1": rng.normal(size=n), "dmg_TP53": rng.integers(0, 2, n).astype(float),
    })
    x = baselines.design_matrix(f, baselines.scalar_columns(f))
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = (x @ rng.normal(size=(x.shape[1], 40)) + noise * rng.normal(size=(n, 40))).astype(np.float32)
    part = np.array(["train"] * 40 + ["val"] * 15 + ["test"] * (n - 55))
    return f, y, part


def _row_pearson(p, t):
    return np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(p, t)])


def _fit_ridge(f, y, part):
    fit = part != "test"
    val_truth = y[part == "val"]
    model = baselines.RidgeBaseline(**RIDGE_KW)
    model.fit(TrainData(features=f[fit].reset_index(drop=True), targets=y[fit], genes=[f"G{i}" for i in range(40)], part=part[fit],
                        val_score=lambda p: float(np.median(_row_pearson(p, val_truth)))))
    return model, model.predict(f[part == "test"].reset_index(drop=True))


def test_ridge_recovers_linear_map_without_noise():
    f, y, part = _ridge_data(noise=0.0)
    model, pred = _fit_ridge(f, y, part)
    assert _row_pearson(pred, y[part == "test"]).min() > 0.99
    assert set(model.info()) == {"alpha", "val_de_pearson"}
    assert PREDICTORS["ridge"] is baselines.RidgeBaseline


def test_ridge_picks_smaller_alpha_without_noise():
    clean, _ = _fit_ridge(*_ridge_data(noise=0.0))
    noisy, _ = _fit_ridge(*_ridge_data(noise=5.0))
    assert clean.alpha < noisy.alpha


def test_ridge_is_deterministic():
    data = _ridge_data(noise=1.0)
    np.testing.assert_array_equal(_fit_ridge(*data)[1], _fit_ridge(*data)[1])


def test_ridge_constant_column():
    f, y, part = _ridge_data(noise=1.0)
    f.loc[part != "test", "dmg_TP53"] = 0.0
    f.loc[part == "test", "dmg_TP53"] = 1.0
    _, pred = _fit_ridge(f, y, part)
    assert pred.dtype == np.float32 and np.isfinite(pred).all()


@pytest.mark.parametrize("all_train,with_score", [(True, True), (False, False)])
def test_ridge_requires_val(all_train, with_score):
    f, y, part = _ridge_data(noise=1.0)
    part = np.where(all_train, "train", part)
    model = baselines.RidgeBaseline(**RIDGE_KW)
    with pytest.raises(ValueError, match="val"):
        model.fit(TrainData(features=f, targets=y, genes=[f"G{i}" for i in range(40)], part=part, val_score=(lambda p: 0.0) if with_score else None))


def test_design_matrix_log_transforms_ipc():
    f = pd.DataFrame({"morgan_counts": [np.zeros(4, np.uint8)] * 2, "Ipc": [1e3, 2e14], "log10_dose_um": [0.0, 0.0]})
    x = baselines.design_matrix(f, baselines.scalar_columns(f))
    np.testing.assert_allclose(x[:, 4], np.log1p([1e3, 2e14]))  # Ipc spans 11 orders of magnitude on the real drugs


def test_ridge_with_infinite_alpha_is_the_per_dose_mean():
    f, y, part = _ridge_data(noise=1.0)
    y = y + np.where(f["log10_dose_um"].to_numpy()[:, None] == 0.0, 5.0, -5.0).astype(np.float32)  # strong dose effect
    fit = part != "test"
    model = baselines.RidgeBaseline(alpha_log10_start=12.0, alpha_log10_stop=12.0, alpha_log10_step=1.0)
    model.fit(TrainData(features=f[fit].reset_index(drop=True), targets=y[fit], genes=[f"G{i}" for i in range(40)], part=part[fit],
                        val_score=lambda p: 0.0))
    test = f[part == "test"].reset_index(drop=True)
    gm = baselines.GlobalMean()
    gm.fit(TrainData(features=f[fit].reset_index(drop=True), targets=y[fit], genes=[f"G{i}" for i in range(40)], part=part[fit]))
    np.testing.assert_allclose(model.predict(test), gm.predict(test), atol=1e-3)
