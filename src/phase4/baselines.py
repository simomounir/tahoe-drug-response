"""Baseline ladder (evaluation.md §6; phase 4b spec §4).

Every baseline keys on dose (B1): a prediction for dose d uses training conditions at dose d only. "Training rows"
are train + val (B3). A baseline with nothing to look up falls back one level and counts it; `info()["fallback_rate"]`
is the share of the last `predict` call's rows that fell back (B2).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from phase3.drugs import DESCRIPTOR_NAMES
from phase3.groups import tanimoto_matrix

if TYPE_CHECKING:
    from phase4.predictors import TrainData

DOSE = "log10_dose_um"


def _keys(frame: pd.DataFrame, cols: list[str]) -> list[tuple]:
    """One tuple per row; the dose is rounded so float keys from different tables compare equal."""
    return list(zip(*[frame[c].round(6).to_numpy() if c == DOSE else frame[c].to_numpy() for c in cols]))


def _group_means(keys: list[tuple], targets: np.ndarray) -> dict[tuple, np.ndarray]:
    rows: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        rows.setdefault(k, []).append(i)
    return {k: np.asarray(targets[idx]).mean(axis=0, dtype=np.float64).astype(np.float32) for k, idx in rows.items()}


class _MeanBaseline:
    """Mean logFC of the training conditions sharing `keys` and the dose; falls back to the dose's global mean."""

    keys: tuple[str, ...] = ()

    def fit(self, train: TrainData) -> None:
        self.n_genes = len(train.genes)
        self.global_ = _group_means(_keys(train.features, [DOSE]), train.targets)
        self.means_ = _group_means(_keys(train.features, [*self.keys, DOSE]), train.targets) if self.keys else self.global_
        self.fallback_rate = 0.0

    def _global(self, dose: float) -> np.ndarray:
        if (dose,) not in self.global_:
            raise ValueError(f"no training condition at {DOSE}={dose}")
        return self.global_[(dose,)]

    def predict(self, conditions: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(conditions), self.n_genes), np.float32)
        fell = 0
        for i, key in enumerate(_keys(conditions, [*self.keys, DOSE])):
            if key in self.means_:
                out[i] = self.means_[key]
            else:
                out[i] = self._global(key[-1])
                fell += 1
        self.fallback_rate = fell / len(conditions) if len(conditions) else 0.0
        return out

    def info(self) -> dict:
        return {"fallback_rate": self.fallback_rate}


class GlobalMean(_MeanBaseline):
    keys = ()


class DrugMean(_MeanBaseline):
    keys = ("drug",)


class CellMean(_MeanBaseline):
    keys = ("cell_line",)


def _fingerprints(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    first = frame.drop_duplicates("drug")
    return dict(zip(first["drug"], (np.asarray(c) for c in first["morgan_counts"])))


class NearestChemical:
    """Response of the most similar training drug (count-Tanimoto on Morgan counts, amendment A4) in the same line
    and dose; ties averaged. With no training row in that line and dose, the nearest drug's mean across training
    lines at the dose (counted as a fallback). A test drug may be absent from training, so similarity is computed
    in `predict`."""

    def fit(self, train: TrainData) -> None:
        f = train.features
        self.n_genes, self.targets, self.fp = len(train.genes), np.asarray(train.targets), _fingerprints(f)
        self.in_line: dict[tuple, dict[str, list[int]]] = {}  # (line, dose) -> {drug: rows}
        for i, (line, drug, dose) in enumerate(_keys(f, ["cell_line", "drug", DOSE])):
            self.in_line.setdefault((line, dose), {}).setdefault(drug, []).append(i)
        self.across: dict[float, dict[str, np.ndarray]] = {}  # dose -> {drug: mean over training lines}
        for (drug, dose), v in _group_means(_keys(f, ["drug", DOSE]), self.targets).items():
            self.across.setdefault(dose, {})[drug] = v
        self.fallback_rate = 0.0

    def predict(self, conditions: pd.DataFrame) -> np.ndarray:
        test_fp = _fingerprints(conditions)
        names = sorted(set(self.fp) | set(test_fp))
        sim = pd.DataFrame(tanimoto_matrix(np.stack([test_fp.get(n, self.fp.get(n)) for n in names])), index=names, columns=names)
        out = np.zeros((len(conditions), self.n_genes), np.float32)
        fell = 0
        for i, (line, drug, dose) in enumerate(_keys(conditions, ["cell_line", "drug", DOSE])):
            in_line = self.in_line.get((line, dose))
            if in_line:
                s = sim.loc[drug, list(in_line)]
                tied = s.index[s == s.max()]
                out[i] = np.mean([self.targets[in_line[d]].mean(axis=0) for d in tied], axis=0)
            else:
                if dose not in self.across:
                    raise ValueError(f"no training condition at {DOSE}={dose}")
                s = sim.loc[drug, list(self.across[dose])]
                out[i] = np.mean([self.across[dose][d] for d in s.index[s == s.max()]], axis=0)
                fell += 1
        self.fallback_rate = fell / len(conditions) if len(conditions) else 0.0
        return out

    def info(self) -> dict:
        return {"fallback_rate": self.fallback_rate}


CELL_PREFIXES = ("pc_", "dmg_", "hot_", "drv_")  # expression PCs, damaging / hotspot mutation flags, Tahoe driver flags


def scalar_columns(features: pd.DataFrame, use_cell_features: bool = True) -> list[str]:
    """Non-fingerprint inputs: RDKit descriptors, dose, and (unless ablated, phase 4c N6) the cell-line features."""
    desc = [c for c in DESCRIPTOR_NAMES if c in features.columns]
    cells = [c for c in features.columns if c.startswith(CELL_PREFIXES)] if use_cell_features else []
    return [*desc, DOSE, *cells]


# Continuous descriptors spanning many orders of magnitude, log1p-transformed (spec amendment B7): Ipc runs from
# 8.6e2 to 2.1e14 over the 92 drugs, so raw it acts as a flag for 5 drugs and extrapolates wildly on held-out ones.
LOG_DESCRIPTORS = ("Ipc",)


def design_matrix(features: pd.DataFrame, cols: list[str]) -> np.ndarray:
    fp = np.log1p(np.stack(features["morgan_counts"].to_numpy()).astype(np.float64))
    scalars = features[cols].to_numpy(dtype=np.float64, copy=True)
    for j, c in enumerate(cols):
        if c in LOG_DESCRIPTORS:
            scalars[:, j] = np.log1p(scalars[:, j])
    return np.hstack([fp, scalars])


class Standardizer:
    """Mean/SD of the rows it is built on; columns with zero SD there are dropped (they carry nothing to fit)."""

    def __init__(self, x: np.ndarray):
        mu, sd = x.mean(axis=0), x.std(axis=0)
        self.keep = sd > 1e-12
        self.mu, self.sd = mu[self.keep], sd[self.keep]

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x[:, self.keep] - self.mu) / self.sd


class _RidgePath:
    """Every ridge solution from one SVD of the standardised X (spec B4); zero-SD columns are dropped.

    X is centred, so Uᵀ·1 = 0 for every s > 0 and Uᵀ Y equals Uᵀ (Y − ȳ) without a centred copy of Y.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.std = Standardizer(x)
        u, self.s, vt = np.linalg.svd(self.std.transform(x), full_matrices=False)
        self.v = vt.T
        self.y_mean = y.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.uty = u.T.astype(np.float32) @ y

    def predict(self, x: np.ndarray, alpha: float) -> np.ndarray:
        a = (self.std.transform(x) @ self.v) * (self.s / (self.s ** 2 + alpha))
        return a.astype(np.float32) @ self.uty + self.y_mean


def _subset(train: TrainData, mask: np.ndarray) -> TrainData:
    return type(train)(features=train.features[mask].reset_index(drop=True), targets=np.asarray(train.targets)[mask], genes=train.genes,
                       part=np.asarray(train.part)[mask])


class RidgeBaseline:
    """Ridge from standardised [log1p Morgan counts ⊕ descriptors ⊕ dose ⊕ cell features] to all genes.

    Fits the residual from the per-dose mean of the fitting rows and adds that mean back (amendment B8), so full
    shrinkage gives exactly `global_mean` instead of the all-dose mean. α from the config grid by validation median
    de_pearson (ties to the larger α), then refit on train + val (B5).
    """

    def __init__(self, alpha_log10_start: float, alpha_log10_stop: float, alpha_log10_step: float, use_cell_features: bool = True):
        self.use_cell_features = use_cell_features
        self.alphas = 10.0 ** np.arange(alpha_log10_start, alpha_log10_stop + alpha_log10_step / 2, alpha_log10_step)

    def fit(self, train: TrainData) -> None:
        is_val = np.asarray(train.part) == "val"
        if train.val_score is None or not is_val.any() or is_val.all():
            raise ValueError("ridge needs train and val rows and TrainData.val_score to choose alpha")
        self.cols = scalar_columns(train.features, self.use_cell_features)
        x = design_matrix(train.features, self.cols)
        y = np.asarray(train.targets, dtype=np.float32)
        fit_rows, val_rows = train.features[~is_val].reset_index(drop=True), train.features[is_val].reset_index(drop=True)
        base = GlobalMean()
        base.fit(_subset(train, ~is_val))
        path = _RidgePath(x[~is_val], y[~is_val] - base.predict(fit_rows))
        offset = base.predict(val_rows)
        scores = np.nan_to_num([train.val_score(offset + path.predict(x[is_val], a)) for a in self.alphas], nan=-np.inf)
        best = max(range(len(scores)), key=lambda i: (scores[i], i))
        self.alpha, self.val_de_pearson = float(self.alphas[best]), float(scores[best])
        del path, offset
        self.base = GlobalMean()
        self.base.fit(train)
        self.path = _RidgePath(x, y - self.base.predict(train.features))

    def predict(self, conditions: pd.DataFrame) -> np.ndarray:
        return self.base.predict(conditions) + self.path.predict(design_matrix(conditions, self.cols), self.alpha)

    def info(self) -> dict:
        return {"alpha": self.alpha, "val_de_pearson": self.val_de_pearson}
