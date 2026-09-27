"""The model interface every predictor implements (evaluation.md §9; phase 4a spec §4), and the dummy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd


@dataclass
class TrainData:
    features: pd.DataFrame  # condition_features rows for train and val
    targets: np.ndarray  # [n_rows, n_genes] logFC, same row order as features
    genes: list[str]
    part: np.ndarray  # "train" or "val" per row
    val_score: Callable[[np.ndarray], float] | None = None  # median val de_pearson of predictions for the val rows


class Predictor(Protocol):
    def fit(self, train: TrainData) -> None: ...

    def predict(self, conditions: pd.DataFrame) -> np.ndarray: ...  # [n, n_genes], no NaN


class DummyPredictor:
    """Predicts no change for every condition. Kept as the harness's own test (evaluation.md §2)."""

    def fit(self, train: TrainData) -> None:
        self.n_genes = len(train.genes)

    def predict(self, conditions: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(conditions), self.n_genes), dtype=np.float32)


from phase4 import baselines  # noqa: E402  (baselines only type-hints TrainData, so no import cycle)

PREDICTORS: dict[str, type] = {
    "dummy": DummyPredictor,
    "global_mean": baselines.GlobalMean,
    "drug_mean": baselines.DrugMean,
    "cell_mean": baselines.CellMean,
    "nearest_chemical": baselines.NearestChemical,
    "ridge": baselines.RidgeBaseline,
}
