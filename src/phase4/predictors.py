"""The model interface every predictor implements (evaluation.md §9; phase 4a spec §4), and the dummy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class TrainData:
    features: pd.DataFrame  # condition_features rows for train and val
    targets: np.ndarray  # [n_rows, n_genes] logFC, same row order as features
    genes: list[str]
    part: np.ndarray  # "train" or "val" per row


class Predictor(Protocol):
    def fit(self, train: TrainData) -> None: ...

    def predict(self, conditions: pd.DataFrame) -> np.ndarray: ...  # [n, n_genes], no NaN


class DummyPredictor:
    """Predicts no change for every condition. Kept as the harness's own test (evaluation.md §2)."""

    def fit(self, train: TrainData) -> None:
        self.n_genes = len(train.genes)

    def predict(self, conditions: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(conditions), self.n_genes), dtype=np.float32)


PREDICTORS: dict[str, type] = {"dummy": DummyPredictor}
