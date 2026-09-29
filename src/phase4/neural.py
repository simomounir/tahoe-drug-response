"""Conditional MLP on ridge's inputs (evaluation.md §8; phase 4c spec N1–N9).

Predicts the residual from the per-dose mean of its fitting rows (N2), so a network that learns nothing gives
`global_mean`. Each fit searches dropout x weight decay on train, scored on val with `TrainData.val_score`
(early stopping on val median de_pearson), then refits on train + val for the chosen config's best epoch count (N5).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from phase4.baselines import GlobalMean, Standardizer, _subset, design_matrix, scalar_columns

if TYPE_CHECKING:
    from phase4.predictors import TrainData


def pick_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class _MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int, n_out: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x))


@dataclass
class _Params:
    hidden: int
    lr: float
    batch_size: int
    max_epochs: int
    patience: int
    seed: int


def _predict(model: _MLP, x: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]).cpu().numpy()


def _train(x: np.ndarray, y: np.ndarray, dropout: float, weight_decay: float, p: _Params, device: torch.device,
           x_val: np.ndarray | None = None, score: Callable[[np.ndarray], float] | None = None,
           epochs: int | None = None) -> tuple[_MLP, int, float]:
    """Train one MLP. With `x_val`: early stopping on `score` (patience, max_epochs), best weights restored.
    Without: exactly `epochs` epochs. Returns (model, best epoch count, best val score or nan)."""
    torch.manual_seed(p.seed)
    rng = np.random.default_rng(p.seed)
    model = _MLP(x.shape[1], p.hidden, y.shape[1], dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p.lr, weight_decay=weight_decay)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device) if x_val is not None else None
    best, best_epoch, best_state, stale = -np.inf, 0, None, 0
    for epoch in range(1, (epochs if xv is None else p.max_epochs) + 1):
        model.train()
        order = torch.as_tensor(rng.permutation(len(xt)), device=device)
        for i in range(0, len(xt), p.batch_size):
            idx = order[i:i + p.batch_size]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        if xv is None:
            continue
        s = float(np.nan_to_num(score(_predict(model, xv, p.batch_size)), nan=-np.inf))  # NaN never wins, as in ridge
        if best_state is None or s > best:
            best, best_epoch, best_state, stale = s, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= p.patience:
                break
    if xv is None:
        return model, epochs, float("nan")
    model.load_state_dict(best_state)
    return model, best_epoch, float(best)


class NeuralPredictor:
    def __init__(self, hidden: int, dropouts: list[float], weight_decays: list[float], lr: float, batch_size: int,
                 max_epochs: int, patience: int, seed: int, use_cell_features: bool = True, device: str | None = None,
                 fixed: dict | None = None):
        self.p = _Params(hidden, lr, batch_size, max_epochs, patience, seed)
        # Tie order (N5): larger weight decay first, then larger dropout; a later config must beat strictly.
        self.configs = sorted(((d, w) for d in dropouts for w in weight_decays), key=lambda c: (-c[1], -c[0]))
        self.use_cell_features = use_cell_features
        self.device = pick_device(device)
        self.fixed = fixed  # {"dropout", "weight_decay", "epochs"}: skip the search, e.g. to refit a recorded choice (phase 5b R6)

    def fit(self, train: TrainData) -> None:
        is_val = np.asarray(train.part) == "val"
        if train.val_score is None or not is_val.any() or is_val.all():
            raise ValueError("neural needs train and val rows and TrainData.val_score for early stopping")
        self.cols = scalar_columns(train.features, self.use_cell_features)
        x = design_matrix(train.features, self.cols)
        y = np.asarray(train.targets, dtype=np.float32)

        if self.fixed is not None:
            self.dropout, self.weight_decay, self.epochs = self.fixed["dropout"], self.fixed["weight_decay"], self.fixed["epochs"]
            self.val_de_pearson = float(self.fixed.get("val_de_pearson", float("nan")))
            self._refit(train, x, y)
            return
        tr = _subset(train, ~is_val)
        base = GlobalMean()
        base.fit(tr)
        std = Standardizer(x[~is_val])
        offset = base.predict(train.features[is_val].reset_index(drop=True))
        resid = y[~is_val] - base.predict(tr.features)
        best = None
        for dropout, wd in self.configs:
            _, epochs, s = _train(std.transform(x[~is_val]), resid, dropout, wd, self.p, self.device,
                                  x_val=std.transform(x[is_val]), score=lambda pv: train.val_score(offset + pv))
            if best is None or s > best[3]:
                best = (dropout, wd, epochs, s)
        self.dropout, self.weight_decay, self.epochs, self.val_de_pearson = best
        self._refit(train, x, y)

    def _refit(self, train: TrainData, x: np.ndarray, y: np.ndarray) -> None:
        self.base = GlobalMean()
        self.base.fit(train)
        self.std = Standardizer(x)
        self.model, _, _ = _train(self.std.transform(x), y - self.base.predict(train.features), self.dropout, self.weight_decay,
                                  self.p, self.device, epochs=self.epochs)

    def predict(self, conditions: pd.DataFrame) -> np.ndarray:
        x = torch.as_tensor(self.std.transform(design_matrix(conditions, self.cols)), dtype=torch.float32, device=self.device)
        return (self.base.predict(conditions) + _predict(self.model, x, self.p.batch_size)).astype(np.float32)

    def info(self) -> dict:
        return {"dropout": self.dropout, "weight_decay": self.weight_decay, "epochs": int(self.epochs),
                "val_de_pearson": float(self.val_de_pearson), "device": self.device.type}
