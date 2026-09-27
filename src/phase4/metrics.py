"""Per-condition metrics (evaluation.md §5; phase 4a spec E6–E7).

Primary: `de_pearson`, `topk_overlap_<k>`, discrimination (normalized rank and top-1, across all test conditions
and within cell line). Secondary: `de_mse`, `sign_accuracy`, `all_gene_pearson`.

Constant predictions: correlations count as 0 and a top-k set counts as overlap 0 (no information); in
discrimination every candidate then ties, which gives the average rank 0.5 (chance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64) - x.mean()
    y = y.astype(np.float64) - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else 0.0


def _row_standardize(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float64)
    m = m - m.mean(axis=1, keepdims=True)
    norm = np.sqrt((m * m).sum(axis=1, keepdims=True))
    return np.divide(m, norm, out=np.zeros_like(m), where=norm > 0)


def _normalized_rank(dist: np.ndarray, own: int, candidates: np.ndarray) -> tuple[float, float]:
    """(normalized rank, top-1) of `own` among `candidates` by distance; ties count half. NaN with no other candidate."""
    others = candidates[candidates != own]
    if len(others) == 0:
        return float("nan"), float("nan")
    d_own = dist[own]
    closer = (dist[others] < d_own).sum()
    ties = (dist[others] == d_own).sum()
    rank = (closer + 0.5 * ties) / len(others)
    return float(rank), float(closer == 0 and ties == 0)


def score_conditions(
    pred: np.ndarray, truth: np.ndarray, genes: list[str], de_sets: pd.DataFrame, conditions: pd.DataFrame, cfg: dict
) -> pd.DataFrame:
    """One row per condition (in `conditions` order); `pred` and `truth` rows follow the same order."""
    col = {g: i for i, g in enumerate(genes)}
    de_sorted = de_sets.sort_values(["condition_id", "rank"])
    de_idx = {
        cid: np.array([col[g] for g in grp["gene_symbol"] if g in col], dtype=np.int64)
        for cid, grp in de_sorted.groupby("condition_id", sort=False)
    }
    ids = list(conditions["condition_id"])
    topk = cfg["metrics"]["topk"]

    # Discrimination: Pearson distance over genes that are DE in at least one of these conditions.
    union = np.unique(np.concatenate([de_idx.get(c, np.array([], dtype=np.int64)) for c in ids]))
    corr = _row_standardize(pred[:, union]) @ _row_standardize(truth[:, union]).T  # [pred i, truth j]
    lines = conditions["cell_line"].to_numpy()
    everyone = np.arange(len(ids))

    rows = []
    for i, cid in enumerate(ids):
        de = de_idx.get(cid, np.array([], dtype=np.int64))
        p, t = pred[i], truth[i]
        row = {"condition_id": cid, "n_de_genes": len(de)}
        row["de_pearson"] = _pearson(p[de], t[de]) if len(de) > 1 else float("nan")
        row["de_mse"] = float(np.mean((p[de].astype(np.float64) - t[de]) ** 2)) if len(de) else float("nan")
        row["sign_accuracy"] = float(np.mean(np.sign(p[de]) == np.sign(t[de]))) if len(de) else float("nan")
        row["all_gene_pearson"] = _pearson(p, t)
        abs_p = np.abs(p)
        constant = np.ptp(abs_p) == 0
        for k in topk:
            m = min(k, len(de))
            if m == 0 or constant:
                row[f"topk_overlap_{k}"] = 0.0 if m else float("nan")
                continue
            true_top = set(de[:m].tolist())
            pred_top = set(np.argsort(-abs_p, kind="stable")[:m].tolist())
            row[f"topk_overlap_{k}"] = len(true_top & pred_top) / len(true_top | pred_top)
        dist = 1.0 - corr[i]
        row["disc_rank_global"], row["disc_top1_global"] = _normalized_rank(dist, i, everyone)
        row["disc_rank_within_line"], row["disc_top1_within_line"] = _normalized_rank(dist, i, everyone[lines == lines[i]])
        rows.append(row)
    return pd.DataFrame(rows)
