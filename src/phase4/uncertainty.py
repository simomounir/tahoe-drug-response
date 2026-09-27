"""Repeat-aware bootstrap intervals (evaluation.md §7; phase 4a spec E8).

Headline per split = mean over repeats of the per-repeat median. The interval resamples conditions *within*
each repeat, takes each repeat's median, and averages over repeats, so a condition tested in several repeats
is never counted as independent evidence. Paired comparisons do the same on per-condition differences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bootstrap_mean_of_medians(values_by_repeat: list[np.ndarray], cfg: dict) -> tuple[float, float, float]:
    boot = cfg["bootstrap"]
    rng = np.random.default_rng(boot["seed"])
    estimate = float(np.mean([np.median(v) for v in values_by_repeat]))
    draws = np.zeros(boot["resamples"])
    for v in values_by_repeat:
        idx = rng.integers(0, len(v), size=(boot["resamples"], len(v)))
        draws += np.median(v[idx], axis=1)
    draws /= len(values_by_repeat)
    alpha = (1 - boot["interval"]) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    return estimate, float(low), float(high)


def _summarize_column(per: pd.DataFrame, column: str, cfg: dict) -> pd.DataFrame:
    rows = []
    for split, grp in per.groupby("split", sort=True):
        scored = grp.dropna(subset=[column])
        by_repeat = [g[column].to_numpy(dtype=np.float64) for _, g in scored.groupby("repeat", sort=True) if len(g)]
        row = {"split": split, "n_scored": len(scored), "n_nan": int(grp[column].isna().sum())}
        if by_repeat:
            row["estimate"], row["ci_low"], row["ci_high"] = _bootstrap_mean_of_medians(by_repeat, cfg)
            row["iqr_low"], row["iqr_high"] = (float(q) for q in np.quantile(scored[column], [0.25, 0.75]))
            row["repeat_medians"] = [float(np.median(v)) for v in by_repeat]
        else:
            row.update(estimate=np.nan, ci_low=np.nan, ci_high=np.nan, iqr_low=np.nan, iqr_high=np.nan, repeat_medians=[])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(per_condition: pd.DataFrame, metric: str, cfg: dict) -> pd.DataFrame:
    """One row per split: estimate, ci_low, ci_high, iqr_low, iqr_high, repeat_medians, n_scored, n_nan."""
    out = _summarize_column(per_condition[["repeat", "split", "condition_id", metric]], metric, cfg)
    out.insert(1, "metric", metric)
    return out


def paired(per_a: pd.DataFrame, per_b: pd.DataFrame, metric: str, cfg: dict) -> pd.DataFrame:
    """Model A minus model B on the same (repeat, split, condition); `no_detectable_difference` if the interval contains 0."""
    keys = ["repeat", "split", "condition_id"]
    merged = per_a[keys + [metric]].merge(per_b[keys + [metric]], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    merged["diff"] = merged[f"{metric}_a"] - merged[f"{metric}_b"]
    out = _summarize_column(merged, "diff", cfg)
    out.insert(1, "metric", metric)
    out["no_detectable_difference"] = (out["ci_low"] <= 0) & (out["ci_high"] >= 0)
    return out
