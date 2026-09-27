"""The four evaluation splits, drawn `repeats` times with fixed seeds (evaluation.md §4, A1–A2; spec E1–E2).

Held-out units: conditions (`random`), drug similarity clusters with all doses (`unseen_drug`), cell lines
(`unseen_cell_line`), and test lines × test clusters (`both_unseen`, where a condition with only one held-out
side is omitted). Validation is carved from the training side by the same rule as the test part.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ["random", "unseen_drug", "unseen_cell_line", "both_unseen"]


def _draw_units(units: list, rng: np.random.Generator, cfg: dict) -> tuple[set, set]:
    """(test, val) subsets of `units`: test_fraction and val_fraction of the total, at least one each."""
    order = [units[i] for i in rng.permutation(len(units))]
    n_test = max(1, round(cfg["test_fraction"] * len(units)))
    n_val = max(1, round(cfg["val_fraction"] * len(units)))
    return set(order[:n_test]), set(order[n_test:n_test + n_val])


def _assign(keys: pd.Series, test: set, val: set) -> pd.Series:
    return pd.Series(np.where(keys.isin(test), "test", np.where(keys.isin(val), "val", "train")), index=keys.index)


def draw_splits(conditions: pd.DataFrame, drug_groups: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per (repeat, split, condition_id) with part in {train, val, test}; omitted conditions have no row."""
    split_cfg = cfg["splits"]
    cond = conditions[["condition_id", "cell_line", "drug"]].copy()
    cond["cluster"] = cond["drug"].str.strip().map(drug_groups.assign(drug=drug_groups["drug"].str.strip()).set_index("drug")["cluster_id"])
    if cond["cluster"].isna().any():
        missing = sorted(cond.loc[cond["cluster"].isna(), "drug"].unique())
        raise ValueError(f"drugs without a similarity cluster: {missing}")
    lines = sorted(cond["cell_line"].unique())
    clusters = sorted(cond["cluster"].unique())

    frames = []
    for repeat, seed in enumerate(split_cfg["seeds"][: split_cfg["repeats"]]):
        for split_index, split in enumerate(SPLITS):
            rng = np.random.default_rng([seed, split_index])
            if split == "random":
                test, val = _draw_units(sorted(cond["condition_id"]), rng, split_cfg)
                part = _assign(cond["condition_id"], test, val)
            elif split == "unseen_drug":
                test, val = _draw_units(clusters, rng, split_cfg)
                part = _assign(cond["cluster"], test, val)
            elif split == "unseen_cell_line":
                test, val = _draw_units(lines, rng, split_cfg)
                part = _assign(cond["cell_line"], test, val)
            else:
                test_l, val_l = _draw_units(lines, rng, split_cfg)
                test_c, val_c = _draw_units(clusters, rng, split_cfg)
                line_part = _assign(cond["cell_line"], test_l, val_l)
                cluster_part = _assign(cond["cluster"], test_c, val_c)
                part = line_part.where(line_part == cluster_part)  # disagreeing sides are omitted
            frame = pd.DataFrame({"repeat": repeat, "split": split, "condition_id": cond["condition_id"], "part": part})
            frames.append(frame.dropna(subset=["part"]))
    return pd.concat(frames, ignore_index=True)


def split_sizes(splits: pd.DataFrame, conditions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Conditions per part for every (repeat, split), plus distinct test lines and drugs when `conditions` is given."""
    sizes = splits.pivot_table(index=["repeat", "split"], columns="part", values="condition_id", aggfunc="count", fill_value=0)
    sizes = sizes.reindex(columns=["train", "val", "test"], fill_value=0).reset_index()
    sizes.columns.name = None
    if conditions is not None:
        test = splits[splits["part"] == "test"].merge(conditions[["condition_id", "cell_line", "drug"]], on="condition_id")
        counts = test.groupby(["repeat", "split"]).agg(test_lines=("cell_line", "nunique"), test_drugs=("drug", "nunique")).reset_index()
        sizes = sizes.merge(counts, on=["repeat", "split"], how="left").fillna({"test_lines": 0, "test_drugs": 0})
    else:
        sizes["test_lines"] = np.nan
        sizes["test_drugs"] = np.nan
    return sizes
