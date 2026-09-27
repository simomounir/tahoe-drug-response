"""Phase 4a contracts (spec §6) and phase 4b additions (4b spec §6). A violation raises phase1.contracts.ContractError."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from phase1.contracts import ContractError


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def validate_splits(splits: pd.DataFrame, conditions: pd.DataFrame, drug_groups: pd.DataFrame, forbidden_test_lines: set[str]) -> None:
    """Contracts 1-2: no held-out unit on both sides; forbidden lines never tested."""
    cluster = drug_groups.assign(drug=drug_groups["drug"].str.strip()).set_index("drug")["cluster_id"]
    info = conditions[["condition_id", "cell_line", "drug"]].assign(cluster=lambda d: d["drug"].str.strip().map(cluster))
    m = splits.merge(info, on="condition_id", how="left", validate="many_to_one")
    _check(m["cell_line"].notna().all(), "splits: condition_id not in the scoring set")
    _check(m["condition_id"].groupby([m["repeat"], m["split"]]).apply(lambda s: s.is_unique).all(), "splits: a condition appears twice in one split")
    test = m[m["part"] == "test"]
    bad = sorted(set(test["cell_line"]) & forbidden_test_lines)
    _check(not bad, f"splits: forbidden line(s) in a test part: {bad}")
    units = {"unseen_drug": ["cluster"], "unseen_cell_line": ["cell_line"], "both_unseen": ["cluster", "cell_line"]}
    for (repeat, split), grp in m.groupby(["repeat", "split"]):
        for unit in units.get(split, []):
            tested = set(grp.loc[grp["part"] == "test", unit])
            elsewhere = set(grp.loc[grp["part"] != "test", unit])
            both = sorted(tested & elsewhere)
            _check(not both, f"splits: {split} repeat {repeat}: {unit} on both test and train/val sides: {both[:5]}")


def validate_de_sets(de_sets: pd.DataFrame, scoring_ids: set[str], min_de_genes: int) -> None:
    """Contract 3: DE sets only for scoring-set conditions, each with at least min_de_genes genes."""
    outside = sorted(set(de_sets["condition_id"]) - scoring_ids)
    _check(not outside, f"de_sets: {len(outside)} condition(s) outside the scoring set, e.g. {outside[:3]}")
    small = de_sets.groupby("condition_id").size()
    _check((small >= min_de_genes).all(), f"de_sets: {(small < min_de_genes).sum()} condition(s) below min_de_genes")


def validate_prediction(pred: np.ndarray, n_rows: int, n_genes: int) -> None:
    """Contract 4: exactly (test conditions x gene universe), finite."""
    _check(pred.shape == (n_rows, n_genes), f"prediction: shape {pred.shape}, expected {(n_rows, n_genes)}")
    _check(bool(np.isfinite(pred).all()), "prediction: NaN or inf values")



def validate_model_info(info: dict) -> None:
    """Phase 4b spec §6.2: model info is JSON-serialisable; fallback_rate in [0, 1]."""
    try:
        json.dumps(info)
    except TypeError as exc:
        raise ContractError(f"model_info: not JSON-serialisable ({exc})") from exc
    if "fallback_rate" in info:
        _check(0.0 <= info["fallback_rate"] <= 1.0, f"model_info: fallback_rate {info['fallback_rate']} outside [0, 1]")


def validate_targets(targets: np.ndarray, conditions: pd.DataFrame, genes: pd.DataFrame) -> None:
    """Contract 5: target matrix matches its row (conditions) and column (genes) index files."""
    _check(targets.shape == (len(conditions), len(genes)), f"targets: shape {targets.shape} does not match index files {(len(conditions), len(genes))}")
    _check(conditions["condition_id"].is_unique and genes["gene_symbol"].is_unique, "targets: duplicate condition_id or gene_symbol in index files")
