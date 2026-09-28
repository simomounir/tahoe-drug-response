"""Fit, predict and score one model on every repeat x split (phase 4a spec §5 step 2).

Reads an eval directory written by scripts/build_eval_data.py: conditions.parquet (scoring set with features, in
target row order), genes.parquet (target column order), targets.npy, de_sets.parquet, splits.parquet.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from phase1.build import peak_rss_mb
from phase4 import contracts, uncertainty
from phase4.metrics import de_index, de_pearson, score_conditions
from phase4.predictors import PREDICTORS, TrainData

NON_METRIC = {"repeat", "split", "condition_id", "n_de_genes"}


def load_eval_dir(eval_dir: Path) -> dict:
    eval_dir = Path(eval_dir)
    data = {
        "conditions": pd.read_parquet(eval_dir / "conditions.parquet"),
        "genes": pd.read_parquet(eval_dir / "genes.parquet"),
        "targets": np.load(eval_dir / "targets.npy", mmap_mode="r"),
        "de_sets": pd.read_parquet(eval_dir / "de_sets.parquet"),
        "splits": pd.read_parquet(eval_dir / "splits.parquet"),
    }
    contracts.validate_targets(data["targets"], data["conditions"], data["genes"])
    return data


def fit_inputs(data: dict, repeat: int, split: str) -> tuple[TrainData, np.ndarray]:
    """What one fit sees: TrainData for the train + val rows of (repeat, split), with the val scorer, and the test row indices."""
    conditions, targets, splits = data["conditions"], data["targets"], data["splits"]
    genes = list(data["genes"]["gene_symbol"])
    features = conditions.reset_index(drop=True)
    row_of = pd.Series(np.arange(len(conditions)), index=conditions["condition_id"])
    grp = splits[(splits["repeat"] == repeat) & (splits["split"] == split)]
    fit_rows = row_of[grp.loc[grp["part"] != "test", "condition_id"]].to_numpy()
    test_rows = row_of[grp.loc[grp["part"] == "test", "condition_id"]].to_numpy()
    part = grp.set_index("condition_id").loc[features.loc[fit_rows, "condition_id"], "part"].to_numpy()
    val_rows = fit_rows[part == "val"]
    val_ids = list(features.loc[val_rows, "condition_id"])
    val_idx, val_truth = de_index(genes, data["de_sets"], set(val_ids)), np.asarray(targets[val_rows])

    def val_score(pred_val: np.ndarray) -> float:
        contracts.validate_prediction(pred_val, len(val_ids), len(genes))
        return float(np.nanmedian(de_pearson(pred_val, val_truth, val_idx, val_ids)))

    train = TrainData(features=features.iloc[fit_rows].reset_index(drop=True), targets=np.asarray(targets[fit_rows]), genes=genes,
                      part=part, val_score=val_score)
    return train, test_rows


def evaluate(name: str, cfg: dict, eval_dir: Path, results_root: Path) -> Path:
    """Score predictor `name` on all repeats x splits; returns the results directory."""
    data = load_eval_dir(eval_dir)
    targets, genes = data["targets"], list(data["genes"]["gene_symbol"])
    features = data["conditions"].reset_index(drop=True)

    start = time.perf_counter()
    scored, model_info = [], []
    for repeat, split in sorted(set(zip(data["splits"]["repeat"], data["splits"]["split"]))):
        train, test_rows = fit_inputs(data, repeat, split)
        model = PREDICTORS[name](**cfg.get("models", {}).get(name, {}))
        model.fit(train)
        test_features = features.iloc[test_rows].reset_index(drop=True)
        pred = model.predict(test_features)
        contracts.validate_prediction(pred, len(test_rows), len(genes))
        scores = score_conditions(pred, np.asarray(targets[test_rows]), genes, data["de_sets"], test_features[["condition_id", "cell_line"]], cfg)
        scored.append(scores.assign(repeat=repeat, split=split))
        if hasattr(model, "info"):
            info = model.info()
            contracts.validate_model_info(info)
            model_info.append({"repeat": int(repeat), "split": split, **info})
    runtime = time.perf_counter() - start

    per = pd.concat(scored, ignore_index=True)
    out = Path(results_root) / name
    out.mkdir(parents=True, exist_ok=True)
    per.to_parquet(out / "per_condition.parquet", index=False)
    metric_cols = [c for c in per.columns if c not in NON_METRIC]
    summaries = {m: uncertainty.summarize(per, m, cfg).to_dict(orient="records") for m in metric_cols}
    (out / "results.json").write_text(
        json.dumps({"model": name, "runtime_s": round(runtime, 2), "peak_rss_mb": peak_rss_mb(), "summaries": summaries, "model_info": model_info}, indent=1, default=float),
        encoding="utf-8",
    )
    return out
