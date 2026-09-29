"""Recompute the headline from a bundle and compare it with reproduce/expected.json (phase 5b spec R4–R6).

Level 1 recomputes intervals, paired differences and the verdict from saved per-condition scores. Level 2 first refits
the per-dose mean, ridge and the neural network on every `both_unseen` repeat of the eval set and scores them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from phase4 import harness, uncertainty
from phase4.metrics import score_conditions
from phase4.predictors import PREDICTORS
from phase4.report import claim_verdict

MODELS = ["global_mean", "ridge", "neural"]
SPLITS = ["random", "unseen_cell_line", "unseen_drug", "both_unseen"]


class ReproduceError(RuntimeError):
    pass


def _triple(row) -> list[float]:
    return [float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"])]


def headline(results_dir: Path, cfg: dict, splits: list[str] | None = None) -> dict:
    splits = splits or SPLITS
    per = {m: pd.read_parquet(Path(results_dir) / m / "per_condition.parquet") for m in MODELS}
    out = {"de_pearson": {}}
    for m in MODELS:
        summ = uncertainty.summarize(per[m][per[m]["split"].isin(splits)], "de_pearson", cfg).set_index("split")
        out["de_pearson"][m] = {s: _triple(summ.loc[s]) for s in splits}
    de = uncertainty.paired(per["neural"], per["ridge"], "de_pearson", cfg).set_index("split").loc["both_unseen"]
    disc = uncertainty.paired(per["neural"], per["ridge"], "disc_rank_global", cfg).set_index("split").loc["both_unseen"]
    out["claim"] = {"de_pearson": _triple(de), "disc_rank_global": _triple(disc)}
    out["verdict"] = claim_verdict(de.to_dict(), disc.to_dict())
    return out


def compare(got: dict, expected: dict, tol: float, path: str = "") -> list[str]:
    """Every mismatch between `got` and `expected` (numbers within `tol`, everything else exactly)."""
    bad = []
    if isinstance(expected, dict):
        for k, v in expected.items():
            if k not in got:
                bad.append(f"{path}{k}: missing")
            else:
                bad += compare(got[k], v, tol, f"{path}{k}.")
    elif isinstance(expected, list):
        if not isinstance(got, list) or len(got) != len(expected):
            return [f"{path.rstrip('.')}: got {got!r}, expected a list of {len(expected)}"]
        for i, (g, e) in enumerate(zip(got, expected)):
            bad += compare(g, e, tol, f"{path}{i}.")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not abs(float(got) - float(expected)) <= tol:
            bad.append(f"{path.rstrip('.')}: got {float(got):.4f}, expected {float(expected):.4f}")
    elif got != expected:
        bad.append(f"{path.rstrip('.')}: got {got!r}, expected {expected!r}")
    return bad


def restrict(expected: dict, splits: list[str]) -> dict:
    """The part of `expected` a refit of `splits` can check."""
    return {"de_pearson": {m: {s: v for s, v in by.items() if s in splits} for m, by in expected["de_pearson"].items()},
            "claim": expected["claim"], "verdict": expected["verdict"]}


def bundle_config(eval_dir: Path, fallback: dict) -> dict:
    """The eval config shipped in level2.zip, so a bundle refits with its own settings (R6); `fallback` if absent."""
    path = Path(eval_dir) / "eval.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else fallback


def load_expected(path: Path) -> dict:
    if not Path(path).exists():
        raise ReproduceError(f"{path} not found: create it from a full run with `python scripts/reproduce.py --write-expected`")
    return json.loads(Path(path).read_text())


def refit_both_unseen(eval_dir: Path, bundle_results: Path, cfg: dict, out_dir: Path, search: bool = False) -> Path:
    """Refit MODELS on every both_unseen repeat and write their per-condition scores under `out_dir`.
    The neural network reuses each fit's recorded dropout / weight decay / epochs unless `search` (R6)."""
    data = harness.load_eval_dir(Path(eval_dir))
    genes, features = list(data["genes"]["gene_symbol"]), data["conditions"].reset_index(drop=True)
    info = json.loads((Path(bundle_results) / "neural" / "results.json").read_text()).get("model_info", [])
    recorded = {int(i["repeat"]): i for i in info if i.get("split") == "both_unseen"}
    repeats = sorted(set(data["splits"].loc[data["splits"]["split"] == "both_unseen", "repeat"]))
    for name in MODELS:
        scored = []
        for r in repeats:
            kwargs = dict(cfg.get("models", {}).get(name, {}))
            if name == "neural" and not search:
                if r in recorded:
                    rec = recorded[r]
                    kwargs["fixed"] = {"dropout": rec["dropout"], "weight_decay": rec["weight_decay"], "epochs": int(rec["epochs"]),
                                       "val_de_pearson": rec.get("val_de_pearson", float("nan"))}
                else:
                    print(f"repeat {r}: no recorded neural config, running the full search for this fit", flush=True)
            train, test_rows = harness.fit_inputs(data, r, "both_unseen")
            model = PREDICTORS[name](**kwargs)
            model.fit(train)
            test = features.iloc[test_rows].reset_index(drop=True)
            pred = model.predict(test)
            s = score_conditions(pred, np.asarray(data["targets"][test_rows]), genes, data["de_sets"], test[["condition_id", "cell_line"]], cfg)
            scored.append(s.assign(repeat=r, split="both_unseen"))
        d = Path(out_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        pd.concat(scored, ignore_index=True).to_parquet(d / "per_condition.parquet", index=False)
        print(f"refit {name}: {len(repeats)} fits", flush=True)
    return Path(out_dir)
