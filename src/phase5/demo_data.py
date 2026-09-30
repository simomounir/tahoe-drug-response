"""Data for the interactive demo (phase 5c spec D2–D5).

For every split and repeat, the per-dose mean, ridge and the neural network are refitted on exactly that fit's inputs
(the neural network with the configuration it chose in the scored run). Each (condition, split) keeps the predictions
from the first repeat in which the condition was a test condition. Values are stored on the condition's DE genes as
integers ×100, with the refit's own DE-gene Pearson ("score") and the evaluated run's ("scored").

Contract (D3, amended 2026-09-29): the deterministic models (per-dose mean, ridge) must match the evaluated run per
condition within `tol`; the neural network, whose GPU results shift slightly across OS versions, must match per split
on the median over the exported conditions within `tol`.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from phase4 import harness
from phase4.metrics import de_index, de_pearson
from phase4.predictors import PREDICTORS
from phase5 import failure_case

MODELS = ["global_mean", "ridge", "neural"]
PER_CONDITION = {"global_mean", "ridge"}  # deterministic: checked condition by condition
SPLITS = ["random", "unseen_cell_line", "unseen_drug", "both_unseen"]


class DemoError(RuntimeError):
    pass


def shard_name(drug: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", drug)[:40].strip("_") or "drug"
    return f"{slug}-{hashlib.sha1(drug.encode()).hexdigest()[:8]}.json"


def _ints(values: np.ndarray) -> list[int]:
    v = np.asarray(values, dtype=np.float64)
    if not np.isfinite(v).all():
        raise DemoError("non-finite value in exported logFC")
    return np.rint(v * 100).astype(int).tolist()


def export(eval_dir: Path, results_dir: Path, cfg: dict, out_dir: Path, cell_meta: Path | None = None, tol: float = 0.003,
           log=print) -> dict:
    data = harness.load_eval_dir(Path(eval_dir))
    cond = data["conditions"].reset_index(drop=True)
    genes = list(data["genes"]["gene_symbol"])
    idx = de_index(genes, data["de_sets"])
    targets = data["targets"]
    results_dir = Path(results_dir)
    scored = {m: pd.read_parquet(results_dir / m / "per_condition.parquet").set_index(["repeat", "split", "condition_id"])["de_pearson"]
              for m in MODELS}
    info = json.loads((results_dir / "neural" / "results.json").read_text()).get("model_info", [])
    recorded = {(int(i["repeat"]), i["split"]): i for i in info}
    tests = data["splits"][data["splits"]["part"] == "test"]
    first = tests.groupby(["split", "condition_id"])["repeat"].min()

    entry = {c: {"splits": {}} for c in cond["condition_id"]}
    pairs: dict[tuple[str, str], list[tuple[float, float]]] = {}  # (split, model) -> [(refit, scored)]
    for split in SPLITS:
        for repeat in sorted(set(tests.loc[tests["split"] == split, "repeat"])):
            wanted = set(first.loc[split][first.loc[split] == repeat].index)
            if not wanted:
                continue
            train, test_rows = harness.fit_inputs(data, repeat, split)
            rows = [r for r in test_rows if cond.at[r, "condition_id"] in wanted]
            sub = cond.iloc[rows].reset_index(drop=True)
            ids = list(sub["condition_id"])
            truth = np.asarray(targets[rows])
            for name in MODELS:
                kwargs = dict(cfg.get("models", {}).get(name, {}))
                if name == "neural":
                    if (repeat, split) not in recorded:
                        raise DemoError(f"neural {split} repeat {repeat}: no recorded configuration in results.json")
                    rec = recorded[(repeat, split)]
                    kwargs["fixed"] = {"dropout": rec["dropout"], "weight_decay": rec["weight_decay"], "epochs": int(rec["epochs"])}
                model = PREDICTORS[name](**kwargs)
                model.fit(train)
                pred = model.predict(sub)
                refit = de_pearson(pred, truth, idx, ids)
                for i, cid in enumerate(ids):
                    want = float(scored[name].loc[(repeat, split, cid)])
                    if np.isnan(refit[i]) != np.isnan(want):
                        raise DemoError(f"{name} {split} repeat {repeat} {cid}: NaN on one side (refit {refit[i]}, scored {want})")
                    if name in PER_CONDITION and not np.isnan(want) and not abs(refit[i] - want) <= tol:
                        raise DemoError(f"{name} {split} repeat {repeat} {cid}: refit de_pearson {refit[i]:.4f} vs scored {want:.4f}")
                    pairs.setdefault((split, name), []).append((refit[i], want))
                    s = entry[cid]["splits"].setdefault(split, {"repeat": int(repeat), "pred": {}, "score": {}, "scored": {}})
                    s["pred"][name] = _ints(pred[i, idx[cid]]) if cid in idx else []
                    s["score"][name] = None if np.isnan(refit[i]) else round(float(refit[i]), 4)
                    s["scored"][name] = None if np.isnan(want) else round(want, 4)
            log(f"{split} repeat {repeat}: {len(ids)} conditions")
        a = np.array(pairs.get((split, "neural"), []), dtype=np.float64)
        if len(a):
            got, want = float(np.nanmedian(a[:, 0])), float(np.nanmedian(a[:, 1]))
            if not abs(got - want) <= tol:
                raise DemoError(f"neural {split}: refit median de_pearson {got:.4f} vs scored median {want:.4f}")

    names = {}
    if cell_meta is not None and Path(cell_meta).exists():
        meta = pd.read_parquet(cell_meta).drop_duplicates("Cell_ID_Cellosaur")
        names = dict(zip(meta["Cell_ID_Cellosaur"], meta["cell_name"]))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    drugs = {}
    for drug, grp in cond.groupby("drug", sort=True):
        shard = {"drug": drug, "conditions": {}}
        for r in grp.itertuples():
            gi = idx.get(r.condition_id, np.array([], dtype=np.int64))
            shard["conditions"][r.condition_id] = {"genes": gi.tolist(), "measured": _ints(np.asarray(targets[r.Index])[gi]),
                                                   "splits": entry[r.condition_id]["splits"]}
        fname = shard_name(drug)
        text = json.dumps(shard, separators=(",", ":"))
        (out / fname).write_text(text, encoding="utf-8")
        total += len(text)
        drugs[drug] = {"name": str(grp["drug_name"].iloc[0]) if "drug_name" in grp else drug, "shard": fname}
    medians = {}
    for m in MODELS:
        for row in json.loads((results_dir / m / "results.json").read_text())["summaries"]["de_pearson"]:
            medians.setdefault(row["split"], {})[m] = round(row["estimate"], 4)
    index = {"genes": genes, "lines": {l: str(names.get(l, l)) for l in sorted(cond["cell_line"].unique())}, "drugs": drugs,
             "conditions": [[r.condition_id, r.cell_line, r.drug, float(r.dose), str(getattr(r, "unit", "uM")), str(getattr(r, "plate", ""))]
                            for r in cond.itertuples()],
             "medians": medians, "models": MODELS, "splits": SPLITS,
             "tested": {s: sorted(c for c, e in entry.items() if s in e["splits"]) for s in SPLITS},
             "featured": failure_case.select_condition(data)}  # the case study's failure case opens the demo
    text = json.dumps(index, separators=(",", ":"))
    (out / "index.json").write_text(text, encoding="utf-8")
    total += len(text)
    return {"conditions": len(cond), "drugs": len(drugs), "bytes": total}
