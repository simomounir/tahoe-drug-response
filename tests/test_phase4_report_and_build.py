from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from phase4 import report, targets

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_eval_data.py"
spec = importlib.util.spec_from_file_location("build_eval_data", SCRIPT)
build_eval_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_eval_data)


def _fake_results(root: Path, name: str, value: float):
    d = root / name
    d.mkdir(parents=True)
    rows = [{"split": s, "metric": "de_pearson", "estimate": value, "ci_low": value - 0.1, "ci_high": value + 0.1,
             "iqr_low": 0, "iqr_high": 1, "repeat_medians": [value] * 5, "n_scored": 10, "n_nan": 0}
            for s in ["random", "unseen_drug", "unseen_cell_line", "both_unseen"]]
    (d / "results.json").write_text(json.dumps({"model": name, "runtime_s": 1.0, "peak_rss_mb": 100.0, "summaries": {"de_pearson": rows}}))


def test_report_renders_models_by_splits(tmp_path):
    _fake_results(tmp_path, "dummy", 0.0)
    _fake_results(tmp_path, "ridge", 0.42)
    md = report.render_results(tmp_path)
    assert "| dummy |" in md and "| ridge |" in md
    assert "0.420 [0.320, 0.520]" in md
    header = next(line for line in md.splitlines() if line.startswith("| model"))
    assert header.index("random") < header.index("unseen_drug") < header.index("unseen_cell_line") < header.index("both_unseen")


def test_assemble_builds_a_consistent_eval_set(tmp_path):
    lines = [f"L{i}" for i in range(6)]
    drugs = [f"D{i}" for i in range(6)]
    view = pd.DataFrame([{"condition_id": f"{l}-{d}", "cell_line": l, "drug": d, "dose": 0.5, "is_control": False,
                          "features_complete": l != "L5", "pc_1": 0.0} for l in lines for d in drugs])
    raw = pd.DataFrame([{"condition_id": c, "gene_symbol": g, "log2FoldChange": 1.0, "padj": 0.01}
                        for c in view["condition_id"] for g in ["TP53", "KRAS", "MYC"]])
    universe = pd.DataFrame({"gene": [3, 4, 6], "gene_symbol": ["KRAS", "MYC", "TP53"]})
    groups = pd.DataFrame({"drug": drugs, "cluster_id": range(6)})
    cfg = {"de_genes": {"padj_below": 0.05, "top_n": 200}, "min_de_genes": 2,
           "splits": {"repeats": 2, "seeds": [1, 2], "test_fraction": 0.2, "val_fraction": 0.15}}
    raw.to_parquet(tmp_path / "raw.parquet")
    de_sets = targets.prepare_de_sets(duckdb.connect(), [tmp_path / "raw.parquet"], set(universe["gene_symbol"]), cfg)
    out = build_eval_data.assemble(view, eligible={"L0", "L1", "L2", "L3", "L4", "L5"}, de_sets=de_sets, universe=universe,
                                   drug_groups=groups, cfg=cfg, forbidden_test_lines={"L5"})
    assert set(out["conditions"]["cell_line"]) == {"L0", "L1", "L2", "L3", "L4"}
    assert out["exclusions"]["features_incomplete"] == 6
    assert set(out["splits"]["condition_id"]) <= set(out["conditions"]["condition_id"])
    assert set(out["de_sets"]["condition_id"]) == set(out["conditions"]["condition_id"])
    assert list(out["genes"]["gene_symbol"]) == ["KRAS", "MYC", "TP53"]
