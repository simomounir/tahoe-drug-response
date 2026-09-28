from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

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


CFG = {"bootstrap": {"resamples": 200, "interval": 0.95, "seed": 1}}


def _fake_run(root: Path, name: str, de_pearson: float, fallback: float | None):
    _fake_results(root, name, de_pearson)
    if fallback is not None:
        res = json.loads((root / name / "results.json").read_text())
        res["model_info"] = [{"repeat": r, "split": s, "fallback_rate": fallback} for r in range(5)
                             for s in ["random", "unseen_drug", "unseen_cell_line", "both_unseen"]]
        (root / name / "results.json").write_text(json.dumps(res))
    rng = np.random.default_rng(0)
    rows = [{"repeat": r, "split": s, "condition_id": f"c{i}", "de_pearson": de_pearson + 0.01 * rng.normal(), "disc_rank_global": 0.5}
            for r in range(5) for s in ["random", "unseen_drug", "unseen_cell_line", "both_unseen"] for i in range(30)]
    pd.DataFrame(rows).to_parquet(root / name / "per_condition.parquet")


def test_report_coverage_and_paired(tmp_path):
    _fake_run(tmp_path, "dummy", 0.0, None)
    _fake_run(tmp_path, "drug_mean", 0.2, 0.25)
    _fake_run(tmp_path, "ridge", 0.5, None)
    md = report.render_results(tmp_path, CFG)
    coverage = md.split("## Coverage")[1].split("\n## ")[0]
    assert "| drug_mean | 25% | 25% | 25% | 25% |" in coverage and "dummy" not in coverage and "ridge" not in coverage
    paired = md.split("## Paired vs ridge")[1]
    row = next(line for line in paired.splitlines() if line.startswith("| drug_mean"))
    estimates = [float(cell.split()[0]) for cell in row.strip("|").split("|")[1:]]
    assert estimates == pytest.approx([-0.3] * 4, abs=0.01)  # de_pearson difference on every split
    assert "n.d." in paired  # disc_rank_global differences are exactly 0


def test_report_without_ridge(tmp_path):
    _fake_run(tmp_path, "dummy", 0.0, None)
    md = report.render_results(tmp_path, CFG)
    assert "| dummy |" in md and "Paired vs ridge" not in md and "Coverage" not in md


def test_report_shows_top1_secondary_and_detail(tmp_path):
    _fake_run(tmp_path, "dummy", 0.0, None)
    res = json.loads((tmp_path / "dummy" / "results.json").read_text())
    for m in ["disc_top1_global", "disc_top1_within_line", "de_mse", "sign_accuracy", "all_gene_pearson"]:
        res["summaries"][m] = res["summaries"]["de_pearson"]
    (tmp_path / "dummy" / "results.json").write_text(json.dumps(res))
    md = report.render_results(tmp_path, CFG)
    for heading in ["## disc_top1_global", "## disc_top1_within_line", "## Secondary metrics", "### all_gene_pearson",
                    "## de_pearson detail"]:
        assert heading in md
    detail = md.split("## de_pearson detail")[1]
    assert "| dummy | random | [0.000, 1.000] | 10 | 0 | 0.000, 0.000, 0.000, 0.000, 0.000 |" in detail


def test_run_eval_all_runs_each_model_in_its_own_process(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"
    spec_ = importlib.util.spec_from_file_location("run_eval", script)
    run_eval = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(run_eval)
    calls = []
    monkeypatch.setattr(run_eval.subprocess, "run", lambda cmd, check: calls.append(cmd[cmd.index("--model") + 1]))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "eval.yaml").write_text((script.parents[1] / "configs" / "eval.yaml").read_text())
    monkeypatch.setattr(run_eval, "ROOT", tmp_path)
    monkeypatch.setattr(run_eval, "REPORT", tmp_path / "results.md")
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["run_eval.py", "--all"])
    run_eval.main()
    assert calls == sorted(run_eval.PREDICTORS)
    assert (tmp_path / "results.md").exists()


@pytest.mark.parametrize("de,disc,verdict", [
    ({"ci_low": 0.01, "estimate": 0.05}, {"estimate": -0.02}, "SUCCESS"),
    ({"ci_low": 0.0, "estimate": 0.05}, {"estimate": -0.02}, "FAILURE"),   # interval touches 0
    ({"ci_low": 0.01, "estimate": 0.05}, {"estimate": 0.0}, "FAILURE"),    # discrimination not in the same direction
    ({"ci_low": -0.10, "estimate": -0.05}, {"estimate": 0.03}, "FAILURE"),
])
def test_claim_verdict_cases(de, disc, verdict):
    assert report.claim_verdict(de, disc) == verdict


def _fake_per_condition(root: Path, name: str, de: float, disc: float):
    d = root / name
    if not d.exists():
        _fake_results(root, name, de)
    rng = np.random.default_rng(1)
    rows = [{"repeat": r, "split": s, "condition_id": f"c{i}", "de_pearson": de + 0.001 * rng.normal(), "disc_rank_global": disc}
            for r in range(5) for s in ["random", "unseen_drug", "unseen_cell_line", "both_unseen"] for i in range(30)]
    pd.DataFrame(rows).to_parquet(d / "per_condition.parquet")


def test_claim_section_renders_first(tmp_path):
    _fake_per_condition(tmp_path, "ridge", 0.60, 0.30)
    _fake_per_condition(tmp_path, "neural", 0.70, 0.20)
    _fake_per_condition(tmp_path, "neural_nocell", 0.65, 0.25)
    md = report.render_results(tmp_path, CFG)
    assert md.index("## Pre-registered claim") < md.index("## de_pearson")
    claim = md.split("## Pre-registered claim")[1].split("\n## ")[0]
    assert "**SUCCESS**" in claim
    abl = next(line for line in claim.splitlines() if line.startswith("| neural |"))
    assert "0.050" in abl  # neural − neural_nocell
    assert next(line for line in claim.splitlines() if line.startswith("| ridge |")).count("—") == 4


def test_claim_without_ablation(tmp_path):
    _fake_per_condition(tmp_path, "ridge", 0.60, 0.30)
    _fake_per_condition(tmp_path, "neural", 0.60, 0.30)
    claim = report.render_results(tmp_path, CFG).split("## Pre-registered claim")[1].split("\n## ")[0]
    assert "**FAILURE**" in claim and "—" in claim


def test_claim_omitted_without_neural(tmp_path):
    _fake_per_condition(tmp_path, "ridge", 0.60, 0.30)
    assert "Pre-registered claim" not in report.render_results(tmp_path, CFG)


def test_claim_states_mps_nondeterminism(tmp_path):
    _fake_per_condition(tmp_path, "ridge", 0.60, 0.30)
    _fake_per_condition(tmp_path, "neural", 0.70, 0.20)
    res = json.loads((tmp_path / "neural" / "results.json").read_text())
    res["model_info"] = [{"repeat": 0, "split": "random", "device": "mps", "epochs": 3}]
    (tmp_path / "neural" / "results.json").write_text(json.dumps(res))
    claim = report.render_results(tmp_path, CFG).split("## Pre-registered claim")[1].split("\n## ")[0]
    assert "trained on MPS" in claim and "last digits" in claim
