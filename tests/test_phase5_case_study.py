from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase4 import harness
from phase5 import failure_case

GENES = [f"G{i}" for i in range(6)]


def _eval_dir(tmp_path, planted="L3-D3-0", tie=False):
    """4 lines x 4 drugs x 2 doses. repeat 0 both_unseen: L3/D3 conditions are test, L2/D2 val, the rest train.
    Targets: dose offset everywhere; `planted` (a test condition) gets a large deviation on its DE genes."""
    rows = [{"condition_id": f"L{l}-D{d}-{k}", "cell_line": f"L{l}", "drug": f"D{d}", "log10_dose_um": [-1.0, 0.0][k],
             "dose": [0.1, 1.0][k], "unit": "uM", "drug_name": f"D{d}",
             "morgan_counts": np.array([d, 1, 0, 0], np.uint8), "MolWt": 100.0 + d, "pc_1": float(l), "features_complete": True}
            for l in range(4) for d in range(4) for k in range(2)]
    cond = pd.DataFrame(rows)
    y = np.where(cond["log10_dose_um"].to_numpy()[:, None] == 0.0, 1.0, -1.0) * np.ones((len(cond), len(GENES)))
    y = y.astype(np.float32)
    for pid in [planted] + (["L3-D3-1"] if tie else []):
        y[cond.index[cond["condition_id"] == pid][0], :3] += 5.0
    parts = []
    for c, l, dr in zip(cond["condition_id"], cond["cell_line"], cond["drug"]):
        if l == "L3" and dr == "D3":
            p = "test"
        elif l == "L3" or dr == "D3":
            continue  # both_unseen discards conditions sharing only one held-out unit
        elif l == "L2" or dr == "D2":
            p = "val"
        else:
            p = "train"
        parts.append({"repeat": 0, "split": "both_unseen", "condition_id": c, "part": p})
    de = pd.DataFrame([{"condition_id": c, "gene_symbol": g, "rank": r + 1} for c in cond["condition_id"] for r, g in enumerate(GENES[:3])])
    d = tmp_path / "eval"
    d.mkdir(parents=True)
    cond.to_parquet(d / "conditions.parquet")
    pd.DataFrame({"gene": range(len(GENES)), "gene_symbol": GENES}).to_parquet(d / "genes.parquet")
    np.save(d / "targets.npy", y)
    de.to_parquet(d / "de_sets.parquet")
    pd.DataFrame(parts).to_parquet(d / "splits.parquet")
    return d


def test_select_condition_picks_farthest_from_dose_mean(tmp_path):
    data = harness.load_eval_dir(_eval_dir(tmp_path))
    assert failure_case.select_condition(data) == "L3-D3-0"


def test_select_condition_tie(tmp_path):
    data = harness.load_eval_dir(_eval_dir(tmp_path, planted="L3-D3-0", tie=True))
    assert failure_case.select_condition(data) == "L3-D3-0"  # equal distance -> smallest condition_id


def test_select_condition_only_test_rows(tmp_path):
    d = _eval_dir(tmp_path, planted="L0-D0-0")  # a training condition with a huge deviation must not be chosen
    data = harness.load_eval_dir(d)
    assert failure_case.select_condition(data) in {"L3-D3-0", "L3-D3-1"}


def test_predictions_for_refits_and_predicts_one(tmp_path):
    data = harness.load_eval_dir(_eval_dir(tmp_path))
    preds = failure_case.predictions_for("L3-D3-0", ["global_mean"], {"models": {}}, data)
    np.testing.assert_allclose(preds["global_mean"], -1.0)  # dose -1 mean over train rows (no planted deviation there)
    with pytest.raises(ValueError, match="not a test condition"):
        failure_case.predictions_for("L0-D0-0", ["global_mean"], {"models": {}}, data)


import re
import xml.etree.ElementTree as ET

from phase5 import figures

DEG_ROWS = [{"model": m, "split": s, "estimate": e, "ci_low": e - 0.01, "ci_high": e + 0.01}
            for m, base in [("global_mean", 0.68), ("ridge", 0.8), ("neural", 0.75)]
            for s, e in zip(["random", "unseen_cell_line", "unseen_drug", "both_unseen"], [base + 0.1, base, base - 0.05, 0.66])]


def _fail_svg(n):
    genes = [f"GENE{i}" for i in range(n)]
    rng = np.random.default_rng(0)
    return figures.failure_case_svg(genes, rng.normal(size=n), {m: rng.normal(size=n) for m in ["global_mean", "ridge", "neural"]})


@pytest.mark.parametrize("svg", [figures.degradation_svg(DEG_ROWS), _fail_svg(20)])
def test_figures_are_wellformed_svg(svg):
    assert svg.lstrip().startswith("<svg")
    ET.fromstring(svg)
    assert "var(--series-1)" in svg and "<title>" in svg  # themed colours, native hover tooltips


@pytest.mark.parametrize("svg", [figures.degradation_svg(DEG_ROWS), _fail_svg(20)])
def test_svg_uses_current_color(svg):
    hard = re.findall(r"(?:fill|stroke)\s*[:=]\s*\"?\s*(#000000|#ffffff|black|white)\b", svg, flags=re.I)
    assert not hard


def test_failure_chart_fewer_genes():
    svg = _fail_svg(8)
    assert sum(f"GENE{i}" in svg for i in range(8)) == 8


import json
from pathlib import Path

from phase1.contracts import ContractError
from phase5 import case_study

SPLITS = ["random", "unseen_cell_line", "unseen_drug", "both_unseen"]
CS_CFG = {"bootstrap": {"resamples": 100, "interval": 0.95, "seed": 1}, "splits": {"repeats": 1},
          "models": {"ridge": {"alpha_log10_start": -2.0, "alpha_log10_stop": 2.0, "alpha_log10_step": 1.0},
                     "neural": {"hidden": 8, "dropouts": [0.2], "weight_decays": [1e-4], "lr": 1e-2, "batch_size": 8, "max_epochs": 2,
                                "patience": 1, "seed": 1, "device": "cpu"}}}
EVAL_MD = """## 12. Amendments

- **A1 — §4.1 drug grouping.** *Before:* x.
- **A8 — §6 baseline 5 (2026-09-27, post-hoc, owner decision).** *Before:* y.

## 13. Open questions
"""


def _results(root: Path, scores: dict[str, float]):
    for model, de in scores.items():
        d = root / model
        d.mkdir(parents=True)
        rng = np.random.default_rng(0)
        ids = {s: ["L3-D3-0", "L3-D3-1"] if s == "both_unseen" else [f"{s}-{i}" for i in range(10)] for s in SPLITS}
        per = pd.DataFrame([{"repeat": 0, "split": s, "condition_id": c, "de_pearson": de + 0.001 * rng.normal(),
                             "disc_rank_global": 0.3 - (de - 0.6)} for s in SPLITS for c in ids[s]])
        per.to_parquet(d / "per_condition.parquet")
        summ = [{"split": s, "metric": "de_pearson", "estimate": de, "ci_low": de - 0.01, "ci_high": de + 0.01, "iqr_low": 0, "iqr_high": 1,
                 "repeat_medians": [de], "n_scored": 10, "n_nan": 0} for s in SPLITS]
        (d / "results.json").write_text(json.dumps({"model": model, "runtime_s": 12.5, "peak_rss_mb": 100.0,
                                                    "summaries": {"de_pearson": summ}, "model_info": []}))


def _build(tmp_path, scores=None, drug_name=None):
    ev = _eval_dir(tmp_path)
    if drug_name:
        c = pd.read_parquet(ev / "conditions.parquet")
        c["drug_name"] = drug_name
        c.to_parquet(ev / "conditions.parquet")
    _results(tmp_path / "results", scores or {"global_mean": 0.60, "ridge": 0.60, "neural": 0.60})
    (tmp_path / "evaluation.md").write_text(EVAL_MD)
    return case_study.build_context(tmp_path / "results", ev, CS_CFG, tmp_path / "evaluation.md")


@pytest.mark.parametrize("diff,phrase", [({"estimate": 0.05, "ci_low": 0.01, "ci_high": 0.09}, "A beats B"),
                                         ({"estimate": -0.05, "ci_low": -0.09, "ci_high": -0.01}, "A falls short of B"),
                                         ({"estimate": 0.0, "ci_low": -0.01, "ci_high": 0.01}, "no detectable difference between A and B")])
def test_compare_sentence_cases(diff, phrase):
    s = case_study.compare_sentence("A", "B", diff)
    assert phrase in s and f"{diff['estimate']:+.3f}" in s


def test_page_numbers_match_results(tmp_path):
    html = case_study.render(_build(tmp_path, {"global_mean": 0.61, "ridge": 0.62, "neural": 0.63}))
    assert "0.630" in html and "0.620" in html and "0.610" in html
    assert "<svg" in html and ("FAILURE" in html or "SUCCESS" in html)


def test_headline_follows_verdict(tmp_path):
    html = case_study.render(_build(tmp_path, {"global_mean": 0.60, "ridge": 0.60, "neural": 0.70}))
    assert "SUCCESS" in html and "beats" in html
    tie = case_study.render(_build(tmp_path / "t", {"global_mean": 0.60, "ridge": 0.60, "neural": 0.60}))
    assert "FAILURE" in tie and "no detectable difference" in tie


def test_missing_model_fails(tmp_path):
    ev = _eval_dir(tmp_path)
    _results(tmp_path / "results", {"global_mean": 0.6, "ridge": 0.6})
    (tmp_path / "evaluation.md").write_text(EVAL_MD)
    with pytest.raises(ContractError, match="neural"):
        case_study.build_context(tmp_path / "results", ev, CS_CFG, tmp_path / "evaluation.md")


def test_names_are_escaped(tmp_path):
    html = case_study.render(_build(tmp_path, drug_name="<b>Evil & Co</b>"))
    assert "<b>Evil" not in html and "&lt;b&gt;Evil &amp; Co&lt;/b&gt;" in html


def test_render_rejects_nan_and_leftovers(tmp_path):
    ctx = _build(tmp_path)
    with pytest.raises(ContractError, match="nan"):
        case_study.render({**ctx, "headline_neural": "nan"})
    with pytest.raises(ContractError, match="placeholder"):
        case_study.render({k: v for k, v in ctx.items() if k != "headline_neural"})


def test_amendments_parsed(tmp_path):
    ctx = _build(tmp_path)
    assert "A1" in ctx["amendments_html"] and "§6 baseline 5" in ctx["amendments_html"] and ctx["n_amendments"] == "2"


def test_every_context_value_is_on_the_page(tmp_path):
    ctx = _build(tmp_path)
    tmpl = case_study.TEMPLATE.read_text()
    assert [k for k in ctx if f"${k}" not in tmpl and "${" + k + "}" not in tmpl] == []
    html = case_study.render(ctx)
    assert "discrimination" in html.split("Pre-registered claim")[1].split("Why the result")[0]
    with pytest.raises(ContractError, match="unused"):
        case_study.render({**ctx, "extra_value": "x"})


def test_disc_sentence_lower_is_better():
    s = case_study.compare_sentence("A", "B", {"estimate": -0.05, "ci_low": -0.09, "ci_high": -0.01}, lower_is_better=True)
    assert "A beats B" in s


def test_failure_rows_are_top_by_measured_and_shared():
    de_genes = [f"R{i}" for i in range(25)]  # DE rank order
    genes = de_genes + ["OTHER"]
    truth = np.array([0.1 * (i % 5) for i in range(25)] + [9.0])
    truth[24] = -3.0  # ranked last by padj, largest |measured|
    names, idx = case_study.failure_rows(de_genes, genes, truth, max_genes=20)
    assert len(names) == 20 and "R24" in names
    assert "OTHER" not in names and [genes[i] for i in idx] == names
    svg = figures.failure_case_svg(names, truth[idx], {"ridge": np.zeros(len(idx))})
    assert all(n in svg for n in names)


def test_failure_labels_align_when_a_de_gene_is_missing(tmp_path):
    ev = _eval_dir(tmp_path)
    de = pd.read_parquet(ev / "de_sets.parquet")
    extra = de[de["rank"] == 1].assign(gene_symbol="NOTAGENE", rank=0)  # ranked first, absent from the gene universe
    pd.concat([extra, de]).to_parquet(ev / "de_sets.parquet")
    _results(tmp_path / "results", {"global_mean": 0.6, "ridge": 0.6, "neural": 0.6})
    (tmp_path / "evaluation.md").write_text(EVAL_MD)
    html = case_study.render(case_study.build_context(tmp_path / "results", ev, CS_CFG, tmp_path / "evaluation.md"))
    assert "NOTAGENE" not in html
    rows = dict(re.findall(r'<th scope="row">(G\d)</th><td>([+-]\d\.\d\d)</td>', html))
    assert rows["G0"] == "+4.00"  # chosen condition L3-D3-0 is at dose -1 (offset -1) with +5 planted on G0-G2


def test_explanation_follows_numbers_and_states_refit(tmp_path):
    html = case_study.render(_build(tmp_path))
    sec = html.split('id="failure"')[1].split('id="limits"')[0]
    assert "refit" in sec and "%" in sec
    assert "stay close to the average instead" not in sec
    assert "least to most held out" in html
