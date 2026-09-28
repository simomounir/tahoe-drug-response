"""The case study page (phase 5a spec): every number, figure and comparison sentence generated from the saved results.

`build_context` gathers the values (HTML-escaped strings and pre-rendered fragments); `render` fills the template and
refuses unfilled placeholders or "nan"/"None" on the page (spec §6).
"""
from __future__ import annotations

import json
import re
import string
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from phase1.contracts import ContractError
from phase4 import harness, uncertainty
from phase4.report import ABLATIONS, CLAIM_RULE, claim_verdict
from phase5 import failure_case, figures

TEMPLATE = Path(__file__).parent / "templates" / "case_study.html"
REQUIRED = ["global_mean", "ridge", "neural"]
MODEL_ORDER = ["dummy", "global_mean", "drug_mean", "cell_mean", "nearest_chemical", "ridge", "ridge_nocell", "neural", "neural_nocell"]
MODEL_TEXT = {"dummy": "no change (zeros)", "global_mean": "per-dose mean", "drug_mean": "drug mean", "cell_mean": "cell-line mean",
              "nearest_chemical": "nearest chemical", "ridge": "ridge", "ridge_nocell": "ridge, no cell features",
              "neural": "neural network", "neural_nocell": "neural, no cell features"}
SPLIT_TEXT = {"random": "random", "unseen_cell_line": "new cell line", "unseen_drug": "new drug", "both_unseen": "new drug + new line"}
FAILURE_MODELS = ["global_mean", "ridge", "neural"]


def compare_sentence(a: str, b: str, diff: dict, lower_is_better: bool = False) -> str:
    """Wording chosen by the paired interval (spec C6), never hand-written. `diff` is a − b."""
    nums = f"(Δ {diff['estimate']:+.3f} [{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}])"
    better, worse = (diff["ci_high"] < 0, diff["ci_low"] > 0) if lower_is_better else (diff["ci_low"] > 0, diff["ci_high"] < 0)
    if better:
        return f"{a} beats {b} {nums}"
    if worse:
        return f"{a} falls short of {b} {nums}"
    return f"no detectable difference between {a} and {b} {nums}"


def failure_rows(de_genes: list[str], genes: list[str], truth: np.ndarray, max_genes: int = 20) -> tuple[list[str], np.ndarray]:
    """The failure-case genes shown in both the chart and its table: DE genes present in the gene universe, the
    `max_genes` largest |measured| (ties by DE rank), ordered from most up- to most down-regulated.
    Returns their names and their column indices into `genes` / the full target vector."""
    col = {g: i for i, g in enumerate(genes)}
    idx = np.array([col[g] for g in de_genes if g in col], dtype=np.int64)
    top = idx[np.argsort(-np.abs(truth[idx]), kind="stable")[:max_genes]]
    top = top[np.argsort(-truth[top], kind="stable")]
    return [genes[i] for i in top], top


def _cell(row: dict) -> str:
    return f"{row['estimate']:.3f} <span class=\"ci\">[{row['ci_low']:.3f}, {row['ci_high']:.3f}]</span>"


def _table(header: list[str], rows: list[list[str]], cls: str = "") -> str:
    head = "".join(f"<th scope=\"col\">{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<t{'h scope=\"row\"' if i == 0 else 'd'}>{c}</t{'h' if i == 0 else 'd'}>" for i, c in enumerate(r)) + "</tr>"
                   for r in rows)
    return f"<div class=\"table-wrap\"><table class=\"{cls}\"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _amendments(evaluation_md: Path) -> list[tuple[str, str]]:
    text = Path(evaluation_md).read_text(encoding="utf-8")
    section = text[text.index("## 12."):text.index("## 13.")] if "## 13." in text else text[text.index("## 12."):]
    return re.findall(r"^- \*\*(A\d+) — ([^*]+?)\.?\*\*", section, flags=re.M)


def build_context(results_dir: Path, eval_dir: Path, cfg: dict, evaluation_md: Path, cell_meta: Path | None = None) -> dict:
    results_dir = Path(results_dir)
    results = {p.parent.name: json.loads(p.read_text()) for p in sorted(results_dir.glob("*/results.json"))}
    missing = [m for m in REQUIRED if m not in results or not (results_dir / m / "per_condition.parquet").exists()]
    if missing:
        raise ContractError(f"case study: no results for {', '.join(missing)}")
    per = {m: pd.read_parquet(results_dir / m / "per_condition.parquet") for m in results if (results_dir / m / "per_condition.parquet").exists()}
    de = {m: {r["split"]: r for r in res["summaries"]["de_pearson"]} for m, res in results.items()}
    models = [m for m in MODEL_ORDER if m in results] + sorted(m for m in results if m not in MODEL_ORDER)

    def paired(a: str, b: str, metric: str) -> pd.DataFrame:
        return uncertainty.paired(per[a], per[b], metric, cfg).set_index("split")

    claim_de = paired("neural", "ridge", "de_pearson").loc["both_unseen"].to_dict()
    claim_disc = paired("neural", "ridge", "disc_rank_global").loc["both_unseen"].to_dict()
    ridge_vs_mean = paired("ridge", "global_mean", "de_pearson").loc["both_unseen"].to_dict()
    verdict = claim_verdict(claim_de, claim_disc)

    deg_rows = [{"model": m, "split": s, **{k: de[m][s][k] for k in ("estimate", "ci_low", "ci_high")}}
                for m in FAILURE_MODELS for s in figures.SPLIT_ORDER]
    deg_table = _table(["model", *[SPLIT_TEXT[s] for s in figures.SPLIT_ORDER]],
                       [[escape(MODEL_TEXT[m]), *[_cell(de[m][s]) for s in figures.SPLIT_ORDER]] for m in FAILURE_MODELS])
    results_table = _table(["model", *[SPLIT_TEXT[s] for s in figures.SPLIT_ORDER]],
                           [[escape(MODEL_TEXT.get(m, m)), *[_cell(de[m][s]) if s in de[m] else "—" for s in figures.SPLIT_ORDER]] for m in models],
                           "results")
    abl_rows = []
    for full, nocell in ABLATIONS:
        if full in per and nocell in per:
            d = paired(full, nocell, "de_pearson")
            abl_rows.append([escape(MODEL_TEXT[full]), *[_cell(d.loc[s].to_dict()) for s in figures.SPLIT_ORDER]])
    ablation_table = _table(["full − no cell features", *[SPLIT_TEXT[s] for s in figures.SPLIT_ORDER]], abl_rows) if abl_rows else \
        "<p>Ablation models have not been run.</p>"

    data = harness.load_eval_dir(Path(eval_dir))
    cond, genes = data["conditions"], list(data["genes"]["gene_symbol"])
    dist = failure_case.distances(data)
    cid = min(dist.index, key=lambda c: (-dist[c], c))
    try:
        preds = failure_case.predictions_for(cid, FAILURE_MODELS, cfg, data)
    except ValueError as exc:
        raise ContractError(f"case study: failure case {exc}") from exc
    row = cond[cond["condition_id"] == cid].iloc[0]
    de_genes = data["de_sets"][data["de_sets"]["condition_id"] == cid].sort_values("rank")["gene_symbol"].tolist()
    truth_full = np.asarray(data["targets"][cond.index[cond["condition_id"] == cid][0]])
    col = {g: i for i, g in enumerate(genes)}
    all_de = np.array([col[g] for g in de_genes if g in col], dtype=np.int64)
    shown, gi = failure_rows(de_genes, genes, truth_full)
    truth = truth_full[gi]
    sub = {m: p[gi] for m, p in preds.items()}
    names = {}
    if cell_meta is not None and Path(cell_meta).exists():
        meta = pd.read_parquet(cell_meta).drop_duplicates("Cell_ID_Cellosaur")
        names = dict(zip(meta["Cell_ID_Cellosaur"], meta["cell_name"]))
    fc_scores = {m: float(per[m].query("repeat == 0 and split == 'both_unseen' and condition_id == @cid")["de_pearson"].iloc[0]) for m in FAILURE_MODELS}
    fc_table = _table(["gene", "measured", *[escape(MODEL_TEXT[m]) for m in FAILURE_MODELS]],
                      [[escape(g), f"{truth[i]:+.2f}", *[f"{sub[m][i]:+.2f}" for m in FAILURE_MODELS]] for i, g in enumerate(shown)])
    # Departure from the per-dose average over all of the condition's DE genes, measured vs predicted (no fixed conclusion).
    mean_all = preds["global_mean"][all_de]
    dev_truth = float(np.mean(np.abs(truth_full[all_de] - mean_all)))
    dev_pred = {m: float(np.mean(np.abs(preds[m][all_de] - mean_all))) for m in ("ridge", "neural")}
    dose = f"{row['dose']:g} {'µM' if row.get('unit', 'uM') == 'uM' else escape(str(row['unit']))}"
    explanation = (
        f"Its measured response is the farthest of the {len(dist)} test conditions from the per-dose average "
        f"(distance {dist[cid]:.1f} over its DE genes; median {float(dist[dist >= 0].median()):.1f}). Over its {len(all_de)} DE genes "
        f"the measurement departs from the average by {dev_truth:.2f} logFC per gene; ridge's prediction departs by "
        f"{dev_pred['ridge']:.2f} ({dev_pred['ridge'] / dev_truth:.0%} of the measured departure) and the neural network's by "
        f"{dev_pred['neural']:.2f} ({dev_pred['neural'] / dev_truth:.0%}). "
        f"DE-gene Pearson on this condition (from the scored run): per-dose mean {fc_scores['global_mean']:.3f}, ridge "
        f"{fc_scores['ridge']:.3f}, neural {fc_scores['neural']:.3f}.")

    amendments = _amendments(evaluation_md)
    runs = _table(["model", "runtime (min)", "peak RSS (GB)"],
                  [[escape(MODEL_TEXT.get(m, m)), f"{results[m]['runtime_s'] / 60:.1f}", f"{results[m]['peak_rss_mb'] / 1024:.1f}"] for m in models])
    mps = [m for m in models for i in results[m].get("model_info", []) if i.get("device") == "mps"]

    return {
        "verdict": verdict, "verdict_class": verdict.lower(), "claim_rule": escape(CLAIM_RULE),
        "headline_neural": f"{de['neural']['both_unseen']['estimate']:.3f}", "headline_ridge": f"{de['ridge']['both_unseen']['estimate']:.3f}",
        "headline_mean": f"{de['global_mean']['both_unseen']['estimate']:.3f}",
        "sentence_neural_ridge": escape(compare_sentence("the neural network", "ridge regression", claim_de)),
        "sentence_ridge_mean": escape(compare_sentence("ridge regression", "the per-dose mean", ridge_vs_mean)),
        "claim_de": _cell(claim_de), "claim_disc": _cell(claim_disc),
        "sentence_disc": escape(compare_sentence("the neural network", "ridge regression", claim_disc, lower_is_better=True)),
        "n_repeats": str(cfg["splits"]["repeats"]), "n_resamples": f"{cfg['bootstrap']['resamples']:,}",
        "n_amendments": str(len(amendments)),
        "amendments_html": "<ul>" + "".join(f"<li><strong>{a}</strong> — {escape(t)}</li>" for a, t in amendments) + "</ul>",
        "degradation_svg": figures.degradation_svg(deg_rows), "degradation_table": deg_table,
        "results_table": results_table, "ablation_table": ablation_table,
        "n_conditions": f"{len(cond):,}", "n_drugs": str(cond["drug"].nunique()), "n_lines": str(cond["cell_line"].nunique()),
        "n_genes": f"{len(genes):,}", "n_doses": str(cond["dose"].nunique()),
        "median_de_genes": str(int(data["de_sets"].groupby("condition_id").size().median())),
        "fc_line": escape(str(names.get(row["cell_line"], row["cell_line"]))), "fc_drug": escape(str(row.get("drug_name", row["drug"]))),
        "fc_dose": dose, "fc_n_genes": str(len(gi)), "fc_svg": figures.failure_case_svg(shown, truth, sub),
        "fc_table": fc_table, "fc_explanation": escape(explanation), "runs_table": runs,
        "mps_note": ("<p class=\"note\">" + escape(", ".join(sorted(set(MODEL_TEXT.get(m, m) for m in mps)))) +
                     " trained on an Apple GPU (MPS), which is not bit-reproducible: a rerun may differ in the last digits.</p>") if mps else "",
    }


def render(context: dict) -> str:
    template = string.Template(TEMPLATE.read_text(encoding="utf-8"))
    unused = sorted(set(context) - set(template.get_identifiers()))
    if unused:
        raise ContractError(f"case study: unused context values {unused} (computed but not on the page)")
    try:
        html = template.substitute(context)
    except KeyError as exc:
        raise ContractError(f"case study: unfilled placeholder {exc}") from exc
    text = re.sub(r"<svg.*?</svg>", "", html, flags=re.S)  # SVG path data may contain arbitrary tokens
    bad = re.findall(r"\b(nan|NaN|None)\b", re.sub(r"<[^>]+>", " ", text))
    if bad:
        raise ContractError(f"case study: page contains {sorted(set(bad))}")
    return html
