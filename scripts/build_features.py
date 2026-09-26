"""One-command build of the Phase 3 feature layer: drugs, drug_groups, cells, depmap_pca and the
`condition_features` view, with contracts enforced at each step (spec P3.5, §4/§6/§8).

`make features` runs this script's `main()`, wiring up the real data paths. `build()` does the
actual work and takes every input path as a parameter so tests can point it at tiny synthetic
fixtures instead of the real data -- in particular the provenance requirement (spec §7): the
feature build must succeed with only `conditions.parquet` present, no `pseudobulk.parquet` or
`logfc.parquet`. `build()` is never even given a path to either of those files, only to
`conditions.parquet`, so nothing it does can depend on the response being predicted.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase1.build import peak_rss_mb
from phase1.config import load_slice_config
from phase1.stream import connect
from phase3.cells import EXPRESSION_FILE, build_cell_features, read_depmap_matrix
from phase3.config import load_features_config
from phase3.contracts import (
    validate_cell_features,
    validate_condition_features,
    validate_drug_features,
    validate_drug_groups,
)
from phase3.drugs import build_drug_features
from phase3.groups import build_drug_groups, cluster_size_report, tanimoto_matrix
from phase3.layer import create_condition_features_view

DRUG_METADATA_PATH = ROOT / "data" / "cache" / "hf" / "metadata" / "drug_metadata.parquet"
DEPMAP_DIR = ROOT / "data" / "cache" / "depmap"
TAHOE_CELLS_PATH = ROOT / "data" / "cache" / "cell_line_metadata.parquet"
CONDITIONS_PATH = ROOT / "data" / "pseudobulk" / "conditions.parquet"
SLICE_CONFIG_PATH = ROOT / "configs" / "slice.yaml"
FEATURES_DIR = ROOT / "data" / "features"
REPORT_PATH = ROOT / "reports" / "features.md"


def _slice_similarity(drugs: pd.DataFrame, slice_drugs: set[str]) -> tuple[np.ndarray, list[str]]:
    """The slice's featurizable drugs, sorted by name, and their pairwise Tanimoto matrix --
    same subset/order `phase3.groups.build_drug_groups` uses, for the cluster-size report."""
    subset = drugs[drugs["drug"].isin(slice_drugs) & drugs["featurizable"].astype(bool)].sort_values("drug")
    names = subset["drug"].tolist()
    if not names:
        return np.zeros((0, 0)), names
    counts = np.stack(subset["morgan_counts"].to_numpy())
    return tanimoto_matrix(counts), names


def build(
    conditions_path: Path | str,
    drug_metadata_path: Path | str,
    depmap_dir: Path | str,
    tahoe_cells_path: Path | str,
    slice_cfg: dict,
    features_dir: Path | str,
    report_path: Path | str,
    cfg: dict | None = None,
) -> dict:
    """Build every P3.5 output, run contracts 1-6, create the view, write `reports/features.md`.

    Returns the summary dict the report and `main()`'s stdout are built from.
    """
    conditions_path = Path(conditions_path)
    depmap_dir = Path(depmap_dir)
    features_dir = Path(features_dir)
    report_path = Path(report_path)
    features_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = cfg if cfg is not None else load_features_config()

    timings: dict[str, dict] = {}
    t_start = time.perf_counter()

    conditions = pd.read_parquet(conditions_path)
    slice_drugs = set(conditions.loc[~conditions["is_control"].astype(bool), "drug"].str.strip())

    t0 = time.perf_counter()
    drug_metadata = pd.read_parquet(drug_metadata_path)
    drugs = build_drug_features(drug_metadata, cfg)
    validate_drug_features(drugs, slice_drugs)
    drugs.to_parquet(features_dir / "drugs.parquet", index=False)
    timings["drugs"] = {"seconds": round(time.perf_counter() - t0, 2), "peak_rss_mb": peak_rss_mb()}

    t0 = time.perf_counter()
    groups = build_drug_groups(drugs, slice_drugs, cfg)
    slice_featurizable = set(drugs.loc[drugs["drug"].isin(slice_drugs) & drugs["featurizable"].astype(bool), "drug"])
    validate_drug_groups(groups, slice_featurizable)
    groups.to_parquet(features_dir / "drug_groups.parquet", index=False)
    sim, names = _slice_similarity(drugs, slice_drugs)
    cluster_sizes = cluster_size_report(sim, names, cfg["drugs"]["report_similarities"]) if names else {}
    timings["drug_groups"] = {"seconds": round(time.perf_counter() - t0, 2), "peak_rss_mb": peak_rss_mb()}

    t0 = time.perf_counter()
    tahoe_cells = pd.read_parquet(tahoe_cells_path)
    cells, loadings = build_cell_features(slice_cfg, depmap_dir, tahoe_cells, cfg)
    expression = read_depmap_matrix(depmap_dir / EXPRESSION_FILE)
    validate_cell_features(cells, loadings=loadings, expression=expression)
    cells.to_parquet(features_dir / "cells.parquet", index=False)
    loadings.to_parquet(features_dir / "depmap_pca.parquet", index=False)
    timings["cells"] = {"seconds": round(time.perf_counter() - t0, 2), "peak_rss_mb": peak_rss_mb()}

    t0 = time.perf_counter()
    con = connect()
    try:
        create_condition_features_view(con, conditions_path, features_dir)
        view = con.execute("SELECT * FROM condition_features").fetchdf()
    finally:
        con.close()
    validate_condition_features(view, conditions)
    timings["view"] = {"seconds": round(time.perf_counter() - t0, 2), "peak_rss_mb": peak_rss_mb()}

    gap_counts = view["feature_gap"].fillna("complete").value_counts().to_dict()

    summary = {
        "n_drugs": int(len(drugs)),
        "n_featurizable": int(drugs["featurizable"].sum()),
        "dropped_descriptors": list(drugs.attrs.get("dropped_descriptors", [])),
        "n_drug_groups": int(len(groups)),
        "cluster_sizes": cluster_sizes,
        "n_cells": int(len(cells)),
        "n_depmap_available": int(cells["depmap_available"].sum()),
        "n_expression_available": int(cells["expression_available"].sum()),
        "n_conditions": int(len(conditions)),
        "n_view_rows": int(len(view)),
        "n_features_complete": int(view["features_complete"].sum()),
        "feature_gap_counts": {str(k): int(v) for k, v in gap_counts.items()},
        "timings": timings,
        "peak_rss_mb": peak_rss_mb(),
        "total_seconds": round(time.perf_counter() - t_start, 2),
    }
    _write_report(report_path, summary)
    return summary


def _write_report(path: Path, summary: dict) -> None:
    lines = ["# Phase 3 features report", ""]
    lines.append(f"Drugs: {summary['n_drugs']} total, {summary['n_featurizable']} featurizable")
    dropped = ", ".join(summary["dropped_descriptors"]) or "none"
    lines.append(f"Descriptors dropped (non-finite for a featurizable drug): {dropped}")
    lines.append(f"Drug groups: {summary['n_drug_groups']} rows (the slice's featurizable drugs, one cluster each)")
    lines.append("")
    lines.append("## Cluster size distribution")
    lines.append("")
    lines.append("| threshold | n_clusters | n_singletons | largest_cluster_size |")
    lines.append("|---|---|---|---|")
    for thr, stats in summary["cluster_sizes"].items():
        lines.append(f"| {thr} | {stats['n_clusters']} | {stats['n_singletons']} | {stats['largest_cluster_size']} |")
    lines.append("")
    lines.append("## Cell lines / DepMap coverage")
    lines.append("")
    lines.append(f"Cell lines: {summary['n_cells']}")
    lines.append(f"DepMap available: {summary['n_depmap_available']} / {summary['n_cells']}")
    lines.append(f"Expression available: {summary['n_expression_available']} / {summary['n_cells']}")
    lines.append("")
    lines.append("## condition_features")
    lines.append("")
    lines.append(f"Conditions: {summary['n_conditions']}; view rows: {summary['n_view_rows']}")
    lines.append(f"features_complete: {summary['n_features_complete']} / {summary['n_view_rows']}")
    lines.append("")
    lines.append("| feature_gap | n_conditions |")
    lines.append("|---|---|")
    for gap, n in sorted(summary["feature_gap_counts"].items()):
        lines.append(f"| {gap} | {n} |")
    lines.append("")
    lines.append(f"Peak RSS: {summary['peak_rss_mb']} MB; total time: {summary['total_seconds']} s")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 3 feature layer from configs/features.yaml.")
    parser.add_argument("--conditions", type=Path, default=CONDITIONS_PATH)
    parser.add_argument("--drug-metadata", type=Path, default=DRUG_METADATA_PATH)
    parser.add_argument("--depmap-dir", type=Path, default=DEPMAP_DIR)
    parser.add_argument("--tahoe-cells", type=Path, default=TAHOE_CELLS_PATH)
    parser.add_argument("--slice-config", type=Path, default=SLICE_CONFIG_PATH)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    slice_cfg = load_slice_config(args.slice_config)
    summary = build(
        args.conditions,
        args.drug_metadata,
        args.depmap_dir,
        args.tahoe_cells,
        slice_cfg,
        args.features_dir,
        args.report,
    )

    print(f"drugs: {summary['n_drugs']} ({summary['n_featurizable']} featurizable)")
    print(f"drug_groups: {summary['n_drug_groups']}")
    print(f"cells: {summary['n_cells']} (depmap {summary['n_depmap_available']}, expression {summary['n_expression_available']})")
    print(f"condition_features: {summary['n_view_rows']} rows, {summary['n_features_complete']} complete")
    for gap, n in sorted(summary["feature_gap_counts"].items()):
        print(f"  feature_gap={gap}: {n}")
    print(f"Contracts: passed. Outputs in {args.features_dir}, report at {args.report}")
    print(f"Peak RSS: {summary['peak_rss_mb']} MB, total time: {summary['total_seconds']} s")


if __name__ == "__main__":
    main()
