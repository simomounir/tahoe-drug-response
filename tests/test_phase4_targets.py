from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from phase4 import targets

CFG = {"de_genes": {"padj_below": 0.05, "top_n": 2}, "min_de_genes": 2}


def _expression(tmp_path, columns):
    path = tmp_path / "expr.parquet"
    pd.DataFrame({c: [1.0] for c in ["column00000", *columns]}).to_parquet(path)
    return path


def _genes():
    return pd.DataFrame({"gene": [3, 4, 5, 6], "gene_symbol": ["TP53", "KRAS", "LINC0001", "MYC"], "ensembl_id": ["e"] * 4})


def test_gene_universe_is_depmap_protein_coding_matched_to_our_symbols(tmp_path):
    expr = _expression(tmp_path, ["TP53 (7157)", "KRAS (3845)", "MYC (4609)", "MYC (99999)", "NOTOURS (1)"])
    uni = targets.gene_universe(expr, _genes())
    assert list(uni["gene_symbol"]) == ["KRAS", "MYC", "TP53"]  # sorted, LINC excluded, one MYC
    assert list(uni["gene"]) == [4, 6, 3]
    assert uni.attrs["n_duplicate_symbols"] == 1
    assert uni.attrs["n_not_in_ours"] == 1


def test_eligible_lines_by_fraction_of_treated_with_logfc():
    conditions = pd.DataFrame(
        {
            "condition_id": ["a1", "a2", "b1", "b2", "c0"],
            "cell_line": ["A", "A", "B", "B", "A"],
            "is_control": [False, False, False, False, True],
        }
    )
    assert targets.eligible_lines(conditions, {"a1", "b1", "b2"}, 0.5) == {"A", "B"}
    assert targets.eligible_lines(conditions, {"b1", "b2"}, 0.5) == {"B"}


def test_prepare_de_sets_filters_ranks_and_caps(tmp_path):
    raw = pd.DataFrame(
        {
            "condition_id": ["c"] * 5,
            "gene_symbol": ["TP53", "KRAS", "MYC", "LINC0001", "TP53X"],
            "log2FoldChange": [1.0, -3.0, 2.0, 5.0, 1.0],
            "padj": [0.01, 0.01, 0.001, 0.0001, 0.2],
        }
    )
    # a second file (= another cell line), with NaN / null rows that must be dropped
    other = pd.DataFrame(
        {
            "condition_id": ["d"] * 4,
            "gene_symbol": ["TP53", "KRAS", "MYC", "MYC"],
            "log2FoldChange": [1.0, np.nan, 2.0, None],
            "padj": [np.nan, 0.001, 0.02, 0.0001],
        }
    )
    paths = [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    raw.to_parquet(paths[0])
    other.to_parquet(paths[1])
    de = targets.prepare_de_sets(duckdb.connect(), paths, {"TP53", "KRAS", "MYC"}, CFG)
    # LINC dropped (not protein-coding), TP53X dropped (padj), MYC first (smallest padj),
    # KRAS before TP53 (same padj, larger |log2FC|), capped at top_n=2
    c = de[de["condition_id"] == "c"]
    assert list(c["gene_symbol"]) == ["MYC", "KRAS"]
    assert list(c["rank"]) == [1, 2]
    # d: only the MYC row with both values present survives
    d = de[de["condition_id"] == "d"]
    assert list(d["gene_symbol"]) == ["MYC"] and list(d["padj"]) == [pytest.approx(0.02)]
    assert list(de.columns) == ["gene_symbol", "log2FoldChange", "padj", "condition_id", "rank"]


def test_scoring_set_applies_every_rule_and_counts_exclusions():
    view = pd.DataFrame(
        {
            "condition_id": ["ok", "control", "ineligible", "gap", "no_de", "few_de"],
            "cell_line": ["A", "A", "Z", "A", "A", "A"],
            "is_control": [False, True, False, False, False, False],
            "features_complete": [True, False, True, False, True, True],
        }
    )
    de = pd.DataFrame({"condition_id": ["ok", "ok", "ineligible", "ineligible", "gap", "gap", "few_de"], "gene_symbol": list("ABCDEFG")})
    kept, excluded = targets.scoring_set(view, {"A"}, de, min_de_genes=2)
    assert list(kept["condition_id"]) == ["ok"]
    assert excluded == {"control": 1, "ineligible_line": 1, "features_incomplete": 1, "no_de_set": 1, "below_min_de_genes": 1}


def test_build_targets_dense_with_exact_zeros(tmp_path):
    logfc = tmp_path / "logfc.parquet"
    pd.DataFrame(
        {"condition_id": ["c1", "c1", "c2", "c9"], "gene": [3, 6, 4, 3], "control_condition_id": ["k"] * 4, "logfc": [0.5, -1.25, 2.0, 9.0]}
    ).to_parquet(logfc)
    uni = pd.DataFrame({"gene": [4, 6, 3], "gene_symbol": ["KRAS", "MYC", "TP53"]})
    conditions = pd.DataFrame({"condition_id": ["c2", "c1"], "cell_line": ["A", "B"]})
    m = targets.build_targets(duckdb.connect(), logfc, conditions, uni)
    assert m.dtype == np.float32
    np.testing.assert_array_equal(m, np.array([[2.0, 0.0, 0.0], [0.0, -1.25, 0.5]], dtype=np.float32))


def test_build_targets_is_deterministic(tmp_path):
    logfc = tmp_path / "logfc.parquet"
    pd.DataFrame({"condition_id": ["c1"], "gene": [3], "control_condition_id": ["k"], "logfc": [0.5]}).to_parquet(logfc)
    uni = pd.DataFrame({"gene": [3], "gene_symbol": ["TP53"]})
    conditions = pd.DataFrame({"condition_id": ["c1"], "cell_line": ["A"]})
    a = targets.build_targets(duckdb.connect(), logfc, conditions, uni)
    b = targets.build_targets(duckdb.connect(), logfc, conditions, uni)
    assert a.tobytes() == b.tobytes()
