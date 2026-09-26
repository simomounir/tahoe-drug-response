from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase3 import crosscheck

DEFAULTS = {
    "n_conditions": 200,
    "seed": 20260926,
    "padj": 0.05,
    "min_median_sign_agreement": 0.90,
    "network_cap_gb": 5,
}


def _conditions(n_treated=5, n_control=2, qc=None):
    rows = []
    for i in range(n_treated):
        rows.append({"condition_id": f"t{i}", "is_control": False, "qc_pass": True if qc is None else qc[i]})
    for i in range(n_control):
        rows.append({"condition_id": f"c{i}", "is_control": True, "qc_pass": True})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# sample_conditions
# ---------------------------------------------------------------------------


def test_sample_conditions_excludes_controls_and_failed_qc():
    qc = [True, True, False, True, True]
    conditions = _conditions(n_treated=5, n_control=2, qc=qc)
    sample = crosscheck.sample_conditions(conditions, n=10, seed=1)
    assert set(sample["condition_id"]) <= {"t0", "t1", "t3", "t4"}
    assert not sample["is_control"].any()
    assert sample["qc_pass"].all()


def test_sample_conditions_is_seeded_and_deterministic():
    conditions = _conditions(n_treated=20, n_control=3)
    a = crosscheck.sample_conditions(conditions, n=5, seed=42)
    b = crosscheck.sample_conditions(conditions, n=5, seed=42)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_sample_conditions_different_seeds_can_differ():
    conditions = _conditions(n_treated=20, n_control=3)
    a = crosscheck.sample_conditions(conditions, n=5, seed=1)
    b = crosscheck.sample_conditions(conditions, n=5, seed=2)
    assert list(a["condition_id"]) != list(b["condition_id"])


def test_sample_conditions_caps_at_pool_size():
    conditions = _conditions(n_treated=3, n_control=1)
    sample = crosscheck.sample_conditions(conditions, n=100, seed=1)
    assert len(sample) == 3


# ---------------------------------------------------------------------------
# compare_condition
# ---------------------------------------------------------------------------


def _ours(symbols, logfc):
    return pd.DataFrame({"gene_symbol": symbols, "logfc": logfc})


def _theirs(symbols, l2fc, padj):
    return pd.DataFrame({"gene_symbol": symbols, "log2FoldChange": l2fc, "padj": padj})


def test_compare_condition_perfect_agreement():
    ours = _ours(["A", "B", "C"], [1.0, -1.0, 0.5])
    theirs = _theirs(["A", "B", "C"], [2.0, -3.0, 1.0], [0.01, 0.01, 0.01])
    result = crosscheck.compare_condition(ours, theirs, padj=0.05)
    assert result["n_genes"] == 3
    assert result["sign_agreement"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)


def test_compare_condition_flipped_signs_gives_zero_agreement():
    ours = _ours(["A", "B", "C"], [1.0, -1.0, 0.5])
    theirs = _theirs(["A", "B", "C"], [-2.0, 3.0, -1.0], [0.01, 0.01, 0.01])
    result = crosscheck.compare_condition(ours, theirs, padj=0.05)
    assert result["n_genes"] == 3
    assert result["sign_agreement"] == pytest.approx(0.0)


def test_compare_condition_ignores_genes_above_padj():
    # D has a huge disagreement but padj is not significant, so it must not affect the result.
    ours = _ours(["A", "B", "D"], [1.0, -1.0, 1.0])
    theirs = _theirs(["A", "B", "D"], [2.0, -3.0, -999.0], [0.01, 0.01, 0.5])
    result = crosscheck.compare_condition(ours, theirs, padj=0.05)
    assert result["n_genes"] == 2
    assert result["sign_agreement"] == pytest.approx(1.0)


def test_compare_condition_handles_symbols_missing_on_either_side_without_crashing():
    # "D" only in ours, "E" only in theirs -- neither should appear in the comparison.
    ours = _ours(["A", "B", "D"], [1.0, -1.0, 5.0])
    theirs = _theirs(["A", "B", "E"], [2.0, -3.0, 5.0], [0.01, 0.01, 0.01])
    result = crosscheck.compare_condition(ours, theirs, padj=0.05)
    assert result["n_genes"] == 2
    assert result["sign_agreement"] == pytest.approx(1.0)


def test_compare_condition_no_de_genes_returns_nan_not_crash():
    ours = _ours(["A", "B"], [1.0, -1.0])
    theirs = _theirs(["A", "B"], [2.0, -3.0], [0.9, 0.9])
    result = crosscheck.compare_condition(ours, theirs, padj=0.05)
    assert result["n_genes"] == 0
    assert np.isnan(result["sign_agreement"])
    assert np.isnan(result["spearman"])


def test_compare_condition_constant_values_spearman_is_nan_but_no_crash():
    # No variance on one side -> Spearman undefined; must not raise.
    ours = _ours(["A", "B", "C"], [1.0, 1.0, 1.0])
    theirs = _theirs(["A", "B", "C"], [2.0, -3.0, 1.0], [0.01, 0.01, 0.01])
    result = crosscheck.compare_condition(ours, theirs, padj=0.05)
    assert result["n_genes"] == 3
    assert np.isnan(result["spearman"])


# ---------------------------------------------------------------------------
# name mapping
# ---------------------------------------------------------------------------


def test_match_drug_name_matches_after_trimming_whitespace():
    assert crosscheck.match_drug_name("Erdafitinib", {"Erdafitinib ", "Afatinib"}) == "Erdafitinib "


def test_match_drug_name_reports_unmatched_as_none():
    assert crosscheck.match_drug_name("Nonexistent Drug", {"Erdafitinib ", "Afatinib"}) is None


def test_match_drug_name_exact_match_preferred():
    assert crosscheck.match_drug_name("Afatinib", {"Afatinib", "Afatinib "}) == "Afatinib"


# ---------------------------------------------------------------------------
# config loader
# ---------------------------------------------------------------------------


REPO_FEATURES = Path(__file__).resolve().parents[1] / "configs" / "features.yaml"


def test_load_crosscheck_config_reads_the_repo_config():
    assert crosscheck.load_crosscheck_config(REPO_FEATURES) == {**DEFAULTS, "network_cap_gb": 25}


def test_load_crosscheck_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        crosscheck.load_crosscheck_config(tmp_path / "features.yaml")


# ---------------------------------------------------------------------------
# P3.1 fix round (review findings): boundaries, NaNs, zeros, duplicate symbols, dose
# ---------------------------------------------------------------------------


def _ours_by_symbol(values: dict) -> pd.DataFrame:
    return pd.DataFrame({"gene_symbol": list(values), "logfc": list(values.values())})


def _theirs_rows(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["gene_symbol", "log2FoldChange", "padj"])


def test_padj_exactly_at_threshold_is_excluded():
    result = crosscheck.compare_condition(_ours_by_symbol({"A": 1.0, "B": 1.0}), _theirs_rows([("A", 2.0, 0.05), ("B", 2.0, 0.01)]), 0.05)
    assert result["n_genes"] == 1


def test_nan_padj_and_nan_log2fc_are_excluded_not_counted_as_disagreement():
    theirs = _theirs_rows([("A", 2.0, 0.01), ("B", np.nan, 0.01), ("C", 2.0, np.nan)])
    result = crosscheck.compare_condition(_ours_by_symbol({"A": 1.0, "B": 1.0, "C": 1.0}), theirs, 0.05)
    assert result["n_genes"] == 1
    assert result["sign_agreement"] == 1.0


def test_exact_zero_agrees_only_with_zero():
    theirs = _theirs_rows([("A", 0.0, 0.01), ("B", 1.0, 0.01), ("C", -1.0, 0.01)])
    result = crosscheck.compare_condition(_ours_by_symbol({"A": 0.0, "B": 0.0, "C": -2.0}), theirs, 0.05)
    assert result["n_genes"] == 3
    assert result["sign_agreement"] == pytest.approx(2 / 3)


def test_duplicate_symbols_on_tahoe_side_are_dropped_and_counted():
    theirs = _theirs_rows([("A", 1.0, 0.01), ("A", -1.0, 0.01), ("B", 1.0, 0.01)])
    result = crosscheck.compare_condition(_ours_by_symbol({"A": 1.0, "B": 1.0}), theirs, 0.05)
    assert result["n_genes"] == 1
    assert result["n_duplicate_symbols"] == 1
    assert result["sign_agreement"] == 1.0


def test_select_condition_rows_matches_dose_and_rejects_ambiguity():
    de = pd.DataFrame(
        {
            "plate": ["1", "1", "1", "2"],
            "drug": ["X", "X", "Y ", "X"],
            "concentration": [0.05, 500.0, 5.0, 0.05],
            "concentration_unit": ["uM", "nM", "uM", "uM"],
            "gene_symbol": ["G1", "G1", "G1", "G1"],
        }
    )
    rows = crosscheck.select_condition_rows(de, plate="1", drug="X", dose_um=0.5)
    assert list(rows["concentration"]) == [500.0]
    assert crosscheck.select_condition_rows(de, plate="1", drug="Y", dose_um=5.0)["drug"].tolist() == ["Y "]
    assert crosscheck.select_condition_rows(de, plate="1", drug="X", dose_um=1.0).empty
    ambiguous = pd.concat([de, de.iloc[[0]]])
    with pytest.raises(ValueError, match="more than one"):
        crosscheck.select_condition_rows(ambiguous.assign(gene_symbol=["G1", "G1", "G1", "G1", "G1"]), plate="1", drug="X", dose_um=0.05)


def test_unknown_concentration_unit_fails_loudly():
    de = pd.DataFrame({"plate": ["1"], "drug": ["X"], "concentration": [1.0], "concentration_unit": ["mg/ml"], "gene_symbol": ["G1"]})
    with pytest.raises(ValueError, match="unit"):
        crosscheck.select_condition_rows(de, plate="1", drug="X", dose_um=1.0)
