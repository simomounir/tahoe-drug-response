from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase1.contracts import ContractError
from phase3.cells import fit_pca, pca_to_loadings_frame, project
from phase3.contracts import (
    validate_cell_features,
    validate_condition_features,
    validate_drug_features,
    validate_drug_groups,
)

# ---------------------------------------------------------------------------
# validate_drug_features (contracts 1 + 2)
# ---------------------------------------------------------------------------


def _drugs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "drug": ["A", "B", "C"],
            "smiles_input": ["CCO", "bad", "CCN"],
            "smiles_parent": ["CCO", None, "CCN"],
            "cleaning_log": ["cleanup,normalize", "parse_failed", "cleanup,normalize"],
            "featurizable": [True, False, True],
            "exclusion_reason": [None, "parse_failed", None],
            "morgan_counts": [np.zeros(4, dtype=np.uint8), None, np.ones(4, dtype=np.uint8)],
            "MolWt": [46.07, np.nan, 45.08],
        }
    )


def test_validate_drug_features_passes_for_a_well_formed_table():
    validate_drug_features(_drugs_frame(), {"A", "B", "C"})


def test_validate_drug_features_fails_when_a_drug_has_neither_flag():
    df = _drugs_frame()
    df.loc[df["drug"] == "B", "exclusion_reason"] = None  # featurizable=False AND no reason
    with pytest.raises(ContractError, match="neither featurizable nor exclusion_reason"):
        validate_drug_features(df, {"A", "B", "C"})


def test_validate_drug_features_fails_on_null_feature_column_for_featurizable_drug():
    df = _drugs_frame()
    df.loc[df["drug"] == "A", "MolWt"] = np.nan  # A is featurizable -- this is a contract 2 violation
    with pytest.raises(ContractError, match="MolWt"):
        validate_drug_features(df, {"A", "B", "C"})


def test_validate_drug_features_fails_when_a_slice_drug_is_missing_entirely():
    with pytest.raises(ContractError, match="missing from drugs table"):
        validate_drug_features(_drugs_frame(), {"A", "B", "C", "Z"})


# ---------------------------------------------------------------------------
# validate_drug_groups (contract 3)
# ---------------------------------------------------------------------------


def _groups_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "drug": ["A", "C"],
            "cluster_id": [0, 1],
            "cluster_size": [1, 1],
            "murcko_scaffold": ["", ""],
            "nearest_drug": ["C", "A"],
            "nearest_tanimoto": [0.1, 0.1],
        }
    )


def test_validate_drug_groups_passes_for_exact_match():
    validate_drug_groups(_groups_frame(), {"A", "C"})


def test_validate_drug_groups_fails_when_a_featurizable_drug_is_missing():
    with pytest.raises(ContractError, match=r"missing=\['D'\]"):
        validate_drug_groups(_groups_frame(), {"A", "C", "D"})


def test_validate_drug_groups_fails_when_a_drug_has_more_than_one_cluster():
    df = pd.concat([_groups_frame(), pd.DataFrame([{"drug": "A", "cluster_id": 2, "cluster_size": 1, "murcko_scaffold": "", "nearest_drug": "C", "nearest_tanimoto": 0.1}])], ignore_index=True)
    with pytest.raises(ContractError, match="more than one cluster"):
        validate_drug_groups(df, {"A", "C"})


# ---------------------------------------------------------------------------
# validate_cell_features (contracts 2, 4, 6)
# ---------------------------------------------------------------------------


def _cells_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_line": ["L1", "L2", "L3"],
            "depmap_id": ["ACH-1", None, "ACH-3"],
            "depmap_release": ["24Q4", "24Q4", "24Q4"],
            "depmap_available": [True, False, True],
            "depmap_absent_reason": [None, "no DepMap entry", None],
            "expression_available": [True, False, False],
            "expression_source": ["ACH-1", None, None],
            "pc_1": [0.5, np.nan, np.nan],
            "pc_2": [0.1, np.nan, np.nan],
            "dmg_G1": [0.0, np.nan, 1.0],
            "hot_G1": [0.0, np.nan, 0.0],
            "drv_G1": [1.0, 0.0, 0.0],
        }
    )


def test_validate_cell_features_passes_with_declared_gaps():
    validate_cell_features(_cells_frame())


def test_validate_cell_features_fails_depmap_unavailable_without_reason():
    df = _cells_frame()
    df.loc[df["cell_line"] == "L2", "depmap_absent_reason"] = None
    with pytest.raises(ContractError, match="without a depmap_absent_reason"):
        validate_cell_features(df)


def test_validate_cell_features_fails_undeclared_null_pc():
    df = _cells_frame()
    df.loc[df["cell_line"] == "L1", "pc_1"] = np.nan  # expression_available=True, so this is a real gap
    with pytest.raises(ContractError, match=r"pc_1 null for 1 line"):
        validate_cell_features(df)


def test_validate_cell_features_fails_undeclared_null_dmg():
    df = _cells_frame()
    df.loc[df["cell_line"] == "L1", "dmg_G1"] = np.nan  # depmap_available=True, so this is a real gap
    with pytest.raises(ContractError, match=r"dmg_G1 null for 1 line"):
        validate_cell_features(df)


def test_validate_cell_features_fails_null_driver_flag():
    df = _cells_frame()
    df.loc[df["cell_line"] == "L2", "drv_G1"] = np.nan  # drv_ never depends on DepMap coverage
    with pytest.raises(ContractError, match="drv_G1"):
        validate_cell_features(df)


def _synthetic_panel(n_lines: int = 6, n_genes: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes)]
    lines = [f"ACH-{i:06d}" for i in range(n_lines)]
    return pd.DataFrame(rng.normal(size=(n_lines, n_genes)), index=lines, columns=genes)


def test_validate_cell_features_contract6_passes_when_pc_reproduces_from_loadings():
    panel = _synthetic_panel()
    pca = fit_pca(panel, k=2)
    loadings = pca_to_loadings_frame(pca)
    coords = project(pca, panel.loc[["ACH-000000"]])[0]

    cells = pd.DataFrame(
        {
            "cell_line": ["L1"],
            "depmap_id": ["ACH-000000"],
            "depmap_release": ["24Q4"],
            "depmap_available": [True],
            "depmap_absent_reason": [None],
            "expression_available": [True],
            "expression_source": ["ACH-000000"],
            "pc_1": [coords[0]],
            "pc_2": [coords[1]],
            "drv_G1": [0.0],
        }
    )
    validate_cell_features(cells, loadings=loadings, expression=panel)


def test_validate_cell_features_contract6_fails_when_pc_does_not_reproduce():
    panel = _synthetic_panel()
    pca = fit_pca(panel, k=2)
    loadings = pca_to_loadings_frame(pca)

    cells = pd.DataFrame(
        {
            "cell_line": ["L1"],
            "depmap_id": ["ACH-000000"],
            "depmap_release": ["24Q4"],
            "depmap_available": [True],
            "depmap_absent_reason": [None],
            "expression_available": [True],
            "expression_source": ["ACH-000000"],
            "pc_1": [999.0],  # wrong on purpose
            "pc_2": [0.0],
            "drv_G1": [0.0],
        }
    )
    with pytest.raises(ContractError, match="does not reproduce"):
        validate_cell_features(cells, loadings=loadings, expression=panel)


# ---------------------------------------------------------------------------
# validate_condition_features (contracts 2, 5)
# ---------------------------------------------------------------------------


def _view_and_conditions() -> tuple[pd.DataFrame, pd.DataFrame]:
    view = pd.DataFrame(
        [
            {
                "condition_id": "c1", "cell_line": "L1", "drug": "A", "is_control": False,
                "log10_dose_um": -1.0, "smiles_input": "CCO", "smiles_parent": "CCO",
                "cleaning_log": "cleanup", "featurizable": True, "exclusion_reason": None,
                "morgan_counts": np.zeros(4, dtype=np.uint8), "MolWt": 46.0,
                "cluster_id": 0, "cluster_size": 1, "murcko_scaffold": "", "nearest_drug": "C", "nearest_tanimoto": 0.1,
                "depmap_id": "ACH-1", "depmap_release": "24Q4", "depmap_available": True, "depmap_absent_reason": None,
                "expression_available": True, "expression_source": "ACH-1", "pc_1": 0.5,
                "dmg_G1": 0.0, "hot_G1": 0.0, "drv_G1": 1.0,
                "feature_gap": None, "features_complete": True,
            },
            {
                "condition_id": "c2", "cell_line": "L1", "drug": "DMSO_TF", "is_control": True,
                "log10_dose_um": np.nan, "smiles_input": None, "smiles_parent": None,
                "cleaning_log": None, "featurizable": None, "exclusion_reason": None,
                "morgan_counts": None, "MolWt": np.nan,
                "cluster_id": np.nan, "cluster_size": np.nan, "murcko_scaffold": None, "nearest_drug": None, "nearest_tanimoto": np.nan,
                "depmap_id": "ACH-1", "depmap_release": "24Q4", "depmap_available": True, "depmap_absent_reason": None,
                "expression_available": True, "expression_source": "ACH-1", "pc_1": 0.5,
                "dmg_G1": 0.0, "hot_G1": 0.0, "drv_G1": 1.0,
                "feature_gap": "control", "features_complete": False,
            },
            {
                "condition_id": "c3", "cell_line": "L2", "drug": "A", "is_control": False,
                "log10_dose_um": -1.0, "smiles_input": "CCO", "smiles_parent": "CCO",
                "cleaning_log": "cleanup", "featurizable": True, "exclusion_reason": None,
                "morgan_counts": np.zeros(4, dtype=np.uint8), "MolWt": 46.0,
                "cluster_id": 0, "cluster_size": 1, "murcko_scaffold": "", "nearest_drug": "C", "nearest_tanimoto": 0.1,
                "depmap_id": "ACH-2", "depmap_release": "24Q4", "depmap_available": True, "depmap_absent_reason": None,
                "expression_available": False, "expression_source": None, "pc_1": np.nan,
                "dmg_G1": 1.0, "hot_G1": 0.0, "drv_G1": 0.0,
                "feature_gap": "no_expression", "features_complete": False,
            },
        ]
    )
    conditions = view[["condition_id", "cell_line", "drug", "is_control"]].copy()
    return view, conditions


def test_validate_condition_features_passes_including_control_and_gap_rows():
    view, conditions = _view_and_conditions()
    validate_condition_features(view, conditions)


def test_validate_condition_features_fails_on_row_count_mismatch():
    view, conditions = _view_and_conditions()
    view = view.iloc[:2]
    with pytest.raises(ContractError, match="one per condition"):
        validate_condition_features(view, conditions)


def test_validate_condition_features_fails_on_null_feature_for_complete_row():
    view, conditions = _view_and_conditions()
    view.loc[view["condition_id"] == "c1", "MolWt"] = np.nan  # c1 is features_complete=True
    with pytest.raises(ContractError, match="MolWt"):
        validate_condition_features(view, conditions)


def test_validate_condition_features_fails_on_inconsistent_complete_and_gap():
    view, conditions = _view_and_conditions()
    view.loc[view["condition_id"] == "c1", "features_complete"] = False  # but feature_gap stays null
    with pytest.raises(ContractError, match="disagree"):
        validate_condition_features(view, conditions)


def test_validate_drug_groups_rejects_null_cluster_or_nearest_values():
    # Final review: the phase 4 splitter reads drug_groups.parquet directly, not via the view.
    groups = _groups_frame()
    drugs = set(groups["drug"])
    bad_cluster = groups.astype({"cluster_id": "float"}).copy()
    bad_cluster.loc[bad_cluster.index[0], "cluster_id"] = np.nan
    with pytest.raises(ContractError, match="cluster_id"):
        validate_drug_groups(bad_cluster, drugs)
    bad_nearest = groups.copy()
    bad_nearest.loc[bad_nearest.index[0], "nearest_tanimoto"] = np.nan
    with pytest.raises(ContractError, match="nearest_tanimoto"):
        validate_drug_groups(bad_nearest, drugs)
