from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase1.contracts import ContractError
from phase3.cells import (
    _canonical_svd_sign,
    build_cell_features,
    fit_pca,
    load_features_config,
    loadings_frame_to_pca,
    mutation_genes,
    pca_to_loadings_frame,
    project,
    read_depmap_matrix,
    select_driver_genes,
)


# ---------------------------------------------------------------------------
# read_depmap_matrix: CSV layout (review focus #3)
# ---------------------------------------------------------------------------


def test_read_depmap_matrix_csv_layout(tmp_path: Path):
    """Unnamed first column = ACH ID; gene columns `SYMBOL (ENTREZ)` parsed to plain SYMBOL."""
    csv_text = (
        ",TP53 (7157),KRAS (3845),EGFR (1956),MYC (4609)\n"
        "ACH-000001,1.0,2.5,0.0,3.3\n"
        "ACH-000002,4.1,0.0,5.5,1.2\n"
        "ACH-000003,0.0,0.0,0.0,0.0\n"
    )
    path = tmp_path / "matrix.csv"
    path.write_text(csv_text)

    df = read_depmap_matrix(path)

    assert list(df.index) == ["ACH-000001", "ACH-000002", "ACH-000003"]
    assert list(df.columns) == ["TP53", "KRAS", "EGFR", "MYC"]
    assert df.loc["ACH-000001", "KRAS"] == pytest.approx(2.5)
    assert df.dtypes.unique().tolist() == [np.float32]


def test_read_depmap_matrix_parquet_roundtrip(tmp_path: Path):
    """Same parsing works on the Parquet fetch_depmap.py converts the CSV to (positional index column)."""
    raw = pd.DataFrame(
        {"": ["ACH-1", "ACH-2"], "TP53 (7157)": [1.0, 2.0], "KRAS (3845)": [0.0, 9.0]}
    )
    path = tmp_path / "matrix.parquet"
    raw.to_parquet(path)

    df = read_depmap_matrix(path)

    assert list(df.index) == ["ACH-1", "ACH-2"]
    assert list(df.columns) == ["TP53", "KRAS"]


# ---------------------------------------------------------------------------
# fit_pca / project
# ---------------------------------------------------------------------------


def _synthetic_panel(n_lines: int = 10, n_genes: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes)]
    lines = [f"ACH-{i:06d}" for i in range(n_lines)]
    data = rng.normal(size=(n_lines, n_genes))
    return pd.DataFrame(data, index=lines, columns=genes)


def test_fit_pca_zero_variance_gene_gives_finite_projections():
    panel = _synthetic_panel()
    panel["ZEROVAR"] = 7.0  # constant across every line -> std would be 0 without the guard

    pca = fit_pca(panel, k=3)
    assert np.all(pca["std"] > 0), "zero-std genes must be forced to std = 1"

    coords = project(pca, panel)
    assert np.all(np.isfinite(coords))


def test_project_panel_member_matches_independent_svd():
    panel = _synthetic_panel(n_lines=8, n_genes=5, seed=1)
    k = 3

    pca = fit_pca(panel, k=k)
    coords = project(pca, panel)

    # Re-derive expected scores independently of fit_pca's own internals, applying the same
    # deterministic sign convention fit_pca uses (raw SVD sign is otherwise arbitrary per
    # component -- see test_canonical_svd_sign_fixes_ambiguity for that rule in isolation).
    X = panel.to_numpy(dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    Xs = (X - mean) / std
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    canonical_Vt = _canonical_svd_sign(Vt[:k])
    idx = np.argmax(np.abs(Vt[:k]), axis=1)
    signs = np.sign(Vt[:k][np.arange(k), idx])
    signs[signs == 0] = 1.0
    expected = Xs @ canonical_Vt.T

    np.testing.assert_allclose(coords, expected, atol=1e-8)
    np.testing.assert_allclose(coords, (U[:, :k] * signs[None, :]) * S[:k], atol=1e-8)


def test_fit_pca_variance_explained_sums_to_at_most_one():
    panel = _synthetic_panel(n_lines=12, n_genes=8, seed=2)
    pca = fit_pca(panel, k=5)
    assert pca["variance_explained"].shape == (5,)
    assert 0.0 <= pca["variance_explained"].sum() <= 1.0 + 1e-9
    assert np.all(np.diff(pca["variance_explained"]) <= 1e-9)  # non-increasing


def test_canonical_svd_sign_fixes_ambiguity():
    """Unit test of the sign-fixing rule itself: largest-magnitude entry per row becomes positive."""
    Vt = np.array([[0.1, -0.9, 0.2], [-0.5, 0.3, -0.1]])

    fixed = _canonical_svd_sign(Vt)

    assert fixed[0, 1] > 0  # row 0's largest-magnitude entry was -0.9 -> flipped positive
    assert fixed[1, 0] > 0  # row 1's largest-magnitude entry was -0.5 -> flipped positive
    np.testing.assert_allclose(_canonical_svd_sign(fixed), fixed)  # idempotent
    # The exact scenario the review flagged: an upstream SVD call that happened to flip a
    # component's sign (different LAPACK build/BLAS backend) must canonicalise to the same result.
    np.testing.assert_allclose(_canonical_svd_sign(-Vt), fixed)


def test_fit_pca_row_permutation_invariant_loadings():
    """Reordering panel rows (as a real rebuild with a different library might effectively induce a
    different internal sign choice) must still yield identical stored loadings/projections."""
    panel = _synthetic_panel(n_lines=10, n_genes=6, seed=5)
    shuffled = panel.loc[list(reversed(panel.index))]

    pca1 = fit_pca(panel, k=4)
    pca2 = fit_pca(shuffled, k=4)

    np.testing.assert_allclose(pca1["loadings"], pca2["loadings"], atol=1e-8)

    row = panel.loc[[panel.index[0]]]
    np.testing.assert_allclose(project(pca1, row), project(pca2, row), atol=1e-8)


def test_loadings_roundtrip_through_parquet(tmp_path: Path):
    panel = _synthetic_panel(n_lines=9, n_genes=7, seed=3)
    pca = fit_pca(panel, k=4)
    expected = project(pca, panel)

    loadings_df = pca_to_loadings_frame(pca)
    path = tmp_path / "depmap_pca.parquet"
    loadings_df.to_parquet(path)

    reloaded = pd.read_parquet(path)
    restored_pca = loadings_frame_to_pca(reloaded)
    restored = project(restored_pca, panel)

    np.testing.assert_allclose(restored, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# mutation_genes
# ---------------------------------------------------------------------------


def test_mutation_genes_frequency_rule():
    # 10 lines; GENE_HI damaged in 6 (0.6), GENE_MID in 2 (0.2), GENE_LO in 1 (0.1)
    lines = [f"ACH-{i:06d}" for i in range(10)]
    damaging = pd.DataFrame(
        {
            "GENE_HI": [1] * 6 + [0] * 4,
            "GENE_MID": [1] * 2 + [0] * 8,
            "GENE_LO": [1] * 1 + [0] * 9,
        },
        index=lines,
    )

    assert mutation_genes(damaging, min_frequency=0.05) == ["GENE_HI", "GENE_LO", "GENE_MID"]
    assert mutation_genes(damaging, min_frequency=0.2) == ["GENE_HI", "GENE_MID"]
    assert mutation_genes(damaging, min_frequency=0.5) == ["GENE_HI"]
    assert mutation_genes(damaging, min_frequency=0.9) == []


# ---------------------------------------------------------------------------
# select_driver_genes
# ---------------------------------------------------------------------------


def _tahoe_cells_fixture() -> pd.DataFrame:
    rows = [
        ("A", "CVCL_A", "TP53"),
        ("A", "CVCL_A", "KRAS"),
        ("B", "CVCL_B", "TP53"),
        ("C", "CVCL_C", "TP53"),
        ("D", "CVCL_D", "KRAS"),
        ("E", "CVCL_E", "RARE_ONE_LINE"),
        ("F", "CVCL_F", "NOT_IN_SLICE_GENE"),
    ]
    return pd.DataFrame(rows, columns=["cell_name", "Cell_ID_Cellosaur", "Driver_Gene_Symbol"])


def test_select_driver_genes_min_lines_rule():
    tahoe_cells = _tahoe_cells_fixture()
    selected = ["CVCL_A", "CVCL_B", "CVCL_C", "CVCL_D", "CVCL_E"]  # excludes CVCL_F

    genes = select_driver_genes(tahoe_cells, selected, min_lines=2)

    assert genes == ["KRAS", "TP53"]  # TP53 in 3 lines, KRAS in 2; RARE_ONE_LINE in 1 -> excluded


# ---------------------------------------------------------------------------
# build_cell_features: end to end on tiny synthetic DepMap files
# ---------------------------------------------------------------------------


def _write_depmap_fixture(depmap_dir: Path, lines: list[str], genes: list[str], seed: int = 0) -> None:
    depmap_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    expr = pd.DataFrame(rng.normal(size=(len(lines), len(genes))), index=lines, columns=[f"{g} (1)" for g in genes])
    expr.index.name = ""
    expr.reset_index().to_parquet(depmap_dir / "OmicsExpressionProteinCodingGenesTPMLogp1.parquet")

    # damaging: DAMAGED_GENE mutated in every line (freq 1.0); RARE_GENE in only one
    dmg = pd.DataFrame(
        {"DAMAGED_GENE (1)": [1] * len(lines), "RARE_GENE (2)": [1] + [0] * (len(lines) - 1)},
        index=lines,
    )
    dmg.index.name = ""
    dmg.reset_index().to_parquet(depmap_dir / "OmicsSomaticMutationsMatrixDamaging.parquet")

    # hotspot: only DAMAGED_GENE present, hot in the first line only
    hot = pd.DataFrame({"DAMAGED_GENE (1)": [1] + [0] * (len(lines) - 1)}, index=lines)
    hot.index.name = ""
    hot.reset_index().to_parquet(depmap_dir / "OmicsSomaticMutationsMatrixHotspot.parquet")


def _slice_cfg() -> dict:
    return {
        "selected_cell_lines": ["CVCL_A", "CVCL_B", "CVCL_ABSENT"],
        "cell_line_info": {
            "CVCL_A": {"depmap_id": "ACH-000001", "tissue": "Lung"},
            "CVCL_B": {"depmap_id": "ACH-000002", "tissue": "Bowel"},
            "CVCL_ABSENT": {"depmap_id": None, "depmap_absent": "not in DepMap (declared)", "tissue": "Pancreas"},
        },
    }


def _tiny_cfg() -> dict:
    return {
        "depmap": {"release": "24Q4"},
        "cells": {"pca_components": 2, "mutation_min_frequency": 0.05, "driver_min_lines": 2},
    }


def test_build_cell_features_end_to_end(tmp_path: Path):
    genes = ["DAMAGED_GENE", "RARE_GENE", "OTHER1", "OTHER2", "OTHER3"]
    lines = ["ACH-000001", "ACH-000002", "ACH-000003", "ACH-000004"]
    depmap_dir = tmp_path / "depmap"
    _write_depmap_fixture(depmap_dir, lines, genes)

    tahoe_cells = pd.DataFrame(
        [
            ("A", "CVCL_A", "TP53"),
            ("B", "CVCL_B", "TP53"),
            ("Absent", "CVCL_ABSENT", "TP53"),
        ],
        columns=["cell_name", "Cell_ID_Cellosaur", "Driver_Gene_Symbol"],
    )

    cells_df, loadings_df = build_cell_features(_slice_cfg(), depmap_dir, tahoe_cells, _tiny_cfg())

    assert list(cells_df["cell_line"]) == ["CVCL_A", "CVCL_B", "CVCL_ABSENT"]
    assert set(
        ["pc_1", "pc_2", "dmg_DAMAGED_GENE", "hot_DAMAGED_GENE", "drv_TP53", "expression_available", "expression_source"]
    ).issubset(cells_df.columns)

    absent_row = cells_df[cells_df["cell_line"] == "CVCL_ABSENT"].iloc[0]
    assert absent_row["depmap_available"] == False  # noqa: E712
    assert pd.isna(absent_row["depmap_id"])  # pandas 3's string dtype represents None as NaN
    assert absent_row["depmap_absent_reason"] == "not in DepMap (declared)"
    assert absent_row["expression_available"] == False  # noqa: E712
    assert pd.isna(absent_row["expression_source"])
    assert pd.isna(absent_row["pc_1"]) and pd.isna(absent_row["dmg_DAMAGED_GENE"])
    assert absent_row["drv_TP53"] == 1.0  # driver flags come from Tahoe, independent of DepMap coverage

    present = cells_df[cells_df["depmap_available"]]
    assert present[["pc_1", "pc_2", "dmg_DAMAGED_GENE", "hot_DAMAGED_GENE"]].notna().all().all()
    assert np.isfinite(present[["pc_1", "pc_2"]].to_numpy()).all()

    # damaging frequency in the 4-line fixture panel: DAMAGED_GENE = 1.0, RARE_GENE = 0.25 -> both >= 0.05
    assert set(loadings_df["gene"]) == set(genes)
    assert {"gene", "mean", "std", "pc_1", "pc_2"}.issubset(loadings_df.columns)


def test_build_cell_features_missing_undeclared_line_raises_contract_error(tmp_path: Path):
    genes = ["G1", "G2"]
    lines = ["ACH-000001"]  # only one line in the panel
    depmap_dir = tmp_path / "depmap"
    _write_depmap_fixture(depmap_dir, lines, genes)

    slice_cfg = {
        "selected_cell_lines": ["CVCL_A", "CVCL_MISSING"],
        "cell_line_info": {
            "CVCL_A": {"depmap_id": "ACH-000001", "tissue": "Lung"},
            "CVCL_MISSING": {"depmap_id": "ACH-999999", "tissue": "Bowel"},  # not declared absent, not in panel
        },
    }
    tahoe_cells = pd.DataFrame(columns=["cell_name", "Cell_ID_Cellosaur", "Driver_Gene_Symbol"])

    with pytest.raises(ContractError, match="ACH-999999"):
        build_cell_features(slice_cfg, depmap_dir, tahoe_cells, _tiny_cfg())


def _write_depmap_fixture_partial_expression(
    depmap_dir: Path, expression_ids: list[str], mutation_ids: list[str], genes: list[str], seed: int = 0
) -> None:
    """Like _write_depmap_fixture, but only `expression_ids` (a subset of `mutation_ids`) get an
    expression row -- for testing lines whose own ID has mutation data but no expression row."""
    depmap_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    expr = pd.DataFrame(
        rng.normal(size=(len(expression_ids), len(genes))), index=expression_ids, columns=[f"{g} (1)" for g in genes]
    )
    expr.index.name = ""
    expr.reset_index().to_parquet(depmap_dir / "OmicsExpressionProteinCodingGenesTPMLogp1.parquet")

    dmg = pd.DataFrame(
        {"DAMAGED_GENE (1)": [1] * len(mutation_ids), "RARE_GENE (2)": [1] + [0] * (len(mutation_ids) - 1)},
        index=mutation_ids,
    )
    dmg.index.name = ""
    dmg.reset_index().to_parquet(depmap_dir / "OmicsSomaticMutationsMatrixDamaging.parquet")

    hot = pd.DataFrame({"DAMAGED_GENE (1)": [1] + [0] * (len(mutation_ids) - 1)}, index=mutation_ids)
    hot.index.name = ""
    hot.reset_index().to_parquet(depmap_dir / "OmicsSomaticMutationsMatrixHotspot.parquet")


def test_build_cell_features_expression_proxy_and_absent(tmp_path: Path):
    """Owner decision (fix round 1): a line missing its own 24Q4 expression row is either proxied
    by another model's expression (mutation flags stay its own) or declared expression-absent
    (null PCs, mutation flags kept)."""
    genes = ["DAMAGED_GENE", "RARE_GENE", "OTHER1", "OTHER2", "OTHER3"]
    # ACH-000001 = CVCL_A, has its own expression. ACH-000003 = the proxy model (has expression).
    # ACH-000002 = CVCL_PROXY's own ID (mutation data only). ACH-000004 = CVCL_EXPABSENT's own ID
    # (mutation data only).
    expression_ids = ["ACH-000001", "ACH-000003"]
    mutation_ids = ["ACH-000001", "ACH-000002", "ACH-000003", "ACH-000004"]
    depmap_dir = tmp_path / "depmap"
    _write_depmap_fixture_partial_expression(depmap_dir, expression_ids, mutation_ids, genes)

    slice_cfg = {
        "selected_cell_lines": ["CVCL_A", "CVCL_PROXY", "CVCL_EXPABSENT"],
        "cell_line_info": {
            "CVCL_A": {"depmap_id": "ACH-000001", "tissue": "Lung"},
            "CVCL_PROXY": {"depmap_id": "ACH-000002", "tissue": "Liver"},
            "CVCL_EXPABSENT": {"depmap_id": "ACH-000004", "tissue": "Bowel"},
        },
    }
    cfg = {
        "depmap": {"release": "24Q4"},
        "cells": {
            "pca_components": 2,
            "mutation_min_frequency": 0.05,
            "driver_min_lines": 2,
            "expression_proxy": {"CVCL_PROXY": {"model": "ACH-000003", "reason": "test proxy"}},
            "expression_absent": {"CVCL_EXPABSENT": "no expression, test"},
        },
    }
    tahoe_cells = pd.DataFrame(columns=["cell_name", "Cell_ID_Cellosaur", "Driver_Gene_Symbol"])

    cells_df, _ = build_cell_features(slice_cfg, depmap_dir, tahoe_cells, cfg)
    by_line = cells_df.set_index("cell_line")

    proxy_row = by_line.loc["CVCL_PROXY"]
    assert proxy_row["expression_available"] == True  # noqa: E712
    assert proxy_row["expression_source"] == "ACH-000003"
    assert np.isfinite(proxy_row[["pc_1", "pc_2"]].to_numpy(dtype=float)).all()
    assert proxy_row["dmg_DAMAGED_GENE"] == 1.0  # ACH-000002's own damaging row, not the proxy's
    assert proxy_row["depmap_available"] == True  # noqa: E712

    a_row = by_line.loc["CVCL_A"]
    assert not np.isclose(proxy_row["pc_1"], a_row["pc_1"])  # not accidentally reusing CVCL_A's row

    absent_row = by_line.loc["CVCL_EXPABSENT"]
    assert absent_row["expression_available"] == False  # noqa: E712
    assert pd.isna(absent_row["expression_source"])
    assert pd.isna(absent_row["pc_1"])
    assert absent_row["dmg_DAMAGED_GENE"] == 1.0  # ACH-000004's own damaging row, kept per the owner decision
    assert absent_row["depmap_available"] == True  # noqa: E712 -- it has a DepMap ID, just no expression


def test_build_cell_features_proxy_model_without_expression_raises(tmp_path: Path):
    """A proxy model that itself lacks a 24Q4 expression row is a contract failure, not imputation."""
    genes = ["G1", "G2"]
    expression_ids = ["ACH-000001"]
    mutation_ids = ["ACH-000001", "ACH-000002"]
    depmap_dir = tmp_path / "depmap"
    _write_depmap_fixture_partial_expression(depmap_dir, expression_ids, mutation_ids, genes)

    slice_cfg = {
        "selected_cell_lines": ["CVCL_A", "CVCL_PROXY"],
        "cell_line_info": {
            "CVCL_A": {"depmap_id": "ACH-000001", "tissue": "Lung"},
            "CVCL_PROXY": {"depmap_id": "ACH-000002", "tissue": "Liver"},
        },
    }
    cfg = {
        "depmap": {"release": "24Q4"},
        "cells": {
            "pca_components": 1,
            "mutation_min_frequency": 0.05,
            "driver_min_lines": 2,
            "expression_proxy": {"CVCL_PROXY": {"model": "ACH-999999", "reason": "proxy itself has no expression"}},
        },
    }
    tahoe_cells = pd.DataFrame(columns=["cell_name", "Cell_ID_Cellosaur", "Driver_Gene_Symbol"])

    with pytest.raises(ContractError, match="ACH-999999"):
        build_cell_features(slice_cfg, depmap_dir, tahoe_cells, cfg)


def test_build_cell_features_declared_absent_line_does_not_raise(tmp_path: Path):
    """hTERT-HPNE-style declared absence must not trip contract 4."""
    genes = ["G1", "G2"]
    lines = ["ACH-000001"]
    depmap_dir = tmp_path / "depmap"
    _write_depmap_fixture(depmap_dir, lines, genes)

    slice_cfg = {
        "selected_cell_lines": ["CVCL_A", "CVCL_ABSENT"],
        "cell_line_info": {
            "CVCL_A": {"depmap_id": "ACH-000001", "tissue": "Lung"},
            "CVCL_ABSENT": {"depmap_id": None, "depmap_absent": "declared", "tissue": "Pancreas"},
        },
    }
    tahoe_cells = pd.DataFrame(columns=["cell_name", "Cell_ID_Cellosaur", "Driver_Gene_Symbol"])

    cells_df, _ = build_cell_features(slice_cfg, depmap_dir, tahoe_cells, _tiny_cfg())
    assert len(cells_df) == 2


# ---------------------------------------------------------------------------
# load_features_config
# ---------------------------------------------------------------------------


def test_cells_uses_the_shared_config_loader():
    from phase3 import config

    assert load_features_config is config.load_features_config


# ---------------------------------------------------------------------------
# Final review fix pass: DepMap mutation matrices hold 0/1/2 (none/het/hom)
# ---------------------------------------------------------------------------


def test_mutation_genes_counts_lines_not_alleles():
    # 20 lines; GENE_HOM is homozygous (2) in 1 line = 5% of lines, GENE_HET heterozygous in 1 line.
    # Averaging raw values would give GENE_HOM 10% and select it at 0.06; counting lines must not.
    lines = [f"ACH-{i:06d}" for i in range(20)]
    damaging = pd.DataFrame({"GENE_HOM": [2] + [0] * 19, "GENE_HET": [1] + [0] * 19}, index=lines)
    assert mutation_genes(damaging, min_frequency=0.06) == []
    assert mutation_genes(damaging, min_frequency=0.05) == ["GENE_HET", "GENE_HOM"]


def test_mutation_flags_are_binary(tmp_path: Path):
    genes = ["DAMAGED_GENE", "RARE_GENE", "OTHER1", "OTHER2", "OTHER3"]
    lines = ["ACH-000001", "ACH-000002", "ACH-000003", "ACH-000004"]
    depmap_dir = tmp_path / "depmap"
    _write_depmap_fixture(depmap_dir, lines, genes)
    for name in ("OmicsSomaticMutationsMatrixDamaging.parquet", "OmicsSomaticMutationsMatrixHotspot.parquet"):
        df = pd.read_parquet(depmap_dir / name)
        df.iloc[0, 1:] = df.iloc[0, 1:] * 2  # line ACH-000001 homozygous for every mutated gene
        df.to_parquet(depmap_dir / name)
    cells_df, _ = build_cell_features(_slice_cfg(), depmap_dir, _tahoe_cells_fixture(), _tiny_cfg())
    flags = cells_df[[c for c in cells_df.columns if c.startswith(("dmg_", "hot_"))]]
    values = flags.to_numpy(dtype=float).ravel()
    assert set(values[~np.isnan(values)]) <= {0.0, 1.0}
    assert cells_df.set_index("cell_line").loc["CVCL_A", "dmg_DAMAGED_GENE"] == 1.0


def test_line_missing_from_damaging_matrix_raises(tmp_path: Path):
    genes = ["DAMAGED_GENE", "RARE_GENE", "OTHER1", "OTHER2", "OTHER3"]
    lines = ["ACH-000001", "ACH-000002", "ACH-000003", "ACH-000004"]
    depmap_dir = tmp_path / "depmap"
    _write_depmap_fixture(depmap_dir, lines, genes)
    dmg = pd.read_parquet(depmap_dir / "OmicsSomaticMutationsMatrixDamaging.parquet")
    dmg[dmg.iloc[:, 0] != "ACH-000002"].to_parquet(depmap_dir / "OmicsSomaticMutationsMatrixDamaging.parquet")
    with pytest.raises(ContractError, match="ACH-000002"):
        build_cell_features(_slice_cfg(), depmap_dir, _tahoe_cells_fixture(), _tiny_cfg())
