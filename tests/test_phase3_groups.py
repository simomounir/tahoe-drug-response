from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase3 import groups

CFG = {
    "drugs": {
        "morgan_radius": 2,
        "morgan_n_bits": 2048,
        "butina_similarity": 0.6,
        "report_similarities": [0.5, 0.6, 0.7],
    }
}


def _counts(*vals: int) -> np.ndarray:
    return np.array(vals, dtype=np.uint8)


# ---------------------------------------------------------------------------
# tanimoto_matrix
# ---------------------------------------------------------------------------


def test_tanimoto_matrix_basic_values():
    counts = np.array(
        [
            [5, 0, 0],  # A
            [4, 0, 0],  # B: near duplicate of A
            [0, 0, 5],  # C: unrelated
        ],
        dtype=np.uint8,
    )
    sim = groups.tanimoto_matrix(counts)
    assert sim.shape == (3, 3)
    assert sim[0, 1] == pytest.approx(4 / 5)
    assert sim[1, 0] == pytest.approx(4 / 5)
    assert sim[0, 2] == pytest.approx(0.0)
    assert sim[0, 0] == pytest.approx(1.0)


def test_tanimoto_matrix_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 10, size=(6, 16)).astype(np.uint8)
    sim = groups.tanimoto_matrix(counts)
    assert np.allclose(sim, sim.T)
    assert (sim >= 0).all() and (sim <= 1).all()


def test_tanimoto_matrix_zero_over_zero_is_zero():
    counts = np.zeros((2, 4), dtype=np.uint8)
    sim = groups.tanimoto_matrix(counts)
    assert sim[0, 1] == 0.0
    assert sim[0, 0] == 0.0  # 0/0, not 1.0 -- no special-casing the diagonal


# ---------------------------------------------------------------------------
# butina
# ---------------------------------------------------------------------------


def test_butina_near_duplicates_and_unrelated():
    names = ["A", "B", "C"]
    sim = np.array(
        [
            [1.0, 0.8, 0.0],
            [0.8, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cluster_ids = groups.butina(sim, names, threshold=0.6)
    assert cluster_ids[0] == cluster_ids[1]
    assert cluster_ids[2] != cluster_ids[0]
    assert len(set(cluster_ids)) == 2


def test_butina_singleton_has_no_neighbour_above_threshold():
    names = ["A", "B", "C"]
    sim = np.array(
        [
            [1.0, 0.8, 0.0],
            [0.8, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cluster_ids = groups.butina(sim, names, threshold=0.6)
    c_cluster = cluster_ids[2]
    assert cluster_ids.count(c_cluster) == 1


def test_butina_independent_of_input_order():
    names = ["A", "B", "C"]
    sim = np.array(
        [
            [1.0, 0.8, 0.0],
            [0.8, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    baseline = dict(zip(names, groups.butina(sim, names, threshold=0.6)))

    order = [2, 0, 1]  # C, A, B
    shuffled_names = [names[i] for i in order]
    shuffled_sim = sim[np.ix_(order, order)]
    shuffled = dict(zip(shuffled_names, groups.butina(shuffled_sim, shuffled_names, threshold=0.6)))

    assert (baseline["A"] == baseline["B"]) == (shuffled["A"] == shuffled["B"])
    assert (baseline["A"] == baseline["C"]) == (shuffled["A"] == shuffled["C"])


def test_butina_tie_break_by_neighbour_count_desc_then_name_asc():
    # Two disjoint pairs, each member has exactly one neighbour (a tie on neighbour count).
    # Presented out of alphabetical order to prove the tie-break uses name, not list position.
    names = ["B", "A", "D", "C"]
    # positions: 0=B, 1=A, 2=D, 3=C
    sim = np.eye(4)
    sim[0, 1] = sim[1, 0] = 0.7  # B-A
    sim[2, 3] = sim[3, 2] = 0.7  # D-C

    cluster_ids = groups.butina(sim, names, threshold=0.6)

    # 'A' is alphabetically first among all tied candidates, so the {A,B} pair becomes cluster 0;
    # then 'C' is alphabetically first among the remaining tied candidates, so {C,D} is cluster 1.
    assert cluster_ids[names.index("A")] == 0
    assert cluster_ids[names.index("B")] == 0
    assert cluster_ids[names.index("C")] == 1
    assert cluster_ids[names.index("D")] == 1


def test_butina_matches_rdkit_reordering_variant_as_a_partition():
    # Fix round 1, Important: back the "equivalent to RDKit ClusterData(..., reordering=True)"
    # docstring claim with an actual comparison. Neighbour counts here are deliberately
    # non-tied (2, 2, 2 vs 2, 2) within each group so the two implementations' different
    # tie-break rules can't cause a spurious mismatch -- only the partition is compared, since
    # cluster id numbering (and which tied member becomes the labelled "centroid") is an
    # implementation detail neither the brief nor RDKit's own docs promise a particular value
    # for.
    from rdkit.ML.Cluster import Butina as rdkit_butina

    names = ["A", "B", "C", "D", "E"]
    sim = np.array(
        [
            [1.00, 0.80, 0.70, 0.00, 0.00],
            [0.80, 1.00, 0.65, 0.00, 0.00],
            [0.70, 0.65, 1.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 1.00, 0.75],
            [0.00, 0.00, 0.00, 0.75, 1.00],
        ]
    )
    threshold = 0.6

    ours = groups.butina(sim, names, threshold)
    our_partition = {}
    for name, cid in zip(names, ours):
        our_partition.setdefault(cid, set()).add(name)
    our_sets = {frozenset(members) for members in our_partition.values()}

    n = len(names)
    dists = [1 - sim[i, j] for i in range(n) for j in range(i)]
    rdkit_clusters = rdkit_butina.ClusterData(
        dists, n, 1 - threshold, isDistData=True, reordering=True
    )
    rdkit_sets = {frozenset(names[m] for m in cluster) for cluster in rdkit_clusters}

    assert our_sets == rdkit_sets == {frozenset({"A", "B", "C"}), frozenset({"D", "E"})}


# ---------------------------------------------------------------------------
# nearest_drug / nearest_tanimoto tie-break (fix round 1, Minor)
# ---------------------------------------------------------------------------


def test_build_drug_groups_nearest_drug_tie_picks_name_ascending():
    # X is exactly equidistant (Tanimoto 0.5) from Zeta and Alpha; both are below the 0.6
    # clustering threshold so this exercises nearest_drug's own tie-break in isolation, not
    # cluster membership. Rows are deliberately ordered with Zeta before Alpha so a correct
    # result can't come from accidentally picking "whichever comes first in the table".
    drugs_df = pd.DataFrame(
        {
            "drug": ["X", "Zeta", "Alpha"],
            "featurizable": [True, True, True],
            "smiles_parent": ["CCO", "CCO", "CCO"],
            "morgan_counts": [_counts(5, 5), _counts(5, 0), _counts(0, 5)],
        }
    )
    slice_drugs = {"X", "Zeta", "Alpha"}

    out = groups.build_drug_groups(drugs_df, slice_drugs, CFG)

    x_row = out.loc[out["drug"] == "X"].iloc[0]
    assert x_row["nearest_tanimoto"] == pytest.approx(0.5)
    assert x_row["nearest_drug"] == "Alpha"


# ---------------------------------------------------------------------------
# murcko_scaffold
# ---------------------------------------------------------------------------


def test_murcko_scaffold_simple_ring():
    # Toluene -> benzene ring scaffold.
    assert groups.murcko_scaffold("Cc1ccccc1") == "c1ccccc1"


def test_murcko_scaffold_acyclic_is_empty_string():
    assert groups.murcko_scaffold("CCO") == ""


def test_murcko_scaffold_multi_fragment_uses_largest_organic_fragment():
    # Oxaliplatin's standardised parent: DACH ligand (8 heavy atoms, has a ring) + oxalate
    # (6 heavy atoms, acyclic) + bare Pt ion (1 heavy atom). Rule: scaffold of the largest
    # fragment by heavy-atom count.
    parent = "NC1CCCCC1N.O=C([O-])C(=O)[O-].[Pt+2]"
    assert groups.murcko_scaffold(parent) == "C1CCCCC1"


def test_murcko_scaffold_multi_fragment_largest_fragment_acyclic_is_empty():
    # Largest fragment (by heavy atoms) is acyclic -> scaffold is ''.
    parent = "CCCCCCCC.CC"
    assert groups.murcko_scaffold(parent) == ""


# ---------------------------------------------------------------------------
# cluster_size_report
# ---------------------------------------------------------------------------


def test_cluster_size_report_distribution_across_thresholds():
    names = ["A", "B", "C", "D"]
    sim = np.eye(4)
    sim[0, 1] = sim[1, 0] = 0.55  # A-B
    sim[2, 3] = sim[3, 2] = 0.65  # C-D

    report = groups.cluster_size_report(sim, names, thresholds=[0.5, 0.6, 0.7])

    assert report[0.5] == {"n_clusters": 2, "n_singletons": 0, "largest_cluster_size": 2}
    assert report[0.6] == {"n_clusters": 3, "n_singletons": 2, "largest_cluster_size": 2}
    assert report[0.7] == {"n_clusters": 4, "n_singletons": 4, "largest_cluster_size": 1}


# ---------------------------------------------------------------------------
# build_drug_groups
# ---------------------------------------------------------------------------


def _drugs_df():
    return pd.DataFrame(
        {
            "drug": ["DrugA", "DrugB", "DrugC", "NotInSlice", "Unfeaturizable"],
            "featurizable": [True, True, True, True, False],
            "smiles_parent": ["CCO", "CCO", "c1ccccc1", "CCO", None],
            "morgan_counts": [
                _counts(5, 0, 0),
                _counts(4, 0, 0),  # near duplicate of DrugA
                _counts(0, 0, 5),  # unrelated
                _counts(5, 0, 0),
                None,
            ],
        }
    )


def test_build_drug_groups_two_near_duplicates_and_one_unrelated():
    drugs_df = _drugs_df()
    slice_drugs = {"DrugA", "DrugB", "DrugC"}

    out = groups.build_drug_groups(drugs_df, slice_drugs, CFG)

    assert set(out["drug"]) == slice_drugs
    a_cluster = out.loc[out["drug"] == "DrugA", "cluster_id"].iloc[0]
    b_cluster = out.loc[out["drug"] == "DrugB", "cluster_id"].iloc[0]
    c_cluster = out.loc[out["drug"] == "DrugC", "cluster_id"].iloc[0]
    assert a_cluster == b_cluster
    assert c_cluster != a_cluster
    assert out["cluster_id"].nunique() == 2


def test_build_drug_groups_only_covers_slice_featurizable_drugs():
    drugs_df = _drugs_df()
    slice_drugs = {"DrugA", "DrugB", "DrugC", "NotInSlice", "Unfeaturizable"}

    out = groups.build_drug_groups(drugs_df, slice_drugs, CFG)

    # "NotInSlice" is a name mismatch trap: it's in slice_drugs and featurizable but a distinct
    # row from DrugA/B/C, so it must appear; "Unfeaturizable" must never appear (D10).
    assert "Unfeaturizable" not in set(out["drug"])
    assert set(out["drug"]) == {"DrugA", "DrugB", "DrugC", "NotInSlice"}


def test_build_drug_groups_no_neighbour_is_a_singleton():
    drugs_df = _drugs_df()
    slice_drugs = {"DrugA", "DrugB", "DrugC"}

    out = groups.build_drug_groups(drugs_df, slice_drugs, CFG)

    row = out.loc[out["drug"] == "DrugC"].iloc[0]
    assert row["cluster_size"] == 1


def test_build_drug_groups_salt_and_free_base_same_parent_are_tanimoto_one_and_same_cluster():
    drugs_df = pd.DataFrame(
        {
            "drug": ["DrugX_salt", "DrugX_freebase", "Unrelated"],
            "featurizable": [True, True, True],
            "smiles_parent": ["CCO", "CCO", "c1ccccc1"],
            "morgan_counts": [_counts(5, 3, 1), _counts(5, 3, 1), _counts(0, 0, 9)],
        }
    )
    slice_drugs = {"DrugX_salt", "DrugX_freebase", "Unrelated"}

    out = groups.build_drug_groups(drugs_df, slice_drugs, CFG)

    salt_row = out.loc[out["drug"] == "DrugX_salt"].iloc[0]
    base_row = out.loc[out["drug"] == "DrugX_freebase"].iloc[0]
    assert salt_row["cluster_id"] == base_row["cluster_id"]
    assert salt_row["nearest_tanimoto"] == pytest.approx(1.0)
    assert salt_row["nearest_drug"] == "DrugX_freebase"
    assert salt_row["cluster_size"] == 2


def test_build_drug_groups_independent_of_input_row_order():
    drugs_df = _drugs_df()
    slice_drugs = {"DrugA", "DrugB", "DrugC"}

    out1 = groups.build_drug_groups(drugs_df, slice_drugs, CFG)
    shuffled = drugs_df.sample(frac=1, random_state=42).reset_index(drop=True)
    out2 = groups.build_drug_groups(shuffled, slice_drugs, CFG)

    out1 = out1.sort_values("drug").reset_index(drop=True)
    out2 = out2.sort_values("drug").reset_index(drop=True)
    pd.testing.assert_frame_equal(out1, out2)


def test_build_drug_groups_has_expected_columns():
    drugs_df = _drugs_df()
    slice_drugs = {"DrugA", "DrugB", "DrugC"}

    out = groups.build_drug_groups(drugs_df, slice_drugs, CFG)

    expected = {"drug", "cluster_id", "cluster_size", "murcko_scaffold", "nearest_drug", "nearest_tanimoto"}
    assert expected <= set(out.columns)
