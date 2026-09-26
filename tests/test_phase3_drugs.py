from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase3 import drugs

CFG = {
    "drugs": {
        "morgan_radius": 2,
        "morgan_n_bits": 2048,
        "butina_similarity": 0.6,
        "report_similarities": [0.5, 0.6, 0.7],
    }
}


def test_standardize_strips_a_simple_salt():
    parent, log = drugs.standardize("CCO.Cl")
    assert parent == "CCO"
    assert "largest_fragment" in log


def test_standardize_keeps_drug_from_tosylate_salt():
    # Lapatinib ditosylate: two tosylate counter-ions + a water of crystallisation + the drug.
    drug_only = "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OCC5=CC(=CC=C5)F)Cl"
    smi = (
        "CC1=CC=C(C=C1)S(=O)(=O)O.CC1=CC=C(C=C1)S(=O)(=O)O."
        f"{drug_only}.O"
    )
    parent, _ = drugs.standardize(smi)
    expected = drugs.Chem.MolToSmiles(drugs.Chem.MolFromSmiles(drug_only))
    assert parent == expected


def test_standardize_oxaliplatin_keeps_pt_and_oxalate_whole():
    # Real oxaliplatin SMILES: DACH ligand + oxalate/oxalic acid + a bare [Pt+2] fragment.
    # Controller ruling: bare-metal-ion fragment => coordination complex => keep every fragment
    # instead of running MetalDisconnector/LargestFragmentChooser.
    smi = "C1CCC(C(C1)[NH-])[NH-].C(=O)(C(=O)O)O.[Pt+2]"
    parent, log = drugs.standardize(smi)
    assert parent is not None
    assert "metal_complex_kept_whole" in log
    assert "metal_disconnect" not in log
    assert "largest_fragment" not in log
    mol = drugs.Chem.MolFromSmiles(parent)
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    assert "Pt" in symbols
    assert "C" in symbols
    # oxalate carbons (2) + DACH ring carbons (6) all survive
    assert sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C") == 8


def test_standardize_sodium_salt_still_reduces_to_organic_part():
    # Na is an alkali counter-ion, not a coordination-complex metal center: normal
    # disconnect-and-choose-largest path still applies.
    parent, log = drugs.standardize("CC(=O)[O-].[Na+]")
    assert parent is not None
    assert "metal_complex_kept_whole" not in log
    assert "metal_disconnect" in log
    assert "largest_fragment" in log
    mol = drugs.Chem.MolFromSmiles(parent)
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    assert "Na" not in symbols
    assert "C" in symbols


def test_standardize_covalent_organoarsenic_is_unaffected_by_metal_complex_rule():
    # Darinaparsin: As is bound covalently within a single fragment, not a bare ion, so it must
    # not trigger the metal-complex path.
    smi = "C[As](C)SCC(C(=O)NCC(=O)O)NC(=O)CCC(C(=O)O)N"
    parent, log = drugs.standardize(smi)
    assert parent is not None
    assert "metal_complex_kept_whole" not in log
    assert "metal_disconnect" in log
    assert "largest_fragment" in log
    mol = drugs.Chem.MolFromSmiles(parent)
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    assert "As" in symbols


def test_standardize_unparseable_smiles_returns_none_with_reason():
    parent, reason = drugs.standardize("not_a_smiles!!!")
    assert parent is None
    assert reason == "parse_failed"


def test_standardize_missing_input_returns_none_with_reason():
    parent, reason = drugs.standardize(None)
    assert parent is None
    assert reason == "no_input"


def test_morgan_counts_equal_for_two_spellings_of_ethanol():
    a = drugs.morgan_counts("CCO", radius=2, n_bits=2048)
    b = drugs.morgan_counts("OCC", radius=2, n_bits=2048)
    assert np.array_equal(a, b)
    assert a.dtype == np.uint8


def test_morgan_counts_deterministic_across_calls():
    a = drugs.morgan_counts("CC(=O)Oc1ccccc1C(=O)O", radius=2, n_bits=2048)  # aspirin
    b = drugs.morgan_counts("CC(=O)Oc1ccccc1C(=O)O", radius=2, n_bits=2048)
    assert np.array_equal(a, b)


def test_morgan_counts_capped_at_255():
    # n_bits=1 forces every set bit to collide into bin 0, so a long chain's raw (uncapped) count
    # genuinely exceeds 255 — verified directly against RDKit's own count fingerprint, so this
    # test would fail if morgan_counts ever dropped its np.clip.
    long_chain = "C" * 300
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(long_chain)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1)
    raw_counts = generator.GetCountFingerprint(mol).GetNonzeroElements()
    assert raw_counts[0] > 255

    arr = drugs.morgan_counts(long_chain, radius=2, n_bits=1)
    assert arr.dtype == np.uint8
    assert arr[0] == 255


def test_descriptors_returns_finite_values_for_a_simple_molecule():
    d = drugs.descriptors("CCO")
    assert isinstance(d, dict)
    assert "MolWt" in d
    assert np.isfinite(d["MolWt"])


def test_build_drug_features_marks_inorganic_talc_excluded():
    meta = pd.DataFrame(
        {
            "drug": ["Talc"],
            "canonical_smiles": [
                "[OH-].[OH-].[O-][Si]12O[Si]3(O[Si](O1)(O[Si](O2)(O3)[O-])[O-])[O-].[Mg+2].[Mg+2].[Mg+2]"
            ],
        }
    )
    out = drugs.build_drug_features(meta, CFG)
    row = out.iloc[0]
    assert row["featurizable"] is np.False_ or row["featurizable"] is False
    assert row["exclusion_reason"] == "inorganic"


def test_build_drug_features_marks_unparseable_excluded_with_reason():
    meta = pd.DataFrame({"drug": ["Bogus"], "canonical_smiles": ["not_a_smiles!!!"]})
    out = drugs.build_drug_features(meta, CFG)
    row = out.iloc[0]
    assert row["featurizable"] == False  # noqa: E712
    assert row["exclusion_reason"] == "parse_failed"


def test_build_drug_features_marks_missing_smiles_excluded():
    meta = pd.DataFrame({"drug": ["NoSmiles"], "canonical_smiles": [None]})
    out = drugs.build_drug_features(meta, CFG)
    row = out.iloc[0]
    assert row["featurizable"] == False  # noqa: E712
    assert row["exclusion_reason"] == "no_input"


def test_build_drug_features_trims_drug_names():
    meta = pd.DataFrame({"drug": ["Aspirin "], "canonical_smiles": ["CC(=O)Oc1ccccc1C(=O)O"]})
    out = drugs.build_drug_features(meta, CFG)
    assert out.iloc[0]["drug"] == "Aspirin"


def test_build_drug_features_featurizable_row_has_morgan_counts_and_descriptors():
    meta = pd.DataFrame({"drug": ["Aspirin"], "canonical_smiles": ["CC(=O)Oc1ccccc1C(=O)O"]})
    out = drugs.build_drug_features(meta, CFG)
    row = out.iloc[0]
    assert row["featurizable"] == True  # noqa: E712
    assert row["exclusion_reason"] is None
    assert isinstance(row["morgan_counts"], np.ndarray)
    assert row["morgan_counts"].shape == (2048,)
    assert "MolWt" in out.columns
    assert np.isfinite(row["MolWt"])


def test_build_drug_features_drops_descriptor_nan_for_one_drug_from_all():
    meta = pd.DataFrame(
        {
            "drug": ["A", "B"],
            "canonical_smiles": ["CCO", "CC(=O)Oc1ccccc1C(=O)O"],
        }
    )

    real_descriptors = drugs.descriptors

    def fake_descriptors(smiles):
        d = real_descriptors(smiles)
        if smiles == "CCO":
            d["MolWt"] = float("nan")
        return d

    orig = drugs.descriptors
    drugs.descriptors = fake_descriptors
    try:
        out = drugs.build_drug_features(meta, CFG)
    finally:
        drugs.descriptors = orig

    assert "MolWt" in out.attrs["dropped_descriptors"]
    assert "MolWt" not in out.columns
    # other descriptors that stayed finite for both rows are still present
    assert "HeavyAtomMolWt" in out.columns


def test_build_drug_features_dropped_descriptors_uses_inf_too():
    meta = pd.DataFrame(
        {
            "drug": ["A", "B"],
            "canonical_smiles": ["CCO", "CC(=O)Oc1ccccc1C(=O)O"],
        }
    )
    real_descriptors = drugs.descriptors

    def fake_descriptors(smiles):
        d = real_descriptors(smiles)
        if smiles == "CCO":
            d["MolWt"] = float("inf")
        return d

    orig = drugs.descriptors
    drugs.descriptors = fake_descriptors
    try:
        out = drugs.build_drug_features(meta, CFG)
    finally:
        drugs.descriptors = orig

    assert "MolWt" in out.attrs["dropped_descriptors"]


def test_build_drug_features_no_descriptor_dropped_when_all_finite():
    meta = pd.DataFrame(
        {
            "drug": ["A", "B"],
            "canonical_smiles": ["CCO", "CC(=O)Oc1ccccc1C(=O)O"],
        }
    )
    out = drugs.build_drug_features(meta, CFG)
    assert out.attrs["dropped_descriptors"] == []


def test_build_drug_features_non_featurizable_row_does_not_affect_finiteness_check():
    # A non-featurizable row's descriptor cells are NaN by construction (never computed);
    # this must not cause a finite descriptor to be dropped.
    meta = pd.DataFrame(
        {
            "drug": ["Bogus", "Aspirin"],
            "canonical_smiles": ["not_a_smiles!!!", "CC(=O)Oc1ccccc1C(=O)O"],
        }
    )
    out = drugs.build_drug_features(meta, CFG)
    assert out.attrs["dropped_descriptors"] == []
    assert "MolWt" in out.columns
