"""Drug features from SMILES: standardisation, Morgan count fingerprints, RDKit 2D descriptors.

Spec: docs/superpowers/specs/2026-09-26-phase3-features-design.md, decisions D2/D3/D10, task P3.2.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

# RDKit logs "explicit valence" / "not removing hydrogen" etc. to stderr by default for every
# malformed input; the build deliberately hits some of these (unparseable SMILES). Silence them
# here so library use never depends on the caller configuring the C++ logger.
RDLogger.DisableLog("rdApp.*")

_DESCRIPTOR_LIST = list(Descriptors.descList)
DESCRIPTOR_NAMES = [name for name, _ in _DESCRIPTOR_LIST]

# Controller ruling (P3.2 fix round 1, recorded in the ledger): a compound whose input has a
# fragment that is a single bare metal ion is a coordination complex, not a salt. For those,
# MetalDisconnector + LargestFragmentChooser would strip the metal center itself (the
# pharmacophore, e.g. Oxaliplatin's Pt) rather than an inert counter-ion, so both steps are
# skipped and every fragment is kept. Alkali/alkaline-earth counter-ions (Li/Na/K/Mg/Ca) are
# still ordinary salts and go through the normal disconnect-and-choose-largest path.
_ORGANIC_SUBSET = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}
_ALKALI_ALKALINE_EARTH = {"Li", "Na", "K", "Mg", "Ca"}


def _is_missing(smiles) -> bool:
    if smiles is None:
        return True
    if isinstance(smiles, float) and math.isnan(smiles):
        return True
    return not str(smiles).strip()


def _has_bare_metal_fragment(mol) -> bool:
    """True if any fragment of `mol` is a single atom that is neither an organic-subset element
    nor an alkali/alkaline-earth counter-ion — i.e. a metal center bound in a coordination
    complex, not an ordinary salt-forming ion."""
    for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
        if frag.GetNumAtoms() != 1:
            continue
        symbol = frag.GetAtomWithIdx(0).GetSymbol()
        if symbol not in _ORGANIC_SUBSET and symbol not in _ALKALI_ALKALINE_EARTH:
            return True
    return False


def standardize(smiles: str) -> tuple[str | None, str]:
    """Standardise a SMILES to its parent molecule.

    Pipeline (spec D3): Cleanup -> MetalDisconnector -> LargestFragmentChooser -> Normalizer ->
    Reionizer -> canonical SMILES. Returns (parent_smiles or None, cleaning_log). The log is
    either the comma-joined list of standardisation steps that ran, or a single reason token
    ("no_input", "parse_failed", "standardize_failed") when standardisation could not produce a
    molecule.

    Coordination complexes (a fragment that is a bare metal ion, e.g. Oxaliplatin's `[Pt+2]`)
    skip MetalDisconnector and LargestFragmentChooser and keep every fragment instead, per
    controller ruling; the log records this as "metal_complex_kept_whole".
    """
    if _is_missing(smiles):
        return None, "no_input"

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, "parse_failed"

    metal_complex = _has_bare_metal_fragment(mol)

    steps: list[str] = []
    try:
        mol = rdMolStandardize.Cleanup(mol)
        steps.append("cleanup")
        if metal_complex:
            steps.append("metal_complex_kept_whole")
        else:
            mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)
            steps.append("metal_disconnect")
            mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
            steps.append("largest_fragment")
        mol = rdMolStandardize.Normalizer().normalize(mol)
        steps.append("normalize")
        mol = rdMolStandardize.Reionizer().reionize(mol)
        steps.append("reionize")
    except Exception:
        return None, "standardize_failed"

    if mol is None or mol.GetNumAtoms() == 0:
        return None, "standardize_failed"

    return Chem.MolToSmiles(mol), ",".join(steps)


def morgan_counts(smiles: str, radius: int, n_bits: int) -> np.ndarray:
    """Hashed Morgan count fingerprint as a dense uint8 array, counts capped at 255 (spec D2)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse SMILES: {smiles!r}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    counts = generator.GetCountFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.uint32)
    for idx, count in counts.GetNonzeroElements().items():
        arr[idx] = count
    return np.clip(arr, 0, 255).astype(np.uint8)


def descriptors(smiles: str) -> dict[str, float]:
    """RDKit 2D descriptors (`Descriptors.descList`) for one molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse SMILES: {smiles!r}")
    values: dict[str, float] = {}
    for name, func in _DESCRIPTOR_LIST:
        try:
            values[name] = float(func(mol))
        except Exception:
            values[name] = float("nan")
    return values


def _is_inorganic(smiles_parent: str) -> bool:
    mol = Chem.MolFromSmiles(smiles_parent)
    return not any(atom.GetSymbol() == "C" for atom in mol.GetAtoms())


def build_drug_features(drug_metadata: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build the per-molecule drug feature table (spec P3.2 output, `data/features/drugs.parquet`).

    Columns: drug (trimmed), smiles_input, smiles_parent, cleaning_log, featurizable,
    exclusion_reason, morgan_counts (uint8[n_bits]), one float column per descriptor that is
    finite for every featurizable drug. Descriptors non-finite for any featurizable drug are
    dropped for all rows; their names are recorded in `df.attrs["dropped_descriptors"]`.
    """
    radius = cfg["drugs"]["morgan_radius"]
    n_bits = cfg["drugs"]["morgan_n_bits"]

    rows: list[dict] = []
    for _, row in drug_metadata.iterrows():
        drug = str(row["drug"]).strip()
        smiles_input = row.get("canonical_smiles")
        parent, log = standardize(smiles_input)

        record: dict = {
            "drug": drug,
            "smiles_input": smiles_input,
            "smiles_parent": parent,
            "cleaning_log": log,
            "featurizable": False,
            "exclusion_reason": None,
            "morgan_counts": None,
        }

        if parent is None:
            record["exclusion_reason"] = log
            rows.append(record)
            continue

        if _is_inorganic(parent):
            record["exclusion_reason"] = "inorganic"
            rows.append(record)
            continue

        record["featurizable"] = True
        record["morgan_counts"] = morgan_counts(parent, radius, n_bits)
        record.update(descriptors(parent))
        rows.append(record)

    df = pd.DataFrame(rows)

    featurizable_mask = df["featurizable"].astype(bool)
    dropped: list[str] = []
    for name in DESCRIPTOR_NAMES:
        if name not in df.columns:
            # Can only happen if there are zero featurizable rows at all; nothing to keep.
            continue
        values = df.loc[featurizable_mask, name].astype(float)
        if not values.empty and not np.isfinite(values).all():
            dropped.append(name)

    df = df.drop(columns=dropped)
    df.attrs["dropped_descriptors"] = dropped
    return df
