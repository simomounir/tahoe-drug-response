"""Drug similarity groups: count-Tanimoto similarity + Butina clustering.

Spec: docs/superpowers/specs/2026-09-26-phase3-features-design.md, decisions D4/D10, task P3.3.

D4: all 92 slice drugs have distinct Bemis-Murcko scaffolds (probe 2026-09-26), so scaffold
grouping groups nothing; drug groups are instead Butina clusters on count-Tanimoto similarity
of the Morgan fingerprints built in `phase3.drugs`. D10: groups are computed for the slice's
drugs only (clusters depend on which drugs are in the set), while drug features cover all
compounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def tanimoto_matrix(counts: np.ndarray) -> np.ndarray:
    """Pairwise count-Tanimoto similarity: sim(i, j) = sum(min(c_i, c_j)) / sum(max(c_i, c_j)).

    `counts` is (n_drugs, n_bits). 0/0 (both fingerprints all-zero for a pair) is defined as 0,
    including on the diagonal -- there is no special-casing of self-similarity.
    """
    counts = np.asarray(counts, dtype=np.int64)
    n = counts.shape[0]
    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        min_sum = np.minimum(counts[i], counts).sum(axis=1)
        max_sum = np.maximum(counts[i], counts).sum(axis=1)
        sim[i] = np.divide(
            min_sum,
            max_sum,
            out=np.zeros(n, dtype=np.float64),
            where=max_sum > 0,
        )
    return sim


def butina(sim: np.ndarray, names: list[str], threshold: float) -> list[int]:
    """Butina clustering: cluster id per drug, in the order of `names`.

    Greedy Butina with reordering, equivalent to RDKit's
    `rdkit.ML.Cluster.Butina.ClusterData(..., reordering=True)`: repeatedly pick the unassigned
    point with the most *currently unassigned* neighbours (sim >= threshold, recomputed against
    the shrinking unassigned set on every iteration -- this is the "reordering" step, not the
    1999 paper's single upfront ranking) as a cluster center, form a cluster from it and those
    neighbours, remove them, repeat. A point with no neighbour above threshold ends up as its
    own singleton cluster.

    Deterministic tie-break (spec): among unassigned candidates, order by neighbour count
    descending, then drug name ascending -- based on the data (name, similarity), never on
    position in the input arrays, so the result does not depend on input order.
    """
    n = len(names)
    neighbor_sets = [
        {j for j in range(n) if j != i and sim[i, j] >= threshold} for i in range(n)
    ]

    cluster_id = [-1] * n
    unassigned = set(range(n))
    next_id = 0
    while unassigned:
        center = min(
            unassigned,
            key=lambda i: (-len(neighbor_sets[i] & unassigned), names[i]),
        )
        members = (neighbor_sets[center] & unassigned) | {center}
        for m in members:
            cluster_id[m] = next_id
        unassigned -= members
        next_id += 1
    return cluster_id


def _nearest(sim: np.ndarray, names: list[str]) -> tuple[list[str | None], list[float]]:
    """For each drug, the single most similar other drug (ties broken by name ascending)."""
    n = len(names)
    nearest_drug: list[str | None] = [None] * n
    nearest_tanimoto: list[float] = [float("nan")] * n
    for i in range(n):
        best_j = None
        best_val = -1.0
        for j in range(n):
            if j == i:
                continue
            val = sim[i, j]
            if val > best_val or (val == best_val and best_j is not None and names[j] < names[best_j]):
                best_val = val
                best_j = j
        if best_j is not None:
            nearest_drug[i] = names[best_j]
            nearest_tanimoto[i] = float(best_val)
    return nearest_drug, nearest_tanimoto


def murcko_scaffold(smiles_parent: str | None) -> str:
    """Bemis-Murcko scaffold SMILES of a (possibly multi-fragment) parent molecule.

    Rule for multi-fragment parents (e.g. a coordination complex kept whole by
    `phase3.drugs.standardize`, such as Oxaliplatin's Pt center + DACH ligand + oxalate):
    the scaffold is that of the largest fragment by heavy-atom count. An acyclic largest
    fragment (or an empty/unparseable input) yields the empty string, not an error.
    """
    if not smiles_parent:
        return ""
    fragments = [Chem.MolFromSmiles(s) for s in str(smiles_parent).split(".")]
    fragments = [f for f in fragments if f is not None]
    if not fragments:
        return ""
    largest = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
    scaffold = MurckoScaffold.GetScaffoldForMol(largest)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold)


def cluster_size_report(sim: np.ndarray, names: list[str], thresholds: list[float]) -> dict:
    """Cluster-size distribution at each threshold: n_clusters, n_singletons, largest_cluster_size."""
    report: dict[float, dict[str, int]] = {}
    for threshold in thresholds:
        cluster_ids = butina(sim, names, threshold)
        sizes = pd.Series(cluster_ids).value_counts()
        report[threshold] = {
            "n_clusters": int(len(sizes)),
            "n_singletons": int((sizes == 1).sum()),
            "largest_cluster_size": int(sizes.max()) if len(sizes) else 0,
        }
    return report


def build_drug_groups(drugs: pd.DataFrame, slice_drugs: set[str], cfg: dict) -> pd.DataFrame:
    """Butina drug-similarity groups over the slice's featurizable drugs only (D10).

    Columns: drug, cluster_id, cluster_size, murcko_scaffold, nearest_drug, nearest_tanimoto.
    Output row order is `drug` ascending, so the result does not depend on the order of `drugs`.
    """
    subset = drugs[drugs["drug"].isin(slice_drugs) & drugs["featurizable"].astype(bool)]
    subset = subset.sort_values("drug").reset_index(drop=True)

    names = subset["drug"].tolist()
    counts = np.stack(subset["morgan_counts"].to_numpy())
    sim = tanimoto_matrix(counts)

    threshold = cfg["drugs"]["butina_similarity"]
    cluster_ids = butina(sim, names, threshold)
    cluster_sizes = pd.Series(cluster_ids).value_counts().to_dict()

    nearest_drug, nearest_tanimoto = _nearest(sim, names)
    scaffolds = [murcko_scaffold(s) for s in subset["smiles_parent"]]

    return pd.DataFrame(
        {
            "drug": names,
            "cluster_id": cluster_ids,
            "cluster_size": [cluster_sizes[c] for c in cluster_ids],
            "murcko_scaffold": scaffolds,
            "nearest_drug": nearest_drug,
            "nearest_tanimoto": nearest_tanimoto,
        }
    )
