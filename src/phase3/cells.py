"""Cell-line features from DepMap 24Q4 expression/mutation panels and Tahoe driver flags (P3.4, spec D6-D9).

Pipeline (`build_cell_features`):
  1. PCA is fit on the *full* DepMap expression panel (~1,700 lines), never on just our 50 lines,
     so a held-out line's features do not leak information about the evaluation population (D7).
  2. Mutation flags (`dmg_<GENE>`, `hot_<GENE>`) are damaging/hotspot indicators for genes damaged
     in >= `mutation_min_frequency` of the DepMap panel (D8), read from the Damaging matrix; the
     Hotspot matrix supplies `hot_<GENE>` for the same gene list, 0 where a gene is absent there.
  3. Driver flags (`drv_<GENE>`) come from Tahoe's own `cell_line_metadata` (one row per driver
     mutation), independent of DepMap coverage, so hTERT-HPNE (no DepMap entry) still gets them.
     Rule (fixed in code, threshold from config, not a hand-picked list): a driver gene gets a
     column when it drives at least `cells.driver_min_lines` of the 50 slice lines.
  4. A slice line whose config declares no DepMap ID at all (`depmap_id: null` in `slice.yaml`,
     e.g. hTERT-HPNE / CVCL_C466, a non-cancer line absent from DepMap) is a declared absence, not
     a failure: it is kept with `depmap_available = False`, `depmap_absent_reason` from the config,
     and null `pc_*`/`dmg_*`/`hot_*`/`expression_*` (it still gets `drv_*`, since those come from
     Tahoe, not DepMap).
  5. A line *with* a DepMap ID whose own row is missing from the 24Q4 expression file (it may still
     have damaging/hotspot rows) needs an explicit `configs/features.yaml` `cells.expression_proxy`
     or `cells.expression_absent` declaration (owner decision, fix round 1):
       - `expression_proxy: {LINE: {model: ACH-..., reason: "..."}}` projects another DepMap
         model's expression (e.g. a parent line) in its place; `expression_source` records which
         ACH ID was actually projected. The proxy model itself must have an expression row, or
         `ContractError`. Mutation flags (`dmg_*`/`hot_*`) always come from the line's *own*
         `depmap_id`, never the proxy's -- only expression is substituted.
       - `expression_absent: {LINE: "reason"}` records that no substitute exists: `pc_*` stay null,
         `expression_available = False`, but `dmg_*`/`hot_*`/`drv_*` are kept (they don't need
         expression).
     An undeclared line whose DepMap ID is missing from 24Q4 expression -- not covered by either
     declaration, nor by `slice.yaml`'s own `depmap_id: null` -- raises `ContractError` (spec §6
     contract 4). No imputation anywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from phase1.contracts import ContractError
from phase3.config import FEATURES_CONFIG_PATH, load_features_config  # noqa: F401  (re-exported)

# D8's driver-flag rule default for callers that pass no config; build_cell_features reads
# cells.driver_min_lines from configs/features.yaml.
DEFAULT_DRIVER_MIN_LINES = 2

EXPRESSION_FILE = "OmicsExpressionProteinCodingGenesTPMLogp1.parquet"
DAMAGING_FILE = "OmicsSomaticMutationsMatrixDamaging.parquet"
HOTSPOT_FILE = "OmicsSomaticMutationsMatrixHotspot.parquet"


def read_depmap_matrix(path: Path | str) -> pd.DataFrame:
    """A DepMap CSV or Parquet matrix, indexed by ACH ID, columns stripped from `SYMBOL (ENTREZ)` to `SYMBOL`.

    The first column is unnamed in the source CSV (pandas/DuckDB give it some placeholder name);
    it is taken positionally, not by name, so this works whether `path` is the original CSV or the
    Parquet DuckDB converted it to.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    index_col = df.columns[0]
    df = df.set_index(index_col)
    df.index.name = "depmap_id"
    df.columns = [str(c).split(" (")[0] for c in df.columns]
    return df.astype(np.float32)


def _canonical_svd_sign(Vt: np.ndarray) -> np.ndarray:
    """Fix SVD's inherent per-component sign ambiguity: flip each row so its largest-magnitude entry is positive.

    `numpy.linalg.svd` (LAPACK) does not guarantee a sign convention -- a different LAPACK build,
    BLAS backend, or even row order can flip any component's sign, silently flipping that PC (and
    every downstream projection) between rebuilds. Fixing the sign from the loadings themselves
    makes the result deterministic and independent of which sign the library happened to produce.
    """
    Vt = np.array(Vt, dtype=np.float64, copy=True)
    idx = np.argmax(np.abs(Vt), axis=1)
    signs = np.sign(Vt[np.arange(Vt.shape[0]), idx])
    signs[signs == 0] = 1.0
    return Vt * signs[:, None]


def fit_pca(panel: pd.DataFrame, k: int) -> dict:
    """PCA via numpy SVD: standardise each gene over `panel`'s rows (zero-std genes get std = 1), keep top k.

    Returns a dict with `genes` (column order), `mean`, `std`, `loadings` (n_genes x k, so that
    `(X - mean) / std @ loadings` gives scores), `variance_explained` (length k, fraction of total
    variance) and the effective `k` (capped at min(n_rows, n_genes)). Each component's sign is
    canonicalised (`_canonical_svd_sign`) so loadings/projections are reproducible across reruns
    regardless of the SVD implementation's arbitrary sign choice.
    """
    genes = list(panel.columns)
    X = panel.to_numpy(dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std = np.where(std == 0.0, 1.0, std)  # zero-variance gene: numerator is 0 too, so this avoids 0/0 = NaN
    Xs = (X - mean) / std
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    k_eff = min(k, Vt.shape[0])
    Vt = _canonical_svd_sign(Vt[:k_eff])
    loadings = Vt.T
    total_var = float(np.sum(S**2))
    variance_explained = (S[:k_eff] ** 2) / total_var if total_var > 0 else np.zeros(k_eff)
    return {
        "genes": genes,
        "mean": mean,
        "std": std,
        "loadings": loadings,
        "variance_explained": variance_explained,
        "k": k_eff,
    }


def project(pca: dict, rows: pd.DataFrame) -> np.ndarray:
    """Standardise `rows` with the fitted mean/std and project onto the fitted loadings."""
    aligned = rows.reindex(columns=pca["genes"])
    X = aligned.to_numpy(dtype=np.float64)
    Xs = (X - pca["mean"]) / pca["std"]
    return Xs @ pca["loadings"]


def pca_to_loadings_frame(pca: dict) -> pd.DataFrame:
    """One row per gene: `gene`, `mean`, `std`, `pc_1..pc_k` -- enough on its own to reproduce `project`."""
    pc_cols = [f"pc_{i + 1}" for i in range(pca["k"])]
    df = pd.DataFrame(np.asarray(pca["loadings"], dtype=np.float64), columns=pc_cols)
    df.insert(0, "gene", pca["genes"])
    df.insert(1, "mean", np.asarray(pca["mean"], dtype=np.float64))
    df.insert(2, "std", np.asarray(pca["std"], dtype=np.float64))
    return df


def loadings_frame_to_pca(df: pd.DataFrame) -> dict:
    """Inverse of `pca_to_loadings_frame`: rebuild the dict `project` needs from the on-disk table."""
    pc_cols = sorted((c for c in df.columns if c.startswith("pc_")), key=lambda c: int(c.split("_")[1]))
    return {
        "genes": df["gene"].tolist(),
        "mean": df["mean"].to_numpy(dtype=np.float64),
        "std": df["std"].to_numpy(dtype=np.float64),
        "loadings": df[pc_cols].to_numpy(dtype=np.float64),
        "k": len(pc_cols),
    }


def mutation_genes(damaging: pd.DataFrame, min_frequency: float) -> list[str]:
    """Genes damaged in >= `min_frequency` of the DepMap panel's lines (Damaging matrix only), sorted.

    DepMap's matrices hold 0/1/2 (none / heterozygous / homozygous); a line counts once however
    many alleles are hit, so frequency is the fraction of lines with a value > 0.
    """
    freq = (damaging.astype(np.float64) > 0).mean(axis=0)
    genes = freq[freq >= min_frequency].index.tolist()
    return sorted(set(str(g) for g in genes))


def select_driver_genes(tahoe_cells: pd.DataFrame, selected_cell_lines: list[str], min_lines: int = DEFAULT_DRIVER_MIN_LINES) -> list[str]:
    """Driver genes that drive >= `min_lines` of `selected_cell_lines`, per Tahoe `cell_line_metadata`."""
    sub = tahoe_cells[tahoe_cells["Cell_ID_Cellosaur"].isin(selected_cell_lines)]
    counts = sub.groupby("Driver_Gene_Symbol")["Cell_ID_Cellosaur"].nunique()
    return sorted(counts[counts >= min_lines].index.tolist())


def _proxy_model_id(declaration) -> str:
    """`expression_proxy` values may be `{model: ACH-..., reason: ...}` or a bare ACH ID string."""
    if isinstance(declaration, dict):
        return declaration["model"]
    return declaration


def build_cell_features(slice_cfg: dict, depmap_dir: Path, tahoe_cells: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The slice's cell-line feature table and the PCA loadings table (for data/features/depmap_pca.parquet).

    Raises `ContractError` (spec §6 contract 4) if a slice line has a DepMap ID that is absent from
    the 24Q4 expression file and is not covered by `slice_cfg`'s `depmap_id: null` or by
    `cfg["cells"]["expression_proxy"]` / `["expression_absent"]`; also if an `expression_proxy`
    model itself has no expression row. No imputation anywhere.
    """
    depmap_cfg = cfg["depmap"]
    cells_cfg = cfg["cells"]
    k = int(cells_cfg["pca_components"])
    min_freq = float(cells_cfg["mutation_min_frequency"])
    driver_min_lines = int(cells_cfg["driver_min_lines"])
    expression_proxy = cells_cfg.get("expression_proxy") or {}
    expression_absent = cells_cfg.get("expression_absent") or {}
    release = depmap_cfg["release"]

    depmap_dir = Path(depmap_dir)
    expression = read_depmap_matrix(depmap_dir / EXPRESSION_FILE)
    damaging = read_depmap_matrix(depmap_dir / DAMAGING_FILE)
    hotspot = read_depmap_matrix(depmap_dir / HOTSPOT_FILE)

    pca = fit_pca(expression, k)
    genes = mutation_genes(damaging, min_freq)
    pc_cols = [f"pc_{i + 1}" for i in range(pca["k"])]
    dmg_cols = [f"dmg_{g}" for g in genes]
    hot_cols = [f"hot_{g}" for g in genes]

    selected = list(slice_cfg["selected_cell_lines"])
    info = slice_cfg["cell_line_info"]

    driver_genes = select_driver_genes(tahoe_cells, selected, min_lines=driver_min_lines)
    drv_cols = [f"drv_{g}" for g in driver_genes]
    driver_lookup = (
        tahoe_cells[tahoe_cells["Cell_ID_Cellosaur"].isin(selected) & tahoe_cells["Driver_Gene_Symbol"].isin(driver_genes)]
        .groupby("Cell_ID_Cellosaur")["Driver_Gene_Symbol"]
        .apply(set)
        .to_dict()
    )

    missing_undeclared: list[tuple[str, str]] = []
    rows = []
    for line in selected:
        meta = info[line]
        depmap_id = meta.get("depmap_id")
        row: dict = {"cell_line": line, "depmap_release": release}
        driver_set = driver_lookup.get(line, set())
        for gene, col in zip(driver_genes, drv_cols):
            row[col] = 1.0 if gene in driver_set else 0.0

        if depmap_id is None:
            row["depmap_id"] = None
            row["depmap_available"] = False
            row["depmap_absent_reason"] = meta.get("depmap_absent")
            row["expression_available"] = False
            row["expression_source"] = None
            for col in pc_cols + dmg_cols + hot_cols:
                row[col] = np.nan
            rows.append(row)
            continue

        row["depmap_id"] = depmap_id
        row["depmap_available"] = True
        row["depmap_absent_reason"] = None

        # Resolve which ACH ID's expression row to project, if any: the line's own row, a declared
        # proxy model, or a declared absence. Anything else missing from 24Q4 expression is
        # undeclared and fails the build below (contract 4) -- no imputation.
        if depmap_id in expression.index:
            expr_source: str | None = depmap_id
        elif line in expression_proxy:
            proxy_model = _proxy_model_id(expression_proxy[line])
            if proxy_model not in expression.index:
                raise ContractError(
                    f"{line}: expression_proxy model {proxy_model} itself has no {release} expression row (no imputation)"
                )
            expr_source = proxy_model
        elif line in expression_absent:
            expr_source = None
        else:
            missing_undeclared.append((line, depmap_id))
            continue

        if expr_source is not None:
            row["expression_available"] = True
            row["expression_source"] = expr_source
            coords = project(pca, expression.loc[[expr_source]])[0]
            for col, value in zip(pc_cols, coords):
                row[col] = float(value)
        else:
            row["expression_available"] = False
            row["expression_source"] = None
            for col in pc_cols:
                row[col] = np.nan

        # Mutation flags always come from the line's own depmap_id, never the expression proxy's.
        # Flags are binary (mutated or not); the 0/1/2 allele count is not a feature.
        if depmap_id not in damaging.index:
            raise ContractError(f"cells: {line} ({depmap_id}) is missing from the DepMap Damaging matrix (no imputation)")
        dmg_row = damaging.loc[depmap_id, genes]
        for gene, col in zip(genes, dmg_cols):
            row[col] = float(dmg_row[gene] > 0)

        if depmap_id in hotspot.index:
            hot_row = hotspot.loc[depmap_id].reindex(genes).fillna(0.0)
        else:
            hot_row = pd.Series(0.0, index=genes)
        for gene, col in zip(genes, hot_cols):
            row[col] = float(hot_row[gene] > 0)

        rows.append(row)

    if missing_undeclared:
        raise ContractError(
            f"cell line(s) with a DepMap ID absent from the {release} expression file and not declared in "
            f"cells.expression_proxy or cells.expression_absent (no imputation): {missing_undeclared}"
        )

    cells_df = pd.DataFrame(rows)
    ordered = (
        ["cell_line", "depmap_id", "depmap_release", "depmap_available", "depmap_absent_reason", "expression_available", "expression_source"]
        + pc_cols
        + dmg_cols
        + hot_cols
        + drv_cols
    )
    cells_df = cells_df[ordered]

    loadings_df = pca_to_loadings_frame(pca)
    return cells_df, loadings_df
