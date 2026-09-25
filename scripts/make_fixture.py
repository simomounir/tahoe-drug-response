"""Build the committed real-schema test fixture from one plate3 shard of Tahoe-100M.

Writes to tests/fixtures/:
  expression_small.parquet   real shard rows (all columns, original schema), 2 lines, ~200 cells
  metadata_small.parquet     same columns as the cached metadata aggregate; n_cells = cells in the fixture
  expected_pseudobulk.parquet, expected_logfc.parquet   aggregate_shard + build_outputs on the fixture

Network: one ~85 MB shard, downloaded to tmp/ and deleted afterwards (--keep-shard to reuse it).
--expected-only: no network; rebuild only the expected outputs from the committed fixture inputs.
Drug/dose per sample come from the local metadata cache (no obs_metadata query).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase1.build import build_outputs
from phase1.parse import is_control_drug
from phase1.pseudobulk import SPECIAL_TOKENS
from phase1.stream import aggregate_shard, connect

REPO_ID = "tahoebio/Tahoe-100M"
SHARD_FILE = "data/train-03300-of-03388.parquet"
PLATE = "plate3"
DOWNLOAD_DIR = ROOT / "tmp" / "t6_shard"
METADATA_CACHE = ROOT / "data" / "cache" / "metadata_by_sample_05435ad905ab.parquet"
FIXTURE_DIR = ROOT / "tests" / "fixtures"
EXPRESSION_PATH = FIXTURE_DIR / "expression_small.parquet"
METADATA_PATH = FIXTURE_DIR / "metadata_small.parquet"
EXPECTED_PSEUDOBULK = FIXTURE_DIR / "expected_pseudobulk.parquet"
EXPECTED_LOGFC = FIXTURE_DIR / "expected_logfc.parquet"
METADATA_COLUMNS = ["cell_line", "plate", "sample", "drug", "drugname_drugconc", "n_cells"]

SEED = 0
CELL_LINES = ["CVCL_0546", "CVCL_0459"]
N_DRUGS = 5
CELLS_PER_DRUG_WELL = 16
# The last drug gets fewer cells than min_cells_per_condition, so QC drops it.
CELLS_LOW_DRUG_WELL = 4
CELLS_PER_CONTROL_WELL = 13
# Genes are subset to keep the fixture small: most-detected genes plus a random sparse tail.
N_TOP_GENES = 400
N_RANDOM_GENES = 400

# Values from configs/slice.yaml, frozen here so the fixture does not change when the slice does.
FIXTURE_CONFIG = {
    "selected_cell_lines": CELL_LINES,
    "cell_line_info": {
        "CVCL_0546": {"depmap_id": "ACH-000842", "tissue": "Bowel"},
        "CVCL_0459": {"depmap_id": "ACH-000463", "tissue": "Lung"},
    },
    "min_cells_per_condition": 10,
    "min_control_cells": 20,
    "control_drugs": ["DMSO_TF"],
}


def download_shard(dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    for wait in (0, 30, 60, 120, 240):
        time.sleep(wait)
        try:
            return Path(hf_hub_download(REPO_ID, SHARD_FILE, repo_type="dataset", local_dir=dest))
        except Exception as exc:  # noqa: BLE001
            if "429" not in str(exc):
                raise
            print(f"HTTP 429, backing off (last wait {wait}s)", flush=True)
    raise RuntimeError("Hugging Face kept returning HTTP 429")


def select_wells(meta: pd.DataFrame, shard_counts: pd.DataFrame, rng: np.random.Generator) -> dict[str, int]:
    """sample -> cells to keep per cell line: N_DRUGS drug wells + every DMSO well of the plate."""
    wells = meta[["sample", "drug"]].drop_duplicates()
    per_sample = shard_counts.pivot_table(index="sample", columns="cell_line", values="n", fill_value=0)
    min_in_shard = per_sample.reindex(columns=CELL_LINES, fill_value=0).min(axis=1)
    wells = wells.assign(min_in_shard=wells["sample"].map(min_in_shard).fillna(0))
    controls = wells[wells["drug"].map(lambda d: is_control_drug(d, FIXTURE_CONFIG["control_drugs"]))]
    treated = wells[~wells.index.isin(controls.index)]

    if (controls["min_in_shard"] < CELLS_PER_CONTROL_WELL).any():
        raise RuntimeError(f"Not enough control cells in the shard:\n{controls}")
    candidates = treated[treated["min_in_shard"] >= CELLS_PER_DRUG_WELL].sort_values("sample")
    candidates = candidates.drop_duplicates("drug")
    picked = sorted(candidates["sample"].iloc[rng.choice(len(candidates), N_DRUGS, replace=False)])

    target = {s: CELLS_PER_DRUG_WELL for s in picked[:-1]}
    target[picked[-1]] = CELLS_LOW_DRUG_WELL
    target.update({s: CELLS_PER_CONTROL_WELL for s in sorted(controls["sample"])})
    return target


def pick_barcodes(cells: pd.DataFrame, target: dict[str, int], rng: np.random.Generator) -> list[str]:
    cells = cells.sort_values(["cell_line_id", "sample", "BARCODE_SUB_LIB_ID"]).reset_index(drop=True)
    chosen = []
    for (line, sample), group in cells.groupby(["cell_line_id", "sample"], sort=True):
        if sample in target:
            idx = rng.choice(len(group), target[sample], replace=False)
            chosen.extend(group["BARCODE_SUB_LIB_ID"].iloc[np.sort(idx)])
    return chosen


def select_genes(genes: pd.Series, rng: np.random.Generator) -> np.ndarray:
    all_genes = np.concatenate([np.unique(g) for g in genes])
    ids, detected = np.unique(all_genes[~np.isin(all_genes, SPECIAL_TOKENS)], return_counts=True)
    order = np.lexsort((ids, -detected))
    top = ids[order[:N_TOP_GENES]]
    rest = np.sort(ids[order[N_TOP_GENES:]])
    tail = rest[rng.choice(len(rest), N_RANDOM_GENES, replace=False)]
    return np.sort(np.concatenate([top, tail]))


def build_fixture(shard: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return (expression rows, metadata rows, number of genes kept) from one shard."""
    rng = np.random.default_rng(SEED)
    con = connect(temp_dir=ROOT / "tmp" / "duckdb_tmp")
    shard_sql = f"read_parquet('{shard}')"
    base_filter = "plate = ? AND list_contains(?, cell_line_id)"

    meta = pd.read_parquet(METADATA_CACHE)
    meta = meta[(meta["plate"] == PLATE) & meta["cell_line"].isin(CELL_LINES)]
    cells = con.execute(
        f"SELECT BARCODE_SUB_LIB_ID, cell_line_id, sample FROM {shard_sql} WHERE {base_filter}", [PLATE, CELL_LINES]
    ).fetchdf()
    if cells["BARCODE_SUB_LIB_ID"].duplicated().any():
        raise RuntimeError("BARCODE_SUB_LIB_ID is not unique within the shard for the fixture lines")
    shard_counts = cells.groupby(["cell_line_id", "sample"], as_index=False).size().rename(
        columns={"cell_line_id": "cell_line", "size": "n"}
    )

    target = select_wells(meta, shard_counts, rng)
    barcodes = pick_barcodes(cells, target, rng)
    rows = con.execute(
        f"SELECT * FROM {shard_sql} WHERE {base_filter} AND list_contains(?, BARCODE_SUB_LIB_ID)",
        [PLATE, CELL_LINES, barcodes],
    ).fetchdf()
    con.close()
    if len(rows) != len(barcodes):
        raise RuntimeError(f"Expected {len(barcodes)} cells, got {len(rows)}")
    rows["genes"] = [np.asarray(g, dtype=np.int64) for g in rows["genes"]]
    rows["expressions"] = [np.asarray(v, dtype=np.float32) for v in rows["expressions"]]

    for genes, values in zip(rows["genes"], rows["expressions"]):
        if genes[0] != 1 or values[0] != -2 or not np.all(np.mod(values[1:], 1) == 0):
            raise RuntimeError("Unexpected cell layout: marker token or non-integer counts")

    keep = select_genes(rows["genes"], rng)
    masks = [np.isin(g, keep) | np.isin(g, SPECIAL_TOKENS) for g in rows["genes"]]
    rows["genes"] = [g[m] for g, m in zip(rows["genes"], masks)]
    rows["expressions"] = [v[m] for v, m in zip(rows["expressions"], masks)]
    rows = rows.sort_values(["cell_line_id", "sample", "BARCODE_SUB_LIB_ID"]).reset_index(drop=True)

    counts = rows.groupby(["cell_line_id", "plate", "sample"], as_index=False).size()
    counts = counts.rename(columns={"cell_line_id": "cell_line", "size": "n_cells"})
    labels = meta[["cell_line", "plate", "sample", "drug", "drugname_drugconc"]].drop_duplicates()
    metadata = counts.merge(labels, on=["cell_line", "plate", "sample"], how="left", validate="one_to_one")
    if metadata["drugname_drugconc"].isna().any():
        raise RuntimeError("Some fixture wells are missing from the metadata cache")
    metadata = metadata[METADATA_COLUMNS].astype({"n_cells": "int64"})
    metadata = metadata.sort_values(["cell_line", "plate", "sample"]).reset_index(drop=True)
    return rows, metadata, len(keep)


def build_fixture_outputs(expression_path: Path, metadata_path: Path, work_dir: Path) -> tuple[dict, Path]:
    """aggregate_shard + build_outputs on the fixture; outputs land in work_dir / 'out'."""
    work_dir.mkdir(parents=True, exist_ok=True)
    con = connect(temp_dir=work_dir)
    gene_file, cell_file = work_dir / "shard_genes.parquet", work_dir / "shard_cells.parquet"
    aggregate_shard(con, expression_path, CELL_LINES, [PLATE], gene_file, cell_file)
    out = work_dir / "out"
    result = build_outputs(
        con, [gene_file], [cell_file], pd.read_parquet(metadata_path), FIXTURE_CONFIG, out, work_dir / "qc_report.md"
    )
    con.close()
    return result, out


def write_expected_outputs() -> dict:
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
        result, out = build_fixture_outputs(EXPRESSION_PATH, METADATA_PATH, Path(tmp))
        shutil.copyfile(out / "pseudobulk.parquet", EXPECTED_PSEUDOBULK)
        shutil.copyfile(out / "logfc.parquet", EXPECTED_LOGFC)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep-shard", action="store_true", help="do not delete the downloaded shard")
    parser.add_argument("--expected-only", action="store_true", help="rebuild only expected_*.parquet from existing inputs")
    args = parser.parse_args()

    if args.expected_only:
        result = write_expected_outputs()
        print(f"logfc conditions: {result['summary']['n_logfc_conditions']}")
        for path in (EXPECTED_PSEUDOBULK, EXPECTED_LOGFC):
            print(f"{path.relative_to(ROOT)}: {path.stat().st_size:,} bytes")
        return

    local = DOWNLOAD_DIR / SHARD_FILE
    if not local.exists():
        local = download_shard(DOWNLOAD_DIR)
    try:
        schema = pq.read_schema(local)
        rows, metadata, n_genes = build_fixture(local)
    finally:
        if not args.keep_shard:
            shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(rows[schema.names], schema=schema, preserve_index=False)
    pq.write_table(table.replace_schema_metadata(schema.metadata), EXPRESSION_PATH, compression="zstd")
    metadata.to_parquet(METADATA_PATH, index=False)
    result = write_expected_outputs()

    print(metadata.to_string())
    print(f"cells: {len(rows)}  genes kept: {n_genes}  logfc conditions: {result['summary']['n_logfc_conditions']}")
    for path in (EXPRESSION_PATH, METADATA_PATH, EXPECTED_PSEUDOBULK, EXPECTED_LOGFC):
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
