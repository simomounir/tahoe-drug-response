from __future__ import annotations

import argparse
import csv
import hashlib
import resource
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase1.build import build_outputs
from phase1.config import load_slice_config
from phase1.stream import aggregate_shard, connect, shards_for_plates

REPO_ID = "tahoebio/Tahoe-100M"
REAL_METADATA_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/metadata/obs_metadata.parquet"
CACHE_DIR = ROOT / "data" / "cache"
RAW_DIR = ROOT / "data" / "raw"
PLATE_MAP_PATH = CACHE_DIR / "shard_plate_map.parquet"
DUCKDB_MEMORY_LIMIT = "2GB"
SHARD_LOG_FIELDS = ["shard", "rows_read", "rows_kept", "bytes", "download_s", "aggregate_s", "peak_rss_mb"]


def _peak_rss_mb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def _connect(remote: bool = False) -> duckdb.DuckDBPyConnection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return connect(DUCKDB_MEMORY_LIMIT, CACHE_DIR / "duckdb_tmp", remote=remote)


def load_metadata_source(path_like: str | Path, selected_lines: list[str], pass_filter: str = "full") -> pd.DataFrame:
    """Return one row per (cell_line, plate, sample, drug, drugname_drugconc) with n_cells.

    Aggregation happens inside DuckDB; the per-cell table (~100M rows) never reaches pandas.
    Per-cell Parquet with a `pass_filter` column counts only cells with that value.
    """
    spec = str(path_like)
    if not (spec.startswith("http") or Path(spec).suffix.lower() in {".parquet", ".pq"}):
        df = pd.read_csv(spec)
        return df[df["cell_line"].isin(selected_lines)].copy()

    key_payload = spec + "|" + ",".join(sorted(selected_lines)) + f"|pass_filter={pass_filter}"
    key = hashlib.sha1(key_payload.encode()).hexdigest()[:12]
    cache_path = CACHE_DIR / f"metadata_by_sample_{key}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    con = _connect(remote=spec.startswith("http"))
    columns = [d[0] for d in con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [spec]).description]
    # Expression shards hold only pass_filter == 'full' cells, so the atlas count must match.
    pass_filter_clause = "AND pass_filter = ?" if "pass_filter" in columns else ""
    placeholders = ", ".join(["?"] * len(selected_lines))
    query = f"""
        SELECT cell_line, plate, sample, drug, drugname_drugconc, COUNT(*) AS n_cells
        FROM read_parquet(?)
        WHERE cell_line IN ({placeholders}) {pass_filter_clause}
        GROUP BY ALL
        ORDER BY cell_line, plate, sample
    """
    params = [spec, *selected_lines] + ([pass_filter] if pass_filter_clause else [])
    df = con.execute(query, params).fetchdf()
    con.close()
    df.to_parquet(cache_path, index=False)
    return df


def partial_dir_for(plate: str, selected_lines: list[str]) -> Path:
    """Per-plate cache, so adding a plate never invalidates plates already streamed."""
    key = hashlib.sha1((plate + "|" + ",".join(sorted(selected_lines))).encode()).hexdigest()[:12]
    return CACHE_DIR / "partials" / key


def process_shards(shards: list[int], selected_lines: list[str], plates: list[str], partial_dir: Path) -> None:
    """Download each shard, reduce it to per-well gene sums, delete it. Already-done shards are skipped."""
    partial_dir.mkdir(parents=True, exist_ok=True)
    log_path = partial_dir / "shard_log.csv"
    new_log = not log_path.exists()
    con = _connect()
    with log_path.open("a", newline="") as fh:
        log = csv.DictWriter(fh, fieldnames=SHARD_LOG_FIELDS)
        if new_log:
            log.writeheader()
        for n, shard in enumerate(shards, 1):
            gene_out = partial_dir / f"shard_{shard:05d}_genes.parquet"
            cell_out = partial_dir / f"shard_{shard:05d}_cells.parquet"
            if cell_out.exists():
                continue
            filename = f"data/train-{shard:05d}-of-03388.parquet"
            t0 = time.perf_counter()
            local = Path(hf_hub_download(REPO_ID, filename, repo_type="dataset", local_dir=RAW_DIR))
            t1 = time.perf_counter()
            kept = aggregate_shard(con, local, selected_lines, plates, gene_out, cell_out)
            t2 = time.perf_counter()
            row = {
                "shard": shard,
                "rows_read": pq.ParquetFile(local).metadata.num_rows,
                "rows_kept": kept,
                "bytes": local.stat().st_size,
                "download_s": round(t1 - t0, 2),
                "aggregate_s": round(t2 - t1, 2),
                "peak_rss_mb": _peak_rss_mb(),
            }
            local.unlink()
            log.writerow(row)
            fh.flush()
            print(f"[{n}/{len(shards)}] shard {shard}: {kept} cells kept, {row['download_s']}s + {row['aggregate_s']}s", flush=True)
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1 pipeline from a YAML slice config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "slice.yaml",
        help="Path to the slice config YAML file.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=REAL_METADATA_URL,
        help="Metadata CSV/Parquet path or remote Parquet URL. Defaults to the live Tahoe-100M metadata subset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "pseudobulk",
        help="Directory for pseudobulk, logfc, conditions and controls tables.",
    )
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "qc_phase1.md", help="QC report path.")
    parser.add_argument("--plates", nargs="+", default=None, help="Override the config's plates, e.g. --plates plate4")
    parser.add_argument("--max-shards", type=int, default=None, help="Only process the first N matching shards.")
    parser.add_argument("--dry-run", action="store_true", help="Report how many shards would be downloaded, then stop.")
    args = parser.parse_args()

    config = load_slice_config(args.config)
    selected_lines = list(config["selected_cell_lines"])
    plates = sorted(set(args.plates or config.get("plates") or []))
    if not plates:
        raise SystemExit("No plates: set `plates` in the config or pass --plates.")

    if not PLATE_MAP_PATH.exists():
        raise SystemExit(f"Missing {PLATE_MAP_PATH}. Run scripts/shard_plate_map.py first.")
    plate_map = pd.read_parquet(PLATE_MAP_PATH)
    # A shard straddling two plates is reduced once per plate, filtered to that plate's cells.
    shards_by_plate = {plate: shards_for_plates(plate_map, [plate]) for plate in plates}
    if args.max_shards is not None:
        shards_by_plate = {plate: shards[: args.max_shards] for plate, shards in shards_by_plate.items()}
    for plate, shards in shards_by_plate.items():
        todo = sum(not (partial_dir_for(plate, selected_lines) / f"shard_{s:05d}_cells.parquet").exists() for s in shards)
        print(f"{plate}: {len(shards)} shards, {todo} to download (~{todo * 85 / 1024:.1f} GB, one at a time, each deleted after use)")
    if args.dry_run or not any(shards_by_plate.values()):
        return

    gene_files, cell_files = [], []
    for plate, shards in shards_by_plate.items():
        partial_dir = partial_dir_for(plate, selected_lines)
        process_shards(shards, selected_lines, [plate], partial_dir)
        gene_files += [partial_dir / f"shard_{s:05d}_genes.parquet" for s in shards]
        cell_files += [partial_dir / f"shard_{s:05d}_cells.parquet" for s in shards]

    metadata = load_metadata_source(args.metadata, selected_lines)
    metadata = metadata[metadata["plate"].isin(plates)]

    con = _connect()
    result = build_outputs(
        con,
        gene_files,
        cell_files,
        metadata,
        config,
        Path(args.output_dir),
        Path(args.report),
    )
    con.close()
    summary, conditions = result["summary"], result["conditions"]

    print(f"Cells processed: {conditions['n_cells'].sum()} of {conditions['n_cells_atlas'].sum()} in the atlas for {plates}")
    print(f"Conditions retained: {summary['n_conditions_kept']} / {summary['n_conditions_total']}")
    print(f"Controls retained: {summary['n_control_conditions_kept']} / {summary['n_control_conditions_total']}")
    print(f"Treated conditions with logFC: {summary['n_logfc_conditions']}")
    print(f"Status: {summary['status']}")
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Contracts: passed. Outputs in {args.output_dir}, report at {args.report}")


if __name__ == "__main__":
    main()
