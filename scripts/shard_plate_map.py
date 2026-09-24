"""Record which plates each Tahoe-100M expression shard holds, reading only Parquet footers.

Resumable: shards already in the output file are skipped.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase1.stream import connect, sql_list

N_SHARDS = 3388
URL = "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/data/train-{:05d}-of-03388.parquet"
OUT = ROOT / "data" / "cache" / "shard_plate_map.parquet"
BATCH = 50
MAX_RETRIES = 6


def fetch_batch(con: duckdb.DuckDBPyConnection, batch: list[int]) -> pd.DataFrame:
    query = f"""
        SELECT file_name, row_group_id, row_group_num_rows AS num_rows,
               stats_min_value AS plate_min, stats_max_value AS plate_max
        FROM parquet_metadata({sql_list(URL.format(i) for i in batch)})
        WHERE path_in_schema = 'plate'
    """
    for attempt in range(MAX_RETRIES):
        try:
            return con.execute(query).fetchdf()
        except duckdb.HTTPException as exc:
            if "429" not in str(exc):
                raise
            wait = min(300, 30 * 2**attempt)
            print(f"Rate limited by Hugging Face; waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
    raise SystemExit("Still rate limited. Progress is saved; re-run later to resume.")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["shard"])
    todo = sorted(set(range(N_SHARDS)) - set(done["shard"].astype(int)))
    con = connect("1GB", remote=True)

    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        df = fetch_batch(con, batch)
        df["shard"] = df.pop("file_name").str.extract(r"train-(\d+)-of")[0].astype(int)
        done = pd.concat([done, df], ignore_index=True) if len(done) else df
        done.to_parquet(OUT, index=False)
        print(f"{done['shard'].nunique()}/{N_SHARDS} shards mapped", flush=True)

    con.close()
    missing_stats = done["plate_min"].isna().sum()
    if missing_stats:
        print(f"WARNING: {missing_stats} row groups have no plate statistics")
    by_plate = done.groupby("plate_min")["shard"].nunique()
    print("Shards per plate (by row-group min):")
    print(by_plate.to_string())


if __name__ == "__main__":
    main()
