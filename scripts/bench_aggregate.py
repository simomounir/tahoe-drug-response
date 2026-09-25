"""Time aggregate_shard on the same local shards with the same code, in whichever interpreter runs it.

Used for the arm64 vs x86_64 (Rosetta) comparison in docs/performance_baseline.md:
    .venv/bin/python scripts/bench_aggregate.py && .venv-x86/bin/python scripts/bench_aggregate.py
Downloads 3 shards (~300 MB) to data/raw/ once; delete them afterwards.
"""
import platform, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import duckdb
from huggingface_hub import hf_hub_download
from phase1.config import load_slice_config
from phase1.stream import aggregate_shard, connect

SHARDS, PLATE, REPEATS = [0, 10, 19], "plate4", 5
raw, out = ROOT / "data/raw", ROOT / "tmp/bench_aggregate"
out.mkdir(parents=True, exist_ok=True)
lines = list(load_slice_config(ROOT / "configs/slice.yaml")["selected_cell_lines"])
con = connect("2GB", ROOT / "tmp/bench_aggregate/duckdb_tmp")
print(f"{platform.machine()} Python {platform.python_version()} DuckDB {duckdb.__version__}")
for shard in SHARDS:
    local = Path(hf_hub_download("tahoebio/Tahoe-100M", f"data/train-{shard:05d}-of-03388.parquet", repo_type="dataset", local_dir=raw))
    times = []
    for _ in range(REPEATS + 1):  # first run is warm-up, discarded
        t = time.perf_counter()
        aggregate_shard(con, local, lines, [PLATE], out / "g.parquet", out / "c.parquet")
        times.append(time.perf_counter() - t)
    print(f"shard {shard}: median {statistics.median(times[1:]):.3f} s (min {min(times[1:]):.3f}, max {max(times[1:]):.3f})")
