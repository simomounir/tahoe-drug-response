"""Export the demo's data to site/demo/data/ (`make demo-data`; phase 5c spec)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase5 import demo_data  # noqa: E402


def main() -> None:
    start = time.perf_counter()
    cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    out = ROOT / "site" / "demo" / "data"
    s = demo_data.export(ROOT / "data" / "eval", ROOT / "data" / "eval" / "results", cfg, out,
                         cell_meta=ROOT / "data" / "cache" / "cell_line_metadata.parquet", log=lambda m: print(m, flush=True))
    print(f"demo data: {s['conditions']} conditions, {s['drugs']} shards, {s['bytes'] / 1e6:.1f} MB in "
          f"{out.relative_to(ROOT)} ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
