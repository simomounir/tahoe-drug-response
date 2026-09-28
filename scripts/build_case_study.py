"""Build reports/case_study.html from data/eval and its results (`make case-study`; phase 5a spec)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase5 import case_study  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
OUT = ROOT / "reports" / "case_study.html"


def main() -> None:
    start = time.perf_counter()
    cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    ctx = case_study.build_context(EVAL_DIR / "results", EVAL_DIR, cfg, ROOT / "evaluation.md",
                                   cell_meta=ROOT / "data" / "cache" / "cell_line_metadata.parquet")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(case_study.render(ctx), encoding="utf-8")
    print(f"case study: {OUT.relative_to(ROOT)} ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
