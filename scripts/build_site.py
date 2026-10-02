"""Assemble the project site into dist/site-<version>/ (`make site`; phase 5d spec). Needs `make case-study` and `make demo-data`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase5 import site  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    try:
        out = site.build(ROOT / "data" / "eval" / "results", cfg, ROOT / "reports" / "case_study.html", ROOT / "site" / "demo",
                         ROOT / "dist", args.version)
    except site.SiteError as exc:
        sys.exit(f"site: {exc}")
    size = (out / "site.zip").stat().st_size / 1e6
    print(f"site: {out.relative_to(ROOT)} (site.zip {size:.0f} MB, links OK)")


if __name__ == "__main__":
    main()
