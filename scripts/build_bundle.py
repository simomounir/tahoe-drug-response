"""Build dist/bundle-<version>/ from data/eval (`make bundle`; phase 5b spec R1, R9)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase5 import bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    out = bundle.build(ROOT / "data" / "eval", ROOT / "dist", args.version, ROOT / "configs" / "eval.yaml")
    print(f"bundle: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
