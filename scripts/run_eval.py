"""Fit and score one model on every split repeat (`make eval MODEL=<name>`), or regenerate reports/results.md (`--report`)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase4 import harness, report  # noqa: E402
from phase4.predictors import PREDICTORS  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
RESULTS_DIR = EVAL_DIR / "results"
REPORT = ROOT / "reports" / "results.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(PREDICTORS))
    parser.add_argument("--report", action="store_true", help="regenerate reports/results.md from saved results")
    args = parser.parse_args()
    if args.model:
        cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
        out = harness.evaluate(args.model, cfg, EVAL_DIR, RESULTS_DIR)
        print(f"{args.model}: results in {out.relative_to(ROOT)}")
    if args.report or args.model:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(report.render_results(RESULTS_DIR), encoding="utf-8")
        print(f"report: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
