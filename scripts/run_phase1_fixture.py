from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase1.conditions import load_fixture_metadata
from phase1.pipeline import run_phase1


FIXTURE_METADATA = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "metadata_small.csv"


def make_expression_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"condition_id": "c1", "genes": [0, 1], "expressions": [3.0, 4.0], "n_cells": 1},
            {"condition_id": "c1", "genes": [1, 2], "expressions": [2.0, 5.0], "n_cells": 1},
            {"condition_id": "c2", "genes": [0, 1], "expressions": [1.0, 1.0], "n_cells": 1},
            {"condition_id": "c3", "genes": [0, 1], "expressions": [2.0, 2.0], "n_cells": 1},
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1 fixture pipeline.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/phase1_fixture"),
        help="Directory for the generated CSV and QC report outputs.",
    )
    parser.add_argument(
        "--selected-lines",
        default="CVCL_0546,CVCL_0459",
        help="Comma-separated cell lines to include from the fixture metadata.",
    )
    args = parser.parse_args()

    metadata = load_fixture_metadata(FIXTURE_METADATA)
    selected = [line.strip() for line in args.selected_lines.split(",") if line.strip()]

    result = run_phase1(
        metadata=metadata,
        expression=make_expression_fixture(),
        selected_lines=selected,
        min_cells_per_condition=5,
        min_control_cells=6,
        output_dir=args.output_dir,
    )

    print(f"Conditions retained: {result['summary']['n_conditions_kept']} / {result['summary']['n_conditions_total']}")
    print(f"Controls retained: {result['summary']['n_control_conditions_kept']} / {result['summary']['n_control_conditions_total']}")
    print(f"Status: {result['summary']['status']}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
