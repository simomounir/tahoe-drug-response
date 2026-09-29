"""Check the headline against reproduce/expected.json (phase 5b spec).

    make reproduce              level 1: recompute statistics from the bundle's saved scores
    make reproduce-refit        level 2: refit global_mean / ridge / neural on both_unseen, then recompute
    BUNDLE_DIR=dist/bundle-v1   use a locally built bundle instead of downloading
    --write-expected            record the current full run (data/eval/results) as the expected values
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase5 import bundle, reproduce  # noqa: E402

EXPECTED = ROOT / "reproduce" / "expected.json"
BUNDLE_JSON = ROOT / "reproduce" / "bundle.json"
CACHE = ROOT / "data" / "cache" / "bundle"
TOL = {1: 0.0005, 2: 0.003}  # R5: level 1 exact at 3 decimals, level 2 within 0.003


def _fetch(level: int) -> Path:
    local = os.environ.get("BUNDLE_DIR")
    if local is None and not BUNDLE_JSON.exists():
        raise reproduce.ReproduceError("no published bundle yet (reproduce/bundle.json missing): build one with `make bundle` "
                                       "and run with BUNDLE_DIR=dist/bundle-v1")
    return bundle.fetch(level, BUNDLE_JSON, CACHE, local_dir=Path(local) if local else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--search", action="store_true", help="with --refit: rerun the neural hyperparameter search")
    parser.add_argument("--write-expected", action="store_true")
    parser.add_argument("--write-bundle-json", metavar="TAG", help="after publishing: record the release's download URLs + checksums")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    start = time.perf_counter()
    if args.write_bundle_json:
        manifest = json.loads((ROOT / "dist" / f"bundle-{args.write_bundle_json.removeprefix('bundle-')}" / "MANIFEST.json").read_text())
        spec = bundle.bundle_json(manifest, "simomounir/tahoe-drug-response", args.write_bundle_json)
        BUNDLE_JSON.write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {BUNDLE_JSON.relative_to(ROOT)}")
        return
    if args.write_expected:
        EXPECTED.parent.mkdir(exist_ok=True)
        EXPECTED.write_text(json.dumps(reproduce.headline(ROOT / "data" / "eval" / "results", cfg), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {EXPECTED.relative_to(ROOT)}")
        return
    try:
        expected = reproduce.load_expected(EXPECTED)
        level1 = _fetch(1)
        if args.refit:
            eval_dir = _fetch(2)
            cfg = reproduce.bundle_config(eval_dir, cfg)
            out = reproduce.refit_both_unseen(eval_dir, level1 / "results", cfg, CACHE / "refit", search=args.search or os.environ.get("REFIT_SEARCH") == "1")
            got, expected, level = reproduce.headline(out, cfg, splits=["both_unseen"]), reproduce.restrict(expected, ["both_unseen"]), 2
        else:
            got, level = reproduce.headline(level1 / "results", cfg), 1
    except (reproduce.ReproduceError, bundle.BundleError) as exc:
        sys.exit(f"reproduce: {exc}")
    bad = reproduce.compare(got, expected, TOL[level])
    seconds = time.perf_counter() - start
    if bad:
        print(f"reproduce level {level}: {len(bad)} value(s) differ from {EXPECTED.relative_to(ROOT)}:\n  " + "\n  ".join(bad))
        sys.exit(1)
    d = got["claim"]["de_pearson"]
    print(f"reproduce level {level}: OK in {seconds:.0f} s — verdict {got['verdict']}, both_unseen neural − ridge de_pearson "
          f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")


if __name__ == "__main__":
    main()
