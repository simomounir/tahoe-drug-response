from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINES = ["L2", "L1"]


def _load():
    spec = importlib.util.spec_from_file_location("run_phase1_partials", ROOT / "scripts" / "run_phase1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_dir_is_per_plate_and_order_independent():
    rp = _load()
    assert rp.partial_dir_for("plate3", LINES) == rp.partial_dir_for("plate3", sorted(LINES))
    assert rp.partial_dir_for("plate3", LINES) != rp.partial_dir_for("plate1", LINES)
    assert rp.partial_dir_for("plate3", LINES).parent == rp.CACHE_DIR / "partials"


def test_single_plate_key_matches_previous_cache_layout():
    # Caches built before per-plate keying (plates=[plate3]) must still be found.
    rp = _load()
    old_key = hashlib.sha1(("plate3" + "|" + ",".join(sorted(LINES))).encode()).hexdigest()[:12]
    assert rp.partial_dir_for("plate3", LINES).name == old_key
