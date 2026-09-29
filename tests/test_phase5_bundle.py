from __future__ import annotations

import json
from pathlib import Path

import pytest

from phase5 import bundle
from test_phase5_case_study import _eval_dir, _results


def _built(tmp_path):
    ev = _eval_dir(tmp_path)
    _results(ev / "results", {"global_mean": 0.6, "ridge": 0.6, "neural": 0.6})
    cfg = tmp_path / "eval.yaml"
    cfg.write_text("splits: {repeats: 1}\n")
    return bundle.build(ev, tmp_path / "dist", "v1", cfg)


def test_build_and_verify(tmp_path):
    out = _built(tmp_path)
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert {"level1.zip", "level2.zip"} <= set(manifest["files"])
    bundle.verify(out, manifest)
    assert (out / "README.md").exists()


def test_verify_flipped_byte(tmp_path):
    out = _built(tmp_path)
    p = out / "level1.zip"
    data = bytearray(p.read_bytes())
    data[len(data) // 2] ^= 0xFF
    p.write_bytes(bytes(data))
    with pytest.raises(bundle.BundleError, match="level1.zip"):
        bundle.verify(out, json.loads((out / "MANIFEST.json").read_text()))


def test_fetch_local_ok(tmp_path):
    out = _built(tmp_path)
    got = bundle.fetch(1, None, tmp_path / "cache", local_dir=out)
    assert (got / "results" / "ridge" / "per_condition.parquet").exists()
    got2 = bundle.fetch(2, None, tmp_path / "cache", local_dir=out)
    assert (got2 / "targets.npy").exists() and (got2 / "eval.yaml").exists()


def test_fetch_local_missing(tmp_path):
    out = _built(tmp_path)
    (out / "level2.zip").unlink()
    with pytest.raises(bundle.BundleError, match="level2.zip"):
        bundle.fetch(2, None, tmp_path / "cache", local_dir=out)


def test_bundle_json_uses_published_urls_and_manifest_shas(tmp_path):
    out = _built(tmp_path)
    manifest = json.loads((out / "MANIFEST.json").read_text())
    spec = bundle.bundle_json(manifest, "o/r", "bundle-v1")
    a = spec["assets"]["level2.zip"]
    assert a["url"] == "https://github.com/o/r/releases/download/bundle-v1/level2.zip"
    assert a["sha256"] == manifest["files"]["level2.zip"]["sha256"] and spec["tag"] == "bundle-v1"


def test_fetch_download_error_is_clean(tmp_path):
    out = _built(tmp_path)
    manifest = json.loads((out / "MANIFEST.json").read_text())
    spec = bundle.bundle_json(manifest, "o/r", "bundle-v1")
    spec["assets"]["level1.zip"]["url"] = (tmp_path / "nope.zip").as_uri()
    (tmp_path / "bundle.json").write_text(json.dumps(spec))
    with pytest.raises(bundle.BundleError, match="download"):
        bundle.fetch(1, tmp_path / "bundle.json", tmp_path / "cache")


def test_readme_links_licences(tmp_path):
    text = (_built(tmp_path) / "README.md").read_text()
    assert "creativecommons.org/licenses/by/4.0" in text and "creativecommons.org/publicdomain/zero/1.0" in text
    assert "10.25452/figshare.plus.27993248" in text


def test_ci_skips_level1_until_bundle_published():
    wf = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "reproduce.yml").read_text()
    assert "hashFiles('reproduce/bundle.json')" in wf
