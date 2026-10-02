from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from phase5 import site
from test_phase5_case_study import _results
from test_phase5_reproduce import CFG

NAV = '<nav><a href="../">Home</a><a href="../case-study/">Case study</a><a href="../demo/">Demo</a></nav>'


def _inputs(tmp_path):
    _results(tmp_path / "results", {"global_mean": 0.61, "ridge": 0.62, "neural": 0.63})
    cs = tmp_path / "case_study.html"
    cs.write_text(f"<html><body>{NAV}<p>case</p></body></html>")
    demo = tmp_path / "demo"
    (demo / "data").mkdir(parents=True)
    (demo / "index.html").write_text(f'<html><body>{NAV}<script src="demo.js"></script></body></html>')
    (demo / "demo.js").write_text("// js")
    (demo / "data" / "index.json").write_text("{}")
    return cs, demo


def test_site_builds(tmp_path):
    cs, demo = _inputs(tmp_path)
    out = site.build(tmp_path / "results", CFG, cs, demo, tmp_path / "dist", "v1")
    root = out / "site"
    for f in ["index.html", ".nojekyll", "case-study/index.html", "demo/index.html", "demo/demo.js", "demo/data/index.json"]:
        assert (root / f).exists(), f
    manifest = json.loads((out / "SITE_MANIFEST.json").read_text())
    assert manifest["files"]["site.zip"]["sha256"] == site.sha256(out / "site.zip")
    with zipfile.ZipFile(out / "site.zip") as z:
        assert "index.html" in z.namelist() and "demo/data/index.json" in z.namelist()


def test_landing_numbers(tmp_path):
    cs, demo = _inputs(tmp_path)
    html = (site.build(tmp_path / "results", CFG, cs, demo, tmp_path / "dist", "v1") / "site" / "index.html").read_text()
    assert "0.630" in html and "0.620" in html and "0.610" in html and ("FAILURE" in html or "SUCCESS" in html)
    assert "simomounir" in html and "<svg" in html


def test_check_links_catches_broken(tmp_path):
    root = tmp_path / "s"
    (root / "a").mkdir(parents=True)
    (root / "index.html").write_text('<a href="a/">ok</a><a href="missing/">bad</a><a href="https://x.org">ext</a><a href="#top">anchor</a>')
    (root / "a" / "index.html").write_text('<img src="../nope.png">')
    broken = site.check_links(root)
    assert sorted(broken) == ["a/index.html -> ../nope.png", "index.html -> missing/"]


def test_build_fails_on_broken_link(tmp_path):
    cs, demo = _inputs(tmp_path)
    cs.write_text('<a href="../nowhere/">x</a>')
    with pytest.raises(site.SiteError, match="nowhere"):
        site.build(tmp_path / "results", CFG, cs, demo, tmp_path / "dist", "v1")


def test_pages_workflow():
    wf = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pages.yml").read_text()
    assert "actions/deploy-pages" in wf and "site.zip" in wf and "sha256" in wf


def test_check_links_escape_root(tmp_path):
    root = tmp_path / "s"
    (root / "a").mkdir(parents=True)
    (tmp_path / "outside.html").write_text("x")
    (root / "index.html").write_text('<a href="a/">ok</a>')
    (root / "a" / "index.html").write_text('<a href="../../outside.html">escapes</a>')
    assert site.check_links(root) == ["a/index.html -> ../../outside.html"]


def test_check_links_case_and_quotes(tmp_path):
    root = tmp_path / "s"
    (root / "demo").mkdir(parents=True)
    (root / "demo" / "index.html").write_text("x")
    (root / "index.html").write_text("<a href='Demo/'>case</a><a href='demo/'>ok</a>")
    assert site.check_links(root) == ["index.html -> Demo/"]


def test_build_checks_demo_shards(tmp_path):
    cs, demo = _inputs(tmp_path)
    (demo / "data" / "index.json").write_text(json.dumps({"drugs": {"D0": {"shard": "D0-abc.json"}}}))
    with pytest.raises(site.SiteError, match="D0-abc.json"):
        site.build(tmp_path / "results", CFG, cs, demo, tmp_path / "dist", "v1")


def test_landing_has_no_handwritten_timings(tmp_path):
    cs, demo = _inputs(tmp_path)
    html = (site.build(tmp_path / "results", CFG, cs, demo, tmp_path / "dist", "v1") / "site" / "index.html").read_text()
    assert "minutes" not in html and "in seconds" not in html


def test_pages_workflow_validates_tag():
    wf = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pages.yml").read_text()
    assert "^site-v[0-9]+$" in wf
