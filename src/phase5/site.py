"""The project site (phase 5d spec S2, S5, S6): landing page + case study + demo, zipped for a release.

dist/site-<version>/site/ is the site root (index.html, case-study/, demo/, .nojekyll); site.zip holds it and
SITE_MANIFEST.json records its sha256. The landing page's numbers come from `reproduce.headline` on the saved results.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import string
import zipfile
from html import escape
from pathlib import Path

from phase5 import figures, reproduce
from phase5.bundle import sha256

LANDING = Path(__file__).parent / "templates" / "landing.html"
REPO_URL = "https://github.com/simomounir/tahoe-drug-response"
AUTHOR = "simomounir"


class SiteError(RuntimeError):
    pass


def _exists_exact(root: Path, rel: Path) -> bool:
    """`rel` exists under `root` with exactly this case in every segment (GitHub Pages is case-sensitive; macOS is not)."""
    here = root
    for part in rel.parts:
        if not here.is_dir() or part not in os.listdir(here):
            return False
        here = here / part
    return True


def check_links(root: Path) -> list[str]:
    """Internal href/src targets in root/**/*.html that do not exist inside the site ("page -> target")."""
    root = Path(root).resolve()
    broken = []
    for page in sorted(root.rglob("*.html")):
        for target in re.findall(r"""(?:href|src)=["']([^"']+)["']""", page.read_text(encoding="utf-8")):
            if re.match(r"^(?:[a-z]+:|//|#)", target):
                continue
            path = target.split("#")[0].split("?")[0]
            resolved = Path(os.path.normpath(page.parent / path))
            if path in ("", ".", "./") or path.endswith("/") or resolved.is_dir():
                resolved = resolved / "index.html"
            ok = resolved.is_relative_to(root) and _exists_exact(root, resolved.relative_to(root))
            if not ok:
                broken.append(f"{page.relative_to(root)} -> {target}")
    return broken


def _landing(results_dir: Path, cfg: dict) -> str:
    h = reproduce.headline(results_dir, cfg)
    de = h["de_pearson"]
    rows = [{"model": m, "split": s, "estimate": v[0], "ci_low": v[1], "ci_high": v[2]}
            for m in reproduce.MODELS for s, v in de[m].items()]
    claim = h["claim"]["de_pearson"]
    ctx = {
        "verdict": h["verdict"], "verdict_class": h["verdict"].lower(),
        "neural": f"{de['neural']['both_unseen'][0]:.3f}", "ridge": f"{de['ridge']['both_unseen'][0]:.3f}",
        "mean": f"{de['global_mean']['both_unseen'][0]:.3f}",
        "claim_diff": f"{claim[0]:+.3f} [{claim[1]:+.3f}, {claim[2]:+.3f}]",
        "ridge_random": f"{de['ridge']['random'][0]:.3f}",
        "figure": figures.degradation_svg(rows), "repo": REPO_URL, "author": escape(AUTHOR),
    }
    return string.Template(LANDING.read_text(encoding="utf-8")).substitute(ctx)


def build(results_dir: Path, cfg: dict, case_study_html: Path, demo_dir: Path, out_dir: Path, version: str) -> Path:
    for need in (Path(case_study_html), Path(demo_dir) / "index.html", Path(demo_dir) / "data" / "index.json"):
        if not need.exists():
            raise SiteError(f"{need} missing: run `make case-study` and `make demo-data` first")
    out = Path(out_dir) / f"site-{version}"
    root = out / "site"
    if root.exists():
        shutil.rmtree(root)
    (root / "case-study").mkdir(parents=True)
    (root / "index.html").write_text(_landing(results_dir, cfg), encoding="utf-8")
    (root / ".nojekyll").write_text("", encoding="utf-8")
    shutil.copyfile(case_study_html, root / "case-study" / "index.html")
    shutil.copytree(demo_dir, root / "demo")
    index = json.loads((root / "demo" / "data" / "index.json").read_text(encoding="utf-8"))
    missing = [d["shard"] for d in index.get("drugs", {}).values() if not _exists_exact(root, Path("demo/data") / d["shard"])]
    broken = check_links(root) + [f"demo/data/index.json -> {m}" for m in missing]
    if broken:
        raise SiteError("broken internal links: " + "; ".join(broken))
    with zipfile.ZipFile(out / "site.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(root).as_posix())
    manifest = {"version": version, "files": {"site.zip": {"sha256": sha256(out / "site.zip"), "bytes": (out / "site.zip").stat().st_size}}}
    (out / "SITE_MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out
