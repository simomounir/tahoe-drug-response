"""Reproduction bundle (phase 5b spec R1, R7, R9): build, verify and fetch.

level1.zip — every model's results.json + per_condition.parquet (the statistics are recomputed from these).
level2.zip — the eval set (conditions with features, genes, targets, DE sets, splits) + the eval config, for refitting.
MANIFEST.json — sha256 and size of each archive, git commit, package versions.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import shutil
import subprocess
import urllib.request
import zipfile
from importlib import metadata
from pathlib import Path
from string import Template

README = Path(__file__).parent / "templates" / "bundle_README.md"
LEVEL2_FILES = ["conditions.parquet", "genes.parquet", "targets.npy", "de_sets.parquet", "splits.parquet"]
PACKAGES = ["numpy", "pandas", "pyarrow", "duckdb", "torch", "rdkit", "matplotlib"]


class BundleError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> dict[str, str]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def _commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(eval_dir: Path, out_dir: Path, version: str, config_path: Path) -> Path:
    eval_dir, out = Path(eval_dir), Path(out_dir) / f"bundle-{version}"
    out.mkdir(parents=True, exist_ok=True)
    results = sorted(p for p in (eval_dir / "results").glob("*/*") if p.name in ("results.json", "per_condition.parquet"))
    if not results:
        raise BundleError(f"no results under {eval_dir / 'results'}")
    with zipfile.ZipFile(out / "level1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in results:
            z.write(p, f"results/{p.parent.name}/{p.name}")
    with zipfile.ZipFile(out / "level2.zip", "w", zipfile.ZIP_STORED) as z:  # targets barely compress (~0.7) and zlib is slow on them
        for name in LEVEL2_FILES:
            z.write(eval_dir / name, name)
        z.write(config_path, "eval.yaml")
    files = {n: {"sha256": sha256(out / n), "bytes": (out / n).stat().st_size} for n in ("level1.zip", "level2.zip")}
    manifest = {"version": version, "git_commit": _commit(eval_dir), "packages": _versions(), "files": files,
                "models": sorted({p.parent.name for p in results})}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "README.md").write_text(Template(README.read_text(encoding="utf-8")).substitute(
        version=version, commit=manifest["git_commit"], models=", ".join(manifest["models"]),
        level1_mb=f"{files['level1.zip']['bytes'] / 1e6:.0f}", level2_mb=f"{files['level2.zip']['bytes'] / 1e6:.0f}"), encoding="utf-8")
    return out


def verify(directory: Path, manifest: dict, names: list[str] | None = None) -> None:
    for name in names or list(manifest["files"]):
        path = Path(directory) / name
        if not path.exists():
            raise BundleError(f"{name}: missing from {directory}")
        if sha256(path) != manifest["files"][name]["sha256"]:
            raise BundleError(f"{name}: sha256 does not match the manifest (corrupted or partial download)")


def bundle_json(manifest: dict, repo: str, tag: str) -> dict:
    """reproduce/bundle.json for a published release: permanent download URLs + the manifest's checksums (R7)."""
    base = f"https://github.com/{repo}/releases/download/{tag}"
    return {"tag": tag, "version": manifest["version"], "git_commit": manifest["git_commit"],
            "assets": {n: {"url": f"{base}/{n}", "sha256": f["sha256"], "bytes": f["bytes"]} for n, f in manifest["files"].items()}}


def fetch(level: int, bundle_json: Path | None, cache_dir: Path, local_dir: Path | None = None) -> Path:
    """Directory with the unpacked level `level` archive, verified. `local_dir` uses a built bundle instead of downloading."""
    name, cache = f"level{level}.zip", Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    if local_dir is not None:
        manifest = json.loads((Path(local_dir) / "MANIFEST.json").read_text())
        verify(local_dir, manifest, [name])
        archive = Path(local_dir) / name
    else:
        spec = json.loads(Path(bundle_json).read_text())
        asset = spec["assets"][name]
        archive = cache / name
        if not archive.exists() or sha256(archive) != asset["sha256"]:
            tmp = cache / f"{name}.part"
            try:
                with urllib.request.urlopen(asset["url"], timeout=60) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
            except (OSError, http.client.HTTPException) as exc:  # URLError, timeouts, IncompleteRead
                raise BundleError(f"{name}: download from {asset['url']} failed ({exc})") from exc
            tmp.replace(archive)
        verify(cache, {"files": {name: asset}}, [name])
    target = cache / f"level{level}"
    if target.exists():
        shutil.rmtree(target)
    with zipfile.ZipFile(archive) as z:
        z.extractall(target)
    return target
