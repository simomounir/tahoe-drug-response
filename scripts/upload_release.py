"""Create a DRAFT GitHub release with the reproduction bundle (`make bundle-upload`; phase 5b spec R3).

Needs GITHUB_TOKEN in the environment (a fine-grained token with Contents: read and write on the repository); the token
is never written anywhere. The release stays a draft, invisible to the public, until the owner reviews it and clicks
Publish on GitHub. This script never publishes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "simomounir/tahoe-drug-response"
ASSETS = ["level1.zip", "level2.zip", "MANIFEST.json", "README.md"]
TYPES = {".zip": "application/zip", ".json": "application/json", ".md": "text/markdown"}


class UploadError(RuntimeError):
    pass


def _http(method: str, url: str, headers: dict, body) -> tuple[int, dict]:
    if isinstance(body, dict):
        data, headers = json.dumps(body).encode(), {**headers, "Content-Type": "application/json"}
    elif isinstance(body, Path):
        data, headers = open(body, "rb"), {**headers, "Content-Length": str(body.stat().st_size)}
    else:
        data = None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    finally:
        if hasattr(data, "close"):
            data.close()


def _check(status: int, payload: dict, what: str) -> dict:
    if status >= 300:
        codes = ", ".join(e.get("code", "") for e in payload.get("errors", []))
        raise UploadError(f"{what}: HTTP {status} {payload.get('message', '')} {codes}".strip())
    return payload


def upload(bundle_dir: Path, version: str, repo: str, token: str | None, http=_http) -> str:
    if not token:
        raise UploadError("set GITHUB_TOKEN (fine-grained, Contents: read and write on the repository)")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    readme = (Path(bundle_dir) / "README.md").read_text(encoding="utf-8") if (Path(bundle_dir) / "README.md").exists() else ""
    release = _check(*http("POST", f"https://api.github.com/repos/{repo}/releases", headers,
                           {"tag_name": f"bundle-{version}", "target_commitish": "main", "name": f"Reproduction bundle {version}",
                            "body": readme, "draft": True}), "create draft release")
    if release.get("draft") is not True:
        raise UploadError(f"release {release.get('id')} is not a draft; stopping without uploading")
    base = release["upload_url"].split("{")[0]
    for name in ASSETS:
        path = Path(bundle_dir) / name
        _check(*http("POST", f"{base}?name={name}", {**headers, "Content-Type": TYPES[path.suffix]}, path), f"upload {name}")
        print(f"uploaded {name}", flush=True)
    return release.get("html_url", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--repo", default=REPO)
    args = parser.parse_args()
    try:
        url = upload(ROOT / "dist" / f"bundle-{args.version}", args.version, args.repo, os.environ.get("GITHUB_TOKEN"))
    except UploadError as exc:
        sys.exit(f"upload: {exc}")
    print(f"draft release created: {url}\nReview it on GitHub and click Publish; then run `python scripts/reproduce.py --write-bundle-json bundle-{args.version}` and commit reproduce/bundle.json.")


if __name__ == "__main__":
    main()
