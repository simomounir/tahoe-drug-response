from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "upload_release.py"
spec = importlib.util.spec_from_file_location("upload_release", SCRIPT)
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)


def _bundle(tmp_path):
    d = tmp_path / "bundle-v1"
    d.mkdir()
    for n in ["level1.zip", "level2.zip", "MANIFEST.json", "README.md"]:
        (d / n).write_bytes(b"x" * 10)
    return d


def test_upload_creates_draft_only(tmp_path):
    calls = []

    def http(method, url, headers, body):
        calls.append((method, url, body if isinstance(body, dict) else None))
        if url.endswith("/releases"):
            return 201, {"id": 7, "draft": True, "html_url": "https://github.com/o/r/releases/tag/untagged-1",
                         "upload_url": "https://uploads.github.com/repos/o/r/releases/7/assets{?name,label}"}
        return 201, {"name": url.split("name=")[1]}

    url = upload.upload(_bundle(tmp_path), "v1", "o/r", "tok", http=http)
    assert calls[0][0] == "POST" and calls[0][2]["draft"] is True and calls[0][2]["tag_name"] == "bundle-v1"
    assert sorted(c[1].split("name=")[1] for c in calls[1:]) == ["MANIFEST.json", "README.md", "level1.zip", "level2.zip"]
    assert not any(c[2] and c[2].get("draft") is False for c in calls)
    assert "releases" in url


def test_upload_requires_token(tmp_path):
    with pytest.raises(upload.UploadError, match="GITHUB_TOKEN"):
        upload.upload(_bundle(tmp_path), "v1", "o/r", None, http=lambda *a: (201, {}))


def test_upload_api_error(tmp_path):
    with pytest.raises(upload.UploadError, match="422.*already_exists"):
        upload.upload(_bundle(tmp_path), "v1", "o/r", "tok", http=lambda *a: (422, {"message": "Validation Failed", "errors": [{"code": "already_exists"}]}))


def test_upload_refuses_non_draft(tmp_path):
    with pytest.raises(upload.UploadError, match="not a draft"):
        upload.upload(_bundle(tmp_path), "v1", "o/r", "tok", http=lambda *a: (201, {"id": 1, "draft": False, "upload_url": "u{?name}"}))
