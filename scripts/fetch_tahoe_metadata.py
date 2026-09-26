"""Fetch Tahoe-100M's drug and cell-line metadata at the revision pinned in configs/features.yaml.

Both files are small. A file already on disk is kept if its sha256 matches the one Hugging Face
records for the pinned revision; otherwise it is downloaded again. A mismatch after download fails.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url  # noqa: E402

from phase3.config import load_features_config  # noqa: E402

DESTINATIONS = {
    "metadata/drug_metadata.parquet": ROOT / "data" / "cache" / "hf" / "metadata" / "drug_metadata.parquet",
    "metadata/cell_line_metadata.parquet": ROOT / "data" / "cache" / "cell_line_metadata.parquet",
}


def planned_downloads(cfg: dict) -> list[dict]:
    return [
        {"repo": cfg["tahoe"]["repo"], "repo_file": repo_file, "revision": cfg["tahoe"]["revision"], "dest": dest}
        for repo_file, dest in DESTINATIONS.items()
    ]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_sha256(item: dict) -> str:
    url = hf_hub_url(item["repo"], item["repo_file"], repo_type="dataset", revision=item["revision"])
    return get_hf_file_metadata(url).etag.strip('"')


def main() -> None:
    for item in planned_downloads(load_features_config()):
        dest, expected = item["dest"], expected_sha256(item)
        if dest.exists() and sha256_of(dest) == expected:
            print(f"ok (cached, sha256 verified): {dest.relative_to(ROOT)}")
            continue
        downloaded = Path(hf_hub_download(item["repo"], item["repo_file"], repo_type="dataset", revision=item["revision"]))
        if sha256_of(downloaded) != expected:
            raise SystemExit(f"sha256 mismatch for {item['repo_file']} at {item['revision']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, dest)
        print(f"downloaded: {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
