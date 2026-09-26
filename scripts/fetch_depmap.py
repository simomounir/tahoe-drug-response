"""Download DepMap 24Q4 files from Figshare+ (article 27993248), verify MD5, convert to Parquet.

Names, file IDs and MD5s come from the Figshare API at run time
(https://api.figshare.com/v2/articles/<article>), never hard-coded, so a re-run always checks
against the live record. Each CSV is downloaded to data/cache/depmap/, MD5-checked against the
API's `computed_md5`, converted to Parquet with DuckDB, and the CSV is deleted. Resumable: a file
whose Parquet + `.md5` sidecar already match the API's MD5 is skipped.

Network: ~660 MB once (Model.csv, OmicsExpressionProteinCodingGenesTPMLogp1.csv,
OmicsSomaticMutationsMatrixDamaging.csv, OmicsSomaticMutationsMatrixHotspot.csv), per
configs/features.yaml `depmap.files` (or the spec defaults if that file does not exist yet).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase1.stream import connect, sql_str
from phase3.config import load_features_config

API_URL = "https://api.figshare.com/v2/articles/{article}"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/{file_id}"
OUT_DIR = ROOT / "data" / "cache" / "depmap"
MAX_RETRIES = 6


def fetch_article_files(article: int) -> dict[str, dict]:
    """name -> {id, computed_md5, size, ...} for every file on the article, read from the Figshare API."""
    req = urllib.request.Request(API_URL.format(article=article), headers={"User-Agent": "phase3-fetch-depmap"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    return {f["name"]: f for f in payload["files"]}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "phase3-fetch-depmap"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp.replace(dest)
            return
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            wait = 30 * (attempt + 1)
            print(f"Rate limited by Figshare; waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
    raise SystemExit(f"Still rate limited downloading {url}; re-run later.")


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_to_parquet(con, csv_path: Path, parquet_path: Path) -> None:
    """Convert with DuckDB (columnar, handles ~19,000-column files without loading a dense pandas frame)."""
    con.execute(
        f"COPY (SELECT * FROM read_csv_auto({sql_str(csv_path)}, ALL_VARCHAR=FALSE)) "
        f"TO {sql_str(parquet_path)} (FORMAT parquet, COMPRESSION zstd)"
    )


def fetch_one(con, name: str, meta: dict, out_dir: Path) -> dict:
    parquet_path = out_dir / (Path(name).stem + ".parquet")
    md5_sidecar = parquet_path.with_suffix(".md5")
    expected_md5 = meta["computed_md5"]
    if parquet_path.exists() and md5_sidecar.exists() and md5_sidecar.read_text().strip() == expected_md5:
        print(f"Skipping {name}: already downloaded, MD5-verified ({expected_md5}), and converted")
        return {"name": name, "md5": expected_md5, "size": meta.get("size"), "skipped": True}

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / name
    url = DOWNLOAD_URL.format(file_id=meta["id"])
    print(f"Downloading {name} ({meta.get('size', 0) / 1e6:.1f} MB) from {url}", flush=True)
    download(url, csv_path)
    actual_md5 = md5sum(csv_path)
    if actual_md5 != expected_md5:
        csv_path.unlink(missing_ok=True)
        raise SystemExit(f"{name}: MD5 mismatch, expected {expected_md5}, got {actual_md5}")
    print(f"{name}: MD5 verified ({actual_md5})", flush=True)

    print(f"Converting {name} -> {parquet_path.name}", flush=True)
    csv_to_parquet(con, csv_path, parquet_path)
    csv_path.unlink()
    md5_sidecar.write_text(expected_md5 + "\n")
    return {"name": name, "md5": actual_md5, "size": meta.get("size"), "skipped": False}


def main() -> None:
    config = load_features_config()
    depmap_cfg = config["depmap"]
    article = depmap_cfg["figshare_article"]
    print(f"Reading file list for Figshare article {article}")
    files = fetch_article_files(article)
    # 4 GB, not the 2 GB used elsewhere (rule 4): this connection only ever reads a local CSV and
    # writes local Parquet (no `remote=True`, no join against the ~100M-row metadata), so the
    # tighter limit that protects the streaming aggregation doesn't apply here.
    con = connect("4GB", ROOT / "data" / "cache" / "duckdb_tmp")
    results = []
    try:
        for name in depmap_cfg["files"]:
            if name not in files:
                raise SystemExit(f"{name} not found in Figshare article {article}: have {sorted(files)[:5]}...")
            results.append(fetch_one(con, name, files[name], OUT_DIR))
    finally:
        con.close()
    print("\nSummary:")
    for r in results:
        status = "skipped (cached)" if r["skipped"] else "downloaded + verified"
        print(f"  {r['name']}: md5={r['md5']} ({status})")


if __name__ == "__main__":
    main()
