"""Download Tahoe-100M gene metadata once and write data/pseudobulk/genes.parquet (gene -> symbol, Ensembl ID).

The expression shards' `genes` values are the `token_id` column of metadata/gene_metadata.parquet.
"""
from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase1.stream import connect, sql_str

URL = "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/metadata/gene_metadata.parquet"
CACHE = ROOT / "data" / "cache" / "gene_metadata.parquet"
OUT = ROOT / "data" / "pseudobulk" / "genes.parquet"
MAX_RETRIES = 6


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
                fh.write(resp.read())
            tmp.replace(dest)
            return
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            wait = 30 * (attempt + 1)
            print(f"Rate limited by Hugging Face; waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
    raise SystemExit("Still rate limited; re-run later.")


def write_genes(con, source: Path, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT token_id::BIGINT AS gene, gene_symbol, ensembl_id
            FROM read_parquet({sql_str(source)}) ORDER BY gene
        ) TO {sql_str(out)} (FORMAT parquet, COMPRESSION zstd)
        """
    )
    return con.execute(f"SELECT COUNT(*) FROM read_parquet({sql_str(out)})").fetchone()[0]


def main() -> None:
    if CACHE.exists():
        print(f"Using cached {CACHE.relative_to(ROOT)}")
    else:
        print(f"Downloading {URL}")
        download(URL, CACHE)
    con = connect("1GB")
    n = write_genes(con, CACHE, OUT)
    lo, hi, n_distinct = con.execute(
        f"SELECT MIN(gene), MAX(gene), COUNT(DISTINCT gene) FROM read_parquet({sql_str(OUT)})"
    ).fetchone()
    print(f"Wrote {OUT.relative_to(ROOT)}: {n} rows, {n_distinct} distinct genes, gene range {lo}-{hi}")
    con.close()


if __name__ == "__main__":
    main()
