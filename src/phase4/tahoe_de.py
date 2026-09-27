"""Locate and download Tahoe-100M's per-condition DESeq2 table (metadata/pseudobulk_differential_expression).

The 1,026 shards are not named by content. Each cell line occupies one contiguous block of shards,
sorted within the block by (plate, drug, gene), but blocks are not in any obvious order, so a line is
found by a coarse fixed-stride scan of shard footers (stride 5) followed by a short walk back to the
block start (see docs/target_crosscheck.md). Footer probes are cached in memory and can be persisted
with save_probe_cache / load_probe_cache so they are paid once. Moved from
scripts/crosscheck_tahoe_de.py (phase 4a, task P4a.1); downloads are pinned to a dataset revision.
"""
from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

from phase1.stream import sql_list, sql_str
from phase3.crosscheck import drug_filter_values

N_FILES = 1026
FILE_PATH = "metadata/pseudobulk_differential_expression/train-{:05d}-of-01026.parquet"
REVISION = "main"  # overridden by set_revision(); callers pass configs/features.yaml tahoe.revision
OUR_PLATES = {"1", "2", "3"}
MAX_RETRIES = 6
SAFETY_MARGIN_BYTES = 120_000_000  # don't start a step likely to blow past the cap
EXTRACT_COLUMNS = [
    "gene_name", "log2FoldChange", "padj", "n_cells_trt", "n_cells_ctrl", "drug", "plate",
    "concentration", "concentration_unit",
]


def file_url(file_idx: int, revision: str) -> str:
    return f"https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/{revision}/" + FILE_PATH.format(file_idx)


def set_revision(revision: str) -> None:
    global REVISION
    REVISION = revision


class NetworkBudget:
    """Tracks bytes spent (Parquet-footer probes + full-shard downloads) against the network cap (`crosscheck.network_cap_gb`)."""

    def __init__(self, cap: int):
        self.cap = cap
        self.spent = 0
        self.probe_bytes = 0
        self.download_bytes = 0
        self.n_probes = 0
        self.n_downloads = 0
        self.stopped = False

    def has_room_for(self, estimated_bytes: int) -> bool:
        return self.spent + estimated_bytes <= self.cap - SAFETY_MARGIN_BYTES

    def add_probe(self, n_bytes: int) -> None:
        self.spent += n_bytes
        self.probe_bytes += n_bytes
        self.n_probes += 1

    def add_download(self, n_bytes: int) -> None:
        self.spent += n_bytes
        self.download_bytes += n_bytes
        self.n_downloads += 1


def with_retry(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (duckdb.HTTPException, urllib.error.HTTPError) as exc:
            code = getattr(exc, "code", None) or (str(exc))
            if "429" not in str(code) and "429" not in str(exc):
                raise
            wait = 30 * (attempt + 1)
            print(f"Rate limited by Hugging Face; waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
    raise SystemExit("Still rate limited by Hugging Face; re-run later.")


# ---------------------------------------------------------------------------
# Locating shards: footer-only probes (coarse stride scan, then local refine)
# ---------------------------------------------------------------------------
#
# A first version of this script assumed the shards were sorted globally by
# (cell line, plate, drug, gene) and used a binary search to locate each cell line's block.
# Direct probing disproved that: sampling row 0 of shards 0, 300, 500, 700, 900 showed cell
# lines CVCL_0023, CVCL_0334, CVCL_0293, CVCL_1056, CVCL_0131 -- not monotonic, so binary search
# silently produced false "not found" results for real cell lines (caught before the real run:
# see docs/target_crosscheck.md). The true structure (empirically, from many probes): each cell
# line occupies one *contiguous* block of shards -- everything within a block is sorted by
# (plate, drug, gene) -- but the *order of blocks* across the file range is not the cell lines'
# alphabetical order (apparently it's whatever order Tahoe's own export pipeline produced them
# in). Blocks are large: CVCL_0023 alone spans at least 20 shards. So instead of binary search,
# this does one coarse, fixed-stride scan across all 1,026 shards to find every cell line's
# block at least once (blocks average ~20 shards; a stride of 5 guarantees at least one sample
# lands in any block of 5+ shards), then a short local walk from the nearest hit to the block's
# true start.

PROBE_SIZE_ESTIMATE = 6_000_000  # observed footer size on this table is ~5.75 MB; round up
COARSE_STRIDE = 5


def probe_file(con: duckdb.DuckDBPyConnection, file_idx: int, cache: dict, budget: NetworkBudget) -> pd.DataFrame | None:
    """Per-row-group (cell_line, plate) of shard `file_idx`, from its Parquet footer only.

    A shard's footer carries stats for *every* row group regardless of which row groups or
    columns we ask for, so this costs the same ~6 MB whether we read one row group or all of
    them -- reading all of them is free and necessary: a cell line's block does not always start
    at a shard boundary (a shard is a fixed ~90 MB chunk of the globally sorted table, so a small
    cell line's entire block can sit in the *middle* of a shard that starts with the previous
    cell line's tail). Checking only row group 0 of each shard misses those cell lines entirely;
    this returns the full per-row-group sequence so the caller can search and download correctly.
    """
    if file_idx in cache:
        return cache[file_idx]
    if not (0 <= file_idx < N_FILES):
        return None
    url = file_url(file_idx, REVISION)
    df = with_retry(
        con.execute,
        f"""
        SELECT row_group_id, stats_min_value, path_in_schema
        FROM parquet_metadata({sql_list([url])})
        WHERE path_in_schema IN ('Cell_ID_Cellosaur', 'plate')
        ORDER BY row_group_id
        """,
    )
    df = df.fetchdf()
    budget.add_probe(PROBE_SIZE_ESTIMATE)
    piv = df.pivot(index="row_group_id", columns="path_in_schema", values="stats_min_value")
    piv = piv.rename(columns={"Cell_ID_Cellosaur": "cell_line"})
    cache[file_idx] = piv
    return piv


def build_coarse_index(con, cache: dict, budget: NetworkBudget) -> dict[str, list[int]]:
    """cell_line -> sorted list of sampled shard indices where it was observed.

    One fixed-stride pass over all 1,026 shards; every probe is cached, so this is the *only*
    time most shards are touched at all.
    """
    hits: dict[str, list[int]] = {}
    for file_idx in range(0, N_FILES, COARSE_STRIDE):
        if not budget.has_room_for(PROBE_SIZE_ESTIMATE):
            print(f"Network cap reached during coarse index scan at shard {file_idx}/{N_FILES}", flush=True)
            break
        piv = probe_file(con, file_idx, cache, budget)
        if piv is None or "cell_line" not in piv.columns:
            continue
        for cl in piv["cell_line"].dropna().unique():
            hits.setdefault(cl, []).append(file_idx)
    return hits


def locate_cell_line_start(con, target: str, coarse_hits: dict[str, list[int]], cache: dict, budget: NetworkBudget) -> int | None:
    """The true first shard of `target`'s block, refined from the nearest coarse-scan hit.

    Blocks are contiguous, so walking backward from any shard known to be inside the block,
    one shard at a time, reaches the true start in at most `COARSE_STRIDE` probes.
    """
    hits = coarse_hits.get(target)
    if not hits:
        return None  # not seen anywhere in the coarse scan -- see docs/target_crosscheck.md caveat
    file_idx = min(hits)
    while file_idx > 0 and budget.has_room_for(PROBE_SIZE_ESTIMATE):
        prev = probe_file(con, file_idx - 1, cache, budget)
        if prev is None or "cell_line" not in prev.columns or target not in set(prev["cell_line"]):
            break
        file_idx -= 1
    return file_idx


def download_file(file_idx: int, dest: Path, budget: NetworkBudget) -> None:
    url = file_url(file_idx, REVISION)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            budget.add_download(dest.stat().st_size)
            return
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            wait = 30 * (attempt + 1)
            print(f"Rate limited by Hugging Face; waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
    raise SystemExit("Still rate limited by Hugging Face; re-run later.")


def fetch_cell_line_plates_1_3(
    con_probe, con_local: duckdb.DuckDBPyConnection, cell_line: str, coarse_hits: dict[str, list[int]],
    cache: dict, budget: NetworkBudget, drugs_needed: set[str], scratch: Path,
) -> tuple[pd.DataFrame, int | None, list[str]]:
    """Download this cell line's plate-1..3 shards in full, return their DE rows for our drugs.

    Returns (rows_df, start, notes); `start` is None if the cell line could not be located at
    all (absent from the coarse scan, or the cap was hit before any of its shards downloaded).
    """
    start = locate_cell_line_start(con_probe, cell_line, coarse_hits, cache, budget)
    notes = []
    if start is None:
        if not budget.has_room_for(PROBE_SIZE_ESTIMATE):
            notes.append("network cap reached while searching")
        else:
            notes.append("cell line not observed anywhere in the coarse shard scan")
        return pd.DataFrame(), None, notes

    parts = []
    file_idx = start
    while True:
        probed = probe_file(con_probe, file_idx, cache, budget)
        if probed is None or "cell_line" not in probed.columns or cell_line not in set(probed["cell_line"]):
            break  # target's block has fully ended (its tail, if any, was in the previous shard)

        our_rows = probed[probed["cell_line"] == cell_line]
        has_our_plates = our_rows["plate"].isin(OUR_PLATES).any()
        has_later_plates = (~our_rows["plate"].isin(OUR_PLATES)).any()

        if has_our_plates:
            if not budget.has_room_for(90_000_000):
                notes.append(f"network cap reached before downloading shard {file_idx}")
                break
            dest = scratch / f"train-{file_idx:05d}.parquet"
            download_file(file_idx, dest, budget)
            rows = extract_rows(con_local, dest, cell_line, drugs_needed)
            dest.unlink()
            if len(rows):
                parts.append(rows)

        if has_later_plates:
            break  # plates 1-3 are fully behind us now (plate is ascending within a cell line)
        file_idx += 1

    rows_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["gene_name", "log2FoldChange", "padj", "n_cells_trt", "n_cells_ctrl", "drug", "plate",
                 "concentration", "concentration_unit"]
    )
    return rows_df, start, notes


def extract_rows(con: duckdb.DuckDBPyConnection, shard: Path, cell_line: str, drugs) -> pd.DataFrame:
    """A local shard's rows for one cell line, plates 1-3, and the given drugs (trimmed on both sides)."""
    return con.execute(
        f"""
        SELECT {", ".join(EXTRACT_COLUMNS)}
        FROM read_parquet({sql_str(shard)})
        WHERE Cell_ID_Cellosaur = {sql_str(cell_line)}
          AND plate IN ({", ".join(sql_str(p) for p in sorted(OUR_PLATES))})
          AND trim(drug) IN ({", ".join(sql_str(d) for d in drug_filter_values(drugs))})
        """
    ).fetchdf()


def save_probe_cache(cache: dict, path: Path) -> None:
    """Persist footer probes: one row per (file_idx, row_group_id); a shard probed as absent has no rows but is listed."""
    frames = []
    for file_idx, piv in cache.items():
        if piv is None:
            frames.append(pd.DataFrame({"file_idx": [file_idx], "row_group_id": [None], "cell_line": [None], "plate": [None]}))
        else:
            df = piv.reset_index()[["row_group_id", "cell_line", "plate"]].copy()
            df.insert(0, "file_idx", file_idx)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["file_idx", "row_group_id", "cell_line", "plate"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def load_probe_cache(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    df = pd.read_parquet(path)
    cache: dict = {}
    for file_idx, group in df.groupby("file_idx", sort=True):
        if group["row_group_id"].isna().all():
            cache[int(file_idx)] = None
        else:
            piv = group.drop(columns="file_idx").astype({"row_group_id": "int64"}).set_index("row_group_id")
            piv.index.name = "row_group_id"
            cache[int(file_idx)] = piv[["cell_line", "plate"]]
    return cache
