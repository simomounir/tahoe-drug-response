"""Cross-check our from-scratch pseudobulk logFC against Tahoe's official DE table (spec P3.1).

Samples `crosscheck.n_conditions` treated, QC-passing conditions from our slice (seeded), then
compares them against Tahoe's official `pseudobulk_differential_expression` table (1,026 remote
Parquet shards on Hugging Face, ~90 MB each).

Locating data is the hard part: the shards are not named by content. Empirically (see
docs/target_crosscheck.md "Tahoe's DE shard layout"): each cell line occupies one *contiguous*
block of shards, sorted within the block by (plate ascending, drug ascending, gene) -- "plate"
runs far beyond our slice's 3 plates because the DE table also covers Tahoe's broader
~1,100-compound program, so a block starts at its own "plate 1" and runs through many more plates
before ending; our slice only needs the first few shards of each ~20-shard block. But the *order
of blocks* across the 1,026 shards is **not** the cell lines' alphabetical (or any obvious) order
-- probing shards 0, 300, 500, 700, 900 directly showed cell lines CVCL_0023, CVCL_0334,
CVCL_0293, CVCL_1056, CVCL_0131, which is not monotonic in any of the plausible sort keys. So
there is no way to binary-search for a cell line's block; this script instead does one coarse,
fixed-stride scan across all 1,026 shard footers (stride 5: any cell line whose block spans at
least 5 shards is hit at least once; blocks average ~20 shards) to find every cell line, then a
short local walk from the nearest hit back to that block's true start. A sampled cell line that the
scan does not find makes the run fail loudly rather than being reported as merely uncovered.

A remote *targeted* Parquet query (`read_parquet(url) WHERE ...`) was tried first and measured at
40-50s per single-drug query, apparently because these shards carry very large footers (~5-6 MB,
observed directly) and thousands of small row groups -- DuckDB's httpfs does not turn a narrow
predicate into a small number of big HTTP range requests here. A plain full-file download of the
same shard via `curl`, by contrast, measured at ~4s for ~90 MB. So this script downloads whole
shards it actually needs (fast, and every byte is used) rather than issuing targeted remote
queries, and only uses cheap footer-only probes (~5-6 MB each, not full downloads) to locate
them. A shard's footer costs the same ~6 MB whether one row group or all of them are read, so
every probe reads *all* row groups' (cell line, plate) for free -- necessary because a cell
line's block does not have to start at a shard boundary (a shard is a fixed ~90 MB chunk of a
much larger sorted block), so checking only a shard's first row group would silently miss cell
lines whose data starts mid-shard.

Cell lines are processed by how many sampled conditions they cover, most first, so a hard stop
from the network cap loses as few matched conditions as possible. A running byte counter (coarse
scan probes + refine probes + full shard downloads) stops the script -- reporting exactly what
was covered -- before the network cap (`crosscheck.network_cap_gb`) would be exceeded. This is expected
to leave some sampled conditions uncovered; that shortfall is reported explicitly, never silently
dropped.

Writes docs/target_crosscheck.md. Deletes every temporary local file it creates.
"""
from __future__ import annotations

import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase1.stream import connect, sql_list, sql_str  # noqa: E402
from phase3 import crosscheck  # noqa: E402

N_FILES = 1026
FILE_URL = (
    "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/"
    "metadata/pseudobulk_differential_expression/train-{:05d}-of-01026.parquet"
)
FEATURES_YAML = ROOT / "configs" / "features.yaml"
CONDITIONS = ROOT / "data" / "pseudobulk" / "conditions.parquet"
LOGFC = ROOT / "data" / "pseudobulk" / "logfc.parquet"
GENES = ROOT / "data" / "pseudobulk" / "genes.parquet"
OUT_DOC = ROOT / "docs" / "target_crosscheck.md"
SCRATCH = ROOT / "data" / "cache" / "de_crosscheck_tmp"

OUR_PLATES = {"1", "2", "3"}
MAX_RETRIES = 6
N_WORST = 5
SAFETY_MARGIN_BYTES = 120_000_000  # don't start a step likely to blow past the cap


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


def df_to_markdown(df: pd.DataFrame) -> str:
    """A minimal Markdown table renderer (the venv has no `tabulate` for pandas.to_markdown)."""
    if df.empty:
        return "*(none)*\n"
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df[cols].itertuples(index=False))
    return "\n".join([header, sep, body]) + "\n"


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
    url = FILE_URL.format(file_idx)
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


# ---------------------------------------------------------------------------
# Downloading the shards a cell line's plates 1-3 actually live in
# ---------------------------------------------------------------------------


def download_file(file_idx: int, dest: Path, budget: NetworkBudget) -> None:
    url = FILE_URL.format(file_idx)
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
    cache: dict, budget: NetworkBudget, drugs_needed: set[str],
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
            dest = SCRATCH / f"train-{file_idx:05d}.parquet"
            download_file(file_idx, dest, budget)
            rows = con_local.execute(
                f"""
                SELECT gene_name, log2FoldChange, padj, n_cells_trt, n_cells_ctrl, drug, plate,
                       concentration, concentration_unit
                FROM read_parquet({sql_str(dest)})
                WHERE Cell_ID_Cellosaur = {sql_str(cell_line)}
                  AND plate IN ({", ".join(sql_str(p) for p in OUR_PLATES)})
                  AND trim(drug) IN ({", ".join(sql_str(d) for d in crosscheck.drug_filter_values(drugs_needed))})
                """
            ).fetchdf()
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


# ---------------------------------------------------------------------------
# Our own data
# ---------------------------------------------------------------------------


def load_sample(con_local: duckdb.DuckDBPyConnection, cfg: dict) -> pd.DataFrame:
    conditions = con_local.execute(f"SELECT * FROM read_parquet({sql_str(CONDITIONS)})").fetchdf()
    sample = crosscheck.sample_conditions(conditions, n=cfg["n_conditions"], seed=cfg["seed"])
    sample = sample.copy()
    sample["plate_de"] = sample["plate"].str.removeprefix("plate")
    return sample


def load_ours(con_local: duckdb.DuckDBPyConnection, condition_ids: list[str]) -> pd.DataFrame:
    ids = sql_list(condition_ids)
    return con_local.execute(
        f"""
        SELECT l.condition_id, g.gene_symbol, l.logfc
        FROM read_parquet({sql_str(LOGFC)}) l
        JOIN read_parquet({sql_str(GENES)}) g ON g.gene = l.gene
        WHERE l.condition_id IN {ids}
        """
    ).fetchdf()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = crosscheck.load_crosscheck_config(FEATURES_YAML)
    print(f"Crosscheck config: {cfg}", flush=True)

    con_local = connect("2GB")  # project rule 4; note: footprint still grows ~55 MB per shard read (see docs/target_crosscheck.md)
    sample = load_sample(con_local, cfg)
    print(f"Sampled {len(sample)} treated, qc_pass conditions (seed={cfg['seed']})", flush=True)

    ours_all = load_ours(con_local, list(sample["condition_id"]))
    print(f"Loaded {len(ours_all)} of our logFC rows for the sample", flush=True)

    con_remote = connect("1GB", remote=True)
    budget = NetworkBudget(int(cfg["network_cap_gb"] * 1e9))
    probe_cache: dict = {}

    by_cell_line = {cl: rows for cl, rows in sample.groupby("cell_line")}
    # Process cell lines with the most sampled conditions first, so a hard stop from the network
    # cap loses as few matched conditions as possible rather than an arbitrary (e.g. alphabetical)
    # subset.
    cell_lines_by_value = sorted(by_cell_line, key=lambda cl: -len(by_cell_line[cl]))

    print(f"Building coarse shard index (stride={COARSE_STRIDE}, one-time cost)...", flush=True)
    coarse_hits = build_coarse_index(con_remote, probe_cache, budget)
    print(
        f"Coarse index: {len(coarse_hits)} distinct cell lines observed in "
        f"{budget.n_probes} probes (~{budget.probe_bytes / 1e6:.0f} MB)",
        flush=True,
    )

    results = []
    unmatched = []
    skipped_cell_lines = []
    not_located = []
    dose_conflicts = []
    control_checks = []

    SCRATCH.mkdir(parents=True, exist_ok=True)
    for cell_line in cell_lines_by_value:
        if not budget.has_room_for(PROBE_SIZE_ESTIMATE):
            skipped_cell_lines.append(cell_line)
            continue

        rows = by_cell_line[cell_line]
        drugs_needed = set(rows["drug"])
        theirs_all, block_start, notes = fetch_cell_line_plates_1_3(
            con_remote, con_local, cell_line, coarse_hits, probe_cache, budget, drugs_needed
        )
        if notes:
            print(f"{cell_line}: {'; '.join(notes)}", flush=True)
        if theirs_all.empty and block_start is None:
            skipped_cell_lines.append(cell_line)
            if any("not observed" in note for note in notes):
                not_located.append(cell_line)
            continue

        theirs_all = theirs_all.rename(columns={"gene_name": "gene_symbol"})
        for row in rows.itertuples():
            try:
                theirs = crosscheck.select_condition_rows(
                    theirs_all, plate=row.plate_de, drug=row.drug, dose_um=row.dose * crosscheck.UNIT_TO_UM[row.unit]
                )
            except ValueError as err:
                dose_conflicts.append({"condition_id": row.condition_id, "error": str(err)})
                continue
            if theirs.empty:
                unmatched.append(
                    {"condition_id": row.condition_id, "cell_line": row.cell_line, "drug": row.drug,
                     "dose": row.dose, "plate": row.plate}
                )
                continue
            ours = ours_all[ours_all["condition_id"] == row.condition_id][["gene_symbol", "logfc"]]
            cmp = crosscheck.compare_condition(ours, theirs, padj=cfg["padj"])
            cmp.update(
                {
                    "condition_id": row.condition_id, "cell_line": row.cell_line, "drug": row.drug,
                    "dose": row.dose, "plate": row.plate,
                }
            )
            results.append(cmp)
            control_checks.append(
                {"condition_id": row.condition_id, "n_cells_trt_match": row.n_cells == theirs["n_cells_trt"].iloc[0]}
            )
            print(
                f"[{len(results)} matched so far] {row.cell_line} {row.drug!r} plate{row.plate_de} "
                f"n_genes={cmp['n_genes']} sign_agreement={cmp['sign_agreement']:.3f} "
                f"(budget spent: {budget.spent / 1e6:.0f} MB)",
                flush=True,
            )

    con_remote.close()
    con_local.close()
    shutil.rmtree(SCRATCH, ignore_errors=True)

    results_df = pd.DataFrame(results)
    unmatched_df = pd.DataFrame(unmatched)
    write_report(cfg, sample, results_df, unmatched_df, control_checks, budget, skipped_cell_lines, dose_conflicts)
    if not_located:
        raise SystemExit(f"sampled cell line(s) not located in the DE shard scan: {not_located}")

    n_pass = len(results_df)
    print(f"\nMatched {n_pass}/{len(sample)} sampled conditions in Tahoe's DE table.")
    print(f"Skipped cell lines (network cap or not found): {skipped_cell_lines}")
    if n_pass:
        median_sign = results_df["sign_agreement"].median()
        median_spear = results_df["spearman"].median()
        verdict = "PASS" if median_sign >= cfg["min_median_sign_agreement"] else "FAIL"
        print(f"Median sign agreement: {median_sign:.4f} ({verdict} against threshold {cfg['min_median_sign_agreement']})")
        print(f"Median Spearman: {median_spear:.4f}")
    print(
        f"Network: {budget.spent / 1e6:.0f} MB spent of {budget.cap / 1e6:.0f} MB cap "
        f"({budget.n_probes} footer probes ~{budget.probe_bytes / 1e6:.0f} MB, "
        f"{budget.n_downloads} shard downloads {budget.download_bytes / 1e6:.0f} MB)."
    )
    print(f"Report written to {OUT_DOC.relative_to(ROOT)}")


def write_report(cfg, sample, results_df, unmatched_df, control_checks, budget: NetworkBudget, skipped_cell_lines, dose_conflicts) -> None:
    n_sampled = len(sample)
    n_matched = len(results_df)
    control_df = pd.DataFrame(control_checks)
    n_control_match = int(control_df["n_cells_trt_match"].sum()) if len(control_df) else 0

    lines = []
    lines.append("# Target cross-check: our logFC vs. Tahoe's official DE table\n")
    lines.append(
        "Cross-checks the phase 1 target (natural-log CPM-ratio logFC in "
        "`data/pseudobulk/logfc.parquet`) against Tahoe-100M's own `pseudobulk_differential_expression` "
        "table (DESeq2 output, 1,026 Parquet shards on Hugging Face), per spec §4 task P3.1.\n"
    )

    lines.append("## Tahoe's DE shard layout (discovered, not documented upstream)\n")
    lines.append(
        "The 1,026 shards are not named by content, and a first attempt to locate them assumed a "
        "global sort by `(Cell_ID_Cellosaur, plate, drug, gene_name)` -- probing shards 0, 300, "
        "500, 700, 900 directly disproved that (`CVCL_0023, CVCL_0334, CVCL_0293, CVCL_1056, "
        "CVCL_0131` -- not monotonic in any plausible key) and was caught before the real run "
        "wasted budget on it. The true structure: each cell line occupies one *contiguous* block "
        "of shards, sorted within the block by `(plate, drug, gene_name)`, but the *order of "
        "blocks* across the 1,026 shards is not the cell lines' alphabetical order (apparently "
        "whatever order Tahoe's export pipeline produced them in). `plate` runs well beyond 3 for "
        "each cell line (Tahoe's DE table also covers its broader ~1,100-compound program beyond "
        "our 92-drug, 3-plate slice), so a block starts fresh at `plate=\"1\"` and runs through "
        "many more plates before ending; our slice only needs the first few shards of each "
        "~20-shard block. Each shard's Parquet footer measured ~5.75 MB (thousands of small row "
        "groups), and a targeted remote query (`read_parquet(url) WHERE ...`) measured 40-50s per "
        "single-drug query -- apparently DuckDB's httpfs does not collapse a narrow predicate into "
        "few large HTTP range requests against a footer this size, whereas a plain full-shard "
        "download measured ~4s for ~90 MB. This script therefore downloads whole shards it needs "
        "(fast, and every byte is used) and locates them with one coarse, fixed-stride scan "
        "(stride 5) across all 1,026 shard footers -- any block spanning 5 or more shards is hit; "
        "a sampled line not found fails the run -- followed by a short local "
        "walk back to each needed cell line's true block start.\n"
    )

    lines.append("## Method\n")
    lines.append(
        f"- Sampled `n_conditions={cfg['n_conditions']}` treated, QC-passing conditions from "
        f"`data/pseudobulk/conditions.parquet` with seed `{cfg['seed']}` "
        "(`phase3.crosscheck.sample_conditions`).\n"
        "- Built one coarse shard index (stride 5, ~205 footer probes), then processed cell lines "
        "in order of how many sampled conditions they cover (most first, so a network-cap stop "
        "loses as few matched conditions as possible): located and downloaded the shards covering "
        "each cell line's plates 1-3, extracted rows for every sampled drug on it, and deleted the "
        "shards immediately after.\n"
        "- Join keys: `Cell_ID_Cellosaur = <our cell_line>` (exact; same Cellosaurus id both sides), "
        "`plate = <our plate, \"plateN\" stripped to \"N\">`, `trim(drug) = trim(<our drug>)` (Tahoe "
        "drug names carry stray whitespace, e.g. `\"Erdafitinib \"`; join key is `trim(drug)` per "
        "spec).\n"
        f"- Over genes with Tahoe `padj < {cfg['padj']}`: sign agreement between our `logfc` and their "
        "`log2FoldChange`, and Spearman correlation (computed as Pearson correlation of ranks). Scale "
        "differs (natural log vs. log2, and ours is not variance-stabilised), so only sign and rank "
        "are compared, never magnitude (`phase3.crosscheck.compare_condition`).\n"
        f"- Pass rule (spec, fixed in advance): median per-condition sign agreement ≥ "
        f"`min_median_sign_agreement={cfg['min_median_sign_agreement']}`.\n"
        f"- Network cap: {budget.cap / 1e9:.0f} GB (`crosscheck.network_cap_gb`). A running byte counter (probes + "
        "downloads) stops the script before exceeding it; any cell lines left unprocessed are "
        "reported below, not silently dropped.\n"
    )

    lines.append("## Tahoe's control\n")
    lines.append(
        "Confirmed directly against the DE table, not assumed: the example row for "
        "`CVCL_0023`/`4EGI-1`/0.05 µM/plate `1` has `n_cells_trt=1378`, `n_cells_ctrl=4862`, which "
        "match our own `conditions.parquet` `n_cells` for that exact condition and our own "
        "`controls.parquet` `n_cells` for `CVCL_0023`/`plate1` exactly. Tahoe's DE table control is "
        "the same plate-matched DMSO pseudobulk this project computes independently. Across the "
        f"matched sample, our `n_cells` (treated) equalled Tahoe's `n_cells_trt` for "
        f"{n_control_match}/{n_matched} conditions (mismatches, if any, are noted below).\n"
    )

    lines.append("## Name mapping and coverage\n")
    lines.append(
        f"- {len(sample['cell_line'].unique())} distinct cell lines and {sample['drug'].nunique()} "
        "distinct drug names appear in the sample; cell line join needs no mapping (`Cell_ID_Cellosaur` "
        "is our own `cell_line`), plate join is a fixed string strip, drug join is `trim()` both "
        "sides.\n"
    )
    if skipped_cell_lines:
        lines.append(
            f"### Cell lines not covered ({len(skipped_cell_lines)}/{len(sample['cell_line'].unique())})\n\n"
            "Reached before the network cap or the search gave up; every condition sampled for these "
            "cell lines is reported as uncovered, not dropped silently:\n\n"
            + "".join(f"- `{cl}`\n" for cl in skipped_cell_lines)
        )
    else:
        lines.append("### Cell lines not covered (0)\n\nEvery sampled cell line was located and processed.\n")

    if len(unmatched_df):
        lines.append(f"### Unmatched conditions ({len(unmatched_df)}/{n_sampled})\n")
        lines.append(
            "Cell line was covered but this exact (plate, drug) row was not found in Tahoe's DE "
            "table for it (name mismatch beyond whitespace trimming, or genuinely absent):\n\n"
        )
        lines.append(df_to_markdown(unmatched_df))
    else:
        lines.append(f"### Unmatched conditions among covered cell lines (0)\n\n")

    lines.append("## Results\n")
    lines.append(f"- Matched {n_matched}/{n_sampled} sampled conditions in Tahoe's DE table.\n")

    if n_matched:
        sign = results_df["sign_agreement"]
        spear = results_df["spearman"].dropna()
        lines.append("### Distribution of per-condition metrics\n")
        lines.append("| metric | min | p25 | median | p75 | max | n |\n|---|---|---|---|---|---|---|\n")
        for name, series in [("sign_agreement", sign), ("spearman", spear)]:
            q = series.quantile([0, 0.25, 0.5, 0.75, 1.0])
            lines.append(
                f"| {name} | {q.iloc[0]:.4f} | {q.iloc[1]:.4f} | {q.iloc[2]:.4f} | {q.iloc[3]:.4f} | "
                f"{q.iloc[4]:.4f} | {len(series)} |\n"
            )

        median_sign = sign.median()
        median_spear = spear.median() if len(spear) else float("nan")
        verdict = "PASS" if median_sign >= cfg["min_median_sign_agreement"] else "FAIL"
        lines.append(
            f"\n**Median sign agreement: {median_sign:.4f} -- {verdict} against "
            f"`min_median_sign_agreement={cfg['min_median_sign_agreement']}`.**\n\n"
            f"Median Spearman: {median_spear:.4f}\n"
        )

        worst = results_df.sort_values("sign_agreement").head(N_WORST)
        lines.append(f"### {N_WORST} worst conditions (lowest sign agreement)\n")
        cols = ["condition_id", "cell_line", "drug", "dose", "plate", "n_genes", "sign_agreement", "spearman"]
        lines.append(df_to_markdown(worst[cols]))

        n_dup = int(results_df["n_duplicate_symbols"].sum()) if "n_duplicate_symbols" in results_df else 0
        lines.append(
            f"\nDuplicate gene symbols on Tahoe's side (dropped, all rows): {n_dup} across all compared "
            f"conditions. Dose conflicts (more than one Tahoe row per gene for one plate/drug/dose): "
            f"{len(dose_conflicts)}.\n"
        )
        if n_matched < n_sampled:
            lines.append(
                f"\n**Coverage:** {n_matched}/{n_sampled} sampled conditions were compared. The "
                f"{n_sampled - n_matched} others are excluded (reasons listed above). Cell lines were "
                "processed most-sampled-first, so an incomplete run covers a *selected*, not a "
                "uniformly random, subset of the seeded sample; the pass/fail call applies only to the "
                "compared conditions.\n"
            )
        else:
            lines.append(f"\n**Coverage:** all {n_sampled} seeded, randomly sampled conditions were compared.\n")
    else:
        lines.append("No sampled conditions matched Tahoe's DE table; no metrics to report. FAIL.\n")

    lines.append("## Network\n")
    lines.append(
        f"- {budget.spent / 1e6:.0f} MB of {budget.cap / 1e9:.0f} GB cap spent: "
        f"{budget.n_probes} Parquet-footer probes (~{budget.probe_bytes / 1e6:.0f} MB, used only to "
        f"locate shards) + {budget.n_downloads} full shard downloads ({budget.download_bytes / 1e6:.0f} "
        "MB, used for the actual comparison). Every downloaded shard was deleted immediately after its "
        "rows were extracted; the scratch directory is removed at the end of the run.\n"
    )

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
