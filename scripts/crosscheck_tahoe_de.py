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
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase1.stream import connect, sql_list, sql_str  # noqa: E402
from phase3 import crosscheck  # noqa: E402
from phase3.config import load_features_config  # noqa: E402
from phase4.tahoe_de import (  # noqa: E402
    COARSE_STRIDE, PROBE_SIZE_ESTIMATE, NetworkBudget, build_coarse_index, fetch_cell_line_plates_1_3, set_revision,
)

FEATURES_YAML = ROOT / "configs" / "features.yaml"
CONDITIONS = ROOT / "data" / "pseudobulk" / "conditions.parquet"
LOGFC = ROOT / "data" / "pseudobulk" / "logfc.parquet"
GENES = ROOT / "data" / "pseudobulk" / "genes.parquet"
OUT_DOC = ROOT / "docs" / "target_crosscheck.md"
SCRATCH = ROOT / "data" / "cache" / "de_crosscheck_tmp"

N_WORST = 5


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
    set_revision(load_features_config(FEATURES_YAML)["tahoe"]["revision"])
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
            con_remote, con_local, cell_line, coarse_hits, probe_cache, budget, drugs_needed, SCRATCH
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
