from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from phase1.pseudobulk import SPECIAL_TOKENS

PARQUET_OPTIONS = "COMPRESSION zstd, COMPRESSION_LEVEL 9"


def sql_str(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_list(values) -> str:
    return "[" + ", ".join(sql_str(v) for v in values) + "]"


def connect(memory_limit: str = "2GB", temp_dir: str | Path | None = None, remote: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit = {sql_str(memory_limit)}")
    if temp_dir is not None:
        con.execute(f"SET temp_directory = {sql_str(temp_dir)}")
    if remote:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
    return con


def shards_for_plates(plate_map: pd.DataFrame, plates: list[str]) -> list[int]:
    """Shards whose row-group plate range [plate_min, plate_max] could contain any requested plate."""
    hit = pd.Series(False, index=plate_map.index)
    for plate in plates:
        hit |= (plate_map["plate_min"] <= plate) & (plate_map["plate_max"] >= plate)
    return sorted(int(s) for s in plate_map.loc[hit, "shard"].unique())


def aggregate_shard(
    con: duckdb.DuckDBPyConnection,
    shard_path: str | Path,
    cell_lines: list[str],
    plates: list[str],
    gene_out: Path,
    cell_out: Path,
) -> int:
    """Write per-(sample, plate, cell_line) gene sums and cell counts for one shard; return cells kept."""
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE shard_cells AS
        SELECT sample, plate, cell_line_id AS cell_line, genes, expressions
        FROM read_parquet(?)
        WHERE list_contains(?, cell_line_id) AND list_contains(?, plate)
        """,
        [str(shard_path), list(cell_lines), list(plates)],
    )
    gene_sql = """
        SELECT sample, plate, cell_line, gene, SUM(value)::DOUBLE AS value
        FROM (SELECT sample, plate, cell_line, UNNEST(genes) AS gene, UNNEST(expressions) AS value FROM shard_cells)
        GROUP BY ALL
    """
    cell_sql = "SELECT sample, plate, cell_line, COUNT(*)::BIGINT AS n_cells FROM shard_cells GROUP BY ALL"

    # cell_out is written last: its presence marks the shard as fully processed.
    for sql, out in ((gene_sql, gene_out), (cell_sql, cell_out)):
        tmp = out.with_suffix(".tmp")
        con.execute(f"COPY ({sql}) TO {sql_str(tmp)} (FORMAT parquet, COMPRESSION zstd)")
        tmp.replace(out)

    kept = con.execute("SELECT COUNT(*) FROM shard_cells").fetchone()[0]
    con.execute("DROP TABLE shard_cells")
    return int(kept)


def _part_path(out_path: Path, *key: str) -> Path:
    return out_path.with_name(f".{out_path.stem}.{'.'.join(key)}.part.parquet")


def _merge_parts(con: duckdb.DuckDBPyConnection, parts: list[Path], out_path: Path) -> None:
    """Concatenate parts into one file sorted by (condition_id, gene); the sort spills to disk."""
    try:
        con.execute(
            f"COPY (SELECT * FROM read_parquet({sql_list(parts)}) ORDER BY condition_id, gene) "
            f"TO {sql_str(out_path)} (FORMAT parquet, {PARQUET_OPTIONS})"
        )
    finally:
        _remove(parts)


def _remove(parts: list[Path]) -> None:
    for part in parts:
        part.unlink(missing_ok=True)


def combine_partials(
    con: duckdb.DuckDBPyConnection,
    gene_files: list[Path],
    cell_files: list[Path],
    lookup: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """Sum shard partials into condition-level pseudobulk (written to out_path); return per-condition totals.

    Conditions never span plates, so each plate is reduced on its own to bound memory.
    """
    keys = ["sample", "plate", "cell_line"]
    lookup = lookup[keys + ["condition_id"]].drop_duplicates(keys)
    parts, per_condition = [], []
    try:
        for plate in sorted(lookup["plate"].unique()):
            parts.append(_part_path(out_path, plate))
            per_condition.append(_combine_plate(con, gene_files, cell_files, lookup[lookup["plate"] == plate], plate, parts[-1]))
    except BaseException:
        _remove(parts)
        raise
    if not parts:
        raise ValueError("combine_partials: lookup has no plates")
    _merge_parts(con, parts, out_path)
    return pd.concat(per_condition, ignore_index=True)


def _combine_plate(
    con: duckdb.DuckDBPyConnection,
    gene_files: list[Path],
    cell_files: list[Path],
    lookup: pd.DataFrame,
    plate: str,
    out_path: Path,
) -> pd.DataFrame:
    con.register("lookup", lookup)
    # Math stays DOUBLE; cast at write time and fail rather than silently round a non-integer count.
    con.execute(
        f"""
        COPY (
            WITH sums AS (
                SELECT l.condition_id, g.gene, SUM(g.value) AS sum_counts
                FROM read_parquet({sql_list(gene_files)}) g JOIN lookup l USING (sample, plate, cell_line)
                WHERE g.plate = {sql_str(plate)} AND g.gene NOT IN ({", ".join(str(t) for t in SPECIAL_TOKENS)})
                GROUP BY ALL
            ),
            cells AS (
                SELECT l.condition_id, SUM(c.n_cells)::BIGINT AS n_cells
                FROM read_parquet({sql_list(cell_files)}) c JOIN lookup l USING (sample, plate, cell_line)
                WHERE c.plate = {sql_str(plate)}
                GROUP BY ALL
            ),
            libs AS (SELECT condition_id, SUM(sum_counts) AS library_size FROM sums GROUP BY ALL),
            pb AS (SELECT s.condition_id, s.gene, s.sum_counts, c.n_cells, b.library_size
                   FROM sums s JOIN cells c USING (condition_id) JOIN libs b USING (condition_id))
            SELECT condition_id, gene::INTEGER AS gene,
                   CASE WHEN sum_counts = round(sum_counts) THEN sum_counts::INTEGER
                        ELSE error('pseudobulk: non-integer sum_counts for ' || condition_id || ', gene ' || gene) END AS sum_counts,
                   n_cells::INTEGER AS n_cells,
                   CASE WHEN library_size = round(library_size) THEN library_size::BIGINT
                        ELSE error('pseudobulk: non-integer library_size for ' || condition_id) END AS library_size
            FROM pb
        ) TO {sql_str(out_path)} (FORMAT parquet, {PARQUET_OPTIONS})
        """
    )
    con.unregister("lookup")
    return con.execute(
        f"""SELECT condition_id, ANY_VALUE(n_cells) AS n_cells, ANY_VALUE(library_size) AS library_size
            FROM read_parquet({sql_str(out_path)}) GROUP BY condition_id"""
    ).fetchdf()


def write_logfc(con: duckdb.DuckDBPyConnection, pseudobulk_path: Path, conditions: pd.DataFrame, out_path: Path) -> int:
    """logfc = ln(1 + cpm) - ln(1 + cpm_control), per gene, against the same line's DMSO on the same plate.

    cpm = sum_counts / library_size * 1e6, computed here in DOUBLE; only logfc is stored (as FLOAT).
    Only QC-passing treated conditions with a QC-passing control; genes seen in either profile.
    Returns the number of treated conditions written.
    """
    ok = conditions[conditions["qc_pass"]]
    usable_controls = set(ok.loc[ok["is_control"], "condition_id"])
    pairs = ok[~ok["is_control"] & ok["control_condition_id"].isin(usable_controls)]
    # A treated condition and its control share plate and cell line, so each pair group is joined on its own.
    parts = []
    try:
        for (plate, line), group in pairs.groupby(["plate", "cell_line"], sort=True):
            parts.append(_part_path(out_path, plate, line))
            _write_logfc_group(con, pseudobulk_path, group, parts[-1])
    except BaseException:
        _remove(parts)
        raise
    if parts:
        _merge_parts(con, parts, out_path)
    else:
        _write_logfc_group(con, pseudobulk_path, pairs, out_path)
    return int(len(pairs))


def _write_logfc_group(con: duckdb.DuckDBPyConnection, pseudobulk_path: Path, pairs: pd.DataFrame, out_path: Path) -> None:
    con.register("pairs", pairs[["condition_id", "control_condition_id"]])
    con.execute(
        f"""
        COPY (
            WITH wanted AS (SELECT condition_id FROM pairs UNION SELECT control_condition_id FROM pairs),
            pb AS (SELECT condition_id, gene, sum_counts::DOUBLE / library_size::DOUBLE * 1e6 AS cpm
                   FROM read_parquet({sql_str(pseudobulk_path)}) SEMI JOIN wanted USING (condition_id)),
            t AS (SELECT p.condition_id, p.control_condition_id, pb.gene, pb.cpm
                  FROM pairs p JOIN pb USING (condition_id)),
            c AS (SELECT p.condition_id, p.control_condition_id, pb.gene, pb.cpm
                  FROM pairs p JOIN pb ON pb.condition_id = p.control_condition_id)
            SELECT COALESCE(t.condition_id, c.condition_id) AS condition_id,
                   COALESCE(t.control_condition_id, c.control_condition_id) AS control_condition_id,
                   COALESCE(t.gene, c.gene) AS gene,
                   (ln(1 + COALESCE(t.cpm, 0)) - ln(1 + COALESCE(c.cpm, 0)))::FLOAT AS logfc
            FROM t FULL OUTER JOIN c ON t.condition_id = c.condition_id AND t.gene = c.gene
        ) TO {sql_str(out_path)} (FORMAT parquet, {PARQUET_OPTIONS})
        """
    )
    con.unregister("pairs")
