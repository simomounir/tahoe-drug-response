# Performance baseline (phase 2)

Where Phase 1 build time goes, measured on native arm64 Python, with the x86_64 (Rosetta)
environment it replaced as the comparison. Measured 2026-09-25.

## Method

| | |
|---|---|
| Machine | Apple M1 (4 performance + 4 efficiency cores), 16 GB RAM, macOS 26.5.2 |
| Native env | `.venv`: uv-managed CPython 3.12.12 aarch64 |
| Rosetta env | `.venv-x86`: Intel Homebrew CPython 3.12.12 x86_64, run under Rosetta 2 |
| Libraries | Identical in both: DuckDB 1.5.5 (8 threads, `memory_limit = 2GB`), pyarrow 25.0.1, pandas 3.0.6 |
| Code | Commit `8b50400` plus stage timings in `build_outputs` (no change to outputs) |
| Slice | `configs/slice.yaml`: plates 1–3, 8 cell lines |
| Cache state | Warm: partials already on disk and read by a previous build, OS file cache not purged |
| Time / memory | `/usr/bin/time -l` for wall time and process peak RSS; per-stage wall time and peak RSS from `build_outputs` (`ru_maxrss`, in-process, so it includes DuckDB) |
| Network | Household broadband, unauthenticated Hugging Face downloads |

Every native rebuild was checked against the previous build with `tmp/compare_builds.py`:
0 differing rows in pseudobulk (56,715,151), logFC (67,362,948) and conditions.

## 1. Rebuild from partials (`make phase1`, no download)

| Run | Env | Wall time | CPU user | Peak RSS |
|---|---|---|---|---|
| P2.1 | Rosetta x86_64 | 529.6 s | 2,361 s | 2.57 GB |
| P2.1 | native arm64 | 317.5 s | 1,234 s | 2.99 GB |
| P2.2 #1 | native arm64 | 346.7 s | 1,223 s | 2.65 GB |
| P2.2 #2 | native arm64 | 376.6 s | 1,248 s | 2.65 GB |
| P2.2 #3 | native arm64 | 341.0 s | 1,206 s | 2.96 GB |

Native median of 4 runs: **344 s (5.7 min)**, range 318–377 s (−8% / +10%), vs 530 s under Rosetta
(one run): about **1.5×** faster, half the CPU time. CPU user time is steady (1,206–1,248 s)
while wall time varies, which points at scheduling or thermals rather than the code.

**Stage breakdown** (native, P2.2 runs #1–#3):

| Stage | What it does | Seconds (min–max) | Share | Peak RSS after stage |
|---|---|---|---|---|
| pseudobulk | `combine_partials`: sum 612 shard partials per (plate, cell line), write `pseudobulk.parquet` | 314–346 | 91% | 2.27–2.35 GB |
| conditions | condition table, QC flags, control assignment | 0.06–0.08 | <0.1% | same |
| logfc | `write_logfc` per (plate, cell line) | 21–25 | 7% | 2.27–2.41 GB |
| checks | contracts, output size, gene coverage, zero-count genes | 4.0–4.2 | 1% | 2.43–2.51 GB |
| write | conditions, controls, QC report | 0.03 | <0.1% | 2.43–2.51 GB |

## 2. Streaming a plate not yet cached (download + reduce per shard)

**Native**, fresh sample: first 20 shards of plate4
(`python scripts/run_phase1.py --plates plate4 --max-shards 20`, 1.7 GB):

| Per shard (median, min–max) | Download | Aggregate | Cells kept |
|---|---|---|---|
| native arm64, plate4, n = 20 | 4.19 s (3.80–4.74) | 0.95 s (0.89–1.12) | 8,491 |
| Rosetta, plates 1–3 logs, n = 612 | 4.95 s (p10–p90 4.18–5.94) | 1.70 s (p10–p90 1.46–2.05) | 8,460–8,880 |

Whole run: 111.7 s wall, peak RSS 1.73 GB. The Rosetta rows come from `shard_log.csv` written
during Phase 1, **before** the aggregation code changed in `572e993` and `8b50400`, so this
comparison mixes architecture and code. The isolated comparison is below.

**Same code, same shards** (`scripts/bench_aggregate.py`: plate4 shards 0, 10, 19 on local
disk, 1 warm-up + 5 timed runs each, median):

| Shard | native arm64 | Rosetta x86_64 | Speedup |
|---|---|---|---|
| 0 | 0.859 s | 1.258 s | 1.46× |
| 10 | 0.949 s | 1.368 s | 1.44× |
| 19 | 1.018 s | 1.415 s | 1.39× |

Run-to-run spread was under 3% in every cell.

## What this says

- **Streaming is network-bound.** Download is 75–80% of per-shard time on both architectures, at
  ~20 MB/s. A new plate (150–275 shards) costs ~13–23 min, of which ~2.5–4.5 min is compute. Faster
  aggregation code cannot change that much; parallel downloads or an HF token might.
- **Rebuild is one stage.** 91% of the 5.7 min is `combine_partials`. The logFC, checks and
  QC stages together take under 30 s. Any future optimisation starts there.
- **Native arm64 is 1.4–1.5× faster on compute and uses about half the CPU**, with identical
  outputs. It also uses 3–16% more peak memory (2.65–2.99 GB vs 2.57 GB, one Rosetta run), well inside the
  8 GB ceiling; the cause is not investigated.
- Memory stays flat with plate count (per (plate, cell line) reduction), so these numbers scale
  roughly linearly with the number of plates.

## Reproduce

```bash
source .venv/bin/activate
/usr/bin/time -l make phase1                     # stage lines + "Build:" line in the output
python tmp/compare_builds.py                     # needs tmp/prev_build/ from an earlier build
python scripts/run_phase1.py --plates plate4 --max-shards 20 \
    --output-dir tmp/p22/plate4_out --report tmp/p22/plate4_qc.md   # per-shard log in data/cache/partials/<key>/shard_log.csv
.venv/bin/python scripts/bench_aggregate.py && .venv-x86/bin/python scripts/bench_aggregate.py
```

Not measured: cold OS file cache (`sudo purge`), Linux, thermal state, repeated Rosetta
rebuilds.
