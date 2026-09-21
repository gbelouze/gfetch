# gfetch throughput investigation - session notes

Tracking doc for an investigation into why gfetch's remote-load throughput was far
below available bandwidth, and whether the download-then-load ("two-stage") design
already planned in `claude/tech-stack.md` is actually faster in practice, not just an
HPC-offline-node necessity. Written at the end of the session to pick this back up
later - see `claude/tech-stack.md`/`claude/dev-stack.md` for gfetch's own architecture
decisions; this doc is specific to the `benchmark/` directory's exploratory scripts.

**Nothing in `src/gfetch/` has been changed as a result of this investigation yet.**
Everything below lives in `benchmark/`, standalone, with no dependency on the `gfetch`
package itself (deliberate - "notebook style", run/edited interactively).

## Files

**2026-09-20: split what used to be one growing `wide_mosaic.py` into dedicated
scripts + a shared `common.py`, once it had accumulated three unrelated sweeps behind
a `--which` flag.** Each of the "wide" scenario's three experiments is now its own
runnable file:

- `common.py` - shared constants (Earth Search URL, collection, bands, resolution),
  the "wide"/"deep" scenario search+dedupe helpers
  (`dedupe_by_processing_baseline()`/`search_wide()`/`search_deep()`), the
  `Scenario = Literal["wide", "deep"]` type plus `SCENARIO_SEARCH`/`SCENARIO_CRS`
  registries every script's `--scenario` CLI param dispatches through (added
  2026-09-20, see below), shared rich-based logging/progress-bar setup
  (`setup_logging()`/`sweep_progress()`, added 2026-09-20, see below), and
  `TaskCounter` (added 2026-09-20, see point 13 below) - a plain dask-callback task
  counter, for reading load progress from a thread other than the one running
  `.load()`. No CLI, not meant to be run directly.
**2026-09-20: renamed `wide_concurrency.py`/`wide_gdal_config.py`/`wide_chunks.py` to
`concurrency.py`/`gdal_config.py`/`chunks.py`** (dropping the `wide_` prefix), now that
`--scenario` lets each one load `"wide"` or `"deep"` - the old names implied wide-only.

- `concurrency.py` - direct remote `odc.stac.load()`, sweeping dask thread count at a
  fixed chunk size. `cyclopts`-based CLI:
  `uv run python benchmark/concurrency.py --scenario deep --worker-counts 128 64 32
  --chunk 2048 --max-time 120`. `--scenario` (`wide`/`deep`, default `wide`) selects
  which scenario to load; results go to `results/concurrency_<scenario>.csv`.
- `gdal_config.py` - direct remote `odc.stac.load()`, comparing GDAL configs
  (baseline / HTTP-multiplex-only / flytemosaic's full config / flytemosaic's
  single-threaded-dask pattern) on a small item subset. `cyclopts`-based CLI,
  parametrizing `--scenario`/`--n-items`/`--worker-counts`/`--memory-gb`/`--debug`;
  results go to `results/gdal_config_<scenario>.csv`.
- `chunks.py` - direct remote `odc.stac.load()`, sweeping dask chunk size at a fixed
  thread count (`--workers`, default 128 - see point 11 below for why this isn't left
  at dask's own default). `cyclopts`-based CLI:
  `uv run python benchmark/chunks.py --chunks 512 1024 ...`. `--scenario` (`wide`/`deep`,
  default `wide`) selects which scenario to load - the "wide" default still reproduces
  the odc-stac-vs-stackstac benchmark's
  ["wide" scenario](https://benchmark-odc-stac-vs-stackstac.netlify.app/) - bbox
  `[27.345815, -14.98724, 27.565542, -7.710992]`, `2020-06-06`, `sentinel-2-l2a`,
  red/green/blue, 10m - against Earth Search instead of Planetary Computer; results go
  to `results/chunks_<scenario>.csv`.
- `two_stage_mosaic.py` - two-stage (`stac_asset` bulk download, then local
  `odc.stac.load()`) benchmark for both the "wide" and "deep" scenarios (deep = MGRS
  tile 35MNM, `2020-06-01/2020-07-31`, 13 dates). Runs top-to-bottom cleanly, ~15-20
  min, ~13GB of real downloads. Imports its scenario search/dedupe (now also
  `Scenario`/`SCENARIO_SEARCH`/`SCENARIO_CRS`) from `common.py` instead of its own copy.
  `cyclopts`-based CLI, parametrizing `--scenarios` (run just `wide` or `deep` instead of
  always both), `--download-concurrencies`/`--best-download-concurrency`/
  `--load-chunk-sizes`/`--load-worker-counts`/`--cache-dir`.

**All four scripts above also take `--max-time`** (added 2026-09-20, see point 13
below), capping how long a single case's `ds.load()` may run (for `two_stage_mosaic.py`,
just its load stage - the download stage is asyncio-based and unaffected). Each case now
runs in its own `multiprocessing` child process specifically so a case that blows past
its budget can be killed outright rather than leaving stray dask worker threads running
in the background to contaminate the next case's measurement; on a timeout, that row's
`mb_s`/`mpix_s` are inferred from the fraction of dask tasks completed by then
(`timed_out`/`completed_tasks`/`total_tasks` columns record this).

**2026-09-20: added verbose logging, `rich` sweep-progress bars, and a `cyclopts` CLI
to every script that didn't already have one** (`concurrency.py`/
`gdal_config.py`/`two_stage_mosaic.py` - `chunks.py` already had a CLI, just
gained the progress bar/extra logging). Every script now takes a `--verbose` flag
(DEBUG-level logging) and shows a `rich.progress.Progress` bar tracking how many of
the sweep's test configurations have completed (one bar per stage for
`two_stage_mosaic.py`'s three-stage run), via `common.py`'s new `setup_logging()`/
`sweep_progress()`. This **obsoletes the "don't `--help` a no-arg script" caveat**
recorded in point 8 below - all four scripts now take a safe `--help`.

Also **replaced every `dask.diagnostics.ProgressBar()` call (see finding 2 below) with
a custom `dask.callbacks.Callback` subclass** (`common.py::_RichDaskCallback`, used via
`dask_load_progress()`) that reports the same per-task completion onto a temporary task
on the *same* `Progress` instance as the sweep bar, instead of `ProgressBar()`'s own
separate ASCII bar - so one dask-backed `.load()` call's progress and the sweep's
overall configuration count render as two bars in one `rich` display, rather than two
unrelated progress indicators fighting for the terminal.
- `visualize.py` - **2026-09-20, new**: plots whatever `results/*.csv` files exist into
  one grid figure (`uv run python benchmark/visualize.py [--output PATH]`), skipping any
  run that hasn't happened yet - safe to run mid-sweep (tolerates a CSV that's been
  created but not yet written to). One panel per benchmark: `concurrency`/`chunks`
  (line per scenario), `gdal_config` (grouped bars, config x worker-count),
  `two_stage_download_concurrency` (line per scenario), `two_stage_download_full` (bar
  per scenario), `two_stage_load` (one sub-panel per scenario, line per worker count).
  Any `timed_out` row (see points 13/14) is marked with an "x" and a legend note, since
  its `mb_s`/`mpix_s` are inferred, not measured. Colors are assigned from a fixed
  categorical order (per the `dataviz` skill's validated default palette) so a given
  scenario/worker-count keeps the same color across every panel. Depends on
  `matplotlib`/`pandas`, both only transitively available today (not added to
  `pyproject.toml` to avoid touching the shared venv while a benchmark run may be using
  it - see the module docstring).
- `gfetch_pipeline.py` - **2026-09-21, new**: the only script here with a dependency on
  the `gfetch` package itself. Calls `gfetch.search.search()` /
  `gfetch.download.download_items()` / `gfetch.mosaic.load()` directly (instead of
  reimplementing them, like `two_stage_mosaic.py` does) on the same "wide"/"deep"
  scenarios, to check whether the real library reaches throughput comparable to the
  reference numbers below now that the two fixes in finding #17 are in place. No
  timeout-enforcing subprocess per case (unlike `concurrency.py`/`chunks.py`/
  `gdal_config.py`/`two_stage_mosaic.py`'s load stage) - this is a one-shot comparison
  check, not an unattended sweep. `cyclopts`-based CLI: `uv run python
  benchmark/gfetch_pipeline.py --scenarios wide deep`; results go to
  `results/gfetch_pipeline_download.csv`/`results/gfetch_pipeline_load.csv`. Downloads
  into its own `cache/gfetch_pipeline/` subdirectory, separate from
  `two_stage_mosaic.py`'s `cache/wide`/`cache/deep`, specifically so its download
  measurement is a real cold download, not a resumed no-op against an already-warm
  cache. `visualize.py` now plots its two CSVs alongside `two_stage_*.csv`'s
  equivalents.
- `wide_mosaic_marimo.py` - the user's own marimo-notebook port of an early version of
  the wide-scenario benchmark; not kept in sync with later fixes (e.g. still has the
  `progress=tqdm.notebook.tqdm` no-op on a dask-backed load). Not touched by request.
- `reference/odc-stac-vs-stackstac-raw.csv` - the *original* benchmark's own raw
  per-run data, pulled directly from a CSV linked on its report page
  (`data/benchmark-results-raw.csv`), not eyeballed off its charts. Ground truth for
  comparison; confirmed our reproduction matches it exactly (same `npix`/`nbytes`/CRS).
- `results/*.csv` - one CSV per script above. Since 2026-09-20's `--scenario` param
  (see point 12 below), `concurrency.py`/`gdal_config.py`/`chunks.py`
  each write `<script>_<scenario>.csv` (e.g. `chunks_wide.csv`,
  `chunks_deep.csv`) instead of a single unsuffixed `<script>.csv`, so a `wide` and
  a `deep` run don't clobber each other; `two_stage_*.csv` is unchanged (already covers
  both scenarios in one run, with a `scenario` column). Old `wide_mosaic_*.csv` files
  from the pre-split single script were removed as part of the split - superseded, not
  historical record worth keeping.
- `cache/` - gitignored (added to `.gitignore` this session). Currently holds ~13GB:
  the full downloaded "wide" (9 items) and "deep" (13 items) item sets from
  `two_stage_mosaic.py`'s last run. Reusable to re-run just the local-load sweep
  without re-downloading; safe to delete for disk space otherwise.

## Findings, in order

1. **Root cause of the original "why is this so slow" complaint**: `odc.stac.load()`
   with no explicit S3 config falls through to botocore's full credential chain (env ->
   shared config -> EC2 instance metadata service). The IMDS lookup
   (`169.254.169.254`) hangs until TCP timeout since this isn't running on an actual
   EC2 instance - confirmed live via `lsof -i -p <pid>` showing recurring `SYN_SENT` to
   that address throughout the run. **Fix**: call
   `odc.stac.configure_s3_access(aws_unsigned=True)` once, before any `odc.stac.load()`.
   This alone took throughput from ~4MB/s to ~11-16MB/s.

2. `odc.stac.load()`'s `progress=`/`pool=` kwargs are documented as "only used in
   non-dask load" - since we always pass `chunks=`, they're silent no-ops. Use
   `dask.diagnostics.ProgressBar()` around `.load()`/`.compute()`/`.to_zarr()` instead.

3. **After the credential fix, throughput still capped far below line rate (~50-100MB/s
   claimed connection speed) - diagnosed as concurrency-limited, not bandwidth-limited.**
   Methodology (all reusable, see the answer given mid-session for exact commands):
   `top -pid <pid>` for CPU (stayed near-idle -> not compute-bound); `nettop -p <pid> -d
   -x` for live per-connection throughput (found ~20 connections each individually slow,
   not one connection hogging everything); raw `curl` against the same bucket, bypassing
   GDAL/dask entirely, as ground truth (1 connection: 4.6MB/s; 6 parallel full-file
   downloads: ~36.5MB/s aggregate).
   - **Directly measured the per-request overhead** with `curl -w` timing fields: a
     512KB range request (mimicking one GDAL block read) spends ~46-48% of its total
     time on TCP+TLS handshake/RTT before any payload byte arrives (~0.55-0.65s fixed
     cost); the identical fixed cost is ~0.4% of a 219MB full-file download's total time.
     **Reusing a connection for a second small request cut it from ~1.3s to ~0.2-0.36s**
     (4-7x). Conclusion: it's not that S3/this bucket is slow, it's that many small
     *windowed* COG range-reads each pay a mostly-fixed connection-setup tax that a
     sustained full-file download doesn't.
   - The concurrency sweep now in `wide_concurrency.py` (`results/wide_concurrency.csv`)
     confirms this: throughput scales cleanly with dask `num_workers` (11 -> 13.4MB/s,
     32 -> 26.2MB/s, 64 -> 31.9MB/s, 128 -> 38.0MB/s), plateauing near the raw-curl
     ceiling. More threads = more overlapped dead-time, not faster individual requests.

4. Reviewed a GDAL config from a public reference (flytemosaic's
   `gdal_configs.py::get_worker_config`, see below) and added an A/B comparison cell
   (baseline vs. that config, same chunk/worker/item count, now `wide_gdal_config.py`) -
   **added but never actually run** (blocked by a stray `breakpoint()` upstream before we
   pivoted to the two-stage benchmark instead). Annotated expectations, not yet
   confirmed:
   - Expected to matter: `GDAL_HTTP_MULTIPLEX`/`GDAL_HTTP_VERSION=2` (HTTP/2
     multiplexing - avoids re-paying handshake per request over a shared connection),
     `CPL_VSIL_CURL_CHUNK_SIZE=12MB` (fewer/bigger range requests -> better
     overhead/payload ratio, same reasoning as chunk-size tuning).
   - Confirmed **irrelevant in this pipeline**: `VSI_CACHE`/`VSI_CACHE_SIZE` - odc-loader
     source (`odc/loader/_rio.py::_rio_read`) forces `VSI_CACHE=False` on every pixel
     read unconditionally, regardless of global config.
   - **Open item for next session**: actually run this cell (or a standalone version)
     and check whether HTTP/2 multiplexing lets us hit a similar throughput ceiling with
     far fewer dask workers than 128.

5. Reviewed [github.com/ljstrnadiii/flytemosaic](https://github.com/ljstrnadiii/flytemosaic)
   (cloned to scratchpad, not kept in this repo) as a reference mosaic-downloader
   architecture, per user request, explicitly ignoring its Flyte orchestration:
   - Builds a **GDAL Raster Tile Index (GTI)** file up front (`ogr2ogr`-built FlatGeobuf
     embedding dtype/extent/CRS/band-count metadata) so GDAL never opens/inspects each
     source COG just to plan a mosaic. odc-stac solves the same problem differently (via
     STAC item properties directly), so less directly applicable, but confirms
     metadata-probe overhead is a real, known cost class in this space.
   - Has its own separate ingest stage that **re-downloads and re-encodes** every source
     file into a uniformly-tiled COG in their own bucket (explicit `BLOCKSIZE=512`)
     before ever mosaicking - a stronger version of gfetch's planned `download` stage
     (which currently just caches original bytes, not re-encodes). Flagged as a possible
     future refinement, not committed to.
   - **Most instructive divergence**: each worker runs dask in
     `scheduler="single-threaded"` mode, relying on `GDAL_NUM_THREADS=ALL_CPUS` +
     HTTP/2 multiplexing for I/O concurrency, and scales out via many separate
     *processes* (Flyte-scheduled pods) rather than one big Python thread pool. Real
     hint that our 128-dask-thread brute force in point 3 above may be compensating for
     HTTP/2 not being enabled - ties directly into the open item in point 4.

5b. **Re-reviewed 2026-09-20** (`flytemosaic/gdal_configs.py` + `flyte/build.py`, this
    time reading the actual task code, not just the config dict) to pin down exactly how
    the single-threaded/GDAL-threads pattern above is wired, before designing the A/B
    test in point 7 below:
    - `get_worker_config()` (the source of `aggressive_gdal_config()`, now in
      `wide_gdal_config.py`) is unchanged from what was captured previously - confirmed
      byte-for-byte, nothing new there.
    - `flyte/build.py::write_mosaic_partition_task` is a **Flyte task per chunk
      partition** (one pod each, `map_task(..., concurrency=32)` fans them out), each
      with `environment=gdal_configs.get_worker_config(8, debug=True)` set as **process
      environment variables** on that pod (not `odc.stac.configure_rio()`/per-thread
      `rasterio.Env`), and inside the task body: `with dask.config.set(scheduler=
      "single-threaded"): ...` - literally zero Python-level thread fan-out for the
      pixel read/warp/write. The comment in their own source states the intent
      explicitly: *"single threaded dask scheduler but ALL_CPUS for GDAL_NUM_THREADS in
      environment"*.
    - So the concurrency story has three independent layers in their design, not one:
      (1) GDAL's own internal warp/read thread pool (`GDAL_NUM_THREADS=ALL_CPUS`,
      within one read), (2) HTTP/2 multiplexing (many in-flight range requests over few
      TCP connections, hiding per-request handshake latency without needing separate
      threads to do it), and (3) **process**-level parallelism across independent
      chunk-partitions (32 concurrent pods), each pod internally single-threaded at the
      dask level. Our own concurrency sweep (point 3 above) only ever exercised one
      knob - dask thread count within a single process - which is a different lever
      from any of these three.
    - This directly reframes the open question from "does the GDAL config help a
      little" to "can (1)+(2) replace dask thread-count entirely, i.e. does a
      single-threaded dask scheduler with GDAL_NUM_THREADS=ALL_CPUS and HTTP/2 come
      anywhere near the 128-thread throughput ceiling on its own" - which is exactly
      what `single_threaded_gdal_threads` in `wide_gdal_config.py`'s `CASES` sweep
      tests.

6. **Built and ran `two_stage_mosaic.py`: the two-stage design is dramatically faster,
   not just an HPC nicety.** Full results in `results/two_stage_*.csv`.
   - **Download stage** (`stac_asset.download_item()`, full-file sustained downloads,
     asyncio-concurrency swept via semaphore - this stage doesn't touch GDAL/rasterio at
     all, it's a separate HTTP stack, so none of the GDAL tuning above applies to it):
     32 concurrent item downloads reached **85.3 MB/s** (wide, 9 items/5.56GB) and
     **77.2 MB/s** (deep, 13 items/7.66GB) - near the claimed line rate, using far less
     concurrency than the windowed-read approach needed to reach less than half that
     throughput.
   - **Load stage** (now purely local disk reads, no network): best throughput 231
     Mpix/s (wide, chunk=7168/workers=11) and 213 Mpix/s (deep, chunk=2048/workers=11) -
     the deep-scenario number actually **exceeds** the original benchmark's own reported
     odc-stac peak for that exact scenario (105 Mpix/s, measured cloud-colocated on
     Planetary Computer's own hub).
   - **Key tuning finding, opposite in character from the download stage**: once local,
     *more worker threads can actively hurt*, not just plateau. At matching chunk sizes,
     `workers=32` was worse than `workers=11` in 5 of 6 comparisons; worst case
     (deep, chunk=7168, workers=32) took **394.7s vs 51.3s** at the same chunk size with
     11 workers - a 7.7x regression, plausibly memory pressure (a single 7168px chunk at
     that scenario's shape is ~3.7GB resident; several concurrent against this machine's
     19.3GB total RAM, already carrying other apps, thrashes). Not independently
     re-verified with a second run - flagged as a strong hypothesis grounded in the
     memory arithmetic, not a confirmed-reproducible fact.
   - **End-to-end**: wide scenario, best configs, ≈78s total (65.1s download + 13.0s
     load) vs. the direct-remote-load approach's own load-only time of ~158s at its best
     (128 workers, 38MB/s ceiling, same 5.99GB dataset) - **two-stage is ~2x faster
     overall despite doing strictly more work** (a real download step plus a load step).
   - Found and fixed a real bug while building this, same class as the one already
     logged in `claude/tech-stack.md`'s POC log: `stac_asset.download_item()` mutates
     its input `Item` in place (rewrites asset hrefs to the local download path).
     Reusing the same search-result items across sweep iterations into different cache
     dirs caused a later iteration to try "downloading" from a previous iteration's
     now-deleted local path. Fixed with `copy.deepcopy(item)` per download attempt
     inside `download_items()` - same fix shape as gfetch's own documented retry bug.

7. **2026-09-20 - added `--which`/`--chunks` CLI args to `wide_mosaic.py`** so any single
   cell (`concurrency`/`gdal_config`/`chunks`) can be run standalone
   (`uv run python benchmark/wide_mosaic.py --which gdal_config`) instead of always
   running the whole file top-to-bottom, and expanded the GDAL-config cell from a
   2-case (baseline/aggressive) A/B into 4 cases: `baseline`, `multiplex_only` (isolates
   just `GDAL_HTTP_MULTIPLEX`/`HTTP_VERSION=2`, since `aggressive` bundles it with a
   bigger chunk size and GDAL-side caching, confounding which knob helped),
   `aggressive` (flytemosaic's config unchanged), and `single_threaded_gdal_threads`
   (flytemosaic's actual production pattern from point 5b - single-threaded dask +
   `GDAL_NUM_THREADS=ALL_CPUS`, zero Python-thread fan-out).

   **Found and fixed an item-selection mistake in `wide_mosaic.py`'s search cell**,
   inherited from the same helper in `two_stage_mosaic.py` (fixed there too). Not a
   library bug - initially misdescribed as one, corrected below. Earth Search lists
   each MGRS tile/date twice here: not two different scenes, but two **processing
   baselines** of the same acquisition. ESA reprocessed this archive's baseline 02.14
   (generated 2020-06-06, same day as acquisition) to baseline 05.00 (generated
   2023-06-16), and Earth Search kept the old item rather than replacing it - confirmed
   via `s2:processing_baseline`/`s2:generation_time`/`s2:datastrip_id`, and independently
   confirmed as a known, documented STAC-community issue (not specific to us) via
   [Element84/earth-search discussions on filtering by datatake/processing_baseline]
   (https://github.com/Element84/earth-search) and the
   [stactools-packages/sentinel2 issue on item IDs across reprocessing]
   (https://github.com/stactools-packages/sentinel2/issues/130).
   `groupby="solar_day"` only discards the older item at *merge* time, after GDAL has
   already fetched pixels from **both**, wasting ~2x the network transfer for the same
   output extent - and the module docstring's claim that this was harmless
   ("groupby...merges them back into 1 time slice, same shape") was wrong: same
   *output* shape, not same *cost*. Worse than the wasted bandwidth: baseline 04.00
   (2022-01-25) changed how DN values encode reflectance (a constant offset for
   negative values), so blending items across baselines in one composite is a
   **radiometric correctness risk**, not just an efficiency one - a reason to resolve to
   one baseline per tile/date before loading regardless of any performance concern.

   The first fix attempt (deduping by lowest `s2:nodata_pixel_percentage`, copying
   `two_stage_mosaic.py`'s existing helper as-is) was itself subtly wrong: in this AOI,
   baseline 02.14 happened to have marginally *lower* nodata% than the reprocessed
   05.00 item in every single tile, so nodata%-based dedup was silently keeping the
   **superseded** baseline, the opposite of what you'd want. Both scripts' dedup
   helpers now select the **highest `s2:processing_baseline`** per tile/date instead.
   Verified: still 18 raw items -> 9 deduped (same count, correct item now kept per
   tile), output shape `90978x10980`/5.99GB - an exact match for the original
   benchmark's own reported `10980x90978px, 5.58 GiB`. Affected every cell of the
   file this lived in at the time (`wide_mosaic.py`, since split - see point 8) - its
   old concurrency-sweep numbers predated this fix (measured against the 18-item,
   wrong-baseline-mixed list) and were treated as stale until re-run (see point 8).

   **Also corrected a live network misdiagnosis, worth recording so it isn't repeated**:
   the first `gdal_config` run (killed mid-flight, see below) showed baseline throughput
   of only ~1.6-1.8 MB/s, and a quick single `curl` against the bucket looked equally
   bad (~2.4 MB/s before timing out), which read as "the network is ~10-20x slower than
   last session." Re-checked properly at the user's prompt: a Cloudflare speed-test
   download hit 55 MB/s (general connection is fine), and a *longer* single-connection
   curl sample (3.93 MB/s) plus 4 parallel curls to the same bucket (~21 MB/s aggregate)
   both landed right in line with last session's own numbers (4.6 MB/s single / 36.5
   MB/s at 6 connections) - i.e. no session-to-session network degradation; the first
   curl was just a short, noisy sample. The benchmark's own bad numbers were real, but
   caused by the baseline-duplication issue above (roughly doubling actual bytes
   transferred per unit of output), not the network - a ~312-task dask graph confirmed
   there was no task-starvation issue either. The run was killed before finishing (see
   "Open threads" below) since a fair comparison across configs needs the corrected
   dedup in place first; not yet re-run.

8. **2026-09-20 - split `wide_mosaic.py` into `wide_concurrency.py`/`wide_gdal_config.py`/
   `wide_chunks.py` + a shared `common.py`**, once the `--which`-flag file from point 7
   had grown to hold three unrelated sweeps behind a dispatcher (see "Files" above for
   the new layout). `common.py` now holds the single copy of `dedupe_by_processing_baseline()`
   that `two_stage_mosaic.py` also imports, instead of two near-duplicate copies of the
   dedup helper drifting apart. `wide_chunks.py`'s CLI was also ported from `argparse` to
   `cyclopts`, matching gfetch's own CLI convention (`dev-stack.md`).

   **Two near-misses while doing this split, worth recording**: (1) a
   `wide_mosaic.py --which concurrency` run the user had started independently in another
   terminal was still active and writing to `results/wide_mosaic_concurrency.csv` when
   the split began - confirmed via `ps aux` before touching anything, and the user
   explicitly OK'd treating that run's output as disposable before it was killed and the
   file restructured. (2) `wide_concurrency.py` and `wide_gdal_config.py` take no CLI
   arguments at all (unlike `wide_chunks.py`), so passing `--help` to them is not a safe
   no-op smoke test - it's silently ignored and the real (network-hitting) script runs
   anyway. Learned this by actually doing it: `--help` on `wide_concurrency.py` ran a
   real ~3-minute sweep against live Earth Search before being caught and killed. One
   good row landed in `results/wide_concurrency.csv` before the kill (128 workers,
   71.58 MB/s) - consistent with the disposable run above, reconfirming the network was
   never the problem and the corrected dedup (point 7) fixes the throughput numbers.
   Neither script was given a CLI for this (no parameters to expose), so the lesson is
   procedural, not a code fix: don't probe a no-argument notebook-style script with
   `--help` expecting an argparse/cyclopts-style safe exit.

9. **2026-09-20 - found and fixed `wide_gdal_config.py`'s `"baseline"` case running with
   a colder GDAL config than every other script's own "no special tuning" baseline**,
   making it an unfair/misleading floor for the multiplex/aggressive comparisons (all of
   them would look better than they should relative to it). Root cause: `odc.stac`
   exposes two different config entry points with **different `cloud_defaults` defaults**
   - `configure_s3_access()` (used by `wide_concurrency.py`/`wide_chunks.py`/
   `two_stage_mosaic.py`) defaults `cloud_defaults=True`; `configure_rio()` (used by
   `wide_gdal_config.py`, since it needs to pass per-case GDAL options) defaults it to
   `False`. `cloud_defaults=True` merges in `GDAL_CLOUD_DEFAULTS`
   (`GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`, `GDAL_HTTP_MAX_RETRY=10`,
   `GDAL_HTTP_RETRY_DELAY=0.5`) - without it, GDAL does a sidecar-file probe (extra
   network round-trip) on every file open. `wide_gdal_config.py`'s `"baseline"` case
   called `configure_rio(aws=...)` with no other kwargs, silently inheriting `False`.
   **Fix**: pass `cloud_defaults=True` explicitly in `wide_gdal_config.py`'s
   `configure_rio()` call, for every case - so the sweep only varies the
   flytemosaic-specific options on top of the same cloud-optimized floor every other
   script already gets. Not yet re-run against live data to confirm the corrected
   `"baseline"` numbers (blocked on the same not-yet-re-run status as point 4/7's open
   `wide_gdal_config.py` re-run).

10. **2026-09-20 - found and fixed a crash in `aggressive_gdal_config()`'s `"aggressive"`/
    `"single_threaded_gdal_threads"` cases**, hit on an actual live run at
    `--worker-counts=64` (the default `memory_gb=4` sizes `GDAL_CACHEMAX` to exactly
    2^31): `TypeError: an integer is required`, raised deep inside
    `odc.loader._rio.restore_env -> rasterio.env.setenv -> set_gdal_config` at `ds.load()`
    time (i.e. once the case's config is actually applied to a dask worker, not at
    `configure_rio()` call time - `baseline`/`multiplex_only` ran fine first since they
    don't set `GDAL_CACHEMAX` at all). Root cause: rasterio's `set_gdal_config()` special
    -cases `GDAL_CACHEMAX`, calling `GDALSetCacheMax64(val)` directly with the raw
    Python value instead of the `str(val).encode()` path every other option goes
    through - so it needs an actual `int`, not the `str(...)` every other value in that
    dict is. flytemosaic's original sets these as **process environment variables**
    (see point 5b), where a decimal-string `GDAL_CACHEMAX` is fine (GDAL's own C-level
    `CPLGetConfigOption`/`atoi` parsing handles it) - the crash only shows up going
    through rasterio's Python `Env()`/`configure_rio()` path instead, which is specific
    to how this benchmark reuses that config dict. **Fix**: keep `GDAL_CACHEMAX` as a
    plain `int` in `aggressive_gdal_config()`'s returned dict (now typed
    `dict[str, str | int]`), every other key unchanged. Verified directly against
    `rasterio.env.Env(**aggressive_gdal_config(memory_gb=4))`, not yet re-run through
    the full script end-to-end.

11. **2026-09-20 - found and fixed `wide_chunks.py` running its whole chunk-size sweep
    at an implicit, unusually low dask thread count**, after the user reported it
    running far slower than even `wide_concurrency.py`'s slowest measured point.
    `wide_chunks.py` never called `dask.config.set(scheduler="threads",
    num_workers=...)` around its `ds.load()`, unlike every other load-sweeping script
    here - so it silently ran at dask's own fallback (`dask.threaded.get()`'s
    `ContextAwareThreadPoolExecutor(CPU_COUNT)`, i.e. `dask.system.CPU_COUNT` = 11
    threads on this machine), which `wide_concurrency.py`'s own sweep already found to
    be the single slowest concurrency it tested (13.4 MB/s vs. 38.0 MB/s at 128
    threads). But that alone doesn't explain "far slower than the slowest
    `wide_concurrency` case" (which itself ran at this same 11-thread floor, just at a
    fixed `chunk=2048`) - the rest comes from chunk size itself: for the "wide"
    scenario's `10980x90978px` output shape, `chunk=512` (the smallest in
    `DEFAULT_CHUNK_SIZES`) produces **3916 output chunks vs. 270 at `chunk=2048`** (a
    14.5x difference) and 150x more than `wide_chunks.py`'s own largest (`chunk=7168`,
    26 chunks) - each output chunk is a separate GDAL windowed read paying the
    ~0.55-0.65s fixed TCP/TLS handshake cost from finding #3, regardless of payload
    size, so the small end of the sweep combines the slowest concurrency tested
    anywhere in this benchmark suite with far more overhead-dominated requests than any
    point `wide_concurrency.py` ever ran - the two effects compound multiplicatively.
    **Fix**: added a `--workers` CLI param (default 128, matching
    `wide_concurrency.py`'s best-measured setting) so the chunk-size sweep no longer
    also varies concurrency by accident; `num_workers` is now recorded in
    `results/wide_chunks_<scenario>.csv` for traceability. Not yet re-run against live
    data.

12. **2026-09-20 - added a `--scenario` CLI param to every script that was hardcoded to
    the "wide" scenario** (`wide_concurrency.py`/`wide_gdal_config.py`/`wide_chunks.py`
    - `two_stage_mosaic.py` already had `--scenarios`, plural, for benchmarking both in
    one run). `common.py` now exposes a shared `Scenario = Literal["wide", "deep"]`
    type plus `SCENARIO_SEARCH`/`SCENARIO_CRS` registries, which `two_stage_mosaic.py`
    also now imports instead of keeping its own local copies. `cyclopts` renders
    `Literal["wide", "deep"]` as `[choices: wide, deep]` in `--help` and rejects any
    other value before the script does anything - the CLI-level validation
    `two_stage_mosaic.py` previously did by hand (`set(scenarios) -
    SCENARIO_SEARCH.keys()`) is gone, no longer needed. Each of the three
    newly-parametrized scripts now writes `results/<script>_<scenario>.csv` instead of
    a single unsuffixed file, so running both scenarios back-to-back doesn't clobber
    the previous run's results (see the `results/*.csv` note above). Default scenario
    is `"wide"` everywhere, preserving prior behavior when `--scenario` is omitted.

13. **2026-09-20 - added `--max-time` to `wide_concurrency.py`, capping how long a
    single case's `ds.load()` may run, by moving each case into its own
    `multiprocessing` child process.** Motivation: dask's local `threads` scheduler has
    no API to cancel an in-flight task - a naive in-process timeout (e.g. `future =
    executor.submit(ds.load); future.result(timeout=...)`) would give up *waiting*, but
    the abandoned dask worker threads keep running regardless, consuming
    network/CPU/GIL time that would bleed into whatever case runs next in the same
    process, undermining exactly the kind of apples-to-apples fairness this investigation
    has repeatedly had to fix elsewhere (points 9/10/11). A genuinely separate OS process
    sidesteps this: it can simply be killed outright, reclaiming its threads and sockets
    atomically, with nothing left to interfere with the next case.
    - **Verified empirically before implementing** (not just from docs): a
      `multiprocessing` child started with the `"spawn"` context inherits the parent's
      stdio, including a real pseudo-terminal (`sys.stdout.isatty()`/`rich`'s own
      terminal detection both see the same terminal as the parent, confirmed via a
      `pty.fork()`-based test) - so each case's own logging/progress bar (built exactly
      as if it were the top-level script, via the same `setup_logging()`/
      `sweep_progress()`/`dask_load_progress()` used everywhere else) renders normally
      with no special-casing needed for the fact that it's a subprocess. Also confirmed
      `pystac.Item` pickles/unpickles cleanly (needed to hand the parent's already-
      searched items to the child via `multiprocessing.Process(args=...)`).
    - **Inferring throughput from an incomplete load**: `odc.stac.load()`'s underlying
      `Dataset.load()` is one blocking `dask.compute()` call - xarray only writes results
      into the `Dataset` once *every* chunk finishes, so there is no way to read out
      already-completed chunks from a still-running computation. What *is* available
      mid-flight is a plain **count** of completed dask tasks, via a new `TaskCounter`
      (`common.py`) - a `dask.callbacks.Callback` with no UI, only a `.completed`
      counter, read from the coordinating thread rather than mutated by it. On a
      timeout, `mb_s`/`mpix_s` are scaled by `completed / total` dask tasks - an
      estimate (some tasks are cheap reshape/concat work, not uniform GDAL reads, so the
      scaling is approximate), flagged via new `timed_out`/`completed_tasks`/
      `total_tasks` CSV columns rather than presented as a real measurement.
    - **Two-layer timeout, both verified with synthetic (non-network) dask computations**
      before running against real data: (1) inside the child, `ds.load()` runs on a
      **daemon** thread; the child's main thread does `thread.join(timeout=max_time)`,
      and on timeout builds the inferred row, sends it over a `multiprocessing.Pipe()`,
      then calls **`os._exit()`** rather than returning normally - a normal return would
      block at interpreter shutdown waiting to join dask's own internal worker-thread
      pool (which registers its own `atexit` shutdown hook, independently of whether the
      *calling* thread is a daemon), so only a hard `os._exit()` guarantees the process
      actually dies on schedule. (2) the parent enforces its own safety-net timeout
      (`max_time` + 30s grace, via `Connection.poll(deadline)`) in case a child hangs
      *before* ever reaching its own timed section (e.g. stuck in
      `configure_s3_access()`/graph-building) - confirmed with a child that never touches
      dask at all, killed cleanly via `terminate()`/`kill()` escalation. All three paths
      (normal completion, internal timeout, safety-net timeout) verified against small
      synthetic dask arrays with an artificial per-chunk delay before touching Earth
      Search, to isolate the process-lifecycle mechanism from network variability.
    - **Not yet applied to `wide_gdal_config.py`/`wide_chunks.py`** - `wide_concurrency.py`
      was the first (trial) implementation, at the user's request; the same pattern
      should generalize to those scripts' own per-case loads if it proves out in
      practice. Not yet re-run against live Earth Search data with a real `--max-time`.

14. **2026-09-20 - rolled `--max-time` out to every remaining `ds.load()` sweep**
    (`gdal_config.py`, `chunks.py`, and `two_stage_mosaic.py`'s load stage - see point
    13 for the trial implementation and full rationale). Factored the per-case
    subprocess/timeout plumbing that was inline in `concurrency.py` into two shared
    `common.py` helpers so the pattern isn't copy-pasted four times: `run_with_timeout()`
    (run a callable on a daemon thread, `thread.join(timeout=...)`, report whether it
    finished) and `run_case_in_subprocess()` (spawn a case's child process, enforce the
    `max_time + SAFETY_GRACE_S` safety net via `Connection.poll()`, escalate to
    `terminate()`/`kill()`). `concurrency.py` itself was refactored onto these same
    helpers, so all four scripts now share one implementation.
    - `gdal_config.py` needed one extra wrinkle: its per-case GDAL config
      (`configure_rio(**rio_kwargs)`) has to be set *inside* the child, since GDAL/rasterio
      config doesn't cross process boundaries (same reasoning as `configure_s3_access()`
      in `concurrency.py`'s child).
    - `two_stage_mosaic.py` needed the opposite trade-off from the other three: it keeps
      one shared `sweep_progress()` `Live` display open across all three of its stages
      (download sweep, full download, load sweep), and a load-case child rendering its
      *own* `Progress`/`Live` at the same time would fight the parent's for the terminal.
      Its load-case child therefore uses `TaskCounter` alone (no rich display) for the
      timeout-inference bookkeeping, while the parent's already-existing "load sweep"
      task bar still advances once per completed case, same as before.
    - **Verified without touching live Earth Search data** (each case's `_run_case_child`/
      `_run_load_case_child` was exercised directly against a synthetic, `dask.array`
      -backed `xarray.Dataset` with an artificial per-chunk delay, `odc.stac.load` mocked
      out - confirming both the success and timed-out/inferred-throughput code paths for
      `two_stage_mosaic.py`'s new load-case function specifically, since its shape
      (dropping the rich display) differed enough from `concurrency.py`'s already-verified
      version to warrant its own check). Not yet run against real Earth Search data for
      any of the three newly-updated scripts.

15. **2026-09-21 - fixed `two_stage_mosaic.py`'s load-stage child silently missing
    `cloud_defaults`, found while reviewing the first real (non-synthetic) results from
    `concurrency.py`/`chunks.py`/`gdal_config.py`/`two_stage_mosaic.py` together (via
    `visualize.py`) before a planned final run.** `main()` calls
    `odc.stac.configure_s3_access(aws_unsigned=True)` once, in the **parent** process,
    before either stage runs - its default `cloud_defaults=True` sets
    `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR` (skips a sidecar-file directory listing on
    every GDAL file open; not S3-specific, applies to local files too). Point 14's
    subprocess refactor of the load stage never carried this over into
    `_run_load_case_child`, since GDAL/rasterio config doesn't cross process boundaries
    (same fact behind the point-9 `wide_gdal_config.py` fix) - so every `two_stage_load.csv`
    row collected after that refactor was measured *without* this setting, a real,
    if likely modest, regression from what the pre-refactor in-process version had.
    **Fix**: added `odc.stac.configure_rio(cloud_defaults=True)` (no `aws=`, purely
    local) at the top of `_run_load_case_child`. Also confirmed, reviewing the same
    fresh data: none of `concurrency.py`/`chunks.py`/`gdal_config.py`'s own findings
    (remote chunk-size-vs-overhead tradeoffs, remote concurrency ceilings, HTTP/2
    multiplexing) should transfer to stage 2 - it has zero network I/O once files are
    local, and `two_stage_load.csv`'s own chunk/worker sweep already reconfirms the
    memory-pressure regression from finding #6 (`chunk=7168, workers=32`: 23.82 mb/s vs.
    183.3 mb/s at `workers=11`, deep scenario). Separately, `run_download_sweep`'s
    hardcoded `subset = its[:3]` means its `max_concurrency=8` vs. `32` rows can't
    actually differ meaningfully (only 3 items to download, so concurrency above 3 is
    never exercised) - not fixed (both scenarios only have 9-13 items total, so no
    subset size would let `max_concurrency=32` mean anything more than "fully
    parallel" here), just flagged so those two rows aren't over-interpreted. Not yet
    re-run against live data with the `cloud_defaults` fix in place.

16. **2026-09-21 - found and fixed a `ValueError: too many values to unpack` crash in
    `_download_one()`, hit on the first real run reusing a fully-populated `cache/` from
    a prior session.** Pre-existing bug, unrelated to any change from this session -
    only surfaces on a cache-hit resume, which hadn't happened before since earlier
    runs always started from an empty cache. Root cause: for a band already downloaded
    (`key not in pending`), the code looks up its local path via
    `item_dir.glob(f"{key}.*")` expecting exactly one match - but the completion
    sentinel touched a few lines above it, `{item_dir}/{key}.complete`, also matches
    that same glob pattern (`"red.*"` matches `"red.complete"` just as much as
    `"red.tif"`), so a fully-cached item always returned two matches, not one. **Fix**:
    filter the sentinel out explicitly (`p.suffix != ".complete"`) rather than relying
    on the glob returning exactly one match. Verified directly against the actual
    `cache/wide/` directory left on disk (`.suffix`-filtered glob resolves to exactly
    the one real asset file); not yet re-run through the full script.

17. **2026-09-21 - fed two of this investigation's findings back into `src/gfetch/`**,
    the first code changes to the actual package this investigation has produced
    (everything before this was benchmark-only, per the top-of-file note above):
    - `mosaic.py::load()` now calls `odc.stac.configure_s3_access(aws_unsigned=True)`
      before `odc.stac.load()` - finding #1's fix, previously only applied in
      `benchmark/`'s own scripts. `aws_unsigned=True` is safe unconditionally for
      every source gfetch currently supports (Earth Search's public S3 bucket,
      Planetary Computer's SAS-signed Azure Blob hrefs), since it only affects AWS
      credential resolution.
    - `search()` now dedupes Sentinel-2 results to one item per (tile, date), keeping
      the highest `s2:processing_baseline` - finding #7's fix (this investigation's own
      `dedupe_by_processing_baseline()`), previously only applied in the sibling
      `wide_mosaic.py`/`two_stage_mosaic.py` benchmark scripts, not in gfetch itself.
      Verified `gfetch.search.search()` now returns the identical 9/13 deduped
      wide/deep item counts these benchmark scripts get from their own copy of the
      same logic.
    - **Not applied**: dask thread-count tuning for the compute step (opposite-
      direction effects between remote and post-download local loads, no single good
      default - see point 6's memory-pressure finding vs. point 3's remote-concurrency
      finding) and the GDAL config tuning from `gdal_config.py` (still has two
      pathologically-hanging cases per the "Open threads" section below, not
      confirmed-good).
    - Added `gfetch_pipeline.py` (see "Files" above) to check gfetch's real
      end-to-end throughput against the reference numbers above now that both fixes
      are in place.

18. **2026-09-21 - ran `gfetch_pipeline.py` for the first time and found two things
    worth recording, not a gfetch performance gap.**
    - **`results/two_stage_download_full.csv`'s deep/wide numbers were inflated
      by stale cache, not measuring real throughput.** `gfetch_pipeline.py`'s first
      run measured 80.46 MB/s (wide, 5.56GB/9 items) and 90.21 MB/s (deep, 7.93GB/13
      items) - both markedly below `two_stage_download_full.csv`'s 203.79 MB/s
      (wide, 6.8GB) and 182.52 MB/s (deep, 15.59GB), which read at first like a real
      gap between gfetch's `download_items()` and `two_stage_mosaic.py`'s own
      reimplementation. Root cause, confirmed by inspecting `cache/deep`/`cache/wide`
      directly: those directories still held 15 directories' worth (8.3GB) of
      **superseded-processing-baseline item downloads** (`*_0_L2A`, `.jp2` assets)
      left over from *before* finding #7's dedup fix was applied to
      `wide_mosaic.py`/`two_stage_mosaic.py`'s own item search - one stale extra
      directory per tile/date, sitting alongside the correct, current
      (`*_1_L2A`, `.tif`) one. `two_stage_mosaic.py`'s `cache_size_bytes()` sums
      every file under the whole cache directory recursively, with no awareness that
      an item it once downloaded is no longer part of the current deduped list, so
      it silently counted both the fresh, correct download *and* the leftover stale
      one - while `elapsed_s` only timed the fresh (correct-item) download, since
      the stale files' sentinels made them no-ops. Net effect: `mb_s` inflated by
      roughly the stale/fresh byte ratio (~1.2x wide, ~2x deep, matching the 2/13 and
      13/13 stale-to-kept directory ratios respectively) - confirmed quantitatively,
      not just qualitatively: subtracting the measured 8.3GB of stale files from
      `two_stage_download_full.csv`'s reported totals (6.8GB-1.19GB=5.6GB wide,
      15.59GB-7.66GB=7.93GB deep) reproduces `gfetch_pipeline.py`'s own measured
      `n_bytes` almost exactly. **The stale directories have been deleted** (disposable,
      gitignored cache, safe per this file's own "reuse or delete as needed" note);
      `two_stage_download_full.csv`'s existing deep/wide rows are now known-inflated
      and shouldn't be trusted as a comparison baseline until `two_stage_mosaic.py` is
      re-run against the now-clean cache - not done here (would re-run its download
      *and* load sweeps, not just the full-download step there's no standalone flag
      for). `gfetch_pipeline.py`'s own numbers (`results/gfetch_pipeline_download.csv`)
      are the trustworthy current reference for gfetch's actual download throughput:
      comparable to, not slower than, the corrected reference figures.
    - **Confirmed the Earth Search collection gfetch actually queries doesn't have
      the processing-baseline duplication bug finding #7 fixed.** `sources.py` maps
      earthsearch's Sentinel-2 collection to `sentinel-2-c1-l2a` ("Collection 1"),
      not the legacy `sentinel-2-l2a` collection `benchmark/common.py` searches.
      Querying both directly for the same deep-scenario tile/date range: legacy
      returns 26 items (13 tile/date pairs, each with a 02.14 and a 05.00 baseline
      duplicate); `c1` returns 13, already resolved to one (05.00) item per
      tile/date - Element84's Collection 1 reprocessing has already deduped this at
      the catalog level. `search()`'s new dedup logic is therefore currently a no-op
      against gfetch's actual configured source (confirmed: `gfetch_pipeline.py`'s
      search returned the same 9/13 counts with zero duplicates logged as dropped) -
      still worth keeping as defensive/correct code (e.g. if gfetch's source config
      ever points at the legacy collection, or if Planetary Computer's own
      `sentinel-2-l2a` collection turns out to have the same duplication - not yet
      checked either way), just not the reason gfetch's own numbers differ from the
      old reference ones (that was purely the stale-cache issue above).

## Open threads / next steps

- **Re-run `two_stage_mosaic.py`** now that its cache is clean (point 18) - its
  `two_stage_download_full.csv` deep/wide rows are currently known-inflated and
  shouldn't be used as a comparison baseline until then.
- **Check whether Planetary Computer's `sentinel-2-l2a` collection has the same
  processing-baseline duplication as Earth Search's legacy collection** (point 18) -
  not yet checked either way; matters for how much `search()`'s dedup logic actually
  does once gfetch's Planetary Computer source is exercised end-to-end (see
  `tech-stack.md`'s "Next steps" item 1).
- **`--max-time` now run against real Earth Search data** (points 13/14) - `aggressive`/
  `single_threaded_gdal_threads` both hit their `max_time` in `gdal_config_wide.csv`
  with almost nothing completed (`aggressive`@64 workers: 0/564 tasks in the full
  150s incl. safety-net grace - looks stuck, not just slow; `single_threaded_gdal_threads`:
  37/564 in 120s). Worth a closer look with `--verbose`/a longer `--max-time` before
  trusting those two configs are actually worse rather than pathologically hung.
- **Re-run `gdal_config.py`** now that the double-listed-item/processing-baseline
  dedup bug (point 7) is fixed - `baseline`/`multiplex_only` now have real numbers
  (`multiplex_only` showed no real improvement over `baseline` for either scenario,
  contrary to the original hypothesis), but `aggressive`/`single_threaded_gdal_threads`
  need the timeout investigation above before they're trustworthy.
- **Re-run `wide_concurrency.py`** fully, once the above is done - only one row
  (128 workers, 71.58 MB/s) exists so far post-fix (see point 8), from a run killed
  early by accident.
- Reconcile chunk size: this investigation's best local-load chunk sizes (2048-7168px)
  are much larger than `tech-stack.md`'s current documented default (~5km/500px at
  10m). Different scenario shape (multi-tile wide mosaic vs. a typical single-AOI
  gfetch job) though - needs a dedicated check against gfetch's actual typical AOI
  scale before changing the documented default, not a direct transfer.
- **Partially done (point 17)**: the S3-credential and processing-baseline fixes are
  now in `src/gfetch/`. Still open: feed the two-stage-is-faster finding (point 6) into
  `claude/tech-stack.md`'s download/write stage design as a stated rationale (currently
  only HPC-necessity is documented there) - not a code change, gfetch's CLI already
  exposes `search`/`download`/`mosaic` as separate stages.
- flytemosaic's GTI/re-COG-on-ingest pattern - explicitly parked as a "future idea," not
  decided on.
- `benchmark/cache/` is sitting on disk, gitignored - reuse or delete as needed
  (`wide`/`deep` from `two_stage_mosaic.py`'s runs, `gfetch_pipeline/` from
  `gfetch_pipeline.py`'s - kept separate deliberately, see point 17).

## Repo hygiene note

Several files were left uncommitted at the end of this session (by design - nothing was
asked to be committed): `.gitignore` (added `benchmark/cache/`), `claude/tech-stack.md`
(8 lines, pre-dating this session), `pyproject.toml`/`uv.lock` (user added `tqdm`), and
all of `benchmark/` (untracked, never committed). Check `git status` before assuming a
clean tree in the next session.
