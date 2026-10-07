# gfetch — tasks log

Running, dated log of confirmed bugs and design issues found during development, plus
the chronological decision log of how `claude/tech-stack.md`'s architecture was arrived
at. Mirrors `lsatfetch`'s `claude/tasks.md` convention (see `claude/dev-stack.md`): each
entry records what was confirmed, why it matters, and the fix/decision taken.

`claude/tech-stack.md`/`claude/dev-stack.md` hold the current, synthesized state of the
architecture and tooling decisions — this file holds the journal of how/when each part of
that state was reached, and should be read chronologically, not as reference material.

## POC log

- **2026-09-19 — odc-stac local-href POC: PASSED.** Searched one Sentinel-2 item on
  Earth Search, downloaded a band via `stac_asset.download_item()`, and compared
  `odc.stac.load()` output between the original remote-href item and the
  downloaded/rewritten local-href item over the same small window: identical shape,
  dtype, and pixel values. Also confirmed a resumed/mixed download (one asset
  pre-cached, one freshly downloaded in the same call) loads correctly and never
  re-downloads the pre-cached asset.
  - **First attempt hit a self-inflicted pitfall, not a library bug** — worth recording
    since it's an easy mistake to repeat: manually filtering an item's assets before
    calling `download_item()` (`item.assets = {key: item.assets[key]}`) breaks
    `Asset.owner`, because pystac's `.assets` is a plain dict attribute — reassigning it
    does **not** re-point the contained assets' `owner` backref at the item you just
    assigned them onto. The asset object keeps pointing at whatever item it was taken
    from. `odc.stac.load()` resolves hrefs via `Asset.get_absolute_href()`, which joins
    a relative href against `asset.owner.get_self_href()` — so a wrongly-owned asset
    silently resolves against the *remote* STAC endpoint instead of the local cache,
    producing bogus URLs (404/403) instead of an obvious error.
  - **The fix is to not do that.** `stac_asset.download_item()`'s own `Config.include`/
    `exclude` parameters are the documented way to restrict which assets get downloaded,
    and they don't disturb ownership — this is also exactly what's needed for resuming
    a partial download (pass `include=<pending asset keys>`, `keep_non_downloaded=True`
    to keep already-cached assets' entries in the returned item). Verified working
    pattern for gfetch's download stage (atomic + resumable, per the durability design
    above):
    ```python
    pending = [k for k in wanted_keys if not (item_dir / f"{k}.complete").exists()]
    with tempfile.TemporaryDirectory(dir=item_dir) as tmp:
        result = await download_item(
            item, Path(tmp), config=Config(include=pending), keep_non_downloaded=True
        )
        for key in pending:
            # .href is relative once a self href is set (which download_item always
            # does) - get_absolute_href() is required to get the real temp-dir path.
            tmp_path = Path(result.assets[key].get_absolute_href())
            final_path = item_dir / f"{key}{tmp_path.suffix}"
            shutil.move(tmp_path, final_path)  # atomic on the same filesystem
            (item_dir / f"{key}.complete").touch()
            result.assets[key].href = str(final_path)
    # assets not in `pending` keep their original remote href here - point them at
    # their already-downloaded local path too before handing the item off.
    result.set_self_href(str(item_dir / f"{item.id}.json"))
    ```
  - **Verified independently, re-reading `stac-asset==0.4.7`'s source**: its writes
    really are non-atomic against a killed process, confirming the concern that
    motivated this whole design (see "Download-stage durability" above) — `Client.
    download_href()` streams straight into the final target path via `aiofiles.open
    (path, "wb")`; the partial file is only deleted in the `except` block, which a
    SIGKILL/walltime-preemption never reaches. gfetch's own temp-dir + rename layer
    (above) is genuinely necessary, not defensive overengineering.
  - Confirmed against `stac-asset==0.4.7`, `odc-stac==0.5.3`, `pystac==1.15.2`.

- **2026-09-19 — Full pipeline POC (search → load/mosaic → write, incl. disjoint-region
  writes): PASSED end-to-end**, at the user's request to de-risk the whole pipeline
  before writing production code, not just the download↔load handoff. Searched Earth
  Search for a month of Sentinel-2 over a ~6.7×5.6km AOI near Paris (8 items, 2 adjacent
  MGRS tiles, cloud cover < 40%), loaded 3 bands + SCL at native 10m resolution with
  `odc.stac.load(items, bands=[...], geobox=geobox, groupby="solar_day", chunks={...})`
  (6 solar-day groups), masked clouds/shadow/cirrus/no-data via the SCL band (classes
  `{0,1,3,8,9,10}`), and took a `median(dim="time", skipna=True)` composite — the
  recommended-stack shape from Pangeo thread 5010 ("Best practices for large-scale
  Sentinel-2 mosaics"), now confirmed to actually run correctly end-to-end rather than
  just cited.
  - **Zarr write**: `computed.to_zarr(path, mode="w")` then `xr.open_zarr(path)` —
    round-tripped values match exactly (`np.allclose(..., equal_nan=True)`, since the
    masked composite legitimately contains NaNs where every timestep at a pixel was
    cloud-masked).
  - **Pre-planned disjoint-region write (the HPC write-stage pattern from "Compute-stage
    output durability" above): PASSED**, confirming the design is actually implementable
    with the tools we chose, not just plausible on paper. Pattern: write the store's
    metadata/coords once with `template.to_zarr(path, compute=False, mode="w")`, then
    have each independent "worker" write only its own non-overlapping slice with
    `worker_slice.to_zarr(path, region={"x": slice(...), "y": slice(None)})`. One real
    gotcha: xarray's region-write rejects the call if *any* variable being written lacks
    a dimension in common with the region (our case: the scalar `spatial_ref` CRS
    coordinate) — fix is `worker_slice.drop_vars([c for c in worker_slice.coords if c
    not in worker_slice.dims])` before the per-region write (the coordinate was already
    written once, correctly, by the metadata-only pass). Two workers writing disjoint
    x-slices reassembled to values identical to the single in-memory mosaic.
  - **Rough benchmark, single laptop over home/office internet (not HPC/cloud-colocated)**:
    computing the 6-solar-day × 3-band, ~6.06M-pixel mosaic (full dask graph: remote COG
    reads + reprojection + SCL masking + median reduction) took **~250-300s** (~0.02
    Mpix/s) across three runs. This is not comparable to the Pangeo benchmark's cited
    74.7 Mpix/s for odc-stac wide/mosaic workloads — that number almost certainly reflects
    compute co-located with the data (in-region cloud) or cached reads, whereas this run
    is dominated by real WAN download bandwidth for genuinely fresh COG reads, not CPU or
    library overhead. Useful as gfetch's own real-world expectation-setting data point
    (rough budget: single-machine, home-internet, full-resolution multi-band monthly
    mosaics over a small AOI take a few minutes, dominated by network, not compute) —
    not as a library performance verdict. Proper benchmarking (cloud-colocated, varying
    AOI size/resolution/band count, Dask cluster vs. local threads) is future work, not
    done here.
  - **Dependency gap found and fixed**: `zarr` itself was missing from gfetch's chosen
    dependency list (only `pystac-client`/`stac-asset`/`odc-stac`/`odc-geo`/`rich`/
    `cyclopts`/`omegaconf`/`retry` had been added) despite Zarr being the v1 output
    format since the very first architecture decision — added via `uv add zarr`
    (`zarr==3.4.0`).
  - Confirmed against `zarr==3.4.0`, `xarray==2026.7.0`, `odc-stac==0.5.3`, `dask==2026.8.0`.

## Decision log

- **2026-09-19** — Chose `odc-stac` over `stackstac` for array loading (Pangeo community
  consensus + odc-stac's active maintenance vs. stackstac's own "don't use in production"
  disclaimer).
- **2026-09-19** — Chose `stac-asset` for the download layer, `pystac-client` for search,
  `odc-geo` for AOI/CRS handling.
- **2026-09-19** — v1 output target is analysis-ready **Zarr cubes** (not just raw asset
  downloads) — mosaicking/compositing is in scope from the start.
- **2026-09-19** — Icechunk noted as a promising future storage backend; **no POC now**.
- **2026-09-19** — v1 STAC sources: **Planetary Computer + Element84 Earth Search**, built
  against both from day one so the `StacSource` abstraction isn't accidentally PC-shaped.
- **2026-09-19** — Added HPC as a first-class deployment target: gfetch's stages
  (`search`, `download`, `load`/`mosaic`, `write`) must be independently callable so
  external orchestrators (SLURM/Snakemake/Nextflow/Parsl) can pin each to the right node
  class (internet-connected vs. compute-only). No Globus integration, no attempt to
  abstract heterogeneous Dask clusters — both are the user's orchestration concern.
- **2026-09-19** — Download-stage durability is gfetch's own responsibility: atomic
  temp-write + rename, one sentinel file per completed asset (not a central manifest DB,
  to avoid distributed-locking problems on shared HPC filesystems). Rejected relying on
  `stac-asset`'s built-in skip-if-exists (not atomic) or transparent `fsspec` caching
  (same non-atomicity risk) as the sole durability mechanism.
- **2026-09-19** — Reaffirmed Icechunk as future/optional rather than default: its ACID
  guarantees are only proven under concurrent commits on S3-compatible storage, and are
  explicitly disclaimed on local/shared POSIX filesystems — the default HPC storage
  shape. Default output writer is plain Zarr with pre-planned non-overlapping per-worker
  chunk regions instead, which is safe on any POSIX filesystem.
- **2026-09-19** — Identified `geefetch` (own prior GEE-based library) as directly
  relevant prior art, more so than any external reference — see "Prior art" section
  above. Of its patterns: config/`omegaconf` convention adopted; `geobbox`,
  `Tiler`/`TileTracker`, and integrity-check-on-resume all considered and explicitly
  rejected — not everything from a sibling project should be carried over by default,
  and gfetch should default to designing fresh where a prior pattern isn't clearly worth
  reusing.
- **2026-09-19** — Confirmed the write stage's atomic-write story: zarr-python's v3
  `LocalStore` already writes chunks atomically (temp+rename) natively, so the
  download-stage pattern generalizes to the write stage without extra work — but the
  pre-planned disjoint-chunk-region-per-worker plan stays required (per-chunk atomicity
  doesn't cover two workers racing on the same chunk). Investigated virtual chunk
  references (kerchunk/VirtualiZarr/Icechunk) as a possible alternative to materializing
  pixel data — confirmed real, but not applicable to odc-stac's core reprojection/mosaic
  step (which requires real pixel materialization); kept as a later, narrower
  optimization, not part of the v1 design.
- **2026-09-19** — Scaffolded the repo per `dev-stack.md`: `pyproject.toml` repinned to
  Python 3.12 (was left at the `uv init` default of 3.14), chosen dependencies added via
  `uv add`, ruff/pydoclint/pytest config blocks added, `pyrefly` config generated via
  `pyrefly init`, `.pre-commit-config.yaml`/`.flake8` added with hooks pinned to current
  releases, `AGENTS.md`/`README.md`/`CONTRIBUTING.md`/`CHANGELOG.md` added. `uv sync` and
  `pre-commit run --all-files` both pass clean. Runtime deps (incl. `rasterio`) resolved
  and installed on 3.12 with no GDAL/uv conflict — the fallback conda env wasn't needed.
- **2026-09-19** — Ran the odc-stac local-href POC (see "POC log" above): **passed**,
  confirming `odc.stac.load()` reads locally-rewritten STAC items identically to
  remote-href ones.
- **2026-09-19** — At the user's request, re-checked Project Pythia cookbooks (not
  previously reviewed) and re-checked the Earthmover blog and Pangeo threads
  specifically for STAC-download implementation patterns, before finalizing the
  download-stage code — see the new "Project Pythia cookbooks" and "Earthmover blog —
  re-checked" entries above. Net effect: found and fixed a self-inflicted pitfall in the
  first download-stage POC attempt (manually filtering `item.assets` breaks
  `Asset.owner`; use `stac_asset.Config.include`/`exclude` instead, which is also the
  correct mechanism for resume) — see the corrected "POC log" entry above for the
  working pattern. Confirmed gfetch's `download` stage design itself (separate,
  optional, atomic, resumable) is correct and not contradicted by any external source
  reviewed; also confirmed, by re-reading `stac-asset`'s source directly, that its
  writes are genuinely non-atomic, so gfetch's own temp-dir+rename wrapper is
  necessary, not defensive overengineering.
- **2026-09-19** — At the user's request, POC'd the *whole* pipeline end-to-end
  (search → load/mosaic → write, incl. the pre-planned disjoint-region write pattern),
  not just the download↔load handoff, plus a rough real-world throughput benchmark —
  see the new "Full pipeline POC" entry above. Everything passed: the Pangeo-recommended
  `groupby="solar_day"` + SCL-cloud-mask + `median` composite shape actually runs
  correctly against real Earth Search data, `to_zarr()` round-trips a masked
  (NaN-containing) composite exactly, and the disjoint-region write pattern that
  gfetch's HPC write-stage design depends on is confirmed implementable with xarray/zarr
  as chosen (one real gotcha found and fixed: scalar coordinates like `spatial_ref` must
  be dropped before a region-write call). Also found and fixed a real scaffolding gap:
  `zarr` itself was missing from the dependency list despite being the v1 output format
  since the first architecture decision.
- **2026-09-19** — Implemented the pipeline for real: `src/gfetch/{sources,profiles,
  search,download,mosaic,write}.py` plus a `cyclopts`/`omegaconf` CLI
  (`gfetch init`/`search`/`download`/`mosaic`), and a real test suite (23 tests, 2
  network-backed and marked `slow`). All the POC-verified patterns above carried over
  directly. Two real bugs were caught by tests/a CLI smoke test that the earlier POCs
  hadn't exercised:
  - **Retry-across-attempts item corruption**: `download_item()` mutates its input
    `Item` in place (sets its self href, rewrites asset hrefs). The download stage's
    retry wrapper was reusing the *same* item object across retry attempts, so a
    failed first attempt (e.g. a real `TimeoutError` hit during the CLI smoke test)
    left the item in a half-mutated state that broke the next attempt (`KeyError` on
    an asset key that should have existed). Fix: `copy.deepcopy(item)` fresh for each
    retry attempt, inside the retried closure. A dedicated regression test
    (`test_download_items_retry_does_not_corrupt_item`, monkeypatching `stac_asset.
    download_item` to fail once then succeed) reproduces and confirms the fix.
  - **Temp-dir-then-move ordering**: an earlier draft moved the downloaded file out of
    its temp directory *after* the `with tempfile.TemporaryDirectory()` block that
    downloaded it had already exited (and deleted the directory) — a plain
    `FileNotFoundError`, caught immediately by the first download unit test. Fix: the
    move must happen inside the same `with` block as the download.
  - Neither bug was visible in the earlier hand-run POC scripts, which never happened
    to retry or hit this exact code path — real test coverage (including a CLI smoke
    test against live Earth Search data, not just unit tests against local files)
    caught both before they'd have surfaced as confusing failures on an actual HPC job.
  - **CLI stage-granularity decision**: `search` and `download` map to their own CLI
    subcommands (matching the internet-connected-node split), but `mosaic`/`write`
    share a single `gfetch mosaic` subcommand rather than two, since they always run
    on the same compute-only node/job back-to-back with no natural serialization point
    for a lazy dask-backed `xr.Dataset` between them — splitting them into separate
    CLI invocations would only add pointless intermediate I/O. The underlying
    `gfetch.mosaic`/`gfetch.write` *library* functions stay separate and independently
    testable/importable either way.
  - CLI stage hand-off between processes (`search` → `download` → `mosaic`) is done by
    serializing/deserializing `pystac.ItemCollection` JSON files under `output_dir`
    (`items.json`, `cached_items.json`) — confirmed round-trips absolute local hrefs
    and self hrefs correctly before relying on it.
- **2026-09-20** — Started a "Reference implementations (inspiration repos)" list (see
  above) of full end-to-end pipelines worth checking against when building a gfetch
  stage, as opposed to individual libraries. Added `ljstrnadiii/flytemosaic` (already
  under investigation for `benchmark/`'s GDAL-config work) and
  `earth-mover/serverless-datacube-demo` (previously only skimmed at the landing-page
  level in "Source-by-source findings" above, now actually read source-first) as the
  first two entries, each with concrete "worth stealing" vs. "not adopted" call-outs
  rather than a general summary.
- **2026-09-21** — Fed two `benchmark/`-investigation findings back into `src/gfetch/`
  (see `benchmark/README.md` for the full investigation this drew from):
  - `mosaic.py::load()` now calls `odc.stac.configure_s3_access(aws_unsigned=True)`
    before `odc.stac.load()`. Without it, GDAL/rasterio falls through to botocore's
    full credential chain (including an EC2-instance-metadata lookup that hangs until
    TCP timeout off-EC2) on every S3 asset — this alone took the benchmark's measured
    throughput from ~4MB/s to ~11-16MB/s. `aws_unsigned=True` is safe unconditionally:
    both current sources (Earth Search's public S3 bucket, Planetary Computer's SAS-
    signed Azure Blob hrefs) work fine with it, since it only affects AWS credential
    resolution.
  - `search()` now dedupes Sentinel-2 results to one item per (tile, date), keeping the
    highest `s2:processing_baseline`, resolving the open question above. Confirmed via
    the benchmark that Earth Search legitimately double-lists reprocessed
    tiles/dates and that blending baselines in one composite is a radiometric
    correctness risk (baseline 04.00 changed how DN values encode reflectance), not
    just wasted bandwidth.
  - **Not applied**: dask thread-count tuning for the compute step. The benchmark
    found opposite-direction effects (more threads help a remote/direct load, but can
    actively hurt a post-download local load via memory pressure) with no single good
    default across scenarios — left as future, scenario-aware tuning rather than a
    blanket change. GDAL config tuning (`GDAL_HTTP_MULTIPLEX`/aggressive caching) was
    inconclusive in the benchmark (two configs pathologically hung rather than being
    confirmed slower) and also not carried over.
  - Added `benchmark/gfetch_pipeline.py`, a benchmark case that exercises gfetch's own
    `search`/`download`/`mosaic` functions end-to-end against the same "wide"/"deep"
    scenarios as the rest of `benchmark/`, to check gfetch's real throughput against
    these reference numbers now that the fixes above are in place. **Ran it**:
    download throughput is 80-90 MB/s (wide/deep), load throughput up to ~440 Mpix/s
    (wide, chunk=7168) - in line with, not behind, the reference numbers; an initial
    reading that gfetch's download was ~2x slower than `two_stage_mosaic.py`'s own
    reimplementation turned out to be a stale-cache measurement artifact in the
    latter's own reference CSV, not a real gap (see `benchmark/README.md` point 18).
    Also confirmed gfetch's actual configured Earth Search collection
    (`sentinel-2-c1-l2a`) doesn't hit the processing-baseline duplication bug the
    dedup fix above targets - Element84's Collection 1 reprocessing already resolves
    to one item per tile/date - so that fix is currently defensive/dormant against
    gfetch's own default source, confirmed still worth keeping (see `benchmark/
    README.md` point 18 for the full reasoning and the open question of whether
    Planetary Computer's collection has the same issue).
- **2026-09-21** — Reworked the mosaic/write stage to produce **one output per native
  UTM zone an AOI spans**, instead of reprojecting everything into a single zone
  auto-picked from the AOI's centroid (`GeoBox.from_bbox(bbox, crs="utm", ...)`, the
  original v1 design). Prompted by the user asking what CRS a country-scale AOI (e.g.
  Tanzania, spanning UTM zones 35S/36S/37S) would end up in - the honest answer was
  "one arbitrarily-chosen zone, with real distortion growing toward the AOI's edges
  and a needlessly huge single grid (Tanzania: ~123,873 x 120,008px at 10m in one
  zone)." Decided **not** to keep this as the default and add a `crs` override later;
  instead, native-per-zone is now the only mode (a `crs` override to force one from
  the CLI/config, e.g. for users who explicitly want a single-zone output despite the
  distortion, is a plausible small follow-up but not built).
  - `gfetch.mosaic.group_by_utm_zone()`: groups items by their **own** footprint's
    UTM zone (resolved via `odc.geo.crs.CRS.utm()` on each item's STAC `bbox`), not
    the AOI as a whole - correct even if a single search AOI spans items whose tiles
    fall in different zones near a boundary.
  - `gfetch.mosaic.zone_geobox()`: builds each zone's output grid from the AOI bbox
    **clipped to that zone's natural 6-degree-wide longitude band** (derived from the
    zone number), not from the union of whichever items happened to be returned -
    deterministic and reproducible independent of item availability/gaps near a zone
    boundary.
  - `gfetch.mosaic.mosaic_by_zone()`: the new top-level entry point, returns
    `dict[CRS, xr.Dataset]`; `cli/mosaic.py` calls `write()` once per zone, to
    `Config.zarr_path(crs)` (now a method, not a fixed property) -
    `<output_dir>/mosaic_epsg<code>.zarr` per zone.
  - **Real bug found while implementing this, worth recording**: `GeoBox.from_bbox()`
    only reprojects its `bbox` argument when `crs` is literally the string `"utm"` (a
    special-cased sentinel) - passing a concrete, already-resolved `CRS` object
    instead makes it treat the bbox's raw numeric values as **already being in that
    CRS's units**, with no reprojection at all (confirmed by reading
    `GeoBox.from_bbox`'s source directly: a plain tuple or a `BoundingBox` with a
    non-None `.crs` bypasses the "utm"-string reprojection branch entirely). This
    silently produced a nonsensical 1x1-pixel geobox near the UTM false-origin in an
    early version of `zone_geobox()` before being caught by a unit test. **Fix**:
    explicitly `.to_crs(crs)` the EPSG:4326 intersection bbox before calling
    `GeoBox.from_bbox()`, never relying on `from_bbox`'s own `crs=` kwarg to
    reproject a resolved (non-"utm"-string) CRS.
  - Verified end-to-end against live Earth Search data: a real AOI straddling the
    35S/36S boundary near 30°E correctly split into two `mosaic_epsg327{35,36}.zarr`
    stores via the full `gfetch search` → `gfetch mosaic` CLI path, both readable back
    with `xarray.open_zarr()` and the expected CRS.
- **2026-09-22** — Went from sparse to routine logging across every stage
  (`search`/`download`/`mosaic`/`write`, their CLI wrappers, config loading), at the
  user's explicit request after a large `gfetch search` (Tanzania, 10,456 matched
  items) looked hung with zero log output between "Searching..." and "Found N items" -
  it wasn't hung, just silent through a long STAC API pagination loop (confirmed live
  via `lsof`/`nettop` on the running process: an active connection, steadily receiving
  bytes). **Policy going forward: prefer adding `log.debug`/`log.info` at any
  potentially-long or otherwise-opaque step over leaving it silent** - the user
  explicitly said not to be shy about this. Concretely: `search()` now logs the STAC
  API's reported match count up front (`ItemSearch.matched()`) and one `log.debug` per
  page fetched; `download_items()` logs a start/end summary; `mosaic()`/`load()`/
  `mosaic_by_zone()` log item/zone counts and geobox shape before the actual
  (potentially slow, silent) `odc.stac.load()` call; `write()`/`prepare_template()`
  log before the blocking `to_zarr()` call; `cli/config.py` gained a logger it didn't
  have at all before.
- **2026-09-22** — Reworked `download_items()`'s progress display from one bar for the
  whole batch to **one bar per concurrently-downloading item, plus the existing
  overall bar** - matching `benchmark/`'s dask-callback pattern (a single `rich.
  Progress` instance can render multiple live bars, whether fed by dask callbacks or,
  here, `stac_asset`'s own per-asset message stream), and confirmed the same
  single-process precondition holds (`download_items()` is pure `asyncio.gather` +
  a semaphore, no multiprocessing, so one shared `Progress` object works). Considered
  and rejected building a separate message-router task: instead, each item's own
  download coroutine (`_download_item()`) is fully self-contained - it adds its own
  task via the existing `temporary_task()` helper (same add/cleanup pattern already
  used for the main bar) and, if a `progress` tracker was passed, creates its own
  local `asyncio.Queue` for `stac_asset.download_item(messages=...)`, running a small
  helper (`_report_asset_progress()`) concurrently via `asyncio.gather()` to turn
  `OpenUrl`/`WriteChunk` messages into real byte counts. No message queue or
  progress-reporting task lives outside the function that owns it. Verified live
  (`max_concurrent_items=3`, real Earth Search downloads, forced-terminal `rich`
  output): exactly one main bar plus up to 3 simultaneous per-item bars, each with
  independent byte counts/speed/ETA, correctly relabeling as each slot picks up its
  next item, all bars cleanly removed on completion (asserted via a `mock.patch.
  object` spy on `Progress.add_task` in the new `test_download_items_reports_progress`
  test: exactly `1 + n_items` tasks created, zero left over).
- **2026-09-22 — found and fixed a real HPC-only bug: `gfetch download` could hang
  forever on a network requiring an HTTP(S) proxy for egress, even though `gfetch
  search` and plain `curl` worked fine on the same node.** Reported by the user on an
  HPC cluster; root-caused without direct access to that machine, then confirmed fixed
  there. Diagnosis, in order:
  - Ruled out DNS/general connectivity (curl succeeded) and ruled out `odc.stac.
    load()`'s known S3-credential-chain hang (point 17 above - `download` never calls
    `odc.stac.load()`, and Earth Search asset hrefs are plain `https://`, not `s3://`,
    so `stac_asset` routes them through its `HttpClient`, not an S3 client, confirmed
    by reading `stac_asset.client.Clients.get_client()`'s scheme-dispatch directly).
  - Root cause: `stac_asset.HttpClient` builds a bare `aiohttp.ClientSession(timeout=
    ..., headers=..., middlewares=...)` with no `trust_env=True` - confirmed by
    reading `stac_asset/http_client.py` directly. Unlike `requests`/`curl` (which read
    `HTTP_PROXY`/`HTTPS_PROXY` automatically), `aiohttp` silently ignores those env
    vars unless `trust_env=True` is passed explicitly, and `stac_asset.Config` exposes
    no field for it. On a network that mandates a proxy for outbound access, this
    means `search` (via `pystac_client`/`requests`) and `curl` work while `download`
    (via `stac_asset`/`aiohttp`) attempts a direct connection that the network drops
    silently - explaining exactly the reported symptom (search fine, download hangs
    at the very start, no error).
  - Confirmed on the user's actual HPC node before writing any fix: `env | grep -i
    proxy` showed proxy vars set, and a bare `aiohttp.ClientSession()` (no
    `trust_env`) reproduced the hang against the same host `curl` reached fine.
  - **Fix**: `download.py` now patches `stac_asset.http_client.ClientSession` (the
    name that module's `from_config()` calls) with a plain factory function
    defaulting `trust_env=True` - not a subclass, since `aiohttp` explicitly
    discourages subclassing `ClientSession` (confirmed via a `DeprecationWarning` from
    an earlier subclassing attempt, corrected before shipping). Applied unconditionally
    at module import time; safe when no proxy is configured (`trust_env=True` is a
    strict superset of default behavior).
  - **Verified the mechanism itself**, not just "no more hang": pointed
    `HTTP_PROXY`/`HTTPS_PROXY` at a closed local port *after* a real search had
    already succeeded (isolating `aiohttp`'s behavior specifically), then ran a real
    download. Before the fix this would succeed (proxy silently ignored); after the
    fix it failed with `ClientProxyConnectionError: Cannot connect to host
    127.0.0.1:1` - proof the patch is genuinely routing through the configured proxy,
    not a false positive. **Confirmed on the user's HPC cluster afterward: downloads
    now work.**

- **2026-09-22** — Split this file out of `claude/tech-stack.md`: the "POC log" and
  "Decision log" sections above were dated journal entries, not part of the current
  architecture snapshot `tech-stack.md` is meant to hold, so they moved here wholesale
  (per lsatfetch's `claude/tasks.md` convention, see `claude/dev-stack.md`). Cross-
  references inside `tech-stack.md` that used to say "see POC log below"/"see Decision
  log" were repointed at this file.
- **2026-09-22** — Added a resampling override, resolving the open question flagged the
  same day about `SatelliteProfile`'s never-built "default resampling". `load()` gained
  a `resampling: str | dict[str, str] | None` parameter threaded straight through to
  `odc.stac.load(resampling=...)`; `mosaic()`/`mosaic_by_zone()` gained the same
  parameter plus a new `_pin_mask_band_resampling()` helper that always forces the
  cloud-mask band (Sentinel-2 `scl`) to `"nearest"`, overriding even an explicit
  per-band request for it — it holds categorical class values, so any other resampling
  method would fabricate class values that were never in the source data. `Config`
  gained a `resampling: dict[str, str]` field (plain dict only, not the full
  `str | dict | None` union `load()`/`mosaic()` accept — confirmed by a failing
  `OmegaConf.structured()` call that omegaconf's structured-config typing doesn't
  support arbitrary `Union`s of a primitive and a container type); a `"*"` key sets the
  default for bands not otherwise listed, reusing odc-stac's own wildcard convention
  (`odc.loader._reader.resolve_load_cfg`'s `resampling.get(name, resampling.get("*",
  fallback))`) rather than inventing gfetch's own. Deliberately not built: any
  per-satellite automatic default (e.g. bilinear for Sentinel-2's coarser 60m
  `coastal`/`nir09` bands) — the override stays manual/opt-in per job.
- **2026-09-22** — Discussed and explicitly deferred a disk-bounded streaming
  download+mosaic design (motivated by jobs whose intermediate downloaded assets can't
  all fit on disk at once): interleave `download`/`mosaic` tile-by-tile instead of
  running each stage once for the whole job, deleting a tile's cached assets once every
  Zarr chunk it feeds has been written. Confirmed feasible in principle, built entirely
  from primitives gfetch already has (sentinel-file readiness, `write.py`'s
  `prepare_template`/`write_region`) plus three new pieces (chunk↔tile dependency map,
  reference-counted cache eviction, a disk-budget-aware loop). Two coordination
  mechanisms considered: a filesystem-mediated watch-loop (no new dependency, stays
  inside the existing "stage separation, not orchestration" non-goal) vs. submitting an
  explicit task DAG to HyperQueue (more scheduling power, but a new dependency and a
  real scope change toward being a workflow engine — and overkill given the actual
  per-tile dependency graph is shallow, download → mosaic → delete). **Decision: not
  building this now** — recorded as a deferred long-term idea, not scheduled; full
  reasoning kept in `claude/tech-stack.md`'s "Future: disk-bounded streaming
  download+mosaic" section rather than only here, since (unlike this file) that's where
  someone would look for *why* a future feature is shaped a certain way, not just that
  it was discussed.
- **2026-09-22** — Added `sentinel-1` support: a new `SatelliteProfile` (`default_bands=
  ("vv", "vh")`, `cloud_mask_band=None` - SAR has no cloud-masking equivalent) and a
  `sentinel-1-grd` collection registered on both `earthsearch` and `planetary-computer`
  in `sources.py`. Verified the real collection metadata before building anything (per
  `claude/tech-stack.md`'s "Sentinel-1" entry in "STAC source landscape") rather than
  assuming: both sources' GRD is public/anonymous, uncalibrated raw amplitude, natively
  **EPSG:4326** (not UTM); PC's `sentinel-1-rtc` (the analysis-ready, terrain-corrected
  product) requires a PC account, so it's deliberately not registered - a separate,
  larger follow-up once gfetch has any credentialed-source mechanism at all. Kept
  `median` as the composite default (not `mean`) given the raw/uncalibrated caveat -
  see `claude/tech-stack.md` for the full reasoning.
  - **`group_by_utm_zone` generalized from a partition to an overlap-based
    assignment**, the one real design change this required. It used to assign each
    item to exactly one "native" zone via `CRS.utm(item.bbox)` - correct only because
    Sentinel-2 items happen to fit within a single zone by construction (MGRS tiling).
    Confirmed via a real Earth Search item that Sentinel-1 GRD breaks that assumption
    (a scene over Japan spanned ~3.7° longitude, crossing two UTM zones on its own),
    so the old function would have silently dropped a wide item's contribution from
    every zone but its arbitrarily-chosen "native" one. New signature
    `group_by_utm_zone(items, aoi_bbox)` enumerates every UTM zone the AOI itself spans
    (via `pyproj.database.query_utm_crs_info` - the same primitive `odc.geo.crs.CRS.
    utm()` uses internally to pick its single best-fit zone, just relaxed to return
    every candidate instead of one) and assigns an item to every such zone whose
    AOI-clipped extent it overlaps - a strict generalization that produces the
    identical result for Sentinel-2 (verified by the existing tests, updated to pass
    an `aoi_bbox`, plus a new `test_group_by_utm_zone_item_spanning_zones_appears_in_
    both`). Added `pyproj` as an explicit direct dependency (`uv add pyproj`) rather
    than relying on it transitively through odc-geo, since gfetch's own code now
    imports it directly.
  - `sat:orbit_state` (ascending/descending) confirmed to be a plain queryable STAC
    property on real items from both sources - no new source-level mechanism needed,
    just a `Config.orbit_state: str | None` field and generalizing `cli/search.py`'s
    single-filter ternary into `_build_query()`, which merges `max_cloud_cover` and
    `orbit_state` into one query dict (raises on an invalid `orbit_state` value).
  - Verified live end-to-end, not just unit-tested: `gfetch search` → `gfetch mosaic`
    against real Earth Search `sentinel-1-grd` data (Paris-area AOI, ascending-only
    filter, 10 items, `median` composite) produced a fully-finite `vv`/`vh` Zarr store
    with sane DN-range values (84-1527). Also added `@pytest.mark.slow` tests
    exercising real multi-zone-spanning Sentinel-1 data through `mosaic_by_zone` and
    the `sat:orbit_state` query filter through `search()`, alongside unit tests for
    the new profile/collection/`_build_query` additions.
- **2026-09-22** — Found and fixed a real download-throughput bug, reported by the user
  from an actual HPC run: Sentinel-1 downloads measured ~1MB/s per thread vs. Sentinel-2's
  ~20MB/s on the same cluster. Root-caused before writing any fix, same discipline as the
  earlier HPC proxy bug:
  - Sentinel-1 asset hrefs are raw `s3://sentinel-s1-l1c/...` URIs (Sentinel-2's are
    already plain `https://...s3.us-west-2.amazonaws.com/...`), so `stac_asset` routes
    them through its `S3Client`, not its `HttpClient`. `S3Client` hardcodes its default
    region to `us-west-2` (`stac_asset.config.DEFAULT_S3_REGION_NAME`) and gfetch's
    `download.py` never overrode it. Confirmed via a direct HEAD request that
    `sentinel-s1-l1c` is actually in `eu-central-1`
    (`x-amz-bucket-region: eu-central-1` on the 301 redirect from the wrong-region
    endpoint) - every request was paying a wrong-region redirect round trip.
    `us-west-2` happens to be correct for Sentinel-2's bucket, which is why this never
    surfaced before Sentinel-1.
  - **Measured the effect directly** (boto3, same object, only the region parameter
    changed): `us-west-2` 19.67 MB/s vs. `eu-central-1` 63.83 MB/s - 3.2x, from this
    sandbox; the gap on the user's actual HPC node was presumably larger given the
    reported ~20x.
  - **Fix**: `download.py::_rewrite_s3_hrefs()` rewrites every `s3://bucket/key` asset
    href to its public `https://bucket.s3.amazonaws.com/key` equivalent before calling
    `download_item()`, routing it through `stac_asset`'s `HttpClient` instead of its
    region-guessing `S3Client` entirely - confirmed via `curl` that this global
    virtual-hosted-style endpoint reaches the bucket with **no redirect at all**,
    regardless of the bucket's actual region. General rather than Sentinel-1-specific
    (no hardcoded region string anywhere), so it protects any future `s3://`-hosted
    source the same way. Considered and rejected: hardcoding `s3_region_name=
    "eu-central-1"` for this one bucket (works, but bakes in bucket-specific knowledge
    gfetch has no other reason to track); pointing `stac_asset.Config.s3_endpoint_url`
    at the global endpoint directly (tested - botocore raises `PermanentRedirect`
    instead of auto-correcting once a custom `endpoint_url` is set, so this doesn't
    work as a fix).
  - **Verified live, not just structurally**: downloaded a real 720MB Sentinel-1 `vv`
    asset end-to-end through `gfetch.download.download_items()` after the fix - 7.37s,
    **97.7 MB/s**.

- **2026-09-21** — Added progress bars to `gfetch mosaic`, previously silent through
  its two actually-slow steps (`ds.compute()` per zone, then the write). Ported
  `benchmark/common.py`'s dask-callback pattern
  (`_RichDaskCallback`/`dask_load_progress`, itself the same "feed a shared `rich.
  Progress` from a dask scheduler callback" idea already decided for
  `download_items()`'s per-item bars) from the benchmark scripts into production code
  as `gfetch.utils.progress.dask_progress()`, rather than leaving it benchmark-only or
  reaching for `dask.diagnostics.ProgressBar()`'s separate non-`rich` bar. `cli/
  mosaic.py` now renders one overall "Computing zones" bar (advanced once per
  zone, matching `download_items()`'s main-bar convention) plus one per-zone task-count
  bar driven by `dask_progress()` around that zone's `ds.compute()` call.
  - `gfetch.utils.progress` gained a second bar constructor, `count_bar()` (task-count
    columns: `MofNCompleteColumn`/`TimeElapsedColumn`/`TimeRemainingColumn`), alongside
    the existing `default_bar()` (byte-oriented, `DownloadColumn`/`TransferSpeedColumn`)
    - a dask task count or a zone count isn't a byte quantity, so reusing
    `default_bar()` for `mosaic` would have rendered nonsensical byte-formatted counts.
  - `dask` added as an explicit direct dependency (`uv add dask`) since
    `gfetch.utils.progress` now imports `dask.callbacks.Callback` directly, rather than
    only depending on it transitively through `odc-stac`/`xarray` - same precedent as
    `pyproj` being added explicitly once gfetch's own code started importing it.
  - Verified live under a forced pseudo-tty (`script -q /dev/null uv run python ...`,
    since this session has no real interactive terminal): a two-zone fake dask
    workload rendered both the "Computing zones" bar and each zone's own task-count
    bar with live frames, confirming the wiring actually renders rather than just
    type-checking.

- **2026-09-21** — Exposed `mosaic` stage memory controls on `Config`, after the user
  hit real OOMs running `gfetch mosaic` on HPC. Root-caused before changing anything:
  read `odc.loader._builder.resolve_chunks` directly, which confirmed `cli/mosaic.py`
  was always calling `mosaic_by_zone(..., chunks=None)` - resolving (per that
  function's own default-fill logic) to `time=1, y=-1, x=-1`, i.e. **one chunk per
  band/timestep covering the entire zone's full-resolution extent**, with no spatial
  tiling at all. Combined with dask's default threaded scheduler running up to
  `dask.system.CPU_COUNT` chunks concurrently (confirmed cgroup/affinity-aware by
  reading its source, so it does respect a SLURM allocation's actual core count, not
  the whole node), peak memory for a large AOI (e.g. the Tanzania example already in
  `tech-stack.md`, ~120k x 120k px at 10m) is trivially many times the job's memory
  budget - explains the reported OOM without needing to reproduce it directly.
  - `Config` gained two new fields: `chunks: dict[str, int]` (dask chunk sizes passed
    straight through to `mosaic_by_zone()`, which already accepted this parameter but
    had no way to set it from a job's YAML - **default `{"x": 2048, "y": 2048}`**, at
    the user's explicit request, rather than defaulting to the old whole-zone
    behavior) and `n_compute_workers: int | None` (forwarded to
    `ds.compute(num_workers=...)` per zone in `cli/mosaic.py` - **default `None`**, at
    the user's explicit request, meaning dask picks its own default rather than gfetch
    second-guessing it).
  - Deliberately did **not** reuse the existing `n_workers` field for this - it already
    means something different (download-stage `max_concurrent_items`), and conflating
    the two would silently change an unrelated stage's concurrency.
  - Verified via a direct `OmegaConf`-backed `load()` round-trip (not just type
    signatures): unset config resolves to the new defaults (`chunks={"x": 2048, "y":
    2048}`, `n_compute_workers=None`), and both are overridable from YAML.

- **2026-09-21** — Found and fixed the real root cause of the OOM above, which the
  `chunks`/`n_compute_workers` knobs alone did **not** fix: the user correctly pushed
  back that bounding chunk size shouldn't matter if the whole thing gets pulled into
  memory anyway. It doesn't matter, because it was: `cli/mosaic.py` called
  `ds.compute(num_workers=...)` to fully materialize a zone's `xr.Dataset` **before**
  calling `write()`. `.compute()` unconditionally returns one fully in-memory
  `Dataset` as its final step, regardless of chunk size or worker count - those only
  bound *transient per-task* memory on the way there, not the final materialized
  size, which for a country-scale zone is the whole problem. `xr.Dataset.to_zarr()`
  on a still-**lazy** (dask-backed) dataset is dask/xarray's own native fix: it drives
  `dask.array.store()` and writes each chunk to the Zarr store as it's computed,
  never assembling the full array in Python memory - not odc-stac-specific, just what
  dask-backed `to_zarr()` always does when you don't short-circuit it with an eager
  `.compute()` first.
  - **Measured, not assumed** (`/usr/bin/time -l`, a synthetic 4-band/2-timestep
    8000x8000 float32 dask array, `chunks=(1,500,500)`, `num_workers=2`): eager
    `ds.compute()` then `to_zarr()` (the old `cli/mosaic.py` shape) peaked at
    **3.02 GB** RSS; streaming `ds.to_zarr()` directly on the lazy dataset peaked at
    **221 MB** - a 13.6x reduction, for identical chunk size and worker count.
    Re-verified through the real `gfetch.write.write()` function itself (not just the
    isolated synthetic script): **222 MB**, matching.
  - **Fix**: `cli/mosaic.py` no longer calls `.compute()` at all - `ds` stays lazy
    all the way into `write(ds, cfg.zarr_path(crs))`, wrapped in
    `dask.config.set(num_workers=cfg.n_compute_workers)` (verified this context
    manager form, not just the `.compute(num_workers=...)` kwarg form, actually bounds
    a downstream `to_zarr()` call's concurrency the same way) so `n_compute_workers`
    now genuinely bounds peak memory (chunk size x this value) instead of only
    bounding transient per-task memory during an eventually-materializing `.compute()`.
    The per-zone `dask_progress()` bar moved to wrap `write()` instead of the removed
    `.compute()` call - still driven by the same dask scheduler callback hooks, which
    fire for any local-scheduler graph execution (`to_zarr()` included), not just
    `.compute()` specifically.
  - `gfetch.mosaic.load()`/`mask_clouds()`/`composite()` were already confirmed lazy
    end to end (`.where()`/`.median()` on dask-backed arrays don't force computation),
    so `mosaic_by_zone()`'s output needed no changes - the eager materialization was
    entirely `cli/mosaic.py`'s own doing, not something the library layer forced.
  - Did **not** reach for `write.py`'s `prepare_template`/`write_region` (the
    multi-worker disjoint-region-write primitives) for this - that solves a different
    problem (several independent processes writing non-overlapping regions of one
    store concurrently), whereas this fix is single-process streaming, which
    `xr.Dataset.to_zarr()` already does natively for a lazy input with no extra
    machinery needed.

- **2026-09-21** — Reverted `Config.chunks` (the CLI-exposed dask chunk-size field
  added earlier the same day). Two reasons surfaced investigating a real HPC OOM
  report, together making the knob not worth keeping as-is:
  - It wasn't what fixed the OOM - the streaming-write fix above (dropping the eager
    `.compute()` before `write()`) was the actual fix, and works regardless of chunk
    size.
  - **Found it was largely non-functional anyway for gfetch's default compositing
    method**: `composite(method="median")` calls `xr.Dataset.median(dim="time",
    skipna=True)`, which dispatches to `dask.array.nanmedian`. Reading
    `dask/array/reductions.py::nanmedian` directly: it rechunks the reduced axis to
    a single block (correct, unavoidable) but **also rechunks the untouched x/y axes
    to `"auto"`** - dask's own generic ~128MiB-target heuristic - discarding whatever
    `chunks` was configured. Verified live: a `(1,200,200)`-chunked array came out of
    `.median(dim="time")` re-chunked to one single block covering the whole spatial
    extent; the same array through `.mean(dim="time")` (a true associative reduction,
    no full-axis rechunk needed) preserved the configured chunking exactly. So
    `Config.chunks` never actually controlled the final write chunk size once
    compositing ran - a real, confirmed gap between the shipped feature and its own
    docstring.
  - **Decision**: rather than fix `composite()` to re-chunk back to the configured
    size after `.median()` (deferred, not designed - would need its own POC and adds
    real complexity for a knob whose actual usefulness is now in question), the user
    chose to drop `Config.chunks` entirely and let odc-stac/dask pick chunk sizes on
    their own. `Config.n_compute_workers` was kept - it still genuinely bounds
    concurrent dask threads during the streaming write regardless of what chunk size
    ends up being used, independent of the `median`/`nanmedian` issue above.
  - Also surfaced in the same investigation, not yet acted on: a real HPC crash
    (`ValueError: None is not a valid chunk input` from `zarr.core.chunk_grids.
    normalize_chunks_nd`) traced to the user's HPC job mixing an older `xarray` (from
    a shared conda module) with a newer `zarr-python` (from the user's own
    `~/.local` packages) - not gfetch's own `uv`-managed environment at all. gfetch's
    own pinned `xarray` (2026.7.0) already carries the fix
    (`backends/zarr.py`: `if _zarr_v3() and chunks is None: chunks = "auto"`). An
    attempted defensive fix in `gfetch.write.write()` (explicit per-variable
    `encoding=` to avoid relying on that xarray-side patch) was tried, found **not**
    to actually prevent the crash on a direct repro, and reverted rather than shipped
    half-verified - left as an open question, not resolved.

- **2026-09-22** — Designed and implemented resumable/concurrent `mosaic`-stage writes
  (see "Compute-stage output durability" → "Resolved 2026-09-22" in `tech-stack.md` for
  the full design). Discussed in chat first per the usual practice, with the user
  pushing back hard on the first two drafts - both rejections led to a simpler design
  than originally proposed:
  - **First draft** (two-step `init`/`run` CLI, sentinel-file-per-tile bookkeeping,
    tile size decoupled from native chunk size as a separate concept) was rejected
    outright: "I don't like the two-step cli... I don't like the sentinel.complete
    thing... more options = more opportunities to go wrong." Each objection was
    verified against the actual library rather than argued from first principles -
    all three turned out to be droppable: `to_zarr(mode="w-")`
    against an already-initialized store raises a clean `FileExistsError` (tested
    directly), so `prepare_template()` could just catch it and become idempotent,
    removing the need for a separate init step or any ordering between tasks; and
    zarr-python v3's chunk files are themselves already the completion record (traced
    `zarr/storage/_local.py`'s `_atomic_write` - temp file + rename, confirmed same
    guarantee the download stage's sentinel exists to provide), so `region_is_written()`
    could check `store.exists()` on the store's own chunk keys instead of a side-car.
  - **Second draft** introduced a separate "tile size" concept distinct from the native
    chunk size, defended as necessary for job-array task granularity. Also rejected:
    "what's the difference with the native zarr atomic size?" Investigating the actual
    question (how does odc-stac restrict computation to part of an AOI) resolved it -
    `odc.stac.load(geobox=...)` is the only restriction mechanism needed, and it already
    internally filters items by overlap with whatever geobox is passed (traced
    `odc.stac._stac_load`'s own `GeoboxTiles`+`_tyx_bins` binning), so no manual
    per-tile item pre-filtering was needed either, further simplifying the first draft.
  - **User's own counter-proposal, adopted as the final design**: reintroduce a coarser
    unit above the native chunk after all, but as a pure multiplier
    (`Config.patch_chunks`, e.g. patches of 10x10 native chunks) rather than an
    independent size - justified as amortizing odc-stac's per-call item-parsing
    overhead across more pixels, explicitly framed as "just a computation
    hyperparameter" rather than a correctness requirement. Paired with an explicit,
    deliberate simplification: a patch with any missing native chunk is recomputed and
    rewritten **as a whole**, never partially - confirmed as the intended trade-off
    (simplicity over avoiding redundant recompute of already-correct chunks) rather
    than an oversight.
  - **Implementation** (`gfetch/mosaic.py::resolve_chunks`/`patch_geoboxes`,
    `gfetch/write.py::region_is_written`, `Config.patch_chunks`, `cli/mosaic.py`
    rewritten around a flat `patches[task_id::n_tasks]` split, `--task-id`/`--n-tasks`
    added to the `mosaic` CLI command) landed with new unit tests (`patch_geoboxes`
    tiling/edge-clipping, `region_is_written`'s per-variable/partial-write semantics,
    `prepare_template` idempotency, and CLI-level skip-on-resume/task-splitting via a
    call-counting fake `build_mosaic`). Full fast suite (58 tests) and
    `pre-commit run --all-files` (ruff, pyrefly, pydoclint) both pass.
  - **Found along the way**: a `pystac.Item` built with
    `geometry=None` silently loses its `bbox` too on a save/reload round-trip (pystac
    drops both together) - only surfaced because the new CLI-level test round-trips a
    fake item through disk the way `gfetch mosaic` really does, unlike `test_mosaic.py`'s
    existing in-memory-only `_item` helper. Not a real-world concern (real STAC items
    always carry geometry) - noted here only because it cost a debugging pass and could
    resurface in a future test.
  - **Real bug found the same day, from an actual HPC run** (Jean Zay, 3-zone,
    1491-item Sentinel-2 job, `--verbose` log pasted by the user): `cli/mosaic.py`'s
    `prepare_template(build(zone_items, geobox), path)` line called the *full*
    `mosaic()`/`load()` pipeline over the **entire zone's item list and geobox**
    (hundreds of items, geobox shapes up to 66786x119109px) just to read off the
    template dataset's shape/dtype - on **every single invocation**, even a fully
    resumed one where `prepare_template` was about to no-op anyway (confirmed in the
    log: `already initialized, skipping template write` printed right after each
    zone's expensive load). Real, non-trivial waste: odc-stac has
    to bin every item against the whole zone's tile grid to build the graph even at
    `compute=False` (multi-second delays per zone in the log), and `log_chunk_footprint`
    printed the whole throwaway graph's theoretical footprint (943 GB, 1336 GB
    "total") for data that's never read - which is also what made three sequential
    template-builds look, at a glance, like several huge datasets loading at once.
    Multiplied by however many SLURM array tasks a job launches (each redoes this for
    every zone). **Fix**: added `gfetch.write.store_initialized()` (checks for the
    root `zarr.json` - a cheap existence check, no store open) and gated the
    `build()`/`prepare_template()` call on it in `cli/mosaic.py`, so the expensive
    template build only ever runs once per zone, on the store's actual first
    initialization. Updated the call-counting CLI tests accordingly (a resumed run
    now expects **zero** `build_mosaic` calls, not one).

- **2026-09-22** — Built and live-verified a GEDI L2A vector-fetch POC via SlideRule
  (see `tech-stack.md`'s new "Vector data: GEDI via SlideRule" section for the full
  research/design writeup). New `src/gfetch/gedi.py::fetch_gedi_l2a()` + `gfetch gedi`
  CLI command, standalone from the raster pipeline (own AOI/time-range flags, not
  `cli/config.py::Config`), writing GeoParquet.
  - Docs for `docs.slideruleearth.io` 403'd on every page but the landing page
    (Cloudflare) - pulled the same content as raw markdown from the
    `SlideRuleEarth/sliderule` GitHub repo (`docs/user_guide/gedi.md` etc.) via `gh
    api` instead, then verified against a live call anyway - good thing, since the
    rendered docs turned out to be wrong: they document `elevation_lowestmode`/
    `elevation_highestreturn`, but the real returned columns are `elevation_lm`/
    `elevation_hr`, plus three undocumented columns (`orbit`, `track`, `sensitivity`).
  - Live smoke test: `bbox=(2.55, 48.35, 2.75, 48.50)` (Fontainebleau forest, small
    AOI near Paris chosen to keep the POC request fast rather than querying all of
    France), full 2020 - 47-164 granules depending on time range, tens of thousands
    of footprints, few seconds end to end via both the library call and the CLI.
  - No NASA Earthdata credentials were needed against SlideRule's public cluster -
    confirmed via docs wording, not assumed (self-hosted-only requirement).
  - `fields=` (defaulting to `elevation_lm`/`elevation_hr` only) implements gfetch's
    "subset of variables" requirement as a plain post-hoc column filter on
    SlideRule's already-reduced ~8-column schema; a richer server-side field-selection
    mechanism (`anc_fields`, documented for ICESat-2's `atl03x`/`atl24x` but not found
    documented or example-referenced for GEDI) was investigated and left unconfirmed,
    not built.
  - Tests: `tests/test_gedi.py`, one pure unit test (`_bbox_to_poly`) plus two
    `@pytest.mark.slow` live-network tests against the real public cluster, same
    convention as the raster pipeline's `test_search.py`. Full fast suite (60 tests)
    and `pre-commit run --all-files` (ruff, pyrefly, pydoclint) both pass; the two new
    slow tests were also run live and pass.

- **2026-09-22** — Real crash bug in `fetch_gedi_l2a()`, found by the user running
  `benchmark/gedi_aoi_sweep.py`'s default spatial sweep and pasting a traceback: a
  `KeyError: "['elevation_lm', 'elevation_hr'] not in index"` on the sweep's largest
  case, after SlideRule's own log showed `"Unexpected termination of response...
  attempt 3 of 3"` and `"Received 0 footprint(s)"`.
  - **Root cause**: `sliderule.gedi.gedi02ap()` returns `sliderule.emptyframe()` (a
    `GeoDataFrame` with *only* a `geometry` column, no data columns at all) both when
    an AOI/time range genuinely matches zero footprints and when the request fails
    server-side entirely - traced into `gedi.py`'s `__flattenbatches`:
    `if rsps == None: return sliderule.emptyframe(...)`. `fetch_gedi_l2a()`'s column
    selection (`gdf[[*fields, gdf.geometry.name]]`) had no empty-response guard, so
    it crashed with a confusing `KeyError` instead of surfacing the real problem
    (SlideRule already logs the failure loudly on its own - the crash added nothing).
  - **Fix**: `fetch_gedi_l2a()` (`src/gfetch/gedi.py`) now checks for missing
    requested columns before selecting: if the response is empty, it logs a
    `WARNING` (empty could mean genuine zero-match *or* a server-side failure - the
    caller can't tell which from this alone, so it's flagged either way) and adds
    the missing fields as empty columns instead of crashing; if the response is
    **not** empty and a field is still missing (a real caller typo), it now raises a
    clear `KeyError` naming the bad field(s), instead of pandas' own less legible one.
  - **Regression tests** (`tests/test_gedi.py`, fast/non-network, via
    `monkeypatch.setattr` on `gedi_module.gedi.gedi02ap`/`gedi_module.sliderule.init`):
    one reproducing the empty/failed-response case directly against
    `sliderule.emptyframe()` (no live SlideRule failure needed to exercise it), one
    confirming a genuine bad field name against a non-empty response still raises.
    Full fast suite (62 tests) and `pre-commit run --all-files` both pass.
  - **Separately, found while investigating**: a `benchmark/results/gedi_spatial_sweep.csv`
    was found sitting in the working tree mid-session with 5/6 default sweep rows
    filled in (up to `side_deg=1.0`) - turned out to be the user's own local run
    (the traceback above is from it), not something the assistant produced; flagged
    to the user rather than silently deleted or assumed, per the usual "investigate
    unfamiliar state before touching it" practice.

- **2026-09-22** — Closed the GEDI L2A field-selection gap flagged as an open question
  the same day (see `tech-stack.md`'s new "`anc_fields`: reading beyond `gedi02ap`'s
  fixed schema" section for the full writeup). Started from the user pasting a GitHub
  issue comment
  ([SlideRuleEarth/sliderule#539](https://github.com/SlideRuleEarth/sliderule/issues/539#issuecomment-3529180434))
  where the maintainer confirms `anc_fields` works for GEDI and points at
  `clients/python/tests/test_ancillary.py::TestGedi` - fetched via `gh api` (`gh issue
  view` itself failed with a "Projects (classic)" GraphQL error on this repo).
  - Live-verified every field from `download.yaml`'s `gedi_l2a.selected_bands` not
    already in `gedi02ap`'s fixed schema, one at a time then all together, against the
    same small Fontainebleau AOI the existing tests use: all confirmed working under
    their bare name except `landsat_treecover`/`modis_treecover` (need a
    `land_cover_data/` group prefix, discovered after the bare name silently returned
    0 rows rather than erroring) and `shot_number` (client-side bug: response comes
    back with exactly double the expected row count, raises `ValueError` on
    assignment - not fixable from gfetch's side).
  - `rh` returns the full 101-element relative-height percentile array per shot, not
    the named `rh0`/`rh2`/.../`rh100` bands geefetch exposed - added
    `gfetch.gedi.expand_rh()` as a separate pure function to slice it into those named
    columns, rather than folding the slicing into `fetch_gedi_l2a` itself.
  - **Implementation**: `fetch_gedi_l2a()` gained `anc_fields` (forwarded to
    `gedi02ap`'s `parms`, always kept regardless of the `fields` filter - required
    reworking the missing-column check to cover both together, since the old code
    would have silently dropped any requested `anc_fields` column not also listed in
    `fields`). CLI gained `--anc-fields`/`--rh-percentiles` (`gfetch/cli/gedi.py`,
    wired through `gfetch/cli/main.py`), with `rh` auto-added to `anc_fields` if
    `rh_percentiles` is given without it.
  - Tests: `tests/test_gedi.py` gained three fast/non-network tests (`anc_fields`
    forwarded into the request parms, kept in the result regardless of `fields`,
    `expand_rh`'s array-splitting) and one `@pytest.mark.slow` live test requesting
    all fourteen working `anc_fields` at once plus `expand_rh`. Full fast suite (70
    tests) and `pre-commit run --all-files` (ruff, pyrefly, pydoclint) pass; the new
    live test also run and passing against the real public cluster.
  - Updated `~/Documents/jz/src/configs/mozambania/v6/2020/download_gfetch/gedi.yaml`
    (a reference-only file in a separate configs repo, not consumed by `gfetch gedi`
    itself - see its own header comment) to use the newly-available `anc_fields`/
    `rh_percentiles`, closing all but one (`shot_number`) of the fields it previously
    flagged as unmappable from `download.yaml`'s `gedi_l2a.selected_bands`.

- **2026-09-22** — Made `gfetch gedi` load a config file instead of the sprawling
  CLI-flag set the previous entry's `--anc-fields`/`--rh-percentiles` additions left
  it with (surfaced when the user, having just hand-translated `gedi.yaml` into a
  16-line CLI invocation to answer "what command do I run", asked "Can we not use the
  config file anymore?"). Narrower than the design rejected earlier the same day
  (`gfetch gedi CONFIG`, not a raster `Config` field) - see `tech-stack.md`'s
  "Chosen architecture" section for the updated writeup.
  - New `gfetch.cli.gedi_config.GediConfig`/`load()` (`src/gfetch/cli/gedi_config.py`),
    reusing `cli/config.py`'s generic `AOIConfig`/`TimeRangeConfig` (`time_range`
    optional, unlike the raster `Config` - GEDI's own "no time range" meaning, the
    whole mission archive, needed to survive) but not the raster-specific `Config`
    itself, keeping the two pipelines' schemas independent as originally decided.
  - `cli/gedi.py`'s `gedi()` now takes a single `config_path: Path` (mirroring
    `search`/`download`/`mosaic`'s own CLI functions) instead of `output`/`bbox`/
    `start`/`end`/`fields`/`anc_fields`/`rh_percentiles`; `cli/main.py`'s `gedi`
    command shrank to `gfetch gedi CONFIG [--verbose]` to match.
  - Tests: new `tests/test_gedi_config.py` (`GediConfig` defaults/overrides,
    mirroring `test_cli_config.py`'s pattern) and `tests/test_cli_gedi.py` (CLI
    wiring via a monkeypatched `fetch_gedi_l2a`, covering the `rh_percentiles`->
    auto-added-`rh` behavior and the config's defaults). Full fast suite (71 tests)
    and `pre-commit run --all-files` pass. Also live-verified end to end via the real
    `gfetch gedi` command against a real config file and the public SlideRule
    cluster (small Fontainebleau AOI, `rh0`/`rh50`/`rh100`/`quality_flag` all came
    back correctly), not just the mocked CLI tests.
  - Updated `gedi.yaml` to drop its now-inaccurate "reference-only, not consumed by
    `gfetch gedi`" header comment - it's now the literal file to pass to the command.

- **2026-09-22** — Fixed `fetch_gedi_l2a()` failing to connect at all on Jean Zay's
  `archive` partition, found from the user's first real `gfetch gedi --config
  gedi.yaml` run there (`srun --pty ...`): every SlideRule request timed out
  ("Timed-out connecting... attempt 1/2/3 of 3"), even though the node has working
  internet via an IDRIS-mandated HTTP proxy (`https_proxy=http://prodprox.idris.fr:3128`).
  - **Root cause**: `sliderule.init()`'s `Session` hardcodes `requests.Session.trust_env
    = False` and exposes no way to override it - `trust_env=False` makes `requests`
    ignore `https_proxy`/`no_proxy` entirely and always attempt a direct connection,
    which Jean Zay's compute/archive nodes don't have. Traced via `sliderule/session.py`
    (`Session.__init__`'s `trust_env` param, defaulted and not forwarded by `init()`)
    after ruling out several other candidate causes live on the user's node: not node-
    to-node proxy variance (same `salloc` allocation throughout), not proxy env vars
    missing (confirmed present and correctly picked up by `requests.utils.
    get_environ_proxies`), not a `requests.Session`-vs-module-level-`requests.get`
    difference (an exact hand-built replica of `sliderule`'s own `session.get(url,
    data=..., headers=..., timeout=(10,120), verify=True)` call succeeded once
    `trust_env=True` was set), and not a custom transport adapter (`sliderule`'s
    `Session.__init__` mounts none - confirmed by reading the source, not assumed).
  - Also chased down and corrected a red herring along the way: `sliderule/__init__.py`
    does `from .sliderule import *`, so the top-level `sliderule.slideruleSession` a
    user (or a diagnostic script) accesses via a plain `import sliderule` is a stale
    snapshot frozen at package-import time (always `None`) - the live global `init()`
    actually updates lives at `sliderule.sliderule.slideruleSession`. Not gfetch's own
    bug (`gfetch/gedi.py` already does `from sliderule import gedi, sliderule`, which
    imports the *submodule* directly and was never affected), but cost real back-and-
    forth to pin down before realizing it was a dead end for the actual connectivity
    problem.
  - **Fix**: `fetch_gedi_l2a()` now calls `sliderule.create_session(trust_env=True)`
    (which does accept `trust_env` - it's a thin `Session(**kwargs)` wrapper) and
    assigns the result to the module's global `slideruleSession` directly, in place of
    `sliderule.init(verbose=False)`. Trades away `init()`'s client/server version-
    compatibility warning (a nicety, not essential) for working connectivity on a
    proxied network; harmless on an unproxied one since `trust_env=True` with no proxy
    env vars set behaves identically to a direct connection.
  - Live-verified twice: the user confirmed the underlying `create_session(trust_env=
    True)` + explicit-session pattern works end-to-end on Jean Zay's `archive`
    partition (`check_version()`/`source()`/a raw `.session.get()` call all succeeded
    through the proxy); the assistant separately re-ran gfetch's full non-network suite
    (71 tests) plus all `@pytest.mark.slow` live GEDI tests against the real public
    cluster from an unproxied network, confirming no regression there.
  - Tests: `tests/test_gedi.py`'s existing `monkeypatch.setattr(gedi_module.sliderule,
    "init", ...)` calls updated to patch `"create_session"` instead, matching the code
    change; no new test added (there's nothing to unit-test about `trust_env` itself -
    it's a passthrough to a third-party library's own connection behavior, and the
    real coverage here is the live HPC verification plus the existing slow tests
    continuing to pass against the real service).
  - **This fix turned out to be wrong / incomplete**: the user's next real
    `gfetch gedi --config ...` run on the `archive` node (a genuinely fresh process,
    not `srun --pty python -c`) still timed out. Spent a long back-and-forth chasing
    why, including two real dead ends worth recording so they're not re-chased:
    (1) `sliderule/__init__.py` does `from .sliderule import *`, so a **diagnostic
    script** doing plain `import sliderule; sliderule.slideruleSession = ...`
    touches a stale top-level copy, not the live submodule global - real, but not
    gfetch's bug (`gfetch/gedi.py` already imports the submodule directly via `from
    sliderule import gedi, sliderule`); (2) an IPython session with connection
    counts climbing across supposedly-separate `srun --pty python -c` invocations
    revealed the user's diagnostics had been running in one long-lived, likely
    duplicate-module-riddled kernel the whole time - also real, also not gfetch's
    bug. Neither explained the CLI itself failing in a genuinely fresh process.
  - **Actual fix**: patch `sliderule.session.Session.__init__`'s own default instead
    of trying to win a race to set the right global before `gedi02ap()`'s internal
    `checksession()`-triggered lazy re-init runs. `gfetch/gedi.py` now wraps
    `Session.__init__` at import time (`kwargs.setdefault("trust_env", True)`) so
    *every* `Session` constructed anywhere in the process - including SlideRule's
    own internal fallback, whatever exact path that takes - defaults to
    `trust_env=True` unless a caller explicitly asks for `False`. `fetch_gedi_l2a()`
    reverted to plain `sliderule.init(verbose=False)`, now safe since construction
    itself can no longer produce a `trust_env=False` session by default. Why the
    external global-assignment approach specifically failed to reach `gedi02ap()`'s
    session even in a clean process was never conclusively pinned down - the patched
    default sidesteps needing to know, by fixing the one thing every code path
    shares (the constructor).
  - Tests: added `test_session_init_defaults_trust_env_to_true` (constructing a
    bare, unrelated `Session()` picks up `trust_env=True`; an explicit
    `trust_env=False` still wins) plus the existing `monkeypatch` fast tests
    reverted to patching `sliderule.init` again. Full fast suite (72 tests) and all
    three `@pytest.mark.slow` live GEDI tests (including a real `gedi02ap()` call)
    re-run and passing from an unproxied network - **not yet re-confirmed on Jean
    Zay** as of this entry; that's the next thing to verify.

- **2026-09-22** — Real bug found via a live `gfetch mosaic` run on Jean Zay (real
  1491-item, 3-zone, 12-band mozambania Sentinel-2 config, `--task-id 0 --n-tasks 6`):
  a `ValueError: Specified Zarr chunks encoding['chunks']=(632, 632) for variable
  named 'rededge2' would overlap multiple Dask chunks` deep inside `to_zarr`, on the
  very first patch.
  - **Investigation**: reproduced the exact same config/real search results/patch
    (zone 32735, patch 0) locally, both at per-patch scale and at full-zone scale
    (matching `prepare_template`'s own code path) against gfetch's own pinned
    dependency versions - every band, including `rededge2`, came out cleanly and
    consistently chunked at 2048/2048/1747. Could not reproduce the mismatch at all
    with gfetch's own dependency set on the identical real data.
  - **Root cause, best-supported explanation**: `mosaic()`'s `chunks=` parameter only
    pinned the chunk grid at `load()` time - nothing guaranteed `composite()`'s
    reduction (or any per-band resampling needed to reach the common output grid,
    e.g. for `rededge2`, a 20m-native band upscaled to the 10m target) preserved that
    grid identically for every variable across separate calls. `--n-tasks 6` means up
    to 6 SLURM array tasks each independently build the *same* zone's full dataset
    and race (via `prepare_template`'s atomic `mode="w-"`) to initialize its store -
    the code's own correctness relies on every caller's build being byte-for-byte
    identical, which wasn't actually guaranteed. Whichever task won the race baked
    its own build's chunk layout permanently into the store; a different task (or a
    later patch) building a very slightly different version of "the same" dataset no
    longer matched it. A separate, real possibility for the same symptom (a store
    left over from an earlier run under different settings, which `prepare_template`
    silently treats as already-done) was also identified but not confirmed or ruled
    out - both are "data layout" bugs, not a library-version issue (the user pushed
    back correctly on an earlier, weaker "different Jean Zay Python env" framing).
  - **Fix 1** (`src/gfetch/mosaic.py::mosaic`): explicitly `.chunk()` the composited
    output to the resolved `x`/`y` chunk sizes right before returning, removing the
    possibility of any two calls disagreeing, by construction, regardless of what
    `load()`/`composite()`/resampling internally produced. Regression test
    (`test_mosaic_pins_output_chunks_to_requested_grid`) fakes a `load()` return with
    deliberately misaligned chunks and asserts `mosaic()` corrects them - confirmed
    to fail without the fix, pass with it.
  - **Fix 2, at the user's explicit request after pushing back on "just rechunk to
    accommodate" as the complete fix**: added `gfetch.write.validate_chunks()` -
    reads a store's actual on-disk chunk grid straight from its Zarr array metadata
    (same mechanism `region_is_written()` already uses) and raises a clear
    `ValueError` naming the store path, variable, and expected-vs-actual chunk size
    if it disagrees with the current run's config. Wired into `cli/mosaic.py` right
    after the existing `store_initialized()` check, before any patch is built - so an
    incompatible pre-existing store (from an earlier run, different config, or older
    gfetch version) fails immediately and clearly instead of only surfacing deep
    inside `to_zarr` after a wasted patch computation. Two new tests in
    `tests/test_write.py` (passes when matching, raises with a clear message on
    mismatch).
  - Full fast suite (75 tests) and `pre-commit run --all-files` both pass.
    **Not yet re-confirmed against a real Jean Zay run** - the original failure was
    never reproduced locally, so fix 1's effectiveness against the *actual* cause on
    Jean Zay remains unverified; fix 2 is a general hardening independent of that.

- **2026-09-22, same day** — **Fix 1 above (mosaic()'s silent `.chunk()` correction)
  reverted**, on the user's pushback: "the changes you introduced could provoke OOM
  (newly)". Correct: `.chunk()` is cheap (near no-op) when chunks already match, but
  on the exact mismatch it exists to handle, it's a real dask `rechunk` - gathering
  multiple source chunks in memory to build each target chunk. Fix 1 was silently
  trading a cheap, immediate, safe `ValueError` for a potentially memory-hungry
  rechunk on an already resource-constrained HPC allocation (`--cpus-per-task=5`) -
  exactly backwards given the "fail fast, don't silently accommodate" principle the
  user had already pushed for earlier in this same investigation (see the
  `validate_chunks` entry above). `mosaic()` reverted to its pre-fix-1 form (plain
  `return composite(ds, method=method)`); its regression test
  (`test_mosaic_pins_output_chunks_to_requested_grid`) removed since it tested
  behavior that's now deliberately gone. **Fix 2 (`validate_chunks`) stands
  unchanged** - it's a pure metadata read, no rechunk, no data touched, and remains
  the actual defense against the stale/mismatched-store failure class. A genuine
  per-patch chunk divergence not caught by `validate_chunks` (e.g. real non-
  determinism in a single build, if that theory is even correct - still unconfirmed)
  will now fail the same way it did before any of this investigation: loud, cheap,
  at `to_zarr`, no wasted rechunk. Full fast suite (74 tests, one fewer - the removed
  test) and `pre-commit run --all-files` both pass.

- **2026-09-22** — Closed out the Jean Zay GEDI connectivity investigation (see the
  two entries above on the `anc_fields` field-selection work and the
  `Session.__init__` `trust_env` patch): **`gfetch`'s fix is confirmed correct and
  needs no further changes.** Root-caused the user's last remaining failure down to
  hard proof it's IDRIS infrastructure, not gfetch: instrumenting the real
  `fetch_gedi_l2a()` call (not a hand-rolled reproduction) to print the raw
  `requests` exception instead of `sliderule`'s bucketed "Connection error" message
  surfaced `ProxyError(... OSError('Tunnel connection failed: 403 Forbidden'))`, and
  `session used: trust_env= True` confirmed the patched default was correctly active
  - the request was reaching and going through the proxy exactly as intended.
  Re-running the plain `curl -v` connectivity check from earlier (same command,
  same node/allocation) reproduced the identical `403 ERR_ACCESS_DENIED` straight
  from squid - **against `debpro144`, a different squid backend than the earlier
  successful runs' `130.84.11.17`/`debpro17`** (`prodprox.idris.fr` load-balances
  across several squid instances). A follow-up test then narrowed this further and
  contradicted the first read of it: `curl` against the `archive` partition's
  compute-node egress hit `130.84.11.17`/`debpro17` again (the *same* backend that
  gave `200` on the very first test hours earlier) and got `403` this time - so it
  isn't simply "some backends allow the domain, others don't" (a static per-backend
  ACL difference), it's the *same* backend flipping from allow to deny over the
  course of the debugging session, on a compute-node egress path. Separately, the
  **front-end/login node** (`jean-zay3`, no `srun`) routes through an entirely
  different proxy subnet (`130.84.14.27`, vs. the `archive` partition's `130.84.11.x`
  pool) and succeeded cleanly (`HTTP/2 200`) at the same moment the compute-node path
  was failing - login-node and compute-node egress are genuinely separate proxy
  pools at IDRIS, not just different members of one pool. Told the user: nothing
  left to fix in gfetch (the `trust_env` patch is proven correct and gets requests
  all the way to the proxy); practical workaround is running `gfetch gedi` from the
  login node directly, matching the existing "safe to run on an HPC login/data-
  transfer node" guidance already given for `search`/`download`; the compute-node
  proxy's time-varying 403 (same backend, same domain, allow -> deny within one
  session) is worth reporting to IDRIS/GENCI support as its own finding, distinct
  from - and more actionable than - the "different backends, different ACLs" theory
  first floated.
  - **Retrospective note for future sessions**: this whole thread (the `trust_env`
    global-vs-patched-default saga, several dead-end diagnoses along the way, and
    finally this proxy-ACL finding) cost a lot of back-and-forth - the user called out
    partway through that some intermediate diagnostics felt "desperate." The lesson
    that actually shortened the loop each time: get the **raw, unbucketed exception**
    (`repr(e)` on the real `requests`/`urllib3` error) instead of reasoning from
    `sliderule`'s own generic retry-log message ("Connection error", "Timed-out
    connecting") - the raw `ProxyError`/`403 Forbidden` was the single piece of
    evidence that actually resolved things, and should have been reached for much
    earlier than it was.

- **2026-09-22** — **Root cause of the `validate_grid_chunks_alignment` saga (three
  entries above) finally confirmed, not just theorized** - the user deleted the
  Jean Zay Zarr stores and reran the exact same job array, and hit the *same* error
  on a *fresh* store, on a *different* zone (32736) and *different* band (`red`,
  a native-10m band with no resampling need) than the first report. That single
  data point definitively ruled out both prior theories (stale store from an older
  run; a different Python environment on Jean Zay) - a freshly-created store
  reproducing it, locally, means it's neither.
  - **Reproduced deterministically, locally**, by finally testing zone 32736 (979
    items - 5x zone 32735's count, never actually tested before this) instead of
    only 32735 (206 items, always clean): every one of the 12 configured bands came
    out chunked at 632x632 instead of 2048x2048, using gfetch's own pinned
    dependencies, no version mismatch involved at all.
  - **Isolated to `load()` vs. `composite()`** by inspecting `load()`'s raw,
    pre-composite output for zone 32736: `time` came out as **three** chunks -
    `(39, 39, 6)` for 84 distinct dates - not one. x/y were still correctly 2048 at
    that point. Traced straight to `download_gfetch/s2.yaml`'s `chunks: {..., time:
    39}` - an explicit config value that overrides `resolve_chunks()`'s safe
    `time: -1` default (`dict.setdefault` only fills in a default for an *absent*
    key), almost certainly a leftover from before `_DEFAULT_TIME_CHUNK` existed.
    With `time` split into 3 chunks, `composite()`'s `.median(dim="time")` has to
    rechunk internally to consolidate them before reducing - the exact class of bug
    `_DEFAULT_TIME_CHUNK=-1` was originally added to prevent, just reachable again
    via a config value nothing warned about. For zone 32736 specifically (by far
    the widest/largest of the three), that internal rechunk also shrinks the
    spatial `x`/`y` chunks as a side effect down to 632; zone 32735 hits the same
    multi-chunk `time` axis but apparently stays under whatever threshold triggers
    the spatial shrink, which is exactly why it looked clean in every earlier test.
  - **Verified the fix directly** before writing any code: same real zone/items,
    `chunks["time"]` forced to `-1` instead of the config's `39` - every band comes
    out cleanly at 2048/2048.
  - **Two-part fix, `src/gfetch/mosaic.py::mosaic`**:
    1. `mosaic()` now overrides a non-default `chunks["time"]` unconditionally
       (logging a `WARNING` naming the bad value and why) before calling `load()` -
       `mosaic()` always reduces over the whole time axis, so a smaller `time`
       chunk is *never* beneficial there and only ever a footgun; forcing it closes
       the actual, confirmed root cause at the source.
    2. **Fix 1 from three entries above (the output `x`/`y` chunk pin,
       `result.chunk({"y":..., "x":...})`) reinstated** - the user reversed their
       own revert: "there are some mechanism by which zarr can rechunk... I go back
       on my earlier decision and I would say your earlier fix to ensure chunk
       alignment is warranted." With the actual trigger now understood and fixed at
       the source, this pin becomes a rare belt-and-suspenders backstop rather than
       the routine path silently absorbing an expensive rechunk - the OOM objection
       that motivated the revert doesn't really apply once it's not the thing
       catching the common case anymore.
  - **Verified end-to-end**, real zone 32736, the config's **unmodified**
    `chunks.time: 39` (not manually overridden) - `mosaic()` now transparently
    produces clean 2048/2048 chunking for every band with no caller changes needed.
  - Two new tests in `tests/test_mosaic.py`:
    `test_mosaic_overrides_non_default_time_chunk_and_warns` (asserts `load()` is
    called with `time=-1` regardless of what was requested, and that a warning is
    logged) and `test_mosaic_pins_output_chunks_to_requested_grid` (the fix-1
    regression test, re-added). Full fast suite (76 tests) and
    `pre-commit run --all-files` both pass.
  - **Also surfaced along the way, not yet acted on**: `load()`'s own
    `log_chunk_footprint` estimated **96.89 GB** peak memory for zone 32736's
    full-time-axis, 2048-chunk, 12-band composite at 11 concurrent chunks (this
    machine's core count) - not a bug, just the inherent cost of a `median` needing
    the whole time axis per spatial chunk in memory, but a very real number given
    the user had already independently reduced `n_compute_workers` from 5 to 3 in
    the config between sessions, suggesting they were already fighting memory
    pressure on Jean Zay before this specific bug was even found. Worth keeping in
    mind if memory problems continue after this fix - they may be a separate,
    legitimate capacity question, not another correctness bug.
  - **Retrospective note**: the user pointed out, correctly, at multiple points in
    this saga that "it all doesn't seem very robust" and that early theories
    (dependency version, stale store, non-deterministic concurrent build) were
    each accepted too readily without being pinned down against the *actual*
    failing zone. The thing that actually cracked it was finally reproducing
    against zone 32736 specifically instead of continuing to reason from zone
    32735's clean result - a reminder to reproduce against the *exact* failing
    input before trusting a theory that only explains a *similar* one.

- **2026-09-22** — Unified `s1.yaml`/`s2.yaml`/`gedi.yaml` into one per-job config
  file plus a `gfetch <satellite> <verb>` CLI, at the user's explicit request after
  the GEDI HPC investigation above surfaced the friction directly: "The
  configuration is either generic or specific to a certain satellite. I would like,
  as in geefetch, satellite specific configuration and generic configuration, all
  in the same config file." Full design writeup in `tech-stack.md`'s new "Unified
  per-job config" section - this entry covers the process, not the design itself.
  - **Planned via `EnterPlanMode` before writing code**, per the usual practice for
    a change this size - two open questions resolved with the user first
    (`AskUserQuestion`) rather than guessed: `gfetch gedi` stays bare (no verb,
    confirmed over adding one for symmetry), and `custom:` is a dict of *several*
    named entries (the user's own correction to my first proposal of a single
    `custom:` block - "I would put it all under a 'custom' section in the config,
    which actually may hold config for several customs", directly mirroring
    geefetch's `customs_vector:` pattern I hadn't originally matched).
  - **Two real issues found only by actually running the new CLI, not by the type
    checker or the test suite**: (1) `GediConfig`'s strict structured-merge schema
    raised `ConfigKeyError` on a raster-only generic field (`n_workers`) the very
    first live end-to-end test (`gfetch gedi` against a config with an `s2:`
    section too) - neither loader's unit tests had exercised a *shared* file with
    fields belonging to the *other* pipeline, since every test config up to that
    point was single-purpose. Fixed by filtering each loader's "generic" dict to
    only the keys its own target dataclass declares, applied symmetrically to both
    `cli/config.py::load()` and `cli/gedi_config.py::load()`, with regression tests
    added for both directions this time. (2) The plan's own proposed CLI shape for
    custom satellites (`gfetch custom <name> <verb> <config>`) doesn't parse under
    cyclopts - a sub-app routes its *next* token to one of its own registered
    commands, so the verb has to come immediately after `custom`
    (`gfetch custom <verb> <name> <config>`). Caught by a minimal cyclopts repro
    before committing it to the real CLI, not discovered via a failing test after
    the fact.
  - **Migration**: replaced the mozambania job's 3 files with one
    `~/Documents/jz/src/configs/mozambania/v6/2020/download_gfetch/config.yaml` -
    verified it resolves to identical `Config`/`GediConfig` values as the 3
    originals (bands, output paths, chunks, `n_compute_workers`, etc.) via a real
    load against the actual file, not just against synthetic test fixtures, before
    deleting the old files.
  - **Live end-to-end verification**, not just the test suite: `gfetch s2 search`
    -> `download` -> `mosaic` and `gfetch gedi`, both through the new unified-config
    CLI path, against a small real AOI; `gfetch custom search <name> <config>`
    against a deliberately-unregistered satellite name with an explicit
    `collection:` override, confirming it genuinely bypasses `gfetch.sources`'s
    registry lookup rather than accidentally still routing through it. (Also hit,
    and worked around, an unrelated pre-existing `validate_chunks` false positive
    on a tiny test AOI whose native array is smaller than the default 2048px
    chunk size - not this session's bug, from the concurrent mosaic-chunk-
    validation work logged elsewhere in this file; sidestepped with an explicit
    small `chunks:` override for the live test rather than investigated further,
    since it's out of scope here.)
  - Tests: new `tests/test_cli_main.py` (CLI dispatch/routing); substantial
    additions to `tests/test_cli_config.py` (section merging, force-set
    `satellite`, `output_dir` defaulting, `custom` required-field errors, the
    cross-pipeline field-filtering regression) and `tests/test_gedi_config.py`
    (same shape, GEDI side); one new test in `tests/test_search.py` for the
    explicit `collection` passthrough. Full fast suite (107 tests) and
    `pre-commit run --all-files` (ruff, pyrefly, pydoclint) pass.

- **2026-09-23** — Added `countries:` as an alternative to `aoi:` in the unified job
  config, at the user's request to mirror geefetch's own `aoi.country` field
  (`geefetch/cli/download_implementation.py::load_country_filter_polygon`).
  - **Researched geefetch's actual behavior before copying it**, via a forked
    subagent trace rather than assuming: `filter_polygon` is a coarse, tile-level
    `shapely.intersects()` check in `Tiler.split` - whole tiles are kept or skipped,
    never per-pixel clipped, and it never reaches GEE (`.filterBounds()`/`.clip()`
    use each tile's own bbox, not the polygon) or the GEDI vector path at all. This
    directly shaped the scope decision: gfetch resolves `countries` to a bounding
    box everywhere (matching geefetch's own real behavior, not an idealized "clips
    to the exact country shape" reading of it), plus one deliberate improvement
    beyond geefetch - `search()` passes the exact polygon to STAC's `intersects=`
    for genuine server-side filtering, which geefetch's GEE-based search has no
    equivalent of.
  - **New `gfetch/countries.py`**: `resolve_country_polygon(countries)` against the
    same public `world-administrative-boundaries` GeoJSON geefetch uses, with no new
    dependencies (`requests`+a local `XDG_CACHE_HOME` cache instead of `pooch`,
    stdlib `difflib` instead of `thefuzz` - all three of `requests`/`geopandas`/
    `shapely` were already transitively available). The typo-suggestion needed a
    substring-match pass before falling back to `difflib.get_close_matches`: a
    naive `difflib`-only attempt failed the realistic case (`'Tanzania'` found no
    match against `'United Republic of Tanzania'` - too large a length gap for
    ratio-based matching alone), confirmed both broken and then fixed live rather
    than assumed.
  - **Config**: `Config`/`GediConfig` both gained `aoi: AOIConfig | None = None` +
    `countries: list[str] | None = None` (reordering each dataclass's fields, since
    a formerly-required `aoi` can no longer precede other required fields);
    `gfetch.cli.config.resolve_aoi()` centralizes the "exactly one of `aoi`/
    `countries`" validation and country-to-bbox resolution, called from both
    loaders. This reopened the type-safety gap `resolve_bands`/`resolve_cloud_mask`
    had already established a pattern for: code written against `cfg.aoi.bbox`
    assuming non-None broke under pyrefly now that the field is legitimately
    `AOIConfig | None` at the schema level. Fixed the same way - a `resolved_aoi`
    property (assert-non-None) on both dataclasses, `load()` guarantees it's
    populated, every consumer (`cli/search.py`, `cli/gedi.py`, `cli/mosaic.py`,
    `cli/finalize.py`) switched to it instead of the raw optional field.
  - **`gfetch/search.py`**: gained `intersects: dict | None`, dropping `bbox` from
    the actual STAC request when given (the API spec treats them as mutually
    exclusive) while still keeping `bbox` for logging.
  - Tests: new `tests/test_countries.py` (union/typo-hint/substring-match behavior,
    fast + one live `@pytest.mark.slow` sanity check against the real dataset),
    `countries`/both-or-neither coverage added to `tests/test_cli_config.py` and
    `tests/test_gedi_config.py`, `intersects`-drops-`bbox` coverage added to
    `tests/test_search.py`. Full fast suite (127 tests) and
    `pre-commit run --all-files` pass. Live-verified end to end against the real
    mozambania AOI (`gfetch s2 search` with `countries: [Mozambique, United
    Republic of Tanzania]` in place of `aoi:`, 648 items found via a real
    `intersects=`-filtered STAC query) - not just the test suite.

- **2026-09-24** — GEDI tiling: `GEDI_MAX_TILE_SIZE_M` raised from 10 km to 50 km,
  and `fetch_gedi_l2a` gained `polygon`, so `gfetch gedi` with `countries:` only
  requests the tiles of the bbox grid intersecting the countries' union. Estimated
  by hand on the mozambania config (bbox ~11.5° x 25.9°): ~37k tiles before, ~1.5k
  bbox tiles at 50 km, of which roughly half intersect Mozambique/Tanzania. The
  filter is tile-level like geefetch's `filter_polygon`: footprints outside the
  countries but inside a kept tile are still returned. `polygon` isn't recorded in
  the resume `params.json`, since it only selects tiles and never changes a tile's
  own content; `max_size_m` is, so a tile dir left by a 10 km run is refused.

- **2026-09-24** — GEDI quality filter and flat `rh{p}` columns, at the user's
  request (the tech-stack "no filtering" POC scope is superseded).
  - Ported geefetch's `l2AQualityFilter` (`geefetch/data/satellites/gedi.py`) minus
    its full-power-beam condition, which is commented out there too:
    `quality_flag == 1`, `degrade_flag == 0` via SlideRule's `l2_quality_filter`/
    `degrade_filter` (less data transferred); `solar_elevation <= 0`,
    `sensitivity >= 0.9`, `0 <= rh98 <= 80` locally on each tile's response (no
    SlideRule equivalent; `rh` is requested automatically for `rh98`). On by
    default (`quality_filter`), recorded in the resume `params.json`.
  - **Live-verified the server-side params**: Fontainebleau AOI, 2020 full year -
    63,495 raw shots, 7,794 passing all five conditions counted locally on the
    unfiltered response, and exactly 7,794 returned by the filtered request, all
    `quality_flag == 1`/`degrade_flag == 0`. Pitfall hit on the way: the same AOI
    over 2020-06-01..15 returns 0 filtered shots - genuinely, not a failure: none of
    its 2,727 shots is at night (`solar_elevation <= 0`), which in mid-latitude
    summer can empty a short window. The live test now uses the full year.
  - `rh_percentiles` moved from the CLI (`expand_rh` on the final frame) into
    `fetch_gedi_l2a` itself, applied per tile before saving, so the tile files no
    longer carry the 101-element array either. The raw array is only returned when
    `'rh'` is in `anc_fields` and `rh_percentiles` isn't set.

- **2026-09-24** — GEDI L4A, at the user's request, filtered like geefetch's
  `l4AQualityFilter` (`geefetch/data/satellites/gedi.py`).
  - **CLI/config shape (user's choice among three)**: `gfetch gedi l2a|l4a CONFIG`,
    each reading its own reserved section `gedi_l2a:`/`gedi_l4a:` (default output
    `<output_dir>/gedi/<product>.parquet`). Alternatives rejected: a `products:`
    list in one `gedi:` section (one run, but `output` stops being a single file),
    or a scalar `product:` (two config files for both). The bare `gedi:` section is
    gone; `gedi_config.load` raises on one rather than silently ignoring it.
    `GediL2AConfig(GediConfig)` adds `rh_percentiles`, so a top-level
    `rh_percentiles` is ignored for L4A like any other foreign generic field, and
    one inside `gedi_l4a:` raises.
  - Library: `fetch_gedi_l4a` shares the tiling/polygon/resume loop
    (`_fetch_tiles`), request+edge dedup (`_request_tile`) and column selection
    (`_select_columns`) with `fetch_gedi_l2a`; only the per-tile filtering differs.
    Default fields `agbd`/`elevation`.
  - Filter: `l4_quality_flag == 1`, `degrade_flag == 0` server-side
    (`l4_quality_filter`/`degrade_filter`, documented in SlideRule's
    `docs/user_guide/gedi.md`), `sensitivity >= 0.9` locally (`sensitivity` is in
    `gedi04ap`'s fixed schema, no `anc_fields` needed).
  - **Live-verified**: Fontainebleau AOI, 2020 full year - 63,496 raw L4A shots,
    6,622 passing all three conditions counted locally on the unfiltered response,
    exactly 6,622 returned by the server-filtered request (all of which already
    have `sensitivity >= 0.9`), and 6,622 written by `gfetch gedi l4a`. `agbd_se`
    works as an `anc_fields` name. `gedi04ap`'s fixed schema is `sensitivity`,
    `beam`, `agbd`, `elevation`, `track`, `solar_elevation`, `flags`, `orbit`.

- **2026-09-24** — `orbit_state: as_bands`, at the user's request: both orbit
  directions searched, each composited separately into `{band}_{orbit_state}`
  variables (`vv_ascending`, `vv_descending`, `vh_ascending`, `vh_descending`).
  - Split in `gfetch.mosaic.mosaic(split_orbit_states=True)`: one `load` +
    `composite` per orbit state, then merged. A single load grouped by orbit
    state was ruled out: `groupby="solar_day"` can merge a morning descending and
    an evening ascending pass over the same area into one time step.
  - An orbit state with no items in a zone gets all-NaN variables, so every zone's
    store has the same variables. An item without `sat:orbit_state` raises.
  - `resolve_output_variables(cfg)` gives the store's variable names; `mosaic`
    (template check, shard listing, resume) and `pack`/`clean` use it, while
    `download` keeps `resolve_bands` (asset keys).
  - Live-verified: Paris AOI, 2026-06-01..15 (4 ascending, 6 descending items),
    `gfetch s1 search` + `gfetch s1 mosaic` on remote hrefs: store holds the four
    variables, all finite; a rerun skips every shard.
  - Single-direction coverage (e.g. Australia, descending only) is handled, not an
    error: live-checked over Alice Springs, June 2026 (5 descending items, 0
    ascending) - `vv_ascending` all NaN, `vv_descending` fully populated.
  - **Found on the way: `mosaic` never masked `nodata`.** Sentinel-1's `vv`/`vh`
    assets carry `nodata: 0` (odc-stac puts it in each variable's `nodata`
    attribute) and nothing converted it to NaN, so the median counted it as a
    value: partial temporal coverage was biased towards 0 and uncovered pixels came
    out 0.0 rather than NaN (checked: a geobox in Kansas loaded with Paris items
    gave finite 0s). This also affected every non-split S1 mosaic. Sentinel-2
    escaped via SCL class 0 (no data) in its mask. Fixed by `mask_nodata`, applied
    before `mask_clouds` (skipping the mask band, which must stay integer for
    `isin`). Side effect: S1 composites are `float32` (xarray's promotion of
    `uint16` under `where`), not `float64`.

- **2026-09-26** — **`mosaic` OOM building a zone's Zarr template.** Mozambique +
  Tanzania, S2 2020, 64 px chunks, `shard_factor: 128`, `compute_chunk_factor: 1`,
  30-task array on Jean Zay CPU nodes (~40 GB): every task was OOM-killed about 70 min
  into building the template for EPSG:32736 (66786 x 286412 px, 921 items), before any
  shard was assigned.
  - Cause: the template is a full `mosaic()` over the whole zone geobox at the
    compute chunking, only to read its metadata. 4.7M spatial chunks x 13 bands
    ≈ 61M odc-stac load tasks, built eagerly, plus the mask/median layers.
    `prepare_template` then rechunked to the store chunks before rechunking to the
    shards, which rebuilt the same number of tasks: dask builds rechunk graphs
    eagerly (measured: 1M output chunks ≈ 60 s, 1.2 GB). EPSG:32735 (about 6.6x
    smaller) survived, taking 7 min.
  - Fix: `gfetch mosaic` builds the template at shard-sized dask chunks, and
    `prepare_template` (sharded case) writes the inner chunks straight into the
    encoding instead of going through `ds.chunk`. Measured on a synthetic
    zone-32736-sized lazy dataset: 0.9 s, 8 MB peak.
  - Rejected: building the template from the geobox alone, without odc-stac. It
    would restate `mosaic()`'s output variables, dtype (uint16 promoted to float32
    by `.where`), attrs and CRS coord, and could drift from it silently.
  - Still open: (1) ~~each shard is loaded with every item in its zone~~, fixed the
    same day, see the next entry. (2) All array tasks still build the template at the same time on a
    fresh store; that's cheap now, but the work is duplicated.
  - Defaults changed on the way: store chunks 256 → 64 px, `shard_factor` 32 → 128
    (8192 px shards either way), `compute_chunk_factor` 1 → 16 (1024 px dask chunks).
    At a factor of 1, 64 px chunks gave 16k dask chunks per shard and band, mostly
    scheduling overhead. Like the other defaults, a user override isn't adapted: the
    user is responsible for keeping chunks, shards and bricks consistent.

- **2026-09-26** — **`mosaic` drops the items that don't intersect the geobox.**
  - Confirmed in odc-stac 0.5.3 (`_stac_load.py`): the time axis is built from every
    item's `groupby` group, with no geobox filtering. Items whose footprint misses a
    chunk get an empty `tyx_bins` entry, so no read happens, but the time step is
    still filled with nodata, held in memory (`time: -1`, so the whole `T x 1024² px`
    brick per band), masked and reduced. Synthetic check: 10 dates over one area plus
    10 over a disjoint one, loaded onto a geobox in the first, gave `time=20`.
  - Fix: `mosaic()` keeps only items whose `geometry` intersects the geobox padded by
    2 px, in EPSG:4326 (`_items_intersecting`). Applied in `mosaic()` itself, so the
    zone template keeps every item and each shard filters its own. Items without a
    `geometry` are kept. If none intersect, the first item alone is kept: odc-stac
    can't load an empty list, and one non-overlapping item gives a single all-nodata
    step, so an all-NaN shard as before.
  - Equivalence tested on local GeoTIFFs with bilinear resampling, filtered vs.
    unfiltered: identical composites. It stayed identical even without the pad, with
    the geobox's edge 50 m short of an adjacent item: odc warps each item separately,
    so an item outside the geobox never feeds its pixels. The pad only guards against
    `geometry` approximating the footprint (edges straight in lon/lat, curved in UTM).
  - Not measured on the real job yet. Expected gain: per shard, memory and median cost
    scale with the shard's own time steps instead of the zone's.

- **2026-09-27** — **`mosaic` skips the shards outside the AOI polygon.** Mozambique +
  Tanzania, S2 Feb–Mar 2024, `max_cloud_cover: 20`, 2048 px shards: the log was
  mostly "0/29 item(s) intersect the geobox".
  - Not a filtering bug. Zone grids are the AOI *bbox* clipped to each zone, while the
    search uses the country polygons, so much of each grid (Malawi, Zambia, Zimbabwe,
    the Indian Ocean) has no item. 4116/9100 shards had no item at all. The "/29" is
    EPSG:32735, a 0.66°-wide strip at lon 29.34–30 that only 45/700 shards of which
    touch the countries. Cross-checked against item bboxes (a superset of their
    `geometry`): of the ~800 empty shards inside the countries, only 39 were hit by a
    bbox, all from partial edge-of-swath items.
  - A separate, data-side effect: ~330k km² of the countries (mostly lat −7 to −15)
    have no item at all, because `max_cloud_cover: 20` removes every rainy-season scene
    there. Left to the user, it's a config choice.
  - Fix: `Config.aoi_geometry` (and `GediConfig.aoi_geometry`) hold the exact AOI
    shape: the countries' union, or the `aoi` box. Anything selecting data uses it
    (STAC `intersects`, GEDI tile filtering, shard planning); output grids keep using
    the bbox, since a Zarr array is a rectangle anyway. `gfetch mosaic` passes
    `outside_aoi(geobox, aoi_geometry)` to `prepare_template`, which records the
    matching shards under the root group attribute `gfetch:skipped_shards` (dimension
    names + shard indices) before the template's atomic rename. `write_regions` leaves
    them out and `region_is_written`/`store_is_complete` treat them as written, so
    resume and completeness need no config. Skipped shards read back as the fill
    value (NaN).
  - On the job above: 4515/9100 shards kept (45/700, 2750/4620, 1720/3780), about
    0.2 s of planning per zone. Existing stores have no attribute, so nothing is
    skipped for them: rebuild a job's stores to benefit.
  - Rejected: a sentinel file per skipped shard (more files, and the skip decision
    would be made per task at run time instead of once per store); recomputing the
    skip set from the config in `finalize` (a second place the decision could drift).
  - Dropped: cropping each zone's grid to bbox(polygon ∩ zone band). Once shards are
    skipped it only shrinks the array extent.
  - Still open: (1) merge a thin end-zone strip like EPSG:32735 into its neighbour
    zone (at lon 29.34, zone 36's scale factor is ~1.0016 vs ~1.0010 at a normal zone
    edge) — agreed, to implement later; it changes the stores' layouts. (2) Remove
    the pack/zip functionality (`pack_store`, `ZipStore` handling in `gfetch mosaic`
    and `finalize`), made redundant by sharding — agreed, to do later.

- **2026-09-27** — **Thin end-zone strips merge into their neighbour UTM zone.**
  - `_zone_extents` splits the AOI bbox between the zones it spans; an end zone
    whose strip is narrower than `_MIN_ZONE_WIDTH_DEG` (1°) goes to its neighbour,
    whose grid then reaches past its own band. The narrower end merges first, until
    both ends are ≥ 1° or one zone is left; a lone narrow zone is kept. Zones merge
    by number, so both hemispheres of an equator-crossing AOI follow. Decided on the
    bbox, not on items, so every stage (`group_by_utm_zone`, `zone_geobox`,
    `finalize`'s store listing) agrees without reading the items.
  - Distortion: at the equator, 1° past a zone's edge is 4° from its central
    meridian, scale error ~0.20% vs ~0.10% at a normal edge. On the Mozambique +
    Tanzania job, EPSG:32735's 0.66° strip (lon 29.34–30) goes to EPSG:32736.
  - A store's grid is fixed once written, so a changed zone split (or AOI,
    resolution) made a resumed run write shards against a stale grid without
    error: `validate_chunks` only compares chunk/shard sizes. `gfetch mosaic` now
    also runs `validate_geobox`, which fails fast if the store's `x`/`y` coordinates
    differ from the zone geobox. On the job above it rejects the old EPSG:32736
    store and accepts EPSG:32737's.

- **2026-09-27** — **Pack/zip removed; `gfetch <satellite> vrt` added.**
  - Sharding made the post-hoc zip store redundant for inode usage, so `pack_store`,
    `packed_store_path`, `gfetch <satellite> pack` and `mosaic`'s `ZipStore`
    handling are gone; `write_regions` takes a store `Path` again. `clean`
    (`remove_cache`) stays, now requiring `store_is_complete` for every store.
  - QGIS opens a mosaic through GDAL's Zarr driver as one subdataset per band.
    Considered switching the store to a single `(band, y, x)` array (one file per
    shard across all bands, but one dtype/fill value for every band, a band list
    fixed at template time, and every existing store rewritten); kept the
    per-variable layout for now.
  - `gfetch.vrt.write_vrt` writes `mosaic_epsg<code>.vrt` next to each store instead,
    stacking the variables as bands. The XML is built directly from the store's
    metadata, without GDAL: rasterio's bundled GDAL (3.12) can't read sharded
    stores. Each band's `SourceFilename` is the array directory
    (`mosaic.zarr/B04`, `relativeToVRT="1"`): verified on GDAL 3.13.3 that this
    reads correctly and survives moving the pair, while `ZARR:"…":/B04` isn't
    resolved relative to the VRT. Georeferencing is written into the VRT, so it
    doesn't depend on GDAL resolving the store's grid mapping (see `tech-stack.md`'s
    "Consolidated metadata and GDAL").

- **2026-09-27** — **`mosaic` reads Planetary Computer assets without `download`.**
  - PC's STAC items hold unsigned Azure Blob hrefs; `search` saves them as-is, so
    `mosaic` from `items.json` got HTTP 409 on every read (`download` was fine:
    stac-asset signs PC hrefs itself). Found with `sentinel-1-rtc`, which opens
    anonymously once signed despite its `msft:requires_account: true`.
  - `gfetch.sources.planetary_computer_signer()` is passed as `odc.stac.load`'s
    `patch_url`, only when loading from remote hrefs. odc-stac patches hrefs when
    the graph is built, and GDAL opens the files only when dask computes it, so the
    token must outlive the whole shard. PC's SAS API issues a new token on every
    request, valid 45 minutes (verified); the `planetary-computer` package caches
    tokens until under 60 s remain, which could hand a shard a nearly expired one.
    So gfetch requests its own tokens (stdlib `urllib`, ~20 lines, no new
    dependency), from a new signer per shard. A shard computing longer than 45
    minutes still fails, and is recomputed on the next run; not worth read-time
    signing for the download-less path.
  - Verified live: a small `sentinel-1-rtc` job mosaics from remote hrefs.

- **2026-09-27** — **`s1` defaults to Planetary Computer's `sentinel-1-rtc`.**
  - GRD isn't terrain-corrected, and odc-stac warps it through GDAL's default
    GCP polynomial, which doesn't even pass through the GCPs: on one Tanzanian
    scene (GCP heights 539–2027 m) it misses them by 94 m median, 293 m p90, 943 m
    max, correlated with GCP height (r = −0.65). The user measured ~250 m against
    S2 in Tanzania, less in flatter Gabon. RTC is already orthorectified, in UTM.
  - RTC's `msft:requires_account: true` turned out not to block anonymous reads:
    a signed href opens without an account (verified 2026-09-27), which lifts the
    credentials blocker behind the original GRD-only decision.
  - `SatelliteProfile.default_source` (`planetary-computer` for S1, `earthsearch`
    for S2) is used when a config sets no `source`, via `resolve_source`, following
    the `resolve_bands`/`resolve_cloud_mask` pattern. A top-level `source:` still
    applies to every satellite, so a config with `source: earthsearch` at the top
    still gets GRD for `s1`. PC's `sentinel-1` now maps to RTC, so PC GRD is only
    reachable through a `custom:` section.

- **2026-09-27** — **Added `gfetch <satellite> coverage`**; see
  `claude/tech-stack.md`'s "`coverage` stage" section for the full design.
  - Writes a single GeoParquet (EPSG:4326) with one row per shard, holding the
    item counts `mosaic` will use: `n_items`, `n_timesteps` (solar-day groups, the
    user's main target), and `n_ascending`/`n_descending` when orbit states are
    split. Shards outside the AOI shape are kept, flagged `skipped`.
  - The shard grid is recomputed from geobox + shard size rather than read from the
    Zarr store, since the store only exists after an expensive `prepare_template`.
    Accepted as a second source of truth because the output is informational;
    `mosaic` keeps relying on the store.
  - Checked in odc-stac's source: `solar_day` offsets by the loaded geobox's
    centroid longitude, not each item's, so solar-day counts are per shard.
  - Built the same day. `write_geoparquet` moved from `gfetch.gedi` to
    `gfetch.utils.geoparquet`, since `gfetch.gedi` imports sliderule at module
    level. Verified live: an S1 RTC `as_bands` search (14 items, 2 shards) gives
    12/7 items, 8/7 solar days, 9+3/3+4 ascending+descending per shard.

- **2026-09-28** — **Added OmniCloudMask (`gfetch s2 ocm`)** as an opt-in
  replacement for SCL cloud masking; see `claude/tech-stack.md`'s "Cloud masking:
  OmniCloudMask" section.
  - POC first (`poc/omnicloudmask_s2.py`, standalone): on three cached T36M* scenes,
    SCL flagged ~40% of a Rift-lake window as cloud where it was bright lake flats
    and escarpment, which OCM left clear (OCM's own miss there: some salt crust as
    thin cloud). On cumulus scenes both caught the clouds, OCM adding small
    clouds/shadows SCL missed (+10-14% of pixels vs. 1-3% SCL-only). ~3 s per
    40 km window at 20 m, but on MPS (Apple GPU, OCM's auto-pick); no CPU timing.
  - Decided with the user: a separate stage (so it can run on GPU nodes), enabled
    by `ocm: true` under `s2:`. Explicit rather than "use it if present", so a
    composite never mixes SCL- and OCM-masked items. `mosaic` refuses to run until
    every item has a mask.
  - No change to the `cached_items.json` hand-off (a central file concurrent `ocm`
    tasks would race on): `<item>/ocm.tif`'s existence is the completion marker
    (written to a temp file, then renamed), and `mosaic` adds the asset to each
    item in memory (`gfetch.ocm.with_ocm_asset`).
  - Thick cloud, thin cloud and shadow always masked out. No-data written as 255,
    since OCM's own no-data value 0 is also its "clear" class.
  - Inputs red/green/nir (B08, 10 m), averaged to 20 m; `nir` must be in `bands`
    (validated at config load), so it also ends up in the mosaic. Compute nodes are
    offline, so `download` fetches the weights into `ocm_model_dir` (default
    `<output_dir>/ocm_models`): checked in OCM 1.7.1's source that `get_models`
    doesn't touch the network when the file is already there.
  - `omnicloudmask` (torch, timm) is an optional `ocm` extra.

- **2026-09-28** — **Found gfetch's store layout too slow for training reads**; see
  `claude/tech-stack.md`'s "Training reads: stacked bands, larger chunks" section.
  - sprout's dataloader was the bottleneck of training on gfetch stores: a 384 px crop
    over 16 bands makes 784 chunk requests with 64 px chunks, one array per band.
  - Copying 10% of the 2020 Mozambique/Tanzania stores into one `(band, y, x)` array
    per store with 256 px chunks cut the read from 339 ms to 60 ms per sample on Jean
    Zay; training steps went from 1.02 s to 0.36 s (with other, sprout-side fixes).
  - sprout already reads that layout. Next: write it from `mosaic`.

- **2026-09-28** — **`mosaic` writes bands stacked into one `(band, y, x)` float32
  array**; see `claude/tech-stack.md`'s "Training reads: stacked bands, larger chunks"
  section for the layout and defaults.
  - `gfetch.write.stack_bands` stacks the mosaic before `prepare_template`/`write_region`;
    `validate_bands` refuses per-band stores and stores with other bands.
  - `write_region` now drops every variable sharing no dimension with the region (the
    `band` coordinate, besides the grid mapping); `gfetch:skipped_shards` leaves `band`
    out so it keeps indexing `("y", "x")`, as sprout expects.
  - Defaults moved to 256 px chunks, 4096 px shards, 1024 px compute bricks.
  - Verified with GDAL 3.13.3: a sharded stacked store opens as a multi-band raster with
    CRS and `DIM_band_VALUE` names, an unwritten shard reads as nodata; the VRT gives the
    band descriptions.
  - Not measured: `mosaic`'s own write time/memory with every band in one shard task
    (`benchmark/sharding_write.py`'s "sharded, all bands per write" row, at 6-12 bands,
    is the closest data point).
