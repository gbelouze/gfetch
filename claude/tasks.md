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

