# gfetch — tech stack research & architecture

Tracking document for the technical decisions behind `gfetch`, a library for downloading
large-scale EO datasets (Sentinel, Landsat) in a form suitable for ML pipelines.

## Goals / non-goals

**Goals**
- A user provides an AOI (and optionally a date range / satellite) and gets good,
  analysis-ready data with sensible defaults — no GEE-style bespoke query language.
- No dependency on Google Earth Engine, or on any single proprietary vendor whose quota
  policy can change unilaterally (this is exactly what happened with GEE and prompted
  this project).
- Rely on STAC as the generic interface to spatial datasets, so the library is not
  hard-wired to one catalog provider.
- All processing local/self-hosted — no reliance on non-local (server-side) processing.
- Work well both for local/single-machine downloads *and* on HPC clusters, where work is
  commonly split across node types: some nodes have internet access but little compute
  (login nodes / data-transfer nodes), others have compute but no internet (compute
  partitions, often firewalled by policy). The download stage and the compute/mosaic
  stage must be separable, run on different nodes/jobs, and hand off data durably —
  safely resumable if a job is killed (walltime limits, preemption are routine on HPC).

**Non-goals (for now)**
- Not building a hosted service or a workflow orchestration engine (cf. Tilebox). No
  bundled Globus integration, no attempt to abstract heterogeneous Dask clusters —
  orchestration across node types is the user's job (SLURM/Snakemake/Nextflow/Parsl all
  natively support pinning a task to a queue/partition/executor); gfetch just needs to
  expose its stages so that's possible.
- Not building a general-purpose datacube system (cf. Open Data Cube core) — we reuse
  `odc-stac`/`odc-geo` as libraries, not the full ODC stack.
- Not adopting Icechunk as the default output store — see "HPC / distributed execution"
  below for why (its ACID guarantees don't hold on the POSIX filesystems most HPC
  clusters use). Stays a future/optional backend for S3-compatible deployments.

## Prior art: geefetch (own predecessor) and sibling projects

Added 2026-09-19. `/Users/gbelouze/Documents/phd/src/geefetch` is the user's own existing
library — the literal reason this project exists ("GEE is dead, long live GEE"):
a mature, GEE-backed CLI/library for downloading Sentinel-1/2, Landsat-8, GEDI, Dynamic
World, Palsar-2, NASADEM with per-satellite defaults, declarative YAML config
(`omegaconf`), resumable downloads, and a `click` CLI. It's more directly relevant than
any external reference researched so far, since it's the same author solving almost the
same problem one layer down (GEE instead of STAC). gfetch supersedes it; several concrete
pieces of its design are worth carrying forward rather than re-deriving:

- **`geobbox` (`GeoBoundingBox`)** — the user's own separate, published, CRS-aware AOI
  package. **Considered and rejected (2026-09-19)**: gfetch uses `odc-geo` exclusively
  for AOI/CRS/pixel-grid handling; `geobbox` is deprecated for this project (matches lsatfetch,
  which already doesn't use it — it rolls its own bbox/tile logic instead).
- **`data/tiler.py` (`Tiler`/`TileTracker`)** — geefetch's AOI-splitting/file-tracking
  code. **Considered and rejected (2026-09-19)**: not special enough to be worth carrying
  over — gfetch's tiling/write-region planning will be designed from scratch, informed by
  STAC's own tiling (e.g. Sentinel-2 MGRS) rather than adapted from geefetch's arbitrary
  metric grid.
- **Resume strategy**: lsatfetch's atomic temp-file-then-rename (see "HPC / distributed
  execution" below) is the chosen pattern for the download stage. geefetch's alternative
  — validating file integrity on resume instead of/in addition to atomic writes —
  **was considered and explicitly rejected (2026-09-19)**: not a good fit for gfetch, not
  carried over.
- **`utils/progress_multiprocessing.py` (`QueuedProgress`/`ProgressQueueConsumer`)** —
  more complete than lsatfetch's multiprocess story: a `Progress`-shaped proxy object
  (`add_task`/`update`/`advance`/`remove_task`) that workers can call in a
  `ProcessPoolExecutor` (no shared memory), which queues commands for the main process
  to replay against the real `rich.progress.Progress`. lsatfetch doesn't need this
  because its parallelism is thread-based (shared memory, so a `Progress` object can be
  passed directly); geefetch needs it because GEE/geedim work happens in separate
  processes. gfetch likely needs *both* patterns depending on the stage: thread-based
  (direct `Progress` sharing, per lsatfetch/`dev-stack.md`) for the I/O-bound download
  stage, and this queue-proxy pattern if the compute/mosaic stage is
  process-parallelized. See `dev-stack.md`.
- **Config-driven, declarative usage** — both geefetch and lsatfetch use `omegaconf` for
  YAML config plus a CLI `init` subcommand that scaffolds a template config file. This is
  a consistent, real convention across both sibling projects (not just one-off), and
  fits gfetch's "give an AOI, get good defaults" goal well — a config file is also a
  natural artifact to hand to a SLURM job. Should be adopted for gfetch's CLI from the
  start; see `dev-stack.md`.

## Source-by-source findings

### Tilebox (docs.tilebox.com)
- Managed platform + Python SDK/CLI/REST API for querying/processing EO data, plus a
  workflow/task orchestration layer. Sentinel-2 exposed as `open_data.copernicus.sentinel2_msi`.
- Fully proprietary, no visible pricing/quota info in the quickstart — same risk class as
  GEE (undisclosed/changeable free-tier limits).
- Worth stealing: the AOI-first quickstart UX (define polygon → query scenes → submit
  processing) matches exactly the ergonomics we want; the clean dataset namespacing
  convention.
- Rejected as a dependency: proprietary, requires their hosted API, no quota guarantees.

### Earthmover / Icechunk (earthmover.io, blog: cloud-native-dataloader)
- **Icechunk**: OSS (Apache 2.0), a transactional storage layer on top of Zarr — adds
  ACID transactions, versioning, and a manifest/index so chunked arrays can be
  queried/updated without full-file rewrites.
- **Arraylake**: Earthmover's hosted cataloging/governance/compute layer on top of
  Icechunk — proprietary, no published pricing. Rejected for the same reason as Tilebox.
- The **cloud-native-dataloader** post's architecture is directly relevant and doesn't
  require Icechunk/Arraylake at all: Zarr (1–16MB chunks) → Xarray → **Xbatcher** (batch
  generator) → Dask (concurrent chunk loads) → PyTorch DataLoader, tuned via
  `prefetch_factor` (≈ batch-load-time / train-step-time) and `num_workers`. They report
  large throughput gains from tuning alone, no custom caching layer. **This is why
  gfetch's output targets plain Zarr** — it plugs straight into this dataloading pattern.
- Icechunk itself is a strong future candidate for our storage layer (versioned/growing
  cubes) but is **out of scope for v1** per team decision (2026-09-19).

### earth-mover/serverless-datacube-demo (GitHub)
- Blueprint: STAC search (Sentinel-2 L2A COGs, AWS Open Data) → `odc.stac.load` → xarray
  processing → write to Zarr, runnable as serverless functions (Coiled/Modal/Lithops
  behind one CLI).
- OSS demo repo; default storage target is Arraylake (proprietary), but the
  STAC/odc-stac/xarray/Zarr portion is separable and directly validates our architecture.
- Confirms `odc-stac` (not `stackstac`) is the real, production-used choice for this kind
  of pipeline.

### Pangeo discourse: odc-stac vs stackstac (thread 4097, issue opendatacube/odc-stac#54, benchmark-odc-stac-vs-stackstac.netlify.app)
- No single universal winner was declared by maintainers — issue #54 (a Carpentries
  maintainer asking which to teach) is still open, unanswered.
- Correctness differences: pixel-center vs pixel-edge coordinate convention; odc-stac
  keeps per-band dtype (doesn't promote an int QA/mask band to float32) while stackstac
  coerces to a common dtype; stackstac applies STAC raster scale/offset by default,
  odc-stac doesn't (values can silently differ between the two libraries); odc-stac reads
  COG overviews for downsampled loads, stackstac doesn't; odc-stac's
  `groupby="solar_day"` merges same-day overlapping scenes before compositing.
- Benchmark numbers: for **deep/temporal stacking**, stackstac is ~15% faster (GDAL
  metadata caching). For **wide/mosaic-building** (our use case), **odc-stac is ~2.5x
  faster** (74.7 vs 30.1 Mpix/s) because it computes item/chunk overlap at
  graph-construction time instead of building a per-item task graph and merging
  (graph-build time: 0.03s vs 0.14s).
- **Decision: odc-stac**, because gfetch's core operation is large-AOI mosaic/composite
  building, not deep per-pixel time series.

### Pangeo discourse: best practices for large-scale Sentinel-2 mosaics & 4D ML patches (thread 5010)
- Recommended stack: STAC query (Element84 Earth Search cited) → Zarr datacube (Dask) →
  ML patches via **xbatcher**. Cites the same serverless-datacube-demo repo.
- Chunking should follow Sentinel-2's native **MGRS tiling grid**, not arbitrary AOI
  boxes — arbitrary chunking against MGRS produces "a dask nightmare graph."
- ~5-day revisit means daily composites are inherently NaN-heavy; don't expect complete
  daily mosaics.
- For both single-day mosaics and multi-year ML patch extraction, a chunk shape around
  **(1 day × 5km × 5km)** is a reasonable balance — adopted as gfetch's default.

### stackstac (stackstac.readthedocs.io)
- Last release Aug 2024, single maintainer, who states in the docs: "I haven't even
  written tests yet! Don't use this in production."
- **Rejected** — effectively dormant and explicitly unfit for production per its own
  author.

### odc-stac (github.com/opendatacube/odc-stac)
- Actively maintained by the Open Data Cube org (releases into 2026, 777 commits, CI,
  test suite). `odc.stac.load(items, ...) -> xr.Dataset`, dask-backed, lazy.
- Depends on `odc-geo` (geometry/CRS/GeoBox, deliberately extracted from ODC as a
  standalone package) but **not** on `datacube-core` — adopting it does not pull in the
  full Open Data Cube stack.
- **Chosen** as gfetch's array-loading layer.

### cubo (cubo.readthedocs.io)
- Confirmed: thin convenience wrapper for point-centered ML chips. Backends: GEE (via
  `xee`), Planetary Computer (default STAC), or custom STAC endpoint — delegates pixel
  loading to `stackstac` (STAC path) or `xee` (GEE path). No real mosaicking logic, single
  `create()` call, AOI model is point + pixel-edge-size reprojected to local UTM.
- **Not a dependency** (inherits stackstac's staleness on the STAC path). Worth stealing:
  the "reproject to local UTM before cutting the box" convention, if/when gfetch adds
  point-based patch extraction alongside AOI polygons.

### phidown (github.com/ESA-PhiLab/phidown)
- CDSE downloader: OData search + S3 download + Sentinel-1 burst tooling. 104 stars, 7
  forks, 1 open issue — moderately maintained but narrow in scope, CDSE-specific.
- **Not a building block** — confirms the "seems a bit basic" read. CDSE itself stays a
  candidate future source, accessed via generic STAC + S3 rather than phidown's bespoke
  API.

### Project Pythia cookbooks (cookbooks.projectpythia.org)

Added 2026-09-19, re-checked at the user's request specifically to validate/simplify the
download-stage implementation before writing it (see `claude/tasks.md` for the corrected
POC log entry).

- **landsat-ml-cookbook** (`ProjectPythia/landsat-ml-cookbook`, notebook
  `1.0_Data_Ingestion-Geospatial.ipynb`, read from the actual notebook source via the
  GitHub API, not just the rendered page): `pystac_client.Client.open(url,
  modifier=planetary_computer.sign_inplace)` for search — PC's `modifier` hook signs
  every item's asset hrefs *in place* as part of the search call itself — then
  `odc.stac.stac_load([selected_item], bands=..., bbox=..., chunks={})` called
  **directly on the signed remote item, with no separate download step at all**.
  odc-stac/GDAL streams bytes on demand from the signed URL; nothing is persisted
  locally.
- **eo-datascience-cookbook**, **interactive-sentinel-2-cookbook**: landing pages only
  exposed high-level descriptions (STAC + Dask Gateway, Sentinel-2 dashboards); actual
  notebook source wasn't pulled for these since the landsat-ml-cookbook notebook already
  gave a concrete, representative code sample and the pattern it confirms (search with
  a signing modifier, load straight from the remote href) is what mattered for this
  check.
- **Implication for gfetch, confirmed rather than contradicted**: this is the standard
  pattern for *interactive/exploratory* use on an internet-connected machine, and it
  validates that `odc.stac.load()` needs no local copy at all in that setting — gfetch's
  `search` → `load`/`mosaic` stages can be chained directly with no `download` step in
  between when running single-machine with internet access throughout. It does **not**
  contradict gfetch's HPC design: the cookbook has no offline-compute-node constraint,
  so it never needed a persisted local cache. gfetch's `download` stage stays a real,
  separate, *optional* stage — needed specifically when the compute step will run
  offline (HPC compute partitions) or when a durable local cache is wanted for its own
  sake, not needed otherwise.

### Earthmover blog (`cloud-native-dataloader`) — re-checked

Re-read specifically for STAC-asset download/caching patterns (not just the dataloader
tuning findings already captured above). **Confirmed it has none**: the post's pipeline
starts from Zarr arrays already sitting in cloud storage (GCS) and is entirely about
Xarray → Xbatcher → Dask → PyTorch `DataLoader` tuning downstream of that. It doesn't
touch STAC search, COG reading, or asset download/caching at all — not a source for the
download-stage design, confirmed rather than newly discovered.

## Adjacent tooling assessed (not in the user's original notes)

| Library | Verdict | Notes |
|---|---|---|
| `stac-asset` | **Adopted** | Async, concurrent STAC asset downloader; pluggable clients (`HttpClient`, `S3Client` incl. requester-pays, `FilesystemClient`, `PlanetaryComputerClient`, `EarthdataClient`) auto-selected by href. Actively developed. This is gfetch's download layer. |
| `pystac-client` | **Adopted** | De facto standard STAC API search client (`Client.open(url).search(...)`). Minimal deps. gfetch's search layer for `ApiSource`. |
| `planetary-computer` SDK | **Used indirectly** | SAS-token signing for PC assets; already wrapped by `stac-asset`'s `PlanetaryComputerClient`. Only needed directly if signing outside the download path. |
| `odc-geo` | **Adopted** | Standalone geometry/CRS/GeoBox handling, needed for AOI handling regardless of odc-stac. |
| `rioxarray` | **Candidate utility** | GDAL/rasterio-backed xarray accessor, useful for ad hoc COG reads outside the odc-stac path. Not a core dependency yet. |
| `torchgeo` | **Reference only** | PyTorch-side consumption library (samplers, `Dataset`/`DataLoader` over already-available rasters). Doesn't solve acquisition at scale — not our layer, but a useful reference for how to structure output for eventual PyTorch consumption. |
| `sentinelhub-py` / `eo-learn` | **Rejected** | Tied to Sentinel Hub's paid API; EO Browser/Dashboard sunset already announced — same vendor-risk class we're avoiding. |
| `cdsetool` | **Rejected** | Narrow CDSE-specific CLI/API tool, same category as phidown, low priority. |

## Reference implementations (inspiration repos)

Full end-to-end pipelines worth checking against when designing/implementing a gfetch
stage — not dependencies, but real running code solving close variants of the same
problem. Each entry below was actually cloned and read (source, not just the landing
page/README) before being added here.

- **[ljstrnadiii/flytemosaic](https://github.com/ljstrnadiii/flytemosaic)** (reviewed
  2026-09-19, re-reviewed 2026-09-20 — see `benchmark/README.md` for the full
  investigation log). STAC → GDAL Raster Tile Index (GTI) → rioxarray → Zarr mosaic
  pipeline, orchestrated with Flyte (explicitly not adopted — gfetch has no hosted-
  orchestration ambition). Worth stealing:
  - `flytemosaic/gdal_configs.py::get_worker_config()` — a concrete, real-world set of
    cloud-tuned GDAL/CPL options (HTTP/2 multiplexing, bigger `CPL_VSIL_CURL_CHUNK_SIZE`,
    `GDAL_NUM_THREADS=ALL_CPUS`), reused directly in `benchmark/wide_gdal_config.py`.
  - `flyte/build.py::write_mosaic_partition_task` — the real production concurrency
    pattern behind that config: a **single-threaded** dask scheduler per worker, relying
    on GDAL's own internal thread pool + HTTP/2 instead of a big Python thread pool, and
    scaling out via **separate processes** (one per chunk partition) instead. A live,
    not-yet-confirmed candidate for gfetch's own `mosaic`/`write` stage concurrency
    model — see the "Open threads" in `benchmark/README.md`.
  - `flytemosaic/mosaics.py::build_recommended_gti` / `build_gti_xarray` — builds a GDAL
    Raster Tile Index (`ogr2ogr`-built FlatGeobuf, embedding dtype/extent/CRS/band-count)
    so GDAL never has to open/inspect each source COG to plan a mosaic. odc-stac solves
    the same problem differently (via STAC item properties), so not directly portable,
    but confirms metadata-probe overhead is a real, known cost class here.
  - Not adopted: Flyte orchestration itself, and the separate ingest stage that
    re-downloads/re-encodes every source file into a uniformly-tiled COG before
    mosaicking (a stronger, more invasive version of gfetch's `download` stage, which
    only caches original bytes) — flagged as a possible future refinement, not decided.

- **[earth-mover/serverless-datacube-demo](https://github.com/earth-mover/serverless-datacube-demo)**
  (landing-page-level review 2026-09-19, actual source read 2026-09-20). STAC (Earth
  Search) → `odc.stac.load` → SCL cloud-mask → median composite → Zarr/Icechunk cube,
  runnable on Coiled/Modal/Lithops behind one `click` CLI. Demo-quality, "fork and
  modify" per its own README, not a maintained package. Worth stealing:
  - `src/lib.py::JobConfig.tiles` uses `odc.geo.geobox.GeoboxTiles` to derive a strict
    tile grid over the target geobox, then indexes each processing job by
    `(tile_index, year, month)` — a cleaner, library-provided way to get gfetch's
    already-verified disjoint-region-per-worker write plan (see "HPC / distributed
    execution" above) than hand-rolled index math.
  - `src/lib.py::JobConfig.generate_jobs` skips tiles that don't intersect land
    (`cartopy.feature.LAND.intersecting_geometries`) before ever issuing a STAC search
    for that tile — a real "skip known-empty work early" pattern worth considering once
    gfetch does country/continent-scale tiling.
  - `src/lib.py`'s cloud-masking uses `odc.algo.mask_cleanup`/`erase_bad` (morphological
    closing/opening on the SCL-derived mask before compositing) rather than a bare
    boolean class-exclusion mask — more robust than gfetch's current POC-stage SCL
    handling (a plain `{0,1,3,8,9,10}` class-set exclusion, see `claude/tasks.md`); worth
    adopting `odc.algo` for the real `mosaic` stage's cloud-masking instead of hand-rolling.
  - `src/storage.py::AbstractStorage` (`initialize()`/`get_zarr_store()`/`commit()`)
    cleanly abstracts a plain-Zarr/fsspec store vs. an Icechunk-backed one behind one
    interface — the same "plain Zarr default, Icechunk opt-in" shape gfetch already
    decided on (see "Compute-stage output durability" below), concretely implemented.
    Worth using as a reference shape for gfetch's own write-stage storage abstraction
    when that's built, rather than re-deriving the interface from scratch.
  - `icechunk.distributed.merge_sessions()`, used in `ArraylakeStorage.commit()` to
    merge each worker's `icechunk.Session` into one commit — **real, running proof** of
    the "many workers write disjoint chunks via per-worker sessions, then merge via a
    coordinator" pattern that `tech-stack.md`'s Icechunk section flagged as the
    documented distributed-write model but noted as unverified for worker-death
    behavior. Doesn't resolve that open question (worker-death-before-merge still
    isn't exercised here), but confirms the happy-path mechanism is real and used in
    production-adjacent code, not just documented in theory.
  - `src/lib.py::ChunkProcessingJob.process` writes raw numpy arrays directly into a
    `zarr.Array` via slice assignment (`target_array[target_slice] = raw_data[None,
    ...]`) instead of `xarray.Dataset.to_zarr(region=...)` — a lower-level, likely
    faster path that avoids per-write xarray/dask overhead, at the cost of manually
    tracking the time index (their own comment: "not writing with xarray, so have to
    reverse engineer the time index"). An alternative worth benchmarking against
    gfetch's current xarray-region-write pattern (`claude/tasks.md`) before the real
    `write` stage is built, not something to adopt sight-unseen.
  - Not adopted: the multi-backend serverless dispatch (Coiled/Modal/Lithops) and
    Arraylake as a storage target — both out of scope per gfetch's non-goals (no hosted
    service, no proprietary storage backend as anything but an opt-in).

## STAC source landscape

"STAC source" is not one architectural shape — there's a real fork between a live search
API and a static file tree:

- **`ApiSource`**: wraps `pystac_client.Client.open(url).search(...)` against a live
  server with bbox/datetime/CQL2 filtering. Planetary Computer, Earth Search, CDSE, USGS
  LandsatLook all fit here.
- **`StaticSource`** (future, not v1): walks a plain tree of linked STAC JSON
  (`catalog.json` → `collection.json` → `item.json`) via `pystac` + `fsspec`/`s3fs`, no
  search endpoint — filtering happens client-side. This is what "generic S3 bucket"
  really means when there's no STAC API in front of it; a naive walk over a large static
  catalog doesn't scale the way a server-side spatial index does.
- Orthogonal to both: a **signing/auth hook** per source. Planetary Computer needs
  SAS-token signing; CDSE needs OAuth2 client-credentials bearer tokens; Earth Search
  needs neither (public, no-sign-required S3).

| Source | Auth | Sentinel-2 collection | Status for gfetch |
|---|---|---|---|
| Planetary Computer | anonymous search; SAS-signed assets | `sentinel-2-l2a` | **v1** — broadest coverage (S1/S2/Landsat), Hub retired 2024 but STAC/Data API confirmed to remain; single-vendor free-tier risk (same class as GEE, not yet triggered) |
| Element84 Earth Search (AWS) | none — public S3 | `sentinel-2-l2a` / `sentinel-2-c1-l2a` | **v1** — no token dance, architecturally simplest, most robust long-term free option; chosen specifically to keep the source abstraction honest (forces `signer` to be optional) |
| USGS Landsat (LandsatLook) | requester-pays S3 for direct access | `landsat-c2-l2` | Future — in practice, free via PC/Earth Search mirrors |
| CDSE | OAuth2 client-credentials | official Sentinel archive | Future — more friction, had a multi-day STAC outage in Mar 2026; authoritative fallback, not default |
| Generic static S3 catalog | none / bucket policy dependent | n/a | Future — needs `StaticSource`, see above |

**Sentinel-1 (added 2026-09-22)**: both Planetary Computer and Earth Search expose a
public, anonymous `sentinel-1-grd` collection (verified via each collection's actual
STAC metadata, not assumed) — raw detected amplitude (`vv`/`vh`/`hh`/`hv` assets),
**not** radiometrically calibrated or terrain-corrected, natively in **EPSG:4326** (not
UTM — unlike Sentinel-2's MGRS-tiled items, a single GRD scene routinely spans several
UTM zones; confirmed on a real item). PC additionally has `sentinel-1-rtc`
(radiometrically terrain-corrected, analysis-ready, float32) but it's gated behind
`msft:requires_account: true` — the only gfetch-relevant collection anywhere that needs
real credentials, unlike every other anonymous-SAS-signed/public-S3 source gfetch uses.
**Decision: ship GRD only for now** (both sources registered, `median` composite kept
as the default — not `mean`, since averaging raw uncalibrated, non-terrain-corrected
amplitude across different orbit geometries isn't physically rigorous the way it would
be for RTC backscatter). RTC support is a deliberately separate, larger follow-up that
also needs gfetch's first credentialed-source mechanism (`StacSource`/`Config` have no
notion of auth today). `sat:orbit_state` (ascending/descending) is a plain queryable
STAC property on both sources' items (confirmed on real items) — exposed as
`Config.orbit_state`, filtered via the existing generic `query` mechanism, no new
source-level plumbing needed. See `claude/tasks.md`'s 2026-09-22 entry for the full
reasoning and the `group_by_utm_zone` generalization this required.

## Chosen architecture

```
AOI + date range + satellite profile ("Sentinel2", "Landsat")
        │
        ▼
[search]  StacSource (ApiSource[PlanetaryComputer] | ApiSource[EarthSearch], + optional signer)
        │  pystac-client search → pystac.ItemCollection
        ▼
[download]  stac-asset, atomic temp+rename writes, one sentinel file per completed
        │   asset — resumable, safe for many workers sharing one cache dir, no
        │   coordination needed. Runs on internet-connected nodes/jobs.
        │   Rewrites item asset hrefs to the local cache paths.
        ▼
locally-cached STAC items (durable handoff — no network needed downstream)
        │
        ▼
[load/mosaic]  odc-stac.load(items, geobox=odc_geo.GeoBox.from_aoi(...),
        │      groupby="solar_day", ...) — per-band dtype, overview-aware,
        │      satellite-profile defaults (bands, cloud/QA mask, scale/offset,
        │      resampling). Runs on compute nodes/jobs, no internet required.
        ▼
xarray.Dataset (dask-backed, lazy)
        │
        ▼
[write]  plain Zarr by default, pre-planned non-overlapping chunk regions per
        │   worker (safe on any POSIX filesystem, incl. shared HPC storage);
        │   Icechunk as an opt-in upgrade only when the store is S3-compatible.
        │   Chunked ~(1 day × 5km × 5km) by default.
        ▼
Zarr store, Xbatcher/PyTorch-DataLoader-compatible output
```

Each bracketed stage (`[search]`, `[download]`, `[load/mosaic]`, `[write]`) is a
separately callable function and CLI subcommand — see "HPC / distributed execution"
below for why that separation is load-bearing, not just a code-organization nicety.

A **satellite profile** (e.g. `gfetch.Sentinel2`, `gfetch.Landsat`) is the good-defaults
layer: it fixes the collection name per source (`sentinel-2-l2a` vs
`sentinel-2-c1-l2a`; `landsat-c2-l2`, etc.), default bands, and default cloud-mask band
(Sentinel-2 SCL) — so the common case really is "give me an AOI," while every default
stays overridable. `groupby`, chunking, and resampling are not profile fields; they're
job-level knobs (`load()`/`mosaic()` parameters, `Config.resampling`) with library-level
defaults (`groupby="solar_day"`, odc-stac's own chunking/`"nearest"` resampling) —
profile-driven per-satellite defaults for these were considered but not built, see
`claude/tasks.md`'s 2026-09-22 entry.

**2026-09-21 update**: `[load/mosaic]`/`[write]` produce **one `xarray.Dataset`/Zarr
store per UTM zone the AOI spans**, not the single dataset/store the diagram above
shows — see `claude/tasks.md`'s 2026-09-21 entry for why. `mosaic_by_zone()` groups items by
their own native UTM zone (not the AOI's), builds one `GeoBox` per zone (clipped to
that zone's natural longitude band), and mosaics each zone independently; `[write]` is
called once per zone's dataset.

## HPC / distributed execution

Added 2026-09-19 after the initial architecture pass, in response to a hard requirement:
gfetch must work well on HPC clusters, where nodes commonly split into two roles —
internet-connected (login/DTN) and compute-only (firewalled from the internet by policy,
not just convention; confirmed common at major HPC centers). The download stage and the
compute/mosaic stage need to run on different nodes/jobs, with a durable handoff between
them, safe against jobs being killed mid-run.

**Stage separation, not orchestration.** gfetch exposes `search`, `download`, `load`/
`mosaic`, `write` as independently callable functions and CLI subcommands, rather than
one monolithic call. This lets any external orchestrator pin each stage to the right node
class — SLURM job arrays/dependencies, Snakemake per-rule `resources: partition=`,
Nextflow per-process `executor`/`queue`, Parsl per-app `executors=` all support this
natively. gfetch does not attempt to solve this itself:
- **No Globus integration.** Globus (Transfer/Compute) is the standard HPC tool for bulk
  multi-TB institutional transfers and remote orchestration, but gfetch pulls individual
  COG assets over plain HTTPS/S3 from public STAC catalogs — `stac-asset` already covers
  that, and Globus would add an unneeded endpoint/auth dependency. If a site mandates
  Globus for inbound transfer, the user can pre-stage into gfetch's cache dir with their
  own Globus job; gfetch doesn't need to know.
- **No attempt to abstract heterogeneous Dask clusters.** `dask-jobqueue` does not support
  a single cluster with two worker pools on different node classes (confirmed open
  limitation, e.g. dask/dask-jobqueue#616) — the real-world workaround is two separate
  `SLURMCluster` instances handed off via shared filesystem. That's the user's
  `dask-jobqueue` config, not something gfetch should paper over.

**Download-stage durability — this is the part gfetch must own.** Neither of our chosen
libraries gives this for free:
- `stac-asset` has skip-if-exists resume logic, but its writes are **not atomic** — it
  streams directly into the final target path and only cleans up on a caught Python
  exception. A SIGKILL/walltime-preempted job leaves a partial file sitting exactly where
  the resume check looks, so **a killed download is silently treated as complete on
  resume** as-is.
- `fsspec`'s cached filesystems (`filecache`/`simplecache`) have the same class of risk:
  only `simplecache` is documented thread/process-safe, and even it writes to the final
  content-addressed cache path with no visible temp-file+rename — completion metadata is
  written only after download, but the raw bytes may already exist at the cache path
  before that. Transparent fsspec caching would hide exactly the failure mode we need to
  engineer around, so it's rejected as the caching mechanism (see `claude/tasks.md`).
- **Chosen approach:** gfetch's download stage writes to a temp path and atomically
  `rename`s into place, and marks completion with one sentinel file per completed
  unit-of-work (e.g. `<item_id>/<asset_key>.complete`) rather than a central manifest
  database. This sidesteps any need for distributed locking (SQLite over NFS/Lustre has
  known locking problems on some HPC filesystems) — each worker only ever writes its own
  uniquely-named files, so many download workers can safely share one cache directory
  with no coordination. A resumed/retried download job just checks for the sentinel file
  before re-fetching.
- `stac-asset`'s `download_item()` already rewrites each asset's href to its local path as
  part of downloading, so the handoff to the load/mosaic stage is "point odc-stac at the
  rewritten local-path items" — no separate STAC-ecosystem utility needed for that part.

**Compute-stage output durability — default plain Zarr with pre-planned regions, Icechunk
as an opt-in upgrade.** Icechunk (Earthmover, OSS Apache 2.0) is the obvious tool for
"many compute workers write different chunks of one output store concurrently, then
commit" — it has a real, documented, actively-developed distributed-write model
(per-worker `ForkSession`s merged by a coordinator, or optimistic-concurrency writes with
conflict rebase). **But its docs explicitly disclaim safety: "File system Storage is not
safe in the presence of concurrent commits... don't use file system storage in production
if there is the possibility of concurrent commits."** Its concurrent-write guarantees are
proven on S3-compatible object storage, not on the local/shared POSIX filesystems
(Lustre/GPFS/NFS) that most HPC clusters actually provide. So:
- **Default:** plain Zarr, with the mosaic's spatial/temporal tiling planned upfront by a
  coordinating step so each compute worker writes into a strictly non-overlapping chunk
  region (the standard Dask-to-Zarr region-write pattern). Shared array metadata is
  written once before workers start; workers never touch shared metadata files, only
  their own disjoint data chunks.
- **Resolved 2026-09-19**: a local/POSIX Zarr store is not one file — it's one file per
  chunk (both v2 `DirectoryStore` and v3 `LocalStore`), the same shape as the
  download stage's one-file-per-COG-asset case. Better still, zarr-python's v3
  `LocalStore` **already writes each chunk atomically (temp file + rename) natively**,
  specifically to prevent corrupted data — gfetch doesn't need to reimplement the
  download stage's atomic-write pattern for the write stage, it comes for free. This
  does **not** replace the disjoint-region-per-worker plan above — it's still required,
  not just an optimization, given open zarr-python issues about concurrent writes to the
  *same* chunk/store not being fully hardened (zarr-developers/zarr-python#328, #3525).
  The atomicity guarantee is per-chunk-file; two workers racing on one chunk is still
  unsafe, which is exactly what pre-planned disjoint regions prevent.
- **Virtual chunk references (kerchunk/VirtualiZarr/Icechunk) — investigated, not a v1
  mechanism.** These let you build a Zarr-compatible index of byte-range references into
  *existing* files (netCDF, HDF5, GRIB2, TIFF/COG) instead of copying pixel data into new
  Zarr chunks — dask/xarray-compatible, and this is genuinely what the "zarr/icechunk can
  reference data living elsewhere" recollection was pointing at (confirmed real;
  Icechunk's "virtual chunks" feature is built directly on VirtualiZarr). But it doesn't
  fit gfetch's core write step: odc-stac's job is reprojecting/resampling heterogeneous
  STAC items (different UTM tiles, footprints, acquisition dates) onto one common AOI
  grid, which requires materializing real pixel values — you cannot express resampling as
  a pointer into someone else's bytes. Confirmed virtual-reference examples in the docs
  are all same-grid concatenation (e.g. stacking identically-gridded files along time);
  combining sources with different footprints/CRS isn't documented, and concurrency
  safety of virtual-chunk cataloging on POSIX storage is undocumented/unproven (a
  different concern from Icechunk's already-documented real-data-write limitation on
  POSIX). Worth keeping as a **later, narrower optimization** — e.g. serving a single
  MGRS tile's time series without ever touching pixel values — not something to design
  the v1 write stage around.
- **Optional upgrade:** Icechunk, when the output target is genuinely S3-compatible
  (cloud deployment, or an on-prem S3 gateway such as Ceph RGW/Garage/SeaweedFS — note
  MinIO's OSS repo was archived in early 2026 as the company shifted to a commercial
  product, so it's a fading choice for that role). Worth revisiting once gfetch has a
  working default pipeline; still no POC planned yet.
- Icechunk's behavior if a worker dies mid-write (before merging/committing) is not
  documented anywhere found — likely benign for the cooperative pattern (an unmerged
  fork is simply never committed) but unverified, another reason not to lean on it as the
  default for walltime-killed HPC jobs.

**Resolved 2026-09-19**: `odc.stac.load()` does read STAC items whose asset hrefs were
rewritten to local cache paths identically to the remote-href path — GDAL/rasterio-backed
reading is indeed href-scheme-agnostic as expected. But getting there isn't code-change-free:
`stac_asset.download_item()` has an owner-backref bug that leaves asset hrefs resolving
against the *original remote* self href unless explicitly fixed. See `claude/tasks.md` for
the repro and fix — gfetch's download stage must apply this fix before handing items to
`load`/`mosaic`.

## Future: disk-bounded streaming download+mosaic (deferred, not designed for v1)

Added 2026-09-22, from a user design discussion — **not built, not scheduled**; recorded
here so the reasoning isn't lost, per the user's explicit "write it off as desirable long
term" framing. Lower priority than current work; revisit only when actually needed.

**Motivating scenario**: for a large enough job (country-scale AOI, long date range),
every downloaded asset for the whole job cannot fit on local/cache disk at once. Today's
`download` stage has no notion of this — it downloads everything the job needs, then
`mosaic`/`write` runs once the whole batch is on disk (see "HPC / distributed execution"
above and `claude/tasks.md` for why `mosaic` currently falls back to remote hrefs, not an
error, when `download` hasn't finished). The goal would be to interleave the two stages:
download enough to mosaic+write one spatial unit, delete that unit's cached assets, and
move on — bounding disk usage to roughly one unit's footprint instead of the whole job's.

**Feasibility verdict: yes in principle**, and it composes cleanly out of primitives
gfetch already has, rather than needing new low-level machinery:
- **Granularity is the native tile, not the UTM zone or the Zarr chunk.** A zone can span
  dozens of tiles (too coarse to bound disk usage meaningfully); a Zarr chunk (~5km) is
  too fine, because a `median`/`mean` composite needs every item covering that chunk
  across the *whole* date range before it can be written, and that need collapses almost
  everywhere to "every item of the one native tile the chunk sits in" (tiles are revisited
  repeatedly over time, chunks sit inside one tile except at tile boundaries). So the
  natural disk-bounded working set is one tile's full time series at a time. Boundary
  chunks (shared between adjacent tiles) get a two-tile dependency instead of one.
- **This is spatial streaming, not temporal streaming.** The composite reduction itself
  (`mean`/`median` over `dim="time"`) still needs a tile's whole time series in memory at
  once — nothing about it becomes incremental. Only which tiles get processed/evicted
  when is what streams.
- **Readiness detection**: already free, via the same per-asset `.complete` sentinel
  files the download stage already writes for resumability (`claude/tech-stack.md`'s
  download-durability design) — no new manifest needed, same "no central bookkeeping,
  no locking" property gfetch already relies on elsewhere.
- **Writing a tile's chunks in isolation**: already built and tested, just not wired into
  the CLI yet — `gfetch.write.prepare_template()`/`write_region()`
  (`src/gfetch/write.py`) already implement "write the full store's metadata once, then
  write disjoint regions independently, safe on any POSIX filesystem." `cli/mosaic.py`
  currently only calls the one-shot `write()`; switching to template-once +
  region-write-per-ready-tile is the only change needed on the write side.
- **New pieces actually needed**: (1) a chunk↔required-tiles dependency map, computed
  from geometry alone at search time (no data needed); (2) reference-counted cache
  eviction — delete a tile's cached assets only once *every* chunk it feeds has been
  written, not just once its own interior chunks are done, because of the boundary-chunk
  case above; (3) a disk-budget-aware loop that doesn't start downloading the next tile
  before there's room for it.

**Coordination mechanism — two options considered, filesystem-mediated preferred**:
1. **Filesystem-mediated (preferred first cut)**: two independently-running, long-lived
   `gfetch` processes — a download loop and a `mosaic --watch` loop — each still pinned to
   their own node class by SLURM as today, coordinating purely by polling the same
   sentinel files on the shared filesystem. No new dependency; stays inside gfetch's
   existing "stage separation, not orchestration" framing (see "Goals / non-goals"
   above) — the two stages remain independently callable/schedulable, they just now also
   run concurrently and loop instead of running once each. Costs: gfetch has to hand-roll
   the reference-counted eviction and disk-budget backpressure logic itself.
2. **HyperQueue as an explicit task-DAG submitter**: gfetch submits "download tile X" /
   "mosaic chunk Y (depends on tiles it needs)" / "delete tile X (depends on every chunk
   that needed it)" as a real dependency graph and lets HyperQueue schedule it, including
   placing download vs. mosaic tasks on differently-tagged workers spanning both node
   classes within one allocation via its custom resource tags. Gets real scheduling and
   backpressure machinery for free, at the cost of a new external dependency and pushing
   gfetch further toward being a workflow engine than its stated non-goals currently allow
   (see "Goals / non-goals" above: "not building a hosted service or a workflow
   orchestration engine"). Also arguably overkill here — the actual dependency graph per
   tile is shallow (download → mosaic → delete, longest path length 1), not enough
   structure to clearly justify an external scheduler over a simple watch-loop.

**Decision**: **not building this now** — deferred, option 1 preferred if/when it's
picked up, since it doesn't require revisiting gfetch's "no workflow-orchestration-engine"
non-goal. No code, no scaffolding, no `claude/tasks.md` follow-up items beyond this note.

## Explicitly rejected dependencies

| Library/service | Reason |
|---|---|
| `stackstac` | Unmaintained per its own author; slower for mosaic-building workloads |
| `cubo` | Thin wrapper adding nothing beyond what we're building directly |
| `phidown`, `cdsetool` | Narrow CDSE-specific clients, not generic building blocks |
| `sentinelhub-py`, `eo-learn` | Tied to a commercial API with an already-announced product sunset |
| Tilebox | Proprietary, undisclosed quota/pricing — same risk class as GEE |
| Arraylake / Earthmover Compute | Proprietary hosted layer; Icechunk itself (OSS) stays a future option |
| Google Earth Engine | Quota policy change made it unsuitable for large-scale use — the reason this project exists |
| `geobbox` (own package) | Deprecated for this project in favor of `odc-geo` alone (2026-09-19) |
| geefetch's `Tiler`/`TileTracker` | Not carried over — worth designing fresh rather than adapting (2026-09-19) |
| geefetch's integrity-check-on-resume | Considered, explicitly rejected as a fit for gfetch (2026-09-19) |

## Open questions / risks

- Planetary Computer's free-tier sustainability is unproven long-term (no SLA); mitigated
  by treating Earth Search as an equally first-class v1 source rather than a fallback.
- Is `groupby="solar_day"` the right default for all AOIs, or should it be a per-profile
  setting (e.g. disabled for pure time-series use cases if gfetch ever supports those)?
- `StaticSource` (generic S3 bucket without a STAC API) is deferred — is there a concrete
  near-term need, or can it stay a "designed for, not built" abstraction for longer?
- Should `rioxarray` become a real dependency (e.g. for reading assets `stac-asset`
  downloaded locally) or stay opportunistic/optional?
- ~~`SatelliteProfile`'s "default resampling" was never actually built~~ — **resolved
  2026-09-22**: `load()`/`mosaic()`/`mosaic_by_zone()` and `Config.resampling` now
  expose a per-band resampling override (`dict[str, str]`, `"*"` sets the default for
  unlisted bands), threaded through to `odc.stac.load(resampling=...)`; the satellite
  profile's `cloud_mask_band` is always pinned to `"nearest"` inside `mosaic()`
  regardless of what's requested for it, since it holds categorical values. No
  per-satellite automatic default (e.g. bilinear for a coarser-resolution band like
  60m `coastal`/`nir09`) was added — the override is manual/opt-in per job, not a new
  profile field; still a plausible future refinement, not built. See `claude/tasks.md`.
- **`SatelliteProfile.default_bands` for `sentinel-2` is RGB-only (`("red", "green",
  "blue")`), found 2026-09-22 answering the same user question.** Worth remembering
  when estimating job size (per-item byte volume scales directly with band count -
  see `benchmark/README.md`'s per-item throughput numbers, which were all RGB-only)
  or when a user's mosaic looks unexpectedly thin on bands - `bands:` in the config
  fully replaces the default list rather than extending it, so getting RGB *and*
  something else means listing all of them explicitly.
- Icechunk's behavior when a worker dies mid-write (before merge/commit) isn't documented
  anywhere found — probably benign for the cooperative fork/merge pattern, but unverified.
  Matters if/when Icechunk is revisited as an S3-backed output option.
- If an on-prem S3-compatible gateway is ever needed for the Icechunk upgrade path,
  MinIO's OSS repo was archived in early 2026 (company pivoted commercial) — evaluate
  Ceph RGW/Garage/SeaweedFS instead when that need actually arises.
- `dask-jobqueue` can't span two node classes in one cluster (open limitation) — if a
  user wants a Dask-based compute stage alongside a separate download stage, they'll need
  two `SLURMCluster`s handed off via shared filesystem. This is a user-facing
  documentation point for gfetch, not something gfetch needs to solve in code.
- **Long-term goal, flagged pre-production (2026-09-19):** Zarr's default layout is
  one file per chunk per variable. A country-scale AOI at 10m resolution with small
  chunks (e.g. the ~5km default) and several bands can land in the tens/hundreds of
  thousands of files — a real concern on Lustre/GPFS-class HPC filesystems (per-user
  inode quotas, metadata-server overhead), not just a performance nicety. Needs a
  deliberate chunk-size sizing pass (and/or Zarr v3 sharding, which packs multiple
  chunks into one file specifically to address this) before gfetch is used in
  production at country scale. Not blocking v1/POC work.
- ~~Sentinel-2 `search` stage needs processing-baseline resolution~~ — **resolved
  2026-09-21**, see `claude/tasks.md`.

## Next steps

Design, scaffolding, a full pipeline POC, and a working implementation (with tests and
a real CLI smoke test against live data) are all done. Remaining work, in rough order:

1. Exercise the Planetary Computer source end-to-end (currently registered/collection-
   mapped but only Earth Search has actually been run against).
2. Add a `landsat` satellite profile once its collection ids are verified per source
   (currently deferred — see "Explicitly rejected dependencies"/source landscape above).
3. Proper benchmarking (cloud-colocated compute, varying AOI size/resolution/band count,
   Dask cluster vs. local threads) is real future work — the POC's ~0.02 Mpix/s number is
   a home-internet/single-laptop data point, not a library performance ceiling, and
   shouldn't be used for capacity planning as-is.
4. Optional `crs` override for `mosaic`/`mosaic_by_zone`, for users who want a single
   target CRS despite the per-zone default (e.g. matching an existing dataset's grid) —
   requested as a "maybe later" by the user when native-per-zone was decided
   (2026-09-21), not designed or built yet.
5. **Long-term, explicitly deferred**: disk-bounded streaming download+mosaic (interleave
   the two stages, tile-by-tile, deleting cached assets once a tile's dependent chunks are
   written, instead of requiring the whole job's assets to fit on disk at once) — see the
   dedicated "Future: disk-bounded streaming download+mosaic" section above for the full
   design reasoning. Lower priority than everything above; not scheduled.

Dated history of how the above was reached — POCs, bugs found/fixed, and individual
decisions — lives in `claude/tasks.md`, not here.
