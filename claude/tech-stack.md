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
`sentinel-2-c1-l2a`; `landsat-c2-l2`, etc.), default bands, default cloud-mask band
(Sentinel-2 SCL), default resampling, and default `groupby`/chunking — so the common case
really is "give me an AOI," while every default stays overridable.

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
  engineer around, so it's rejected as the caching mechanism (see decision log).
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
against the *original remote* self href unless explicitly fixed. See "POC log" below for
the repro and fix — gfetch's download stage must apply this fix before handing items to
`load`/`mosaic`.

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

## POC log

- **2026-09-19 — odc-stac local-href POC: PASSED, but surfaced a real `stac-asset` bug
  to work around.** Searched one Sentinel-2 item on Earth Search, downloaded its
  smallest band (`coastal`/B01, 60m) via `stac_asset.download_item()`, and compared
  `odc.stac.load()` output between the original remote-href item and the
  downloaded/rewritten local-href item over the same small window: identical shape,
  dtype, and pixel values.
  - **Bug found**: `download_item()`'s returned `Item` has each `Asset.owner` still
    pointing at the *original* (pre-download) item object, whose self href is the
    remote STAC API URL, not the new local one. `odc.stac.load()` resolves asset hrefs
    via `Asset.get_absolute_href()`, which joins `asset.href` against `asset.owner`'s
    self href — so without a fix, it silently re-resolves the (now-relative-looking)
    local href against the *remote* STAC endpoint, producing bogus URLs like
    `https://earth-search.aws.../items/B01.tif` (404) or, after a naive
    `item.make_asset_hrefs_absolute()` fix attempt, worse ones like
    `https://earth-search.aws.../tmp/gfetch_poc_cache/B01.tif` (403) — both `git`-blame
    to the same root cause, not two separate bugs. `Item.make_asset_hrefs_absolute()`
    alone does not fix this, because it only rewrites `.href` — it doesn't touch the
    stale `.owner` backref that `get_absolute_href()` actually reads.
  - **Fix**: after calling `download_item()`, explicitly re-point every asset at the
    returned item before resolving hrefs absolute:
    ```python
    result = await download_item(item, cache_dir, config=Config())
    for asset in result.assets.values():
        asset.set_owner(result)
    result.make_asset_hrefs_absolute()
    ```
  - **Action item**: gfetch's download-stage wrapper around `stac_asset.download_item()`
    must apply this fix before handing items off to the `load`/`mosaic` stage — codify
    it as a small helper (e.g. `gfetch.download.finalize_item()` or similar) rather than
    something call sites have to remember. Confirmed against `stac-asset==0.4.7`,
    `odc-stac==0.5.3`, `pystac==1.15.2` — worth a quick recheck if any of those are
    upgraded later, in case upstream fixes the owner-backref bug and the workaround
    becomes a no-op (harmless either way, since `set_owner` to the already-correct owner
    is a no-op).

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
  remote-href ones, but only after working around a `stac-asset` owner-backref bug.
  gfetch's download stage must apply the documented fix (`asset.set_owner(result)` before
  `make_asset_hrefs_absolute()`) before handing items to the `load`/`mosaic` stage.

## Next steps

Design phase and scaffolding are both done; the odc-stac local-href POC (the only
remaining unverified-but-load-bearing assumption) has passed. Next, in rough order:

1. **Implement `search` and `download` stages first** (against both Planetary Computer
   and Earth Search, per the v1 source decision), since they're the least architecturally
   risky and unblock testing the atomic-write + sentinel-file resume design for real,
   before tackling the `load`/`mosaic`/`write` stages. The download stage must bake in
   the `stac-asset` owner-backref fix found during the POC.
2. Start a `claude/tasks.md` (mirroring lsatfetch's) once real implementation issues
   start turning up.
