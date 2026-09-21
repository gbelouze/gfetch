# gfetch

A library for downloading large-scale Earth observation datasets (Sentinel, Landsat) as
analysis-ready Zarr cubes, built on STAC — no Google Earth Engine, no dependency on any
single proprietary vendor.

## Overview

Give `gfetch` an AOI (and optionally a date range / satellite) and it searches, downloads,
mosaics, and writes the result as a Zarr cube suitable for ML pipelines (Xbatcher/PyTorch
`DataLoader`-compatible). Built on `pystac-client` (search), `stac-asset` (download),
`odc-stac`/`odc-geo` (mosaic/reproject). Designed to run both on a single machine and
across HPC clusters, where the download and compute/mosaic stages typically run on
different node types (internet-connected vs. compute-only) — see `claude/tech-stack.md`
for the full architecture writeup.

## Features

- [x] Search: STAC search against Element84 Earth Search (tested) and Planetary Computer
      (source/collection mapping registered, not yet exercised end-to-end)
- [x] Download: resumable, atomic (temp-write + rename), sentinel-file completion
      markers, works through an HTTP(S) proxy (needed on many HPC compute nodes),
      one live byte-progress bar per concurrently-downloading item
- [x] Mosaic: `odc-stac`-backed load/reproject, one native UTM zone at a time, cloud
      masking + composite (an AOI spanning several zones is never warped into one
      arbitrarily-chosen zone)
- [x] Write: plain Zarr output, one store per UTM zone the AOI spans, incl. pre-planned
      per-worker disjoint-region writes
- [x] Stage separation: `search`/`download`/`mosaic` independently callable CLI subcommands
      (mosaic build + write share one subcommand — no benefit to splitting across SLURM jobs
      when both run compute-only, back to back, in the same process)
- [ ] Satellite profiles: `sentinel-2` implemented; `landsat` deferred (collection ids
      unverified per `claude/tech-stack.md`)
- [x] Config-driven: `omegaconf` YAML config + CLI `init` scaffolding
- [x] CLI: `cyclopts`-based, one subcommand per stage
- [x] Pleasant UX: `rich` logging and progress bars
- [x] Fully typed: complete type annotations, checked with `pyrefly`
- [x] Testing: unit tests for every stage, incl. resumability/atomicity edge cases; a few
      network-backed tests marked `slow`

## Installation

```bash
uv sync
uv run pre-commit install
```

## Usage

```bash
uv run gfetch init config.yaml   # scaffold a config template, then edit its AOI/time range
uv run gfetch search config.yaml     # internet-connected: find matching STAC items
uv run gfetch download config.yaml   # internet-connected: download assets to a local cache
uv run gfetch mosaic config.yaml     # compute-only: load, cloud-mask, composite, write to Zarr
```

`mosaic` writes one Zarr store per UTM zone the AOI spans
(`<output_dir>/mosaic_epsg<code>.zarr`) — a country-scale AOI crossing several zones
produces several stores, each in its own zone's native CRS, rather than one store
reprojected into a single arbitrarily-chosen zone.

On an HPC compute node that requires an HTTP(S) proxy for outbound access, set
`HTTP_PROXY`/`HTTPS_PROXY` as usual — both `search` and `download` respect them (the
latter needed a fix, since its underlying HTTP library doesn't do this by default; see
`claude/tech-stack.md`'s decision log for details).

## Configuration

See [`examples/tanzania-sentinel2-2020h1.yaml`](examples/tanzania-sentinel2-2020h1.yaml)
for a worked example (Sentinel-2 cloud-free median mosaic, Tanzania, Jan-Jun 2020).

See `claude/tech-stack.md` and `claude/dev-stack.md` for the architecture and tooling
decisions behind this project.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
