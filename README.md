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
- [x] Download: resumable, atomic (temp-write + rename), sentinel-file completion markers
- [x] Mosaic: `odc-stac`-backed load/reproject onto a common AOI grid, cloud masking + composite
- [x] Write: plain Zarr output, incl. pre-planned per-worker disjoint-region writes
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

## Configuration

See `claude/tech-stack.md` and `claude/dev-stack.md` for the architecture and tooling
decisions behind this project.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
