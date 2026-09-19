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

- [ ] Search: STAC search against Planetary Computer and Element84 Earth Search
- [ ] Download: resumable, atomic (temp-write + rename), sentinel-file completion markers
- [ ] Mosaic: `odc-stac`-backed load/reproject onto a common AOI grid, satellite-profile defaults
- [ ] Write: plain Zarr output with pre-planned per-worker chunk regions
- [ ] Stage separation: `search`/`download`/`load`/`write` independently callable, for SLURM job splitting
- [ ] Satellite profiles: `Sentinel2`, `Landsat` with sensible per-collection defaults
- [ ] Config-driven: `omegaconf` YAML config + CLI `init` scaffolding
- [ ] CLI: `cyclopts`-based, one subcommand per stage
- [ ] Pleasant UX: `rich` logging and progress bars
- [ ] Fully typed: complete type annotations, checked with `pyrefly`
- [ ] Testing: comprehensive test coverage

## Installation

```bash
uv sync
uv run pre-commit install
```

## Usage

```bash
uv run gfetch --help
```

## Configuration

See `claude/tech-stack.md` and `claude/dev-stack.md` for the architecture and tooling
decisions behind this project.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
