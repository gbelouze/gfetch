# Changelog

All notable changes to this project will be documented in this file.

This changelog should be updated with every pull request with some information about what has been changed. These changes can be added under a temporary title 'pre-release'.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Each release can have sections: "Added", "Changed", "Deprecated", "Removed", "Fixed" and "Security".

## pre-release

### Added

- Project scaffold: `pyproject.toml` (Python 3.12, `ruff`/`pyrefly`/`pydoclint` config), `pre-commit` hooks, `AGENTS.md`.
- `search`, `download`, `mosaic`/`write` pipeline stages, a `sentinel-2` satellite profile, and a `cyclopts`/`omegaconf`-based CLI (`init`/`search`/`download`/`mosaic`), with tests for every stage.
- `mosaic.py::mosaic_by_zone`/`group_by_utm_zone`/`zone_geobox`: mosaic each UTM zone an AOI spans separately, instead of reprojecting everything into one zone chosen from the AOI's centroid. `gfetch mosaic` now writes one Zarr store per spanned zone (`mosaic_epsg<code>.zarr`).
- Logging: every stage (`search`/`download`/`mosaic`/`write`, plus their CLI wrappers and config loading) now reports progress at `INFO`/`DEBUG` as it works, instead of going quiet for the length of a long call - e.g. `search()` logs the total match count up front and one line per STAC API page fetched, so a large search no longer looks hung.
- `download_items()`: with a `progress` tracker passed in, each concurrently-downloading item now gets its own live byte-progress bar (via `stac_asset`'s per-asset message stream), alongside the existing overall "items downloaded" bar - not just one bar for the whole batch.

### Fixed

- `download_items()` (`gfetch download`) could hang indefinitely on networks that require an HTTP(S) proxy for outbound access (e.g. many HPC compute nodes), even though `gfetch search` and plain `curl` worked fine on the same node. Root cause: `stac_asset`'s downloads go through a raw `aiohttp.ClientSession`, which - unlike `requests`/`curl` - silently ignores `HTTP_PROXY`/`HTTPS_PROXY` unless `trust_env=True` is set, and `stac_asset` exposes no way to set it. `download.py` now patches `stac_asset`'s session construction to default `trust_env=True`. Confirmed fixed on the reporting user's HPC cluster.

### Changed

- `Config.zarr_path` is now a method taking a `CRS` (one output store per UTM zone) instead of a fixed property.
