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

### Changed

- `Config.zarr_path` is now a method taking a `CRS` (one output store per UTM zone) instead of a fixed property.
