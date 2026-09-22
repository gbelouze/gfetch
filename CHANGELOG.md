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
- `mosaic.py::group_by_utm_zone`/`zone_geobox`: mosaic each UTM zone an AOI spans separately, instead of reprojecting everything into one zone chosen from the AOI's centroid. `gfetch mosaic` now writes one Zarr store per spanned zone (`mosaic_epsg<code>.zarr`).
- Logging: every stage (`search`/`download`/`mosaic`/`write`, plus their CLI wrappers and config loading) now reports progress at `INFO`/`DEBUG` as it works, instead of going quiet for the length of a long call - e.g. `search()` logs the total match count up front and one line per STAC API page fetched, so a large search no longer looks hung.
- `download_items()`: with a `progress` tracker passed in, each concurrently-downloading item now gets its own live byte-progress bar (via `stac_asset`'s per-asset message stream), alongside the existing overall "items downloaded" bar - not just one bar for the whole batch.
- `gfetch gedi` command and `gfetch.gedi.fetch_gedi_l2a()`: a standalone GEDI L2A vector-footprint fetcher built on [SlideRule](https://slideruleearth.io)'s on-demand HDF5 subsetting, independent of the raster `search`/`download`/`mosaic` pipeline (no local asset cache, no shared config - one AOI/time-range request returns an already-subsetted GeoDataFrame, written to GeoParquet). No NASA Earthdata credentials required against SlideRule's public cluster.

### Fixed

- `download_items()` (`gfetch download`) could hang indefinitely on networks that require an HTTP(S) proxy for outbound access (e.g. many HPC compute nodes), even though `gfetch search` and plain `curl` worked fine on the same node. Root cause: `stac_asset`'s downloads go through a raw `aiohttp.ClientSession`, which - unlike `requests`/`curl` - silently ignores `HTTP_PROXY`/`HTTPS_PROXY` unless `trust_env=True` is set, and `stac_asset` exposes no way to set it. `download.py` now patches `stac_asset`'s session construction to default `trust_env=True`. Confirmed fixed on the reporting user's HPC cluster.
- Written Zarr stores appeared unreferenced to strict CF readers (e.g. GDAL's Zarr driver, so QGIS too) even though the `spatial_ref` coordinate itself carried a full CRS. Root cause: `odc.stac.load` links each band to its CRS coordinate via `.encoding["grid_mapping"]`, but `mask_clouds`/`composite`'s `.where`/`.median` calls drop `.encoding` from their output, so the link never reached `to_zarr`. xarray's own readers (and rioxarray/odc-geo) still resolved the CRS via a coordinate-scanning fallback, masking the gap. `write.py::write`/`prepare_template`/`write_region` now restore the link before every write.
- `gfetch mosaic` could fail deep inside `to_zarr` with a confusing `validate_grid_chunks_alignment` error - confirmed root cause: a `Config.chunks` YAML with an explicit `time` entry (e.g. a value left over from before `_DEFAULT_TIME_CHUNK` existed) defeats gfetch's own safe default, forcing `load()` to split the time axis into several chunks instead of one; `composite()`'s reduction then has to rechunk internally to consolidate them, which for a large enough zone was confirmed to also shrink the *spatial* (`x`/`y`) chunk size as a side effect, breaking alignment with the Zarr store's own chunk grid. `mosaic()` now overrides a non-default `time` chunk unconditionally (logging a warning), and still pins its output's `x`/`y` chunking explicitly as a second, independent safeguard, since nothing else guarantees `composite()`'s reduction (or a band's resampling to the common grid) preserves it for every variable. Also added `write.validate_chunks()`, called once per store before any patch is built, which fails fast with a clear error if an *already-existing* store's on-disk chunk grid doesn't match the current run's configuration (e.g. a store left over from before this fix), instead of only surfacing the same confusing error after a wasted patch computation.

### Changed

- `Config.zarr_path` is now a method taking a `CRS` (one output store per UTM zone) instead of a fixed property.
