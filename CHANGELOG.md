# Changelog

All notable changes to this project will be documented in this file.

This changelog should be updated with every pull request with some information about what has been changed. These changes can be added under a temporary title 'pre-release'.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Each release can have sections: "Added", "Changed", "Deprecated", "Removed", "Fixed" and "Security".

## pre-release

### Added

- Sharded Zarr output (`shard_factor:` config, chunks per shard along each side, default 32): one file per shard instead of per chunk. Each shard is also `mosaic`'s unit of work and resume, listed from the store's own shard grid (`gfetch.write.write_regions`). Reading a sharded store with GDAL/QGIS needs GDAL 3.13+.
- `compute_chunk_factor:` config (default 1): `mosaic` computes in dask chunks of that many store chunks per side, separately from the store's own `chunks`. Must divide `shard_factor`.
- `n_compute_workers` accepts a `[min, max]` range: `mosaic` then tunes the dask thread count per shard while it runs (`gfetch.utils.tuning.WorkerTuner`): min, max and middle first, then the best count predicted by fitting `a + b/w + c*w` to the observed seconds per byte processed, plus exploration noise.
- `benchmark/sharding_write.py` compares unsharded, sharded all-bands and sharded per-band writes.
- Project scaffold: `pyproject.toml` (Python 3.12, `ruff`/`pyrefly`/`pydoclint` config), `pre-commit` hooks, `AGENTS.md`.
- `search`, `download`, `mosaic`/`write` pipeline stages, a `sentinel-2` satellite profile, and a `cyclopts`/`omegaconf`-based CLI (`init`/`search`/`download`/`mosaic`), with tests for every stage.
- `mosaic.py::group_by_utm_zone`/`zone_geobox`: mosaic each UTM zone an AOI spans separately, instead of reprojecting everything into one zone chosen from the AOI's centroid. `gfetch mosaic` now writes one Zarr store per spanned zone (`mosaic_epsg<code>.zarr`).
- Logging: every stage (`search`/`download`/`mosaic`/`write`, plus their CLI wrappers and config loading) now reports progress at `INFO`/`DEBUG` as it works, instead of going quiet for the length of a long call - e.g. `search()` logs the total match count up front and one line per STAC API page fetched, so a large search no longer looks hung.
- `download_items()`: with a `progress` tracker passed in, each concurrently-downloading item now gets its own live byte-progress bar (via `stac_asset`'s per-asset message stream), alongside the existing overall "items downloaded" bar - not just one bar for the whole batch.
- `gfetch gedi` command and `gfetch.gedi.fetch_gedi_l2a()`: a standalone GEDI L2A vector-footprint fetcher built on [SlideRule](https://slideruleearth.io)'s on-demand HDF5 subsetting, independent of the raster `search`/`download`/`mosaic` pipeline (no local asset cache, no shared config - one AOI/time-range request returns an already-subsetted GeoDataFrame, written to GeoParquet). No NASA Earthdata credentials required against SlideRule's public cluster.
- `fetch_gedi_l2a()` (`gfetch gedi`) splits the AOI into tiles of at most 10km x 10km (`max_size_m`, via the new `split_bbox()`) and sends one SlideRule request per tile, since a single request over a large AOI often failed server-side. Footprints on a shared tile edge are kept only once. `gfetch gedi` shows a progress bar over the tiles.
- `gfetch gedi` is resumable: each tile's footprints are saved atomically under `<output>.tiles/` as soon as they're fetched (`fetch_gedi_l2a(tile_dir=...)`), a rerun reuses them, and the directory is removed once the final GeoParquet is written. Resuming with different request parameters raises `ValueError`.
- `gfetch.finalize.pack_store()`: packs a complete Zarr store into a single uncompressed `<store>.zip`, readable in place with `zarr.storage.ZipStore` or GDAL's `/vsizip/`, to cut inode usage on HPC filesystems. Atomic (temp file + rename) and idempotent; optionally removes the source store. `gfetch.finalize.remove_cache()` deletes the download cache and `cached_items.json` once every store is packed or complete. Exposed as `gfetch <satellite> pack [--remove-store]` and `gfetch <satellite> clean`; `mosaic` skips zones that are already packed.
- `gfetch.write.store_is_complete()`.
- `custom:` job config section: one or more arbitrary STAC satellite/source combinations, each requiring an explicit `collection` id and `bands` (no `gfetch.profiles`/`gfetch.sources` registry entry needed). Runs through `gfetch custom <verb> <name> <config>`.
- `countries:` job config field, as an alternative to `aoi:`: a list of country names (matched, with a typo suggestion, against the public `world-administrative-boundaries` dataset) resolved to the bounding box of their union. `search` additionally queries by the exact country polygon (`intersects=`) rather than just its bounding box.

### Fixed

- `search()` (`gfetch <satellite> search`) raises `ValueError` before searching when the time range overlaps a period Earth Search's `sentinel-2-c1-l2a` is known to be missing (Nov 2016 - Nov 2019 and 2022, pending ESA's baseline 05.00 reprocessing). It previously returned a small fraction of the real scenes without warning (45 instead of ~1,300 over a 230 km box in France for Jan-Sep 2022). A time range lying entirely within 2022 is instead searched against Earth Search's older `sentinel-2-l2a`, with a warning; one straddling 2022 and another year still raises.
- `gfetch mosaic` recomputed all-nodata patches on every resume: zarr-python skips writing chunks that are entirely fill value by default, so `region_is_written` never saw them as written. `write`/`write_region` now always write empty chunks.

- `download_items()` (`gfetch download`) could hang indefinitely on networks that require an HTTP(S) proxy for outbound access (e.g. many HPC compute nodes), even though `gfetch search` and plain `curl` worked fine on the same node. Root cause: `stac_asset`'s downloads go through a raw `aiohttp.ClientSession`, which - unlike `requests`/`curl` - silently ignores `HTTP_PROXY`/`HTTPS_PROXY` unless `trust_env=True` is set, and `stac_asset` exposes no way to set it. `download.py` now patches `stac_asset`'s session construction to default `trust_env=True`. Confirmed fixed on the reporting user's HPC cluster.
- Written Zarr stores appeared unreferenced to strict CF readers (e.g. GDAL's Zarr driver, so QGIS too) even though the `spatial_ref` coordinate itself carried a full CRS. Root cause: `odc.stac.load` links each band to its CRS coordinate via `.encoding["grid_mapping"]`, but `mask_clouds`/`composite`'s `.where`/`.median` calls drop `.encoding` from their output, so the link never reached `to_zarr`. xarray's own readers (and rioxarray/odc-geo) still resolved the CRS via a coordinate-scanning fallback, masking the gap. `write.py::write`/`prepare_template`/`write_region` now restore the link before every write.
- `gfetch mosaic` could fail deep inside `to_zarr` with a confusing `validate_grid_chunks_alignment` error - confirmed root cause: a `Config.chunks` YAML with an explicit `time` entry (e.g. a value left over from before `_DEFAULT_TIME_CHUNK` existed) defeats gfetch's own safe default, forcing `load()` to split the time axis into several chunks instead of one; `composite()`'s reduction then has to rechunk internally to consolidate them, which for a large enough zone was confirmed to also shrink the *spatial* (`x`/`y`) chunk size as a side effect, breaking alignment with the Zarr store's own chunk grid. `mosaic()` now overrides a non-default `time` chunk unconditionally (logging a warning), and still pins its output's `x`/`y` chunking explicitly as a second, independent safeguard, since nothing else guarantees `composite()`'s reduction (or a band's resampling to the common grid) preserves it for every variable. Also added `write.validate_chunks()`, called once per store before any patch is built, which fails fast with a clear error if an *already-existing* store's on-disk chunk grid doesn't match the current run's configuration (e.g. a store left over from before this fix), instead of only surfacing the same confusing error after a wasted patch computation.

### Changed

- `patch_chunks` is removed in favour of `shard_factor`: a config still setting it fails to load. `shard_factor: 1` keeps an unsharded store, compatible with stores written before.
- Default `chunks` is 256 px (was 2048), so the default 8192 px shard holds 32x32 chunks.
- Sentinel-2 cloud masking also masks SCL class 7 (unclassified, in practice largely low-probability cloud).
- `Config.zarr_path` is now a method taking a `CRS` (one output store per UTM zone) instead of a fixed property.
- Job configs are now unified: one YAML file per job, with generic top-level defaults overridable per satellite under reserved `s1`/`s2`/`gedi`/`custom` sections, instead of one flat file per satellite. The CLI follows: `gfetch <satellite> search|download|mosaic|pack|clean <config>` (`gfetch custom <verb> <name> <config>` for a `custom:` entry), `gfetch gedi <config>` unchanged.
