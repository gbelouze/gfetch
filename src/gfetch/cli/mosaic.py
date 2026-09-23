import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import dask
import pystac
import xarray as xr
from odc.geo.geobox import GeoBox
from zarr.storage import ZipStore

from gfetch.cli.config import load, resolve_bands, resolve_cloud_mask, resolve_compute_workers
from gfetch.finalize import packed_store_path
from gfetch.mosaic import (
    group_by_utm_zone,
    resolve_chunks,
    resolve_compute_chunks,
    resolve_shards,
    zone_geobox,
)
from gfetch.mosaic import mosaic as build_mosaic
from gfetch.utils.progress import count_bar, dask_progress, temporary_task
from gfetch.utils.tuning import WorkerTuner
from gfetch.write import (
    prepare_template,
    region_is_written,
    store_initialized,
    validate_chunks,
    write_region,
    write_regions,
)

log = logging.getLogger(__name__)


def mosaic(config_path: Path, satellite_key: str, *, task_id: int = 0, n_tasks: int = 1) -> None:
    """
    Load, cloud-mask, composite, and write this job's items to a Zarr mosaic, one
    shard at a time.

    Runs against the local cache if `gfetch download` has been run, otherwise loads
    directly from the items' remote hrefs (fine for a single internet-connected
    machine; a compute-only HPC node needs the local cache). Safe to resume after
    being killed/preempted: an already-written band of a shard is skipped, and an
    unwritten one is computed and written as a whole. Safe for several concurrent invocations to
    split the work via `task_id`/`n_tasks`, since each shard is its own file and is
    assigned to exactly one task.

    Parameters
    ----------
    config_path : Path
        Path to the configuration YAML file.
    satellite_key : str
        Which satellite to load from `config_path` - `'s1'`, `'s2'`, or a name under
        its `custom:` section. See `gfetch.cli.config.load`.
    task_id : int
        This invocation's index among `n_tasks` concurrent invocations (e.g. a SLURM
        job array's `$SLURM_ARRAY_TASK_ID`). Defaults to 0.
    n_tasks : int
        Total number of concurrent invocations splitting this job's shards between
        them (e.g. a SLURM job array's `$SLURM_ARRAY_TASK_COUNT`). Defaults to 1: no
        splitting, this invocation does everything.
    """
    cfg = load(config_path, satellite_key)

    if cfg.cached_items_path.exists():
        items_path = cfg.cached_items_path
    elif cfg.items_path.exists():
        items_path = cfg.items_path
        log.warning("No downloaded cache found, loading directly from remote hrefs.")
    else:
        log.error(f"{cfg.items_path} not found, run `gfetch search` first.")
        return

    items = list(pystac.ItemCollection.from_file(items_path))
    log.debug(f"Loaded {len(items)} item(s) from {items_path}")
    bands = resolve_bands(cfg)
    mask_band, mask_out = resolve_cloud_mask(cfg)

    zones = group_by_utm_zone(items, cfg.resolved_aoi.bbox)
    log.info(f"Items span {len(zones)} UTM zone(s): {[crs.epsg for crs in zones]}")

    chunks = resolve_chunks(cfg.chunks)
    shards = resolve_shards(cfg.shard_factor, chunks)
    compute_chunks = resolve_compute_chunks(chunks, cfg.compute_chunk_factor, shards)

    def build(
        zone_items: list[pystac.Item],
        geobox: GeoBox,
        *,
        log_footprint: bool = True,
        on_load: Callable[[xr.Dataset], object] | None = None,
    ) -> xr.Dataset:
        return build_mosaic(
            zone_items,
            geobox,
            bands,
            mask_band=mask_band,
            mask_out=mask_out,
            resampling=cfg.resampling,
            chunks=compute_chunks,
            log_footprint=log_footprint,
            on_load=on_load,
        )

    units: list[tuple[str, Path, list[pystac.Item], GeoBox, dict[str, slice]]] = []
    for crs, zone_items in zones.items():
        geobox = zone_geobox(crs, cfg.resolved_aoi.bbox, cfg.resolution)
        path = cfg.zarr_path(crs)
        packed = packed_store_path(path)

        if packed.exists():
            log.info(f"EPSG:{crs.epsg}: already packed into {packed}, skipping")
            # A packed zone is complete (`pack_store` only packs complete stores) and
            # its directory may be gone; its shards are still listed, from the zip,
            # so that every task assigns the same shards regardless of when packing
            # happened.
            zip_store = ZipStore(packed, mode="r")
            try:
                regions = write_regions(zip_store, bands[0])
            finally:
                zip_store.close()
        else:
            # Building the template dataset means running the full mosaic() pipeline
            # over the whole zone's item list and geobox (odc-stac has to bin every
            # item against the zone's full tile grid to construct the graph) just to
            # read off its post-composite shape/dtype/coords: real, non-trivial work
            # even though `prepare_template`'s `compute=False` never reads or writes
            # any pixel data. Checking first avoids paying for that on every
            # invocation once the template already exists (the common case once a job
            # is past its first run). Every task still calls `prepare_template`
            # independently and idempotently when it does need to, so there's no
            # ordering dependency between tasks.
            if not store_initialized(path):
                prepare_template(
                    build(zone_items, geobox, log_footprint=False),
                    path,
                    shards=shards,
                    chunks={"y": chunks["y"], "x": chunks["x"]},
                )

            # Fails fast, before any shard is built, if this store's on-disk chunk or
            # shard grid doesn't match what this run's config expects
            validate_chunks(path, bands, {"y": chunks["y"], "x": chunks["x"]}, shards)
            regions = write_regions(path, bands[0])

        for region in regions:
            label = f"EPSG:{crs.epsg} shard (y={region['y'].start}, x={region['x'].start})"
            unit_geobox = cast("GeoBox", geobox[region["y"], region["x"]])
            units.append((label, path, zone_items, unit_geobox, region))

    my_units = units[task_id::n_tasks]
    log.info(f"Task {task_id}/{n_tasks}: {len(my_units)}/{len(units)} shard(s) assigned")

    min_workers, max_workers = resolve_compute_workers(cfg)
    tuner = WorkerTuner(min_workers, max_workers)
    log.info(f"Using {min_workers}-{max_workers} dask worker thread(s)")

    with (
        count_bar() as progress,
        temporary_task(progress, "Computing shards", total=len(my_units)) as units_task,
    ):
        for label, path, zone_items, unit_geobox, region in my_units:
            # Each band's shard is its own file, written atomically, so a shard counts
            # as done once every band's file exists; otherwise all its bands are
            # recomputed and rewritten.
            if packed_store_path(path).exists() or region_is_written(path, region, bands):
                log.debug(f"{label}: already written, skipping")
            else:
                n_workers = tuner.next()
                loaded: list[xr.Dataset] = []
                with dask.config.set(num_workers=n_workers):
                    ds = build(zone_items, unit_geobox, on_load=loaded.append)
                    start = time.perf_counter()
                    with dask_progress(progress, label):
                        write_region(ds, path, region)
                tuner.observe(n_workers, time.perf_counter() - start, loaded[0].nbytes)
            progress.advance(units_task)
