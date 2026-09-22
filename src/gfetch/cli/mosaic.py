import logging
from pathlib import Path

import dask
import pystac
import xarray as xr
from odc.geo.geobox import GeoBox

from gfetch.cli.config import load
from gfetch.mosaic import group_by_utm_zone, patch_geoboxes, resolve_chunks, zone_geobox
from gfetch.mosaic import mosaic as build_mosaic
from gfetch.profiles import get_profile
from gfetch.utils.progress import count_bar, dask_progress, temporary_task
from gfetch.utils.system import available_cpus
from gfetch.write import (
    prepare_template,
    region_is_written,
    store_initialized,
    validate_chunks,
    write_region,
)

log = logging.getLogger(__name__)


def mosaic(config_path: Path, *, task_id: int = 0, n_tasks: int = 1) -> None:
    """
    Load, cloud-mask, composite, and write this job's items to a Zarr mosaic, one
    patch of native Zarr chunks at a time.

    Runs against the local cache if `gfetch download` has been run, otherwise loads
    directly from the items' remote hrefs (fine for a single internet-connected
    machine; a compute-only HPC node needs the local cache). Safe to resume after
    being killed/preempted: an already-written patch is skipped, and a partially
    written one is recomputed and rewritten as a whole, not patched up. Safe for
    several concurrent invocations to split the work via `task_id`/`n_tasks`, since
    every zone's output store is pre-planned into disjoint, chunk-aligned patches,
    so no two tasks ever touch the same one.

    Parameters
    ----------
    config_path : Path
        Path to the configuration YAML file.
    task_id : int
        This invocation's index among `n_tasks` concurrent invocations (e.g. a SLURM
        job array's `$SLURM_ARRAY_TASK_ID`). Defaults to 0.
    n_tasks : int
        Total number of concurrent invocations splitting this job's patches between
        them (e.g. a SLURM job array's `$SLURM_ARRAY_TASK_COUNT`). Defaults to 1: no
        splitting, this invocation does everything.
    """
    cfg = load(config_path)

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
    profile = get_profile(cfg.satellite)
    bands = list(cfg.bands) if cfg.bands else list(profile.default_bands)

    zones = group_by_utm_zone(items, cfg.aoi.bbox)
    log.info(f"Items span {len(zones)} UTM zone(s): {[crs.epsg for crs in zones]}")

    chunks = resolve_chunks(cfg.chunks)
    patch_shape = (cfg.patch_chunks * chunks["y"], cfg.patch_chunks * chunks["x"])

    def build(zone_items: list[pystac.Item], geobox: GeoBox) -> xr.Dataset:
        return build_mosaic(
            zone_items,
            geobox,
            bands,
            mask_band=profile.cloud_mask_band,
            mask_out=profile.cloud_mask_out,
            resampling=cfg.resampling,
            chunks=cfg.chunks,
        )

    patches: list[tuple[str, Path, list[pystac.Item], GeoBox, dict[str, slice]]] = []
    for crs, zone_items in zones.items():
        geobox = zone_geobox(crs, cfg.aoi.bbox, cfg.resolution)
        path = cfg.zarr_path(crs)
        # Building the template dataset means running the full mosaic() pipeline over
        # the whole zone's item list and geobox (odc-stac has to bin every item
        # against the zone's full tile grid to construct the graph) just to read off
        # its post-composite shape/dtype/coords: real, non-trivial work even though
        # `prepare_template`'s `compute=False` never reads or writes any pixel data.
        # Checking first avoids paying for that on every invocation once the template
        # already exists (the common case once a job is past its first run). Every
        # task still calls `prepare_template` independently and idempotently when it
        # does need to, so there's no ordering dependency between tasks.
        if not store_initialized(path):
            prepare_template(build(zone_items, geobox), path)
        # Fails fast, before any patch is built, if this store's on-disk chunk grid
        # doesn't match what this run's config expects - a store built by an earlier
        # run under different settings would otherwise only surface as a confusing
        # low-level error deep inside `write_region`, after paying for a patch build.
        validate_chunks(path, bands, {"y": chunks["y"], "x": chunks["x"]})
        for (iy, ix), patch_geobox, region in patch_geoboxes(geobox, patch_shape):
            label = f"EPSG:{crs.epsg} patch ({iy}, {ix})"
            patches.append((label, path, zone_items, patch_geobox, region))

    my_patches = patches[task_id::n_tasks]
    log.info(f"Task {task_id}/{n_tasks}: {len(my_patches)}/{len(patches)} patch(es) assigned")

    n_compute_workers = cfg.n_compute_workers or available_cpus()
    log.info(f"Using {n_compute_workers} dask worker thread(s)")

    with (
        count_bar() as progress,
        temporary_task(progress, "Computing patches", total=len(my_patches)) as patches_task,
        dask.config.set(num_workers=n_compute_workers),
    ):
        for label, path, zone_items, patch_geobox, region in my_patches:
            if region_is_written(path, region, bands):
                log.debug(f"{label}: already written, skipping")
            else:
                ds = build(zone_items, patch_geobox)
                with dask_progress(progress, label):
                    write_region(ds, path, region)
            progress.advance(patches_task)
